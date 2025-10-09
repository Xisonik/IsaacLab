import torch

class Memory_manager:
    def __init__(self, num_envs: int, embedding_size: int, action_size: int, history_length: int, device: torch.device):
        """
        Инициализация менеджера памяти для хранения эмбеддингов и действий как стека.

        Args:
            num_envs (int): Количество сред.
            embedding_size (int): Размер эмбеддинга (например, 512 для ResNet18).
            action_size (int): Размер действия (например, 2 для линейной и угловой скорости).
            history_length (int): Длина стека (n).
            device (torch.device): Устройство для хранения тензоров (CPU/GPU).
        """
        self.num_envs = num_envs
        self.embedding_size = embedding_size
        self.action_size = action_size
        self.history_length = history_length
        self.device = device
                
        # Создаем индексы для выборки: [0, k, 2k, ..., (m-1)*k]
        m = 4
        k = 4
        indices = torch.arange(0, m * k, k, device="cpu", dtype=torch.long)  # [m]
        # Ограничиваем индексы длиной стека
        self.indices = torch.clamp(indices, 0, self.history_length - 1)

        # Инициализация стека для эмбеддингов: [num_envs, history_length, embedding_size]
        self.embedding_history = torch.zeros((num_envs, history_length, embedding_size), device="cpu")
        self.action_history = torch.zeros((num_envs, history_length, action_size), device="cpu")
        # Флаг, указывающий, что стек еще не заполнен (для первой инициализации)
        self.is_initialized = False

    def update(self, embeddings: torch.Tensor, actions: torch.Tensor):
        """
        Обновление стека: новый эмбеддинг и действие добавляются в начало, старые сдвигаются вправо.

        Args:
            embeddings (torch.Tensor): Эмбеддинги текущего шага, форма [num_envs, embedding_size].
            actions (torch.Tensor): Действия текущего шага, форма [num_envs, action_size].
        """
        # Если стек не инициализирован, заполняем его первым эмбеддингом и действием
        if not self.is_initialized:
            self.embedding_history = embeddings.cpu().unsqueeze(1).expand(-1, self.history_length, -1)
            self.is_initialized = True
        # Сдвигаем историю вправо (удаляем самый старый элемент)
        # Создаем новую историю, объединяя новый эмбеддинг/действие с частью старой истории
        self.embedding_history = torch.cat([embeddings.cpu().unsqueeze(1), self.embedding_history[:, :-1]], dim=1)
        self.action_history = torch.cat([actions.cpu().unsqueeze(1), self.action_history[:, :-1]], dim=1)

    def get_observations(self, m=4, k=4) -> torch.Tensor:
        """
        Получение m эмбеддингов и m действий с частотой k, начиная с последнего (индекс 0).

        Args:
            m (int): Количество возвращаемых эмбеддингов и действий.
            k (int): Шаг выборки (частота).

        Returns:
            torch.Tensor: Тензор формы [num_envs, m * (embedding_size + action_size)] с m эмбеддингами и действиями.
        """
        # Выбираем эмбеддинги и действия для всех сред одновременно
        selected_embeddings = self.embedding_history[:, self.indices].to(self.device)  # [num_envs, m, embedding_size]
        selected_actions = self.action_history[:, self.indices].to(self.device)  # [num_envs, m, action_size]

        # Объединяем эмбеддинги и действия по последней размерности
        combined = torch.cat([selected_embeddings, selected_actions], dim=-1)  # [num_envs, m, embedding_size + action_size]

        # Разворачиваем в плоский вектор
        output = combined.view(self.num_envs, -1).to(self.device)  # [num_envs, m * (embedding_size + action_size)]

        return output

    def reset(self):
        """
        Сброс стека для указанных сред или всех сред.
        """
        self.is_initialized = False

class PathTracker:
    def __init__(self, num_envs: int, T_max: int = 256, device: str = "cuda", pos_dim: int = 2):
        """
        Батчевый менеджер траекторий и управляющих воздействий.

        Args:
            num_envs (int): количество сред
            T_max (int): максимальная длина траектории
            device (str): устройство
            pos_dim (int): размерность позиции (обычно 2 или 3)
        """
        self.num_envs = num_envs
        self.T_max = T_max
        self.device = device
        self.pos_dim = pos_dim

        # [num_envs, T_max, pos_dim]
        self.positions = torch.zeros((num_envs, T_max, pos_dim), device=device, dtype=torch.float32)
        # [num_envs, T_max, 2] (lin, ang)
        self.velocities = torch.zeros((num_envs, T_max, 2), device=device, dtype=torch.float32)
        # Счётчик длины траектории для каждой среды
        self.lengths = torch.zeros(num_envs, device=device, dtype=torch.long)

    @torch.no_grad()
    def add_step(self, env_ids: torch.Tensor, positions: torch.Tensor, velocities: torch.Tensor):
        """
        Добавить позиции и управляющие воздействия в батчевом режиме.
        Args:
            env_ids (torch.Tensor): [K]
            positions (torch.Tensor): [K, pos_dim]
            velocities (torch.Tensor): [K, 2]
        """
        env_ids = env_ids.to(self.device)
        idxs = self.lengths[env_ids]  # текущие индексы вставки
        for i, env_id in enumerate(env_ids):
            if idxs[i] < self.T_max:
                self.positions[env_id, idxs[i]] = positions[i].to(self.device)
                self.velocities[env_id, idxs[i]] = velocities[i].to(self.device)
                self.lengths[env_id] += 1  # увеличиваем счётчик

    def reset(self, env_ids: torch.Tensor):
        """
        Очистить траектории и управляющие воздействия для указанных сред.
        """
        env_ids = env_ids.to(self.device)
        self.positions[env_ids] = 0.0
        self.velocities[env_ids] = 0.0
        self.lengths[env_ids] = 0

    @torch.no_grad()
    def compute_path_lengths(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Подсчитать длину пути для агентов (евклидова сумма).
        Returns: [K]
        """
        env_ids = env_ids.to(self.device)
        pos = self.positions[env_ids]  # [K, T_max, pos_dim]
        L = self.lengths[env_ids]      # [K]

        # Считаем диффы вдоль оси T
        diffs = pos[:, 1:] - pos[:, :-1]       # [K, T_max-1, pos_dim]
        dist = torch.norm(diffs, dim=-1)       # [K, T_max-1]

        # Маска по длине
        mask = torch.arange(self.T_max-1, device=self.device).unsqueeze(0) < (L.unsqueeze(1)-1)
        dist = dist * mask

        return dist.sum(dim=1)

    def get_paths(self, env_ids: torch.Tensor):
        """
        Вернуть пути агентов (с обрезкой до длины).
        """
        env_ids = env_ids.to(self.device)
        out = {}
        for i, env_id in enumerate(env_ids):
            L = self.lengths[env_id].item()
            out[env_id.item()] = self.positions[env_id, :L]
        return out

    def get_velocities(self, env_ids: torch.Tensor):
        """
        Вернуть последовательности управляющих воздействий.
        """
        env_ids = env_ids.to(self.device)
        out = {}
        for i, env_id in enumerate(env_ids):
            L = self.lengths[env_id].item()
            out[env_id.item()] = self.velocities[env_id, :L]
        return out

    @torch.no_grad()
    def compute_jerk(self, env_ids: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """
        Подсчитать количество резких скачков скоростей.
        Args:
            threshold (float): порог
        Returns: [K] количество скачков
        """
        env_ids = env_ids.to(self.device)
        vels = self.velocities[env_ids]  # [K, T_max, 2]
        L = self.lengths[env_ids]

        diffs = torch.norm(vels[:, 1:] - vels[:, :-1], dim=-1)  # [K, T_max-1]

        mask = torch.arange(self.T_max-1, device=self.device).unsqueeze(0) < (L.unsqueeze(1)-1)
        diffs = diffs * mask

        return (diffs > threshold).sum(dim=1).float()
