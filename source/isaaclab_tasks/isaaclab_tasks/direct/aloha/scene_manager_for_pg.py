# ЗАМЕНИТЬ ФАЙЛ: scene_manager.py

import torch
import math
import random
import json
# from .graph_manager import ObstacleGraph
# from .placement_strategies import PlacementStrategy, GridPlacement, OnSurfacePlacement

import importlib.util
def import_class_from_path(module_path, class_name):
    print(f"[DEBUG] Importing class '{class_name}' from module: {module_path}")
    spec = importlib.util.spec_from_file_location("custom_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_obj = getattr(module, class_name)
    print(f"[DEBUG] Successfully imported class: {class_obj}")
    return class_obj

module_path = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/graph_manager.py"
ObstacleGraph = import_class_from_path(module_path, "ObstacleGraph")
module_path = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/placement_strategies_for_pg.py"
PlacementStrategy = import_class_from_path(module_path, "PlacementStrategy")
GridPlacement = import_class_from_path(module_path, "GridPlacement")
OnSurfacePlacement = import_class_from_path(module_path, "OnSurfacePlacement")

class SceneManager:
    def __init__(self, num_envs: int, config_path: str, device: str):
        self.num_envs = num_envs
        self.device = device
        with open(config_path, 'r') as f:
            self.config = json.load(f)['objects']

        self.object_indices = {}
        self.placement_strategies = {}
        self.graphs = [ObstacleGraph(self.config, device) for _ in range(num_envs)]
        self._initialize_strategies()

        # Параметры робота и комнаты
        self.robot_radius = 0.5
        self.room_bounds = {'x_min': -4, 'x_max': 4.0, 'y_min': -3.0, 'y_max': 3.0}
        self.robot_placement_bounds = {
            'x_min': self.room_bounds['x_min'] + self.robot_radius,
            'x_max': self.room_bounds['x_max'] - self.robot_radius,
            'y_min': self.room_bounds['y_min'] + self.robot_radius,
            'y_max': self.room_bounds['y_max'] - self.robot_radius
        }
        # Для дискретизации углов
        self.n_angles = 36
        self.angle_step = math.pi / self.n_angles
        self.discrete_angles = torch.arange(0, 2 * math.pi, self.angle_step, device=self.device)
        self.goal_positions = torch.zeros((num_envs,3),device=self.device)


    def _initialize_strategies(self):
        """Создает и настраивает стратегии размещения на основе конфига."""
        start_idx = 0
        for obj_cfg in self.config:
            name = obj_cfg['name']
            count = obj_cfg['count']
            self.object_indices[name] = list(range(start_idx, start_idx + count))
            start_idx += count

            placement_cfg = obj_cfg.get('placement')
            if not placement_cfg: continue

            placement_cfg = placement_cfg[0]
            strategy_type = placement_cfg['strategy']
            if strategy_type == 'grid':
                print(placement_cfg['grid_coordinates'])
                self.placement_strategies[name] = GridPlacement(
                    self.device, placement_cfg['grid_coordinates']
                )
            elif strategy_type == 'on_surface':
                surface_types = placement_cfg['surface_types']
                # surface_indices = self.object_indices[placement_cfg['surface_object_name']]
                self.placement_strategies[name] = OnSurfacePlacement(
                    self.device, surface_types, placement_cfg['margin']
                )

    def randomize_scene(self, env_ids: torch.Tensor, mess: bool = False, use_obstacles: bool = False, all_defoult = False):
        """
        По-настоящему унифицированная, иерархическая рандомизация сцены.
        Логика оперирует исключительно ТИПАМИ объектов, а не их именами.
        """
        from collections import defaultdict
        for env_id in env_ids:
            graph = self.graphs[env_id.item()]
            # 1. Сброс состояния графа
            graph.active[:] = False
            graph.positions = graph.default_positions.clone()
            # print("def pos ", graph.default_positions)
            graph.on_surface_idx[:] = -1
            graph.surface_level[:] = 0
            if all_defoult:
                pass
            # 2. Сбор всех доступных объектов по их РОЛЯМ (типам)
            all_providers = graph.get_nodes_by_type("surface_provider")
            all_surface_only_objects = graph.get_nodes_by_type("surface_only")
            # print("all_providers :", all_providers)
            # print("all_surface_only_objects :", all_surface_only_objects)
            # Находим препятствия, которые могут стоять на полу
            all_floor_obstacles = [
                i for i in graph.get_nodes_by_type("movable_obstacle")
                if i not in all_surface_only_objects
            ]
            # print("all_floor_obstacles :", all_floor_obstacles)
            # 3. ОПРЕДЕЛЕНИЕ КОЛИЧЕСТВА НА ОСНОВЕ РОЛЕЙ (логика "задом наперед")
            
            # Шаг А: Сколько объектов, которые могут быть только на поверхностях, мы хотим разместить?
            # Сюда входят и цели (possible_goal), и просто декор (stuff), если они surface_only.
            # Для простоты и для гарантии наличия цели, будем размещать хотя бы один такой объект.
            num_surface_only_to_place = random.randint(1, len(all_surface_only_objects)) if len(all_surface_only_objects) > 0 else 0
            # Шаг Б: Сколько поверхностей нам нужно под эти объекты?
            min_providers_needed = num_surface_only_to_place
            num_providers_to_place = 0
            if all_providers:
                max_providers = len(all_providers)
                num_providers_to_place = random.randint(min_providers_needed, max_providers) if min_providers_needed <= max_providers else max_providers

            # Шаг В: Сколько объектов будет стоять на полу?
            num_floor_obstacles_to_place = random.randint(0, len(all_floor_obstacles)) if all_floor_obstacles and use_obstacles else 0

            # 4. ВЫБОР КОНКРЕТНЫХ ЭКЗЕМПЛЯРОВ И ИХ РАЗМЕЩЕНИЕ
            
            # Выбираем случайные экземпляры провайдеров для активации
            providers_to_place = random.sample(all_providers, num_providers_to_place)
            
            # Выбираем случайные экземпляры surface-only объектов для активации
            surface_only_to_place = random.sample(all_surface_only_objects, num_surface_only_to_place)
            # Выбираем случайные экземпляры напольных препятствий для активации
            floor_obstacles_to_place = random.sample(all_floor_obstacles, num_floor_obstacles_to_place)

            # Собираем все объекты, которые нужно разместить, в один список
            all_objects_to_place = providers_to_place + surface_only_to_place + floor_obstacles_to_place
            # Группируем их по имени, чтобы применить правильную стратегию
            grouped_by_name = defaultdict(list)
            
            for node_idx in all_objects_to_place:
                name = graph.graph.nodes[node_idx]['name']
                grouped_by_name[name].append(node_idx)
            # Применяем стратегии в правильном порядке: сначала поверхности, потом все остальное
            placement_order = ["surface_provider", "surface_only", "movable_obstacle"]
            
            for p_type in placement_order:
                for name, indices in grouped_by_name.items():
                    # Проверяем, относится ли группа объектов к текущему этапу размещения
                    node_types = graph.graph.nodes[indices[0]]['types']
                    if p_type in node_types:
                        strategy = self.placement_strategies.get(name)
                        if strategy:
                            strategy.apply(graph, indices, len(indices), mess)

                # 5. Финальное обновление состояния графа
                graph.update_graph_state()
            # graph.print_graph_info(f"______END ________")
        self.chose_active_goal_state(env_ids=env_ids)

    def chose_active_goal_state(self, env_ids: torch.Tensor):
        """Находит случайную активную цель для каждой среды и возвращает ее локальную позицию."""
        goal_positions = torch.zeros(len(env_ids), 3, device=self.device)
        
        for i, env_id in enumerate(env_ids):
            graph = self.graphs[env_id.item()]
            
            # ИЗМЕНЕНИЕ: Используем новый метод get_nodes_by_type, который ищет строку в множестве типов.
            
            active_goal_indices = graph.get_nodes_by_type("possible_goal", only_active=True)
            # print("chose_active_goal_state: ", env_id, active_goal_indices)
            if active_goal_indices:
                # Логика выбора цели остается той же
                chosen_idx = random.choice(active_goal_indices)
                goal_positions[i] = graph.positions[chosen_idx]
            else:
                # Аварийный случай, если нет активных целей
                print(f"[WARNING] Env {env_id.item()}: No active goals found. Using fallback position.")
                goal_positions[i] = torch.tensor([-4.5, 0.0, 0.75], device=self.device)
        self.goal_positions[env_ids] = goal_positions

    def get_active_goal_state(self, env_ids: torch.Tensor):
        return self.goal_positions[env_ids]

    def place_robot_for_goal(self, env_ids: torch.Tensor, mean_dist: float, min_dist: float, max_dist: float, angle_error: float, max_attempts: int = 50):
        """
        Размещает робота на полу (уровень 0) относительно цели,
        проверяя коллизии только с объектами на полу.
        """
        goal_positions = self.goal_positions[env_ids]
        num_envs = len(env_ids)
        final_robot_positions = torch.zeros(num_envs, 2, device=self.device)
        valid_positions_found = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        # Шаг 1: Собираем препятствия на ПОЛУ (surface_idx == -1)
        floor_obstacles_per_env = []
        for env_id in env_ids:
            graph = self.graphs[env_id.item()]
            floor_nodes = [
                (data['position'][:2], data['radius']) for i, data in graph.graph.nodes(data=True)
                if data['active'] and data['on_surface_idx'] == -1 and data['radius'] > 0
            ]
            floor_obstacles_per_env.append(floor_nodes)
        
        # ... (код для создания тензоров all_obstacle_pos, all_obstacle_radii, obstacle_valid_mask
        # ... остается таким же, как в моем предыдущем ответе, он уже векторизован и корректен)
        max_obstacles = max(len(obs) for obs in floor_obstacles_per_env) if floor_obstacles_per_env else 0
        all_obstacle_pos = torch.zeros(num_envs, max_obstacles, 2, device=self.device)
        all_obstacle_radii = torch.zeros(num_envs, max_obstacles, device=self.device)
        obstacle_valid_mask = torch.zeros(num_envs, max_obstacles, dtype=torch.bool, device=self.device)
        for i in range(num_envs):
            if floor_obstacles_per_env[i]:
                num_obs = len(floor_obstacles_per_env[i])
                positions, radii = zip(*floor_obstacles_per_env[i])
                all_obstacle_pos[i, :num_obs] = torch.tensor(positions, device=self.device)
                all_obstacle_radii[i, :num_obs] = torch.tensor(radii, device=self.device)
                obstacle_valid_mask[i, :num_obs] = True
        # print("all_obstacle_pos", all_obstacle_pos)
        # Шаг 2: Итеративный поиск валидной позиции
        # Генерируем все возможные кандидаты сразу для всех сред
        # [num_envs, num_angles, 2]
        std_dev = mean_dist*0.1
        base = 1.3
        radii_not_clamp = torch.randn(num_envs) * std_dev + mean_dist + base
        radii = torch.clamp(radii_not_clamp, min=min_dist, max=max_dist)
        candidate_vectors = torch.stack([torch.cos(self.discrete_angles), torch.sin(self.discrete_angles)], dim=1) # [num_angles, 2]
        candidates = candidates = goal_positions[:, None, :2].to(self.device) + radii[:, None, None].to(self.device) * candidate_vectors[None, :, :].to(self.device)
        # Проверка 1: Границы комнаты
        bounds = self.robot_placement_bounds
        in_bounds_mask = (candidates[..., 0] >= bounds['x_min']) & (candidates[..., 0] <= bounds['x_max']) & \
                        (candidates[..., 1] >= bounds['y_min']) & (candidates[..., 1] <= bounds['y_max'])
        # Проверка 2: Коллизии с препятствиями на полу
        dists = torch.norm(candidates.unsqueeze(2) - all_obstacle_pos.unsqueeze(1), dim=3) # [envs, angles, obs, 2] -> [envs, angles, obs]
        required_dists = self.robot_radius + all_obstacle_radii.unsqueeze(1)
        collisions = (dists < required_dists) & obstacle_valid_mask.unsqueeze(1)
        no_collision_mask = ~torch.any(collisions, dim=2)
        
        # Финальная маска валидных углов
        valid_angle_mask = in_bounds_mask & no_collision_mask
        # Шаг 3: Выбор позиции и итеративное выталкивание
        for i in range(num_envs):
            valid_indices = torch.where(valid_angle_mask[i])[0]
            if valid_indices.numel() > 0:
                has_collisions = True
                while has_collisions:
                    chosen_angle_idx = valid_indices[torch.randint(0, valid_indices.numel(), (1,))]
                    pos = candidates[i, chosen_angle_idx].squeeze()
                    
                    # Итеративное выталкивание (упрощенная версия для примера)
                    # В полноценной реализации здесь должен быть цикл, как в вашем исходном коде

                    pos, has_collisions = self._resolve_collisions_iteratively(pos, all_obstacle_pos[i], all_obstacle_radii[i], obstacle_valid_mask[i])
                    final_robot_positions[i] = pos
            else:
                # Fallback, если не найдено ни одного валидного угла
                print("[ OH NO ]not valid angle")
                print("in_bounds_mask: ", in_bounds_mask)
                print("in_bounds_mask: ", no_collision_mask)
                final_robot_positions[i] = goal_positions[i, :2] + torch.tensor([max_dist, 0.0], device=self.device)
        # Шаг 4: Вычисление ориентации
        robot_quats = torch.zeros(num_envs, 4, device=self.device)
        direction_to_goal = goal_positions[:, :2] - final_robot_positions
        base_yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
        error = (torch.rand(num_envs, device=self.device) - 0.5) * 2 * angle_error
        final_yaw = base_yaw + error
        robot_quats[:, 0] = torch.cos(final_yaw / 2.0)
        robot_quats[:, 3] = torch.sin(final_yaw / 2.0)
        # final_robot_positions = torch.zeros(num_envs, 2, device=self.device)
        return final_robot_positions, robot_quats
    
    def _resolve_collisions_iteratively(self, robot_pos, obs_pos, obs_radii, obs_mask, max_iter=15, safety_margin=0.1):
        """Итеративно отодвигает робота от препятствий, проверяя границы области."""
        # Вычисляем начальные расстояния до препятствий
        robot_pos = self._clamp_to_room_bounds(robot_pos)
        dists = torch.norm(robot_pos - obs_pos, dim=1)
        required = self.robot_radius + obs_radii + safety_margin
        collisions = (dists < required) & obs_mask
        step = 0
        istep = 0
        pos_start = robot_pos
        while torch.any(collisions):
            # Пересчитываем расстояния и коллизии
            dists = torch.norm(robot_pos - obs_pos, dim=1)
            required = self.robot_radius + obs_radii + safety_margin
            
            collisions = (dists < required) & obs_mask
            
            # Проверяем, нет ли коллизий
            if not torch.any(collisions):
                # Проверяем границы комнаты
                # print("rp: ", robot_pos)
                return robot_pos, False  # Нет коллизий, позиция скорректирована

            # Находим ближайшее препятствие с коллизией
            colliding_dists = dists.clone()
            colliding_dists[~collisions] = float('inf')
            closest_idx = torch.argmin(colliding_dists)
            old_robot_pos = robot_pos
            # Вектор от центра препятствия к роботу
            vec_to_robot = robot_pos - obs_pos[closest_idx]
            vec_norm = torch.norm(vec_to_robot)
            
            # Вычисляем расстояние для сдвига
            move_dist = required[closest_idx] - vec_norm
            
            # Сдвигаем робота
            if vec_norm > 1e-6 and step < 3:
                robot_pos += (vec_to_robot / vec_norm) * move_dist
            else:  # Робот в центре препятствия — случайный сдвиг
                robot_pos += torch.randn(2, device=self.device) * move_dist
            robot_pos = self._clamp_to_room_bounds(robot_pos)
            step += 1
            if step > max_iter:
                step = 0
                istep += 1
                print(f"[DEBUG] COLLISION IN _resolve_collisions_iteratively after {istep} iterations")
                print(f"old robot pos: {old_robot_pos} obs: {obs_pos}")
                print(f"dist for obs: {dists} required: {required}")



                return robot_pos, True  # Нет коллизий, позиция скорректирована
        return robot_pos, False  # Нет коллизий, позиция скорректирована
        
    def _clamp_to_room_bounds(self, robot_pos):
        """Ограничивает позицию робота границами комнаты."""
        bounds = self.robot_placement_bounds
        x = torch.clamp(robot_pos[0], bounds['x_min'], bounds['x_max'])
        y = torch.clamp(robot_pos[1], bounds['y_min'], bounds['y_max'])
        return torch.tensor([x, y], device=self.device)
    
    def get_selected_indices(self, env_id: int | torch.Tensor) -> list[int] | None:
        """
        Возвращает отсортированный список индексов активных препятствий (тип 'movable_obstacle') для заданной среды.
        Аналог старого метода для совместимости с path_manager.
        
        Args:
            env_id: Индекс среды (int или torch.Tensor).
        
        Returns:
            list[int]: Sorted indices активных movable_obstacle, или None если ошибка.
        """
        if torch.is_tensor(env_id):
            env_id = env_id.item()
        
        graph = self.graphs[env_id]
        active_obstacles = graph.get_nodes_by_type("movable_obstacle", only_active=True)
        
        if not active_obstacles:
            print(f"[WARNING] No active movable_obstacles in env {env_id}")
            return None
        
        return sorted(active_obstacles)
    
    def set_obstacle_positions(self, env_ids: torch.Tensor, positions: list[list[float]]):
        """
        Детерминировано устанавливает позиции для активных препятствий (movable_obstacle).
        Активирует первые len(positions) movable nodes, устанавливает их позиции, остальным — default и inactive.
        USD выбирается автоматически (как в randomize).
        
        Args:
            env_ids: Индексы сред.
            positions: List of [x,y,z] для каждого активного препятствия (len <= count movable).
        """
        # movable_nodes = self.get_nodes_by_type("movable_obstacle")  # Через graph.get_nodes_by_type, но self нет graph, так что:
        # Найдем movable indices из object_indices
        movable_indices = self.object_indices.get('chair', [])  # По name, предполагая 'chair' — movable
        if len(positions) > len(movable_indices):
            raise ValueError(f"Too many positions: {len(positions)} > movable count {len(movable_indices)}")
        
        for env_id in env_ids:
            graph = self.graphs[env_id.item()]
            graph.active[:] = False
            graph.positions = graph.default_positions.clone()
            graph.on_surface_idx[:] = -1
            graph.surface_level[:] = 0
            
            # Активируем и устанавливаем первые len(positions)
            for i, pos in enumerate(positions):
                node_idx = movable_indices[i]
                graph.positions[node_idx] = torch.tensor(pos, device=self.device)
                graph.active[node_idx] = True
                graph.on_surface_idx[node_idx] = -1  # На полу
                graph.surface_level[node_idx] = 0
            
            graph.update_graph_state()  # Синхронизируем NetworkX

    def get_active_obstacle_positions(self, env_id: int | torch.Tensor) -> list[tuple[float, float, float]]:
        """
        Возвращает отсортированный список позиций активных movable_obstacle для env_id.
        Positions rounded to 1 decimal для matching с generator.
        """
        if torch.is_tensor(env_id):
            env_id = env_id.item()
        graph = self.graphs[env_id]
        obst_indices = graph.get_nodes_by_type("movable_obstacle", only_active=True)
        positions = [(round(p[0].item(), 1), round(p[1].item(), 1), round(p[2].item(), 1)) for p in graph.positions[obst_indices]]
        return sorted(positions)

    def set_goal_position(self, env_ids: torch.Tensor, position: list[float]):
        """
        Детерминировано устанавливает позицию цели (possible_goal).
        Если surface_only, размещает на surface_provider (первом) с z=height_surface + half_size_goal_z, x/y = position (игнор margin для фикса).
        
        Args:
            env_ids: Индексы сред.
            position: [x,y,z] для цели (z игнорируется если on_surface).
        """
        # goal_nodes = self.get_nodes_by_type("possible_goal")  # Через graph, но аналогично
        goal_idx = self.object_indices.get('bowl', [])[0]  # Предполагая один bowl
        
        surface_idx = None
        for obj_cfg in self.config:
            if "possible_goal" in obj_cfg['type'] and "surface_only" in obj_cfg['type']:
                # Находим surface_provider index (active)
                surface_nodes = self.graphs[0].get_nodes_by_type("surface_provider", only_active=False)
                if surface_nodes:
                    surface_idx = surface_nodes[0]  # Первый активный (table)
        
        for env_id in env_ids:
            graph = self.graphs[env_id.item()]
            pos_tensor = torch.tensor(position, device=self.device)
            
            if surface_idx is not None:
                # On surface: z = surface_z + surface_size_z/2 + goal_size_z/2
                surf_pos = graph.positions[surface_idx]
                surf_size = graph.sizes[surface_idx]
                goal_size = graph.sizes[goal_idx]
                # TODO CHANGE FIXED  surf_pos[2] + surf_size[2]/2 + goal_size[2]/2
                pos_tensor = torch.tensor([position[0], position[1], 0.75], device=self.device )  #(pos_tensor, 0.75))
                graph.on_surface_idx[goal_idx] = surface_idx
                graph.surface_level[goal_idx] = graph.surface_level[surface_idx] + 1
            
            graph.positions[goal_idx] = pos_tensor
            graph.active[goal_idx] = True
            self.goal_positions[env_id] = pos_tensor  # Обновляем self.goal_positions
            
            graph.update_graph_state()