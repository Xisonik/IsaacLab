import torch
import math
import random
import json
from collections import defaultdict
from tabulate import tabulate
import importlib.util
# Импортируем обновленные, векторизованные стратегии
# from .placement_strategies import PlacementStrategy, GridPlacement, OnSurfacePlacement # Эти классы остаются как в предыдущем ответе
def import_class_from_path(module_path, class_name):
    spec = importlib.util.spec_from_file_location("custom_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_obj = getattr(module, class_name)
    return class_obj

module_path = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/placement_strategies.py"
PlacementStrategy = import_class_from_path(module_path, "PlacementStrategy")
GridPlacement = import_class_from_path(module_path, "GridPlacement")
OnSurfacePlacement = import_class_from_path(module_path, "OnSurfacePlacement")

class SceneManager:
    def __init__(self, num_envs: int, config_path: str, device: str):
        self.num_envs = num_envs
        self.device = device
        with open(config_path, 'r') as f:
            raw = json.load(f)
        self.raw_config = raw
        self.config = raw['objects']
        self.type_placements_cfg = raw.get('type_placements', {})

        self.colors_dict = {
                        "green": [0.0, 1.0, 0.0],
                        "blue": [0.0, 0.0, 1.0],
                        "yellow": [1.0, 1.0, 0.0],
                        "gray": [0.5, 0.5, 0.5],
                        "red": [1.0, 0.0, 0.0]
                    }
        # --- Начало: Векторизованная структура данных ---
        self.num_total_objects = sum(obj['count'] for obj in self.config)
        self.object_ids = torch.zeros(1, self.num_total_objects, device=self.device)
        # Словари для быстрого доступа к метаданным
        self.object_map = {} # {name: {'indices': tensor, 'types': set, 'count': int}}
        self.type_map = defaultdict(list) # {type_str: [indices...]}
        
        # Глобальные тензоры состояний
        self.positions = torch.zeros(self.num_envs, self.num_total_objects, 3, device=self.device)
        self.sizes = torch.zeros(1, self.num_total_objects, 3, device=self.device)
        self.radii = torch.zeros(1, self.num_total_objects, device=self.device)
        self.colors = torch.ones(1, self.num_total_objects, 3, device=self.device)  # По умолчанию белый для не-changeable
        self.names = [] # Список имен для print_graph_info
        self.active = torch.zeros(self.num_envs, self.num_total_objects, dtype=torch.bool, device=self.device)
        self.on_surface_idx = torch.full((self.num_envs, self.num_total_objects), -1, dtype=torch.long, device=self.device)
        self.surface_level = torch.zeros(self.num_envs, self.num_total_objects, dtype=torch.long, device=self.device)
        
        self._initialize_object_data()
        self.default_positions = self.positions.clone()
        # --- Конец: Векторизованная структура данных ---

        self.placement_strategies = self._initialize_strategies()

        self.robot_radius = 0.5
        self.room_bounds = {'x_min': -1, 'x_max': 1, 'y_min': -1, 'y_max': 1}
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)

        n_angles = 36
        angle_step = 2 * math.pi / n_angles
        self.discrete_angles = torch.arange(0, 2 * math.pi, angle_step, device=self.device)
        self.candidate_vectors = torch.stack([torch.cos(self.discrete_angles), torch.sin(self.discrete_angles)], dim=1)
        # Assign object IDs based on name

    
    def update_prims(self):
        pass
    
    def get_scene_data_dict(self):
        return {
            "positions": self.positions,
            "sizes": self.sizes.expand(self.num_envs, -1, -1),
            "radii": self.radii.expand(self.num_envs, -1),
            "active": self.active,
            "on_surface_idx": self.on_surface_idx,
            "surface_level": self.surface_level,
            "names": self.names,              # <— добавить
        }


    
    def apply_fixed_positions(self, env_ids: torch.Tensor, positions_config: list[dict]):
        """
        positions_config: список словарей по числу сред.
        Каждый словарь: { "chair": [[x,y,z], ...], "table": [...], ... }
        """
        self.active[env_ids] = False
        self.positions[env_ids] = self.default_positions[env_ids]
        self.on_surface_idx[env_ids] = -1
        self.surface_level[env_ids] = 0
        scene_data = self.get_scene_data_dict()
        for env_id in env_ids:
            env_dict = positions_config[env_id.item()]
            for obj_name, pos_list in env_dict.items():
                if obj_name not in self.object_map:
                    continue
                indices = self.object_map[obj_name]["indices"]
                # print(indices)
                for i, pos in enumerate(pos_list):
                    if i >= len(indices):
                        print("errror")
                        break
                    scene_data["positions"][env_id.item(), indices[i]] = torch.tensor(pos, device=self.device)
                    scene_data["active"][env_id.item(), indices[i]] = True
                    scene_data["on_surface_idx"][env_id.item(), indices[i]] = -1
                    scene_data["surface_level"][env_id.item(), indices[i]] = 0

        # for i in env_ids:
        #     self.print_graph_info(i)
        self.chose_active_goal_state(env_ids)


    def _initialize_object_data(self):
        """Заполняет метаданные об объектах и их начальные/дефолтные состояния."""
        start_idx = 0
        
        # Создаем временный тензор для дефолтных позиций
        default_pos_tensor = torch.zeros(1, self.num_total_objects, 3, device=self.device)
        
        # --- Начало: Логика создания "кладбища" ---
        graveyard_start_x = -8.0
        graveyard_start_y = 6.0
        spacing = 1.1 # Расстояние между объектами на кладбище
        max_per_row = 14 # Сколько объектов в ряду на кладбище

        for i in range(self.num_total_objects):
            row = i // max_per_row
            col = i % max_per_row
            default_pos_tensor[0, i, 0] = graveyard_start_x + col * spacing
            default_pos_tensor[0, i, 1] = graveyard_start_y + row * spacing
            default_pos_tensor[0, i, 2] = 0.0
        # --- Конец: Логика создания "кладбища" ---

        for obj_cfg in self.config:
            name = obj_cfg['name']
            count = obj_cfg['count']
            indices = torch.arange(start_idx, start_idx + count, device=self.device, dtype=torch.long) # indeces of objects in general pull (for table 0,1,2)
            types = set(obj_cfg['type'])

            if "changeable_color" in types:
                colors_dict = self.colors_dict
                color_names = list(colors_dict.keys())
                for idx in indices:
                    color_name = random.choice(color_names)
                    self.colors[0, idx] = torch.tensor(colors_dict[color_name], device=self.device)

            self.object_map[name] = {'indices': indices, 'types': types, 'count': count}
            for type_str in types:
                self.type_map[type_str].extend(indices.tolist())
            
            self.names.extend([f"{name}_{i}" for i in range(count)])
            
            size_tensor = torch.tensor(obj_cfg['size'], device=self.device)
            self.sizes[0, indices] = size_tensor
            self.radii[0, indices] = torch.norm(size_tensor[:2] / 2)
            start_idx += count

        for type_str, indices in self.type_map.items():
            self.type_map[type_str] = torch.tensor(sorted(indices), device=self.device, dtype=torch.long)

        # --- Исправленная последовательность ---
        # 1. Присваиваем правильно созданные "кладбищенские" позиции
        self.default_positions = default_pos_tensor.expand(self.num_envs, -1, -1)
        id_map = {"table": 1, "bowl": 2, "chair": 3, "cabinet": 4}
        for name, data in self.object_map.items():
            obj_id = id_map.get(name, 0)  # Default to 0 for unmapped objects
            self.object_ids[0, data['indices']] = obj_id
        # 2. Инициализируем текущие позиции из дефолтных
        self.positions = self.default_positions.clone()

    def _initialize_strategies(self):
        """
        Возвращает стратегии ПО ТИПАМ (а не по именам объектов):
        { 'surface_provider': GridPlacement(...), 'surface_only': OnSurfacePlacement(...), ... }

        Источник стратегий:
        1) Предпочтительно: верхний блок JSON `type_placements`.
        2) Fallback (для обратной совместимости): если блока нет, берём первую стратегию
        из любого объекта соответствующего типа (если у объекта есть ['placement']).
        """
        strategies_by_type = {}

        def _indices_for_types(type_names):
            if isinstance(type_names, str):
                type_names = [type_names]
            acc = []
            for t in type_names:
                inds = self.type_map.get(t, torch.tensor([], dtype=torch.long, device=self.device))
                if len(inds):
                    acc.extend(inds.tolist())
            # уникальные, отсортированные индексы
            return sorted(set(acc))

        # 1) Явно заданные стратегии из блока type_placements
        if self.type_placements_cfg:
            for t, t_cfg in self.type_placements_cfg.items():
                stype = t_cfg["strategy"]
                if stype == "grid":
                    strategies_by_type[t] = GridPlacement(self.device, t_cfg["grid_coordinates"])
                elif stype == "on_surface":
                    surf_types = t_cfg.get("surface_types", ["surface_provider"])
                    surf_inds = _indices_for_types(surf_types)
                    strategies_by_type[t] = OnSurfacePlacement(self.device, surf_inds, t_cfg["margin"])

        # 2) Fallback: нет type_placements -> достаём первую стратегию с уровня объектов
        if not strategies_by_type:
            print("[ ERR ] WE HAVE AN ERROR IN _initialize_strategies")
            seen = set()
            for obj_cfg in self.config:
                # поддержка старого формата
                plc_list = obj_cfg.get("placement") or []
                if not plc_list:
                    continue
                plc = plc_list[0]
                stype = plc["strategy"]
                for t in obj_cfg.get("type", []):
                    if t in seen:
                        continue
                    if stype == "grid":
                        strategies_by_type[t] = GridPlacement(self.device, plc["grid_coordinates"])
                    elif stype == "on_surface":
                        surf_types = plc.get("surface_types", ["surface_provider"])
                        surf_inds = _indices_for_types(surf_types)
                        strategies_by_type[t] = OnSurfacePlacement(self.device, surf_inds, plc["margin"])
                    seen.add(t)

        return strategies_by_type


    def randomize_scene(self, env_ids: torch.Tensor, mess: bool = False, use_obstacles: bool = False, all_defoult: bool = True):
        """
        Простой и надёжный вариант:
        - Сбрасываем env в дефолт
        - Для КАЖДОГО env:
            * выбираем количества для типов с единственным правилом:
            surface_only <= surface_provider, а surface_provider >= max(1, surface_only) если провайдеры есть
            * в ПОРЯДКЕ типов берём случайные НЕАКТИВНЫЕ индексы этого типа и вызываем стратегию, делая их активными
        Так объект не сможет переиспользоваться на следующем типе.
        """
        device = self.device
        num_envs_sel = len(env_ids)

        # 1) reset
        self.active[env_ids] = False
        self.positions[env_ids] = self.default_positions[env_ids]
        self.on_surface_idx[env_ids] = -1
        self.surface_level[env_ids] = 0
        if all_defoult:
            return

        scene_data  = self.get_scene_data_dict()
        type_strats = self.placement_strategies  # {type: strategy}

        # Порядок важен: поверхности -> напольные -> "на поверхности" -> стафф
        placement_order = [t for t in ["surface_provider", "surface_only", "movable_obstacle", "staff_obstacle"]
                        if t in type_strats]

        # Удобные ссылки на пулы индексов по типам (могут быть пустыми тензорами)
        idx_by_type = {
            "surface_provider": self.type_map.get("surface_provider", torch.tensor([], dtype=torch.long, device=device)),
            "surface_only":     self.type_map.get("surface_only",     torch.tensor([], dtype=torch.long, device=device)),
            "movable_obstacle": self.type_map.get("movable_obstacle", torch.tensor([], dtype=torch.long, device=device)),
            "staff_obstacle":   self.type_map.get("staff_obstacle",   torch.tensor([], dtype=torch.long, device=device)),
        }

        # Вспомогательная: выбрать до k случайных НЕАКТИВНЫХ индексов eligible для одного env
        def _sample_available_for_env(env_id: int, eligible: torch.Tensor, k: int) -> torch.Tensor:
            if eligible.numel() == 0 or k <= 0:
                return torch.empty((1, 0), dtype=torch.long, device=device)  # форма [1, 0] для совместимости
            # маска доступности по этому env
            avail_mask = ~self.active[env_id, eligible]  # [M]
            if avail_mask.sum().item() == 0:
                return torch.empty((1, 0), dtype=torch.long, device=device)
            elig_avail = eligible[avail_mask]            # [Ma]
            Ma = elig_avail.numel()
            kk = min(k, Ma)
            # случайная выборка без замены
            perm = torch.randperm(Ma, device=device)[:kk]
            picked = elig_avail[perm]                    # [kk]
            return picked.view(1, -1)                    # [1, kk]

        # Вспомогательная: безопасно вызвать стратегию на одном env
        def _apply_strategy_one_env(p_type: str, env_id: int, obj_idx_row: torch.Tensor):
            # obj_idx_row: [1, kk] Long
            if obj_idx_row.numel() == 0:
                return
            env_row = torch.tensor([env_id], dtype=torch.long, device=device)  # [1]
            type_strats[p_type].apply(env_row, obj_idx_row.to(torch.long), scene_data, mess)

        # 2) основной цикл по env
        for env_id in env_ids.tolist():
            # ---- 2.1 Выбор количества по типам (только верхние ограничения; реальное число урежется по доступности) ----
            max_prov = idx_by_type["surface_provider"].numel()
            max_surf = idx_by_type["surface_only"].numel()
            max_mov  = idx_by_type["movable_obstacle"].numel()
            max_staff= idx_by_type["staff_obstacle"].numel()

            # surface_only: 0..max_surf (если есть такие объекты)
            # TODO: I should unitify goal in every object in scene not only on surfsce
            if max_surf > 0:
                k_surface_only = int(torch.randint(1, max_surf + 1, (1,), device=device).item())
            else:
                k_surface_only = 0
                print("[ ERR ] max_surf == 0")

            # providers: минимум 1, и не меньше k_surface_only; если провайдеров нет — и surface_only обнуляем
            if max_prov > 0:
                low = max(1, k_surface_only)
                low = min(low, max_prov)  # на случай если k_surface_only > max_prov
                k_providers = int(torch.randint(low, max_prov + 1, (1,), device=device).item())
            else:
                print("[ ERR ] max_prov == 0")
                k_providers = 0
                k_surface_only = 0

            # напольные препятствия (если попросили obstacles)
            k_movable = int(torch.randint(7, 10, (1,), device=device).item()) if (use_obstacles and max_mov > 0) else 0
            k_staff   = int(torch.randint(int(max_mov/2), max_staff + 1, (1,), device=device).item()) if (use_obstacles and max_staff > 0) else 0

            # ---- 2.2 Размещение по порядку типов ----
            for p_type in placement_order:
                if p_type == "surface_provider":
                    elig = idx_by_type["surface_provider"]
                    picked = _sample_available_for_env(env_id, elig, k_providers)
                    _apply_strategy_one_env("surface_provider", env_id, picked)

                elif p_type == "movable_obstacle":
                    # исключим экземпляры, которые также помечены surface_only (если такие есть), чтобы не схватить их на пол
                    elig = idx_by_type["movable_obstacle"]
                    so   = set(idx_by_type["surface_only"].tolist()) if idx_by_type["surface_only"].numel() > 0 else set()
                    if so and elig.numel() > 0:
                        elig = torch.tensor([i for i in elig.tolist() if i not in so], dtype=torch.long, device=device)
                    picked = _sample_available_for_env(env_id, elig, k_movable)
                    _apply_strategy_one_env("movable_obstacle", env_id, picked)

                elif p_type == "surface_only":
                    # важно: провайдеры уже размещены, берём доступные surface_only
                    elig = idx_by_type["surface_only"]
                    picked = _sample_available_for_env(env_id, elig, k_surface_only)
                    _apply_strategy_one_env("surface_only", env_id, picked)

                elif p_type == "staff_obstacle":
                    elig = idx_by_type["staff_obstacle"]
                    picked = _sample_available_for_env(env_id, elig, k_staff)
                    _apply_strategy_one_env("staff_obstacle", env_id, picked)

        # 3) цель
        self.chose_active_goal_state(env_ids)

    def get_active_obstacle_positions_for_path_planning(self, env_ids: torch.Tensor) -> list:
        """
        Возвращает позиции активных препятствий в формате списка списков,
        специально для генерации строкового ключа в path_manager.
        """
        obs_indices = self.type_map.get("movable_obstacle", torch.tensor([], dtype=torch.long))
        if len(obs_indices) == 0:
            return [[] for _ in env_ids]
            
        active_mask = self.active[env_ids][:, obs_indices] # (num_envs, num_obstacles)
        positions = self.positions[env_ids][:, obs_indices].cpu().numpy() # (num_envs, num_obstacles, 3)
        
        output_list = []
        for i in range(len(env_ids)):
            # Выбираем только активные позиции для i-й среды
            active_positions = positions[i, active_mask[i].cpu().numpy()]
            # Округляем и сортируем для консистентности ключа
            rounded_pos = [(round(p[0], 1), round(p[1], 1), round(p[2], 1)) for p in active_positions]
            output_list.append(sorted(rounded_pos))
            
        return output_list

    def get_graph_embedding(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Создает тензорный эмбеддинг фиксированного размера для текущего состояния сцены."""
        # [is_active, pos_x, pos_y, pos_z, size_x, size_y, size_z, radius, object_id]
        # Размер фичи: 1 + 3 + 3 + 1 + 1 = 9
        num_features = 9
        embedding = torch.zeros(len(env_ids), self.num_total_objects, num_features, device=self.device)
        # print("bbbb ", len(embedding[0]))
        env_positions = self.positions[env_ids] + 5
        env_active = self.active[env_ids].float().unsqueeze(-1)
        env_sizes = self.sizes.expand(len(env_ids), -1, -1)
        env_radii = self.radii.expand(len(env_ids), -1).unsqueeze(-1)
        env_object_ids = self.object_ids.expand(len(env_ids), -1).unsqueeze(-1)

        embedding[..., 0:1] = env_active
        embedding[..., 1:4] = env_positions * env_active
        embedding[..., 4:7] = env_sizes * env_active
        embedding[..., 7:8] = env_radii * env_active
        embedding[..., 8:9] = env_object_ids * env_active

        # Нормализация для лучшего обучения (применяется ко всем, но неактивные останутся 0)
        embedding[..., 1:4] /= 5.0  # Делим позиции на примерный масштаб комнаты
        embedding[..., 4:7] /= 1.0  # Размеры уже примерно в этом диапазоне
        embedding[..., 7:8] /= 2.0  # Радиусы
        embedding[..., 8:9] /= 3.0  # Нормализация ID (максимум 3 для chair)
        # Возвращаем "плоский" тензор
        return embedding.view(len(env_ids), -1)

    def print_graph_info(self, env_id: int):
        """Печатает детальную информацию о сцене для ОДНОГО окружения."""
        print(f"\n=== Scene Information (Env ID: {env_id}) ===")
        
        # Данные для указанного env_id
        positions = self.positions[env_id]
        active_states = self.active[env_id]
        surface_indices = self.on_surface_idx[env_id]
        surface_levels = self.surface_level[env_id]
        
        table_data = []
        for i in range(self.num_total_objects):
            name = self.names[i]
            pos = positions[i]
            # Ищем типы по индексу
            types = ", ".join([t for t, inds in self.type_map.items() if i in inds])

            row = [
                i, name, types,
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                f"{self.radii[0, i]:.2f}",
                str(active_states[i].item()),
                surface_indices[i].item(),
                surface_levels[i].item()
            ]
            table_data.append(row)
            
        headers = ["ID", "Name", "Types", "Position", "Radius", "Active", "On Surface", "Surface Level"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def chose_active_goal_state(self, env_ids: torch.Tensor):
        goal_indices = self.type_map.get("possible_goal", torch.tensor([], dtype=torch.long))
        if len(goal_indices) == 0:
            print(f"[WARNING] No objects of type 'possible_goal' found in config.")
            self.goal_positions[env_ids] = torch.tensor([-3.75, 0.0, 0.1], device=self.device)
            return
        
        active_goal_mask = self.active[env_ids][:, goal_indices].float()
        
        # Fallback если ни одна цель не активна
        any_active = active_goal_mask.sum(dim=1) > 0
        if not all(any_active):
            print("NO GOAL", any_active)
            # # Для env где нет активных целей, активируем первую попавшуюся
            # fallback_mask = ~any_active
            # active_goal_mask[fallback_mask, 0] = 1.0

        chosen_goal_rel_idx = torch.multinomial(active_goal_mask + 1e-9, 1).squeeze(-1)
        chosen_goal_idx = goal_indices[chosen_goal_rel_idx]
        
        env_indices = env_ids
        self.goal_positions[env_indices] = self.positions[env_indices, chosen_goal_idx]

    def get_active_goal_state(self, env_ids: torch.Tensor):
        return self.goal_positions[env_ids]

    def place_robot_for_goal(self, env_ids: torch.Tensor, mean_dist: float, min_dist: float, max_dist: float, angle_error: float):
        """Размещает робота относительно цели, избегая препятствий и границ."""
        # Этап 1: Получение числа сред
        num_envs = len(env_ids)

        # Этап 2: Извлечение позиций целей
        goal_pos = self.goal_positions[env_ids]

        # Этап 3: Определение активных препятствий на полу
        is_floor_obstacle = (self.active[env_ids] == True) & (self.on_surface_idx[env_ids] == -1)

        # Этап 4: Извлечение позиций и радиусов препятствий
        obstacle_pos_all = self.positions[env_ids, :, :2].clone()

        obstacle_radii_all = self.radii.expand(self.num_envs, -1)[env_ids]
        # Этап 5: Фильтрация неактивных препятствий
        inf_pos = torch.full_like(obstacle_pos_all, 999.0)

        obstacle_pos = torch.where(is_floor_obstacle.unsqueeze(-1), obstacle_pos_all, inf_pos)
        # Этап 6: Генерация радиусов для размещения робота
        mean_dist_with_shift = mean_dist + 1.31
        radii = torch.normal(mean=mean_dist_with_shift, std=mean_dist * 0.1, size=(num_envs, 1), device=self.device).clamp_(min_dist, max_dist)
        # Этап 7: Генерация кандидатов для позиций робота
        candidates = goal_pos[:, None, :2] + radii.unsqueeze(1) * self.candidate_vectors
        # Этап 8: Проверка границ комнаты
        # Этап 8: Проверка границ комнаты (только границы, без коллизий)
        bounds = self.room_bounds
        in_bounds_mask = (
            (candidates[..., 0] >= bounds['x_min'] + self.robot_radius) &
            (candidates[..., 0] <= bounds['x_max'] - self.robot_radius) &
            (candidates[..., 1] >= bounds['y_min'] + self.robot_radius) &
            (candidates[..., 1] <= bounds['y_max'] - self.robot_radius)
        )
        # print(candidates)
        # print(in_bounds_mask)
        # Этап 9: Выбор углов только по границам
        in_bounds_mask_float = in_bounds_mask.float() + 1e-9
        chosen_angle_idx = torch.multinomial(in_bounds_mask_float, 1).squeeze(-1)
        # print(chosen_angle_idx)
        # Этап 10: Выбор финальных позиций робота
        batch_indices = torch.arange(num_envs, device=self.device)
        final_robot_positions = candidates[batch_indices, chosen_angle_idx]

        # Этап 11: fallback если ни одна позиция не в границах
        no_valid_pos_mask = ~in_bounds_mask.any(dim=1)
        if torch.any(no_valid_pos_mask):
            fallback_pos = goal_pos[:, :2] + torch.tensor([max_dist, 0.0], device=self.device) # 0.0!
            final_robot_positions[no_valid_pos_mask] = fallback_pos[no_valid_pos_mask]
        final_robot_positions = torch.zeros_like(final_robot_positions,device=self.device)
        # Этап 15: Вычисление ориентации робота (yaw)
        direction_to_goal = goal_pos[:, :2] - final_robot_positions
        base_yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
        error = (torch.rand(num_envs, device=self.device) - 0.5) * 2 * angle_error
        final_yaw = base_yaw + error
        # Этап 16: Формирование кватернионов ориентации
        robot_quats = torch.zeros(num_envs, 4, device=self.device)
        robot_quats[:, 0] = torch.cos(final_yaw / 2.0)
        robot_quats[:, 3] = torch.sin(final_yaw / 2.0)
        # Этап 17: Возврат результатов
        # Проверяем пересечения с препятствиями
        self.remove_colliding_obstacles(env_ids, final_robot_positions)
        # print("final_robot_positions: ", final_robot_positions)
        return final_robot_positions, robot_quats
    

    def remove_colliding_obstacles(self, env_ids: torch.Tensor, robot_positions: torch.Tensor):
        """Ставит в дефолт все препятствия, пересекающиеся с роботом."""
        # TODO There can be obstacles with suface providing and we should delete alse items on that
        obs_indices = self.type_map.get("movable_obstacle", torch.tensor([], dtype=torch.long))
        if len(obs_indices) == 0:
            return

        # позиции и радиусы препятствий
        obs_pos = self.positions[env_ids][:, obs_indices, :2]
        obs_r = self.radii.expand(len(env_ids), -1)[:, obs_indices]

        # расстояния от робота до препятствий
        dists = torch.norm(obs_pos - robot_positions[:, None, :2], dim=2)
        
        coll_mask = dists < (self.robot_radius + obs_r + 0.2)
        if coll_mask.any():
            # print("coll_mask: ",  coll_mask)
            # for i in env_ids:
            #     self.print_graph_info(i)
            # переносим такие препятствия в дефолт
            default_pos = self.default_positions[env_ids][:, obs_indices]
            batch_idx, obs_idx = torch.where(coll_mask)                 # индексы элементов с коллизией
            env_batch_idx = env_ids[batch_idx]                           # индексы env_ids для batch
            obs_indices_sel = obs_indices[obs_idx]                       # индексы obstacles

            # Присваиваем значения дефолтных позиций в исходный тензор
            self.positions[env_batch_idx, obs_indices_sel] = default_pos[batch_idx, obs_idx]

            # print(self.positions[env_ids][:, obs_indices][coll_mask])
            # print(default_pos[coll_mask])
            
            # self.positions[env_ids][:, obs_indices][coll_mask] = default_pos[coll_mask]
            # print(self.positions[env_ids][:, obs_indices][coll_mask])

            # деактивируем их
            # print(self.active[env_ids][:, obs_indices][coll_mask] )
            self.active[env_batch_idx, obs_indices_sel] = False

            # print(self.active[env_ids][:, obs_indices][coll_mask] )

        obs_pos = self.positions[env_ids][:, obs_indices, :2]
        obs_r = self.radii.expand(len(env_ids), -1)[:, obs_indices]

        # расстояния от робота до препятствий
        dists = torch.norm(obs_pos - robot_positions[:, None, :2], dim=2)
        
        coll_mask = dists < (self.robot_radius + obs_r)
        if coll_mask.any():
            
            # print("coll_mask 2: ",  coll_mask)
            
            for i in env_ids:
                self.print_graph_info(i)

    def get_graph_obs(self, env_ids=None) -> dict[str, torch.Tensor]:
        """Returns a dictionary with full tensorized graph representation for observations.
        
        - node_features: tensor (num_envs, num_objects, 14) - per object: [pos(3), size(3), radius(1), color(3), id(1), active(1), parent_id(1), level(1)].
        - edge_features: tensor (num_envs, num_objects, 6) - per possible edge (from child to parent): [exists(1), z_diff(1), level_diff(1), dist(1), color_diff_norm(1), id_diff(1)]; 0 if no edge.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        num_selected = len(env_ids)
        
        # Select data for env_ids
        positions = self.positions[env_ids]/10  # (num_selected, num_objects, 3)
        sizes = self.sizes.expand(num_selected, -1, -1)/10  # (num_selected, num_objects, 3)
        radii = self.radii.expand(num_selected, -1).unsqueeze(-1)/10  # (num_selected, num_objects, 1)
        colors = self.colors.expand(num_selected, -1, -1)/10  # (num_selected, num_objects, 3)
        object_ids = self.object_ids.expand(num_selected, -1).unsqueeze(-1).float()/10  # (num_selected, num_objects, 1)
        active = self.active[env_ids].unsqueeze(-1).float()/10  # (num_selected, num_objects, 1)
        parents = self.on_surface_idx[env_ids].unsqueeze(-1).float()/10  # (num_selected, num_objects, 1); -1 for no parent
        levels = self.surface_level[env_ids].unsqueeze(-1).float()/10  # (num_selected, num_objects, 1)
        
        # Node features: concat all per object
        node_features = torch.cat([
            positions, sizes, radii, colors, object_ids, active, parents, levels
        ], dim=-1)  # (num_selected, num_objects, 14)
        
        # Edge features: per object (potential edge to parent)
        edge_exists = (parents >= 0).float()  # (num_selected, num_objects, 1)
        
        # For valid parents: z_diff = child_z - parent_z
        valid_mask = (parents >= 0).squeeze(-1)  # (num_selected, num_objects)
        z_diff = torch.zeros(num_selected, self.num_total_objects, 1, device=self.device)
        batch_idx = torch.arange(num_selected, device=self.device)[:, None].expand(-1, self.num_total_objects)[valid_mask]
        obj_idx = torch.arange(self.num_total_objects, device=self.device)[None, :].expand(num_selected, -1)[valid_mask]
        parent_idx = parents.squeeze(-1)[valid_mask].long()
        z_diff[valid_mask] = positions[batch_idx, obj_idx, 2:3] - positions[batch_idx, parent_idx, 2:3]
        
        # level_diff = child_level - parent_level (should be 1 usually)
        level_diff = torch.zeros_like(z_diff)
        level_diff[valid_mask] = levels[batch_idx, obj_idx] - levels[batch_idx, parent_idx]
        
        # dist = norm(child_pos_xy - parent_pos_xy)
        dist = torch.zeros_like(z_diff)
        child_xy = positions[batch_idx, obj_idx, :2]
        parent_xy = positions[batch_idx, parent_idx, :2]
        dist[valid_mask] = torch.norm(child_xy - parent_xy, dim=-1, keepdim=True)
        
        # color_diff_norm = norm(child_color - parent_color)
        color_diff_norm = torch.zeros_like(z_diff)
        child_color = colors[batch_idx, obj_idx]
        parent_color = colors[batch_idx, parent_idx]
        color_diff_norm[valid_mask] = torch.norm(child_color - parent_color, dim=-1, keepdim=True)
        
        # id_diff = child_id - parent_id
        id_diff = torch.zeros_like(z_diff)
        child_id = object_ids[batch_idx, obj_idx]
        parent_id = object_ids[batch_idx, parent_idx]
        id_diff[valid_mask] = child_id - parent_id
        
        # Edge features: [exists, z_diff, level_diff, dist, color_diff_norm, id_diff]
        edge_features = torch.cat([edge_exists, z_diff, level_diff, dist, color_diff_norm, id_diff], dim=-1)  # (num_selected, num_objects, 6)
        
        return {"node_features": node_features, "edge_features": edge_features}