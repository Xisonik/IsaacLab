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
    print(f"[DEBUG] Importing class '{class_name}' from module: {module_path}")
    spec = importlib.util.spec_from_file_location("custom_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_obj = getattr(module, class_name)
    print(f"[DEBUG] Successfully imported class: {class_obj}")
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
            self.config = json.load(f)['objects']
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
        self.room_bounds = {'x_min': -4, 'x_max': 4.0, 'y_min': -2.7, 'y_max': 2.7}
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)

        n_angles = 36
        angle_step = 2 * math.pi / n_angles
        self.discrete_angles = torch.arange(0, 2 * math.pi, angle_step, device=self.device)
        self.candidate_vectors = torch.stack([torch.cos(self.discrete_angles), torch.sin(self.discrete_angles)], dim=1)
        # Assign object IDs based on name

    
    def update_prims(self):
        pass
    
    def get_scene_data_dict(self):
        return {"positions": self.positions, "sizes": self.sizes.expand(self.num_envs, -1, -1), "radii": self.radii.expand(self.num_envs, -1), "active": self.active, "on_surface_idx": self.on_surface_idx, "surface_level": self.surface_level}
    
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
        graveyard_start_x = 0.0
        graveyard_start_y = 6.0
        spacing = 1.0 # Расстояние между объектами на кладбище
        max_per_row = 5 # Сколько объектов в ряду на кладбище

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
            indices = torch.arange(start_idx, start_idx + count, device=self.device, dtype=torch.long)
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
        strategies = {}
        for obj_cfg in self.config:
            name = obj_cfg['name']
            print("name: ", name)
            placement_cfg_list = obj_cfg.get('placement')
            if not placement_cfg_list: continue
            placement_cfg = placement_cfg_list[0]
            strategy_type = placement_cfg['strategy']
            if strategy_type == 'grid':
                strategies[name] = GridPlacement(self.device, placement_cfg['grid_coordinates'])
            elif strategy_type == 'on_surface':
                surface_indices = self.type_map.get(placement_cfg['surface_types'][0], torch.tensor([], dtype=torch.long))
                strategies[name] = OnSurfacePlacement(self.device, surface_indices.tolist(), placement_cfg['margin'])
        return strategies

    def randomize_scene(self, env_ids: torch.Tensor, mess: bool = False, use_obstacles: bool = False, all_defoult: bool = False):
        """Абстрактная, векторизованная рандомизация сцены на основе ТИПОВ объектов."""
        num_to_randomize = len(env_ids)
        
        # 1. Сброс состояния
        self.active[env_ids] = False
        self.positions[env_ids] = self.default_positions[env_ids]
        self.on_surface_idx[env_ids] = -1
        self.surface_level[env_ids] = 0
        if all_defoult:
            return

        # 2. Определение количества объектов для размещения по типам
        num_surface_only = len(self.type_map.get("surface_only", []))
        num_providers = len(self.type_map.get("surface_provider", []))
        num_floor_obs = len(self.type_map.get("movable_obstacle", [])) - num_surface_only
        num_static_floor_obs = len(self.type_map.get("staff_obstacle", [])) - num_surface_only

        num_surface_only_to_place = torch.randint(1, num_surface_only + 1, (num_to_randomize,), device=self.device) if num_surface_only > 0 else torch.zeros(num_to_randomize, dtype=torch.long, device=self.device)
        
        if num_providers > 0:
            # Нижняя граница: минимум 1, и не меньше чем количество surface_only объектов.
            low_bound = torch.max(
                torch.tensor(1, device=self.device), 
                num_surface_only_to_place
            )
            # Верхняя граница (исключающая для генерации)
            high_bound = num_providers + 1
            
            # Генерируем случайные числа с плавающей точкой и масштабируем их до нужного диапазона
            rand_float = torch.rand(num_to_randomize, device=self.device)
            num_providers_to_place = (low_bound + rand_float * (high_bound - low_bound)).long()
        else:
            num_providers_to_place = torch.zeros(num_to_randomize, dtype=torch.long, device=self.device)
        num_floor_obstacles_to_place = torch.randint(0, num_floor_obs + 1, (num_to_randomize,), device=self.device) if use_obstacles and num_floor_obs > 0 else torch.zeros(num_to_randomize, dtype=torch.long, device=self.device)
        num_static_floor_obstacles_to_place = torch.randint(0, num_static_floor_obs + 1, (num_to_randomize,), device=self.device) if use_obstacles and num_static_floor_obs > 0 else torch.zeros(num_to_randomize, dtype=torch.long, device=self.device)

        # 3. Применение стратегий в правильном порядке (сначала поверхности)
        scene_data = self.get_scene_data_dict()
        placement_order = ["surface_provider", "movable_obstacle", "surface_only", "staff_obstacle"]
        
        # Определяем, сколько объектов какого типа нужно разместить
        counts_by_type = {
            "surface_provider": num_providers_to_place,
            "movable_obstacle": num_floor_obstacles_to_place,
            "surface_only": num_surface_only_to_place,
            "staff_obstacle": num_static_floor_obstacles_to_place,
        }
        
        for p_type in placement_order:
            # Находим все объекты, имеющие данный тип
            for name, data in self.object_map.items():
                if p_type in data['types'] and name in self.placement_strategies:
                    # Некоторые типы могут иметь несколько ролей, например, movable_obstacle может быть не surface_only
                    # Проверяем, что мы еще не разместили этот объект в рамках другой роли
                    if p_type == "movable_obstacle" and "surface_only" in data['types']:
                        continue

                    obj_indices = data['indices']
                    num_to_place_per_env = counts_by_type.get(p_type, torch.zeros(num_to_randomize, dtype=torch.long))

                    # Векторизованный выбор случайных экземпляров
                    rand_indices = torch.rand(num_to_randomize, len(obj_indices), device=self.device).argsort(dim=1)
                    
                    max_num_to_place = int(num_to_place_per_env.max())
                    if max_num_to_place == 0: continue
                    
                    indices_to_place = obj_indices[rand_indices[:, :max_num_to_place]]

                    # Маска, чтобы применять стратегию только к тем env, где нужно разместить > 0 объектов
                    valid_envs_mask = num_to_place_per_env > 0
                    
                    # Фильтруем env_ids и indices_to_place
                    active_env_ids = env_ids[valid_envs_mask]
                    if len(active_env_ids) == 0: continue

                    active_indices_to_place = indices_to_place[valid_envs_mask]
                    # print("active_indices_to_place ", active_indices_to_place)
                    self.placement_strategies[name].apply(active_env_ids, active_indices_to_place, scene_data, mess)

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
            self.goal_positions[env_ids] = torch.tensor([-4.5, 0.0, 0.75], device=self.device)
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