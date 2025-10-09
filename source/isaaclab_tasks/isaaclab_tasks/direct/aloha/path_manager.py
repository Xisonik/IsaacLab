import json
import torch
import os
from typing import Dict, List, Tuple, Iterable, Optional
from pathlib import Path
from .config_codec import ConfigCodec

import math

class Path_manager:
    def __init__(
            self, 
            scene_manager, 
            ratio: float = 10.0, 
            shift: list = [5, 5], 
            device: str = 'cpu', 
            obstacle_types: Iterable[str] = ("movable_obstacle", "static_obstacle"),
            config_path="source/isaaclab_tasks/isaaclab_tasks/direct/aloha/scene_items.json"):
        """
        Инициализирует менеджер путей для загрузки путей из all_paths.json и преобразования координат.

        Args:
            log_dir (str): Путь к директории с all_paths.json.
            ratio (float): Масштабный коэффициент для преобразования координат (по умолчанию 4.0).
            shift (list): Смещение координат [shift_x, shift_y] (по умолчанию [5, 4]).
            device (str): Устройство для хранения путей (по умолчанию 'cpu').
        """
        self.device = device
        self.ratio = ratio
        self.shift = torch.tensor(shift, device=device, dtype=torch.float32)
        self.all_paths = {}
        self.paths_file = os.path.join("data", "all_paths.json")
        self._load_paths()
        self.scene_manager = scene_manager
        self.config_path = config_path
        with open(config_path, "r") as f:
            raw = json.load(f)
        self.raw = raw
        self.objects_cfg: List[dict] = raw["objects"]
        self.type_placements: Dict[str, dict] = raw.get("type_placements", {})

        self.type_radii: Dict[str, float] = self._compute_type_radii(self.objects_cfg)
        self.type_grids: Dict[str, List[Tuple[float, float, float]]] = self._collect_type_grids(self.type_placements)
        self.obstacle_types = ("movable_obstacle",)
        self.codec = ConfigCodec(self.type_grids,
                         types=self.obstacle_types,   # важно: тот же порядок
                         quant=8,
                         max_radius=70.0)
        self._quant = getattr(self.codec, "quant", 8)

        # индекс квантизованных клеток -> тип (только для типов-препятствий)
        self._q_index = self._build_quant_index(self.type_grids, self._quant, self.obstacle_types)

    def _build_quant_index(self, type_grids: Dict[str, List[Tuple[float,float,float]]], q: int, use_types: Iterable[str]):
        """Карта (qx,qy,qz) -> type для быстрый классификации точек, квантуем шагом 1/q."""
        idx = {}
        for t, pts in type_grids.items():
            if t not in use_types:
                continue
            for x, y, z in pts:
                key = (int(round(x * q)), int(round(y * q)), int(round(z * q)))
                idx[key] = t
        return idx

    def _classify_raw_obstacles(self, raw_list: List[Tuple[float,float,float]]) -> Dict[str, List[Tuple[float,float,float]]]:
        """
        Пришёл старый формат: список позиций без типов.
        Разносим по типам, сопоставляя к ближайшим клеткам сеток соответствующих типов.
        Сначала точное совпадение в квантизованной решётке, затем небольшой окрестностный поиск.
        """
        by_type = {t: [] for t in self.obstacle_types}
        q = self._quant
        idx = self._q_index

        for (x, y, z) in raw_list:
            qkey = (int(round(x * q)), int(round(y * q)), int(round(z * q)))
            t = idx.get(qkey)
            if t is None:
                # маленькая окрестность в квант-пространстве (±1 шаг) — дёшево и надёжно для сеток
                found = None
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neigh = (qkey[0] + dx, qkey[1] + dy, qkey[2])
                        if neigh in idx:
                            found = idx[neigh]
                            break
                    if found:
                        break
                t = found if found else self.obstacle_types[0]  # дефолтно кидаем в первый тип (обычно movable)
            by_type[t].append((float(x), float(y), float(z)))

        # стабилизируем порядок (важно для детерминированного cfg_id)
        for t in by_type:
            by_type[t].sort(key=lambda p: (p[0], p[1], p[2]))
        return by_type

    
    @staticmethod
    def _compute_type_radii(objects_cfg: List[dict]) -> Dict[str, float]:
        radii: Dict[str, float] = {}
        for obj in objects_cfg:
            types = obj.get("type", [])
            size = obj.get("size", None)
            if not size:
                continue
            r = float(((size[0] / 2.0) ** 2 + (size[1] / 2.0) ** 2) ** 0.5)
            for t in types:
                if t not in radii:
                    radii[t] = r
        for t in list(radii):
            if not math.isfinite(radii[t]):
                radii[t] = 0.35
        return radii
    
    @staticmethod
    def _collect_type_grids(type_placements: Dict[str, dict]) -> Dict[str, List[Tuple[float, float, float]]]:
        grids: Dict[str, List[Tuple[float, float, float]]] = {}
        for t, cfg in type_placements.items():
            if cfg.get("strategy") == "grid":
                coords = cfg.get("grid_coordinates", [])
                grids[t] = [tuple(map(float, c)) for c in coords]
        return grids
    
    def _load_paths(self):
        """
        Загружает пути из all_paths.json.
        Формат: {config_key: {target_node: {start_node: path}}}
        """
        if os.path.exists(self.paths_file):
            try:
                with open(self.paths_file, 'r') as f:
                    loaded_paths = json.load(f)
                for config_key, targets in loaded_paths.items():
                    self.all_paths[config_key] = {}
                    for target_str, nodes in targets.items():
                        target = tuple(map(int, target_str.split(',')))
                        self.all_paths[config_key][target] = {}
                        for node_str, path in nodes.items():
                            node = tuple(map(int, node_str.split(',')))
                            self.all_paths[config_key][target][node] = [tuple(p) for p in path]
                print(f"Loaded {len(self.all_paths)} configurations from {self.paths_file}")
            except Exception as e:
                print(f"Error loading paths file: {e}")
                self.all_paths = {}
        else:
            print(f"No paths file found at {self.paths_file}")
            self.all_paths = {}

    def real_to_grid(self, real_point: torch.Tensor) -> torch.Tensor:
        """
        Преобразует реальные координаты (x, y) в сеточные.

        Args:
            real_point (torch.Tensor): Реальные координаты [num_envs, 2].

        Returns:
            torch.Tensor: Сеточные координаты [num_envs, 2], целочисленные.
        """
        grid_x = torch.round((real_point[:, 0] + self.shift[0]) * self.ratio).to(torch.int32)
        grid_y = torch.round((real_point[:, 1] + self.shift[1]) * self.ratio).to(torch.int32)
        return torch.stack([grid_x, grid_y], dim=-1)

    def grid_to_real(self, grid_point: torch.Tensor) -> torch.Tensor:
        """
        Преобразует сеточные координаты (x, y) в реальные.

        Args:
            grid_point (torch.Tensor): Сеточные координаты [..., 2].

        Returns:
            torch.Tensor: Реальные координаты [..., 2].
        """
        # print(grid_point)
        # print(grid_point[:, 0])
        # print(grid_point[..., 0])
        real_x = grid_point[..., 0] / self.ratio - self.shift[0]
        real_y = grid_point[..., 1] / self.ratio - self.shift[1]
        return torch.stack([real_x, real_y], dim=-1)
    def debug_print_cfg_ids(self, limit: int | None = None):
        """
        Печатает все верхнеуровневые ключи (cfg_id) из self.all_paths.
        Если limit задан — печатает только первые limit штук.
        """
        keys = list(self.all_paths.keys())
        print(f"[all_paths] total configs: {len(keys)}")
        if limit is not None:
            keys = keys[:limit]
        for i, k in enumerate(keys, 1):
            print(f"{i:6d}: {k}", flush=True)

    def get_paths(
        self,
        env_ids: torch.Tensor,
        start_positions: torch.Tensor,      # [N,2] real
        target_positions: torch.Tensor,     # [N,2] real
        active_obstacles_by_type_list: List[Dict[str, List[Tuple[float,float,float]]]],
        device: str = "cuda:0",
        max_obstacle_radius: Optional[float] = None,   # если None — возьмём 7.0
    ):
        """
        Принимает конфигурации препятствий по типам, сама формирует cfg_id (битсеты),
        делает канонизацию (симметрию), достаёт пути и возвращает их в реальных координатах.
        """
        N = len(env_ids)
        assert len(active_obstacles_by_type_list) == N

        # 1) Симметрия целей/стартов в I квадрант
        # mult = self._symmetry_multipliers(target_positions)   # [N,2] ∈ {±1}
        start_sym  = start_positions
        target_sym = target_positions # * mult
        # print("target_sym ", target_sym)
        # 2) Разворот препятствий + отсеивание по радиусу; кодируем cfg_id
        cfg_ids: List[str] = []
        r_use = 70.0 if max_obstacle_radius is None else float(max_obstacle_radius)
        for i in range(N):
            # mx, my = float(mult[i, 0]), float(mult[i, 1])
            raw = active_obstacles_by_type_list[i]
            by_type = self._classify_raw_obstacles(raw)              # ← новый формат: dict[type] -> [(x,y,z),...]
            # by_type_sym = self._apply_symmetry_and_radius_by_type(  # отражаем и режем по радиусу
            #     obstacles_by_type_env=by_type, mx=mx, my=my, max_radius=r_use
            # )
            cfg_ids.append(self.codec.encode_id_hex(by_type))   # m:<hex>|s:<hex>
        # 3) Переводим канонические координаты в узлы сетки
        start_nodes  = self.real_to_grid(start_sym)   # [N,2] → int
        target_nodes = self.real_to_grid(target_sym)

        # 4) Достаём пути из all_paths (ключи-узлы у тебя уже tuple[int,int])
        max_path_length = 15
        self.max_path_length = max_path_length
        paths = []
        for cfg_id, start, target in zip(cfg_ids, start_nodes.tolist(), target_nodes.tolist()):
            start_key  = (start[0], start[1])
            target_key = (target[0], target[1])
            cfg_dict   = self.all_paths.get(cfg_id, {})
            
            path = cfg_dict.get(target_key, {}).get(start_key, [])
            # fallback: ближайшие имеющиеся узлы (если точного нет)
            if not path and cfg_dict:
                tx, ty = target_key
                tgt_keys = list(cfg_dict.keys())
                if tgt_keys:
                    nearest_target_key = min(tgt_keys, key=lambda tk: abs(tx - tk[0]) + abs(ty - tk[1]))
                    start_dict = cfg_dict.get(nearest_target_key, {})
                    if start_dict:
                        sx, sy = start_key
                        nearest_start_key = min(start_dict.keys(), key=lambda sk: abs(sx - sk[0]) + abs(sy - sk[1]))
                        path = start_dict.get(nearest_start_key, [])
            paths.append(path)
        # 5) Сборка тензора пути в СЕТКЕ
        path_tensor = torch.full((N, max_path_length, 2), -7777.0, device=device, dtype=torch.float32)
        for i, (env_id, path) in enumerate(zip(env_ids, paths)):
            if path:
                if len(path) > max_path_length:
                    path = path[-max_path_length:]
                path_tensor[i, -len(path):] = torch.tensor(path, device=device, dtype=torch.float32)
            else:
                path_tensor[i, -1] = start_nodes[i].to(device=device, dtype=torch.float32)

        # 6) Обратно в реальные координаты и разворот из каноники назад
        path_real = self.grid_to_real(path_tensor.to(device=device))   # [N,L,2]
        # path_real = path_real * mult.unsqueeze(1)
        return path_real


    
    def _symmetry_multipliers(self, target_positions: torch.Tensor) -> torch.Tensor:
        """
        Для каждой среды вернёт [mx,my] ∈ {±1}×{±1}, чтобы привести цель в I квадрант.
        """
        mx = torch.where(target_positions[:, 0] < 0, -1.0, 1.0).to(target_positions)
        my = torch.where(target_positions[:, 1] < 0, -1.0, 1.0).to(target_positions)
        return torch.stack([mx, my], dim=-1)  # [N,2]

    def _apply_symmetry_and_radius_by_type(
        self,
        obstacles_by_type_env: dict[str, list[tuple[float,float,float]]],
        mx: float,
        my: float,
        max_radius: float | None = None,
    ) -> dict[str, list[tuple[float,float,float]]]:
        """
        Отразить препятствия по типам и, если задан max_radius, отбросить точки вне круга по центрам.
        Возвращает {type: [(x,y,z), ...]} c сортировкой по (x,y,z) для стабильности.
        """
        out: dict[str, list[tuple[float,float,float]]] = {}
        r2 = None if max_radius is None else max_radius * max_radius
        for t, pts in obstacles_by_type_env.items():
            lst = []
            for (x, y, z) in pts:
                x2, y2 = mx * float(x), my * float(y)
                if (r2 is None) or (x2 * x2 + y2 * y2 <= r2):
                    lst.append((x2, y2, float(z)))
            lst.sort(key=lambda p: (p[0], p[1], p[2]))
            out[t] = lst
        return out

    def find_nearest_node(self, target: tuple, nodes: set) -> tuple:
        """
        Находит ближайший узел из набора узлов к целевой точке (манхэттенское расстояние).

        Args:
            target (tuple): Целевая точка в сеточных координатах (x, y).
            nodes (set): Набор узлов в сеточных координатах.

        Returns:
            tuple: Ближайший узел или None, если набор узлов пуст.
        """
        if not nodes:
            return None
        min_distance = float('inf')
        nearest_node = None
        target_x, target_y = target
        for node in nodes:
            x, y = node
            distance = abs(target_x - x) + abs(target_y - y)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        return nearest_node