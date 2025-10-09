from abc import ABC, abstractmethod
import torch
import random
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

class PlacementStrategy(ABC):
    def __init__(self, device: str, **kwargs):
        self.device = device

    @abstractmethod
    def apply(self, graph: ObstacleGraph, object_indices: list[int], num_to_place: int, mess: bool):
        pass

class GridPlacement(PlacementStrategy):
    def __init__(self, device: str, grid_coordinates: list[list[float]]):
        super().__init__(device)
        self.grid = torch.tensor(grid_coordinates, device=self.device, dtype=torch.float32)

    def apply(self, graph: ObstacleGraph, object_indices: list[int], num_to_place: int, mess: bool):
        num_to_place = min(num_to_place, len(object_indices), len(self.grid))
        if num_to_place == 0: return

        active_indices = random.sample(object_indices, num_to_place)
        pos_indices = torch.randperm(len(self.grid), device=self.device)[:num_to_place]

        graph.positions[active_indices] = self.grid[pos_indices]
        graph.active[active_indices] = True
        graph.on_surface_idx[active_indices] = -1 # Размещение на сетке = на полу
        graph.surface_level[active_indices] = 0

class OnSurfacePlacement(PlacementStrategy):
    def __init__(self, device: str, surface_types: list[str], margin: float):
        super().__init__(device)
        self.surface_types = set(surface_types)
        self.margin = margin

    def apply(self, graph: ObstacleGraph, object_indices: list[int], num_to_place: int, mess: bool):
        num_to_place = min(num_to_place, len(object_indices))
        if num_to_place == 0: return
        
        # Находим все активные поверхности нужного типа
        available_surfaces = [
            i for i, data in graph.graph.nodes(data=True)
            if data['active'] and not self.surface_types.isdisjoint(data['types'])
        ]
        if not available_surfaces: return

        active_indices = random.sample(object_indices, num_to_place)
        for i, obj_idx in enumerate(active_indices):
            # Выбираем поверхность для размещения
            if mess:
                target_surface_idx = random.choice(available_surfaces)
            else: # В режиме не-mess, распределяем по одному на поверхность
                target_surface_idx = available_surfaces[i % len(available_surfaces)]
            surface_pos = graph.positions[target_surface_idx]
            surface_size = graph.sizes[target_surface_idx]
            # Размещаем либо случайно на поверхности (mess), либо в центре
            if mess:
                rand_x = (torch.rand(1).item() - 0.5) * (surface_size[0] - self.margin * 2)
                rand_y = (torch.rand(1).item() - 0.5) * (surface_size[1] - self.margin * 2)
            else:
                rand_x, rand_y = 0.0, 0.0
            
            pos_z = surface_pos[2] + surface_size[2] + graph.sizes[obj_idx][2] / 2.0
            graph.positions[obj_idx] = torch.tensor([surface_pos[0] + rand_x, surface_pos[1] + rand_y, pos_z], device=self.device)
            # Устанавливаем иерархию
            graph.active[obj_idx] = True
            graph.on_surface_idx[obj_idx] = target_surface_idx
            graph.surface_level[obj_idx] = graph.surface_level[target_surface_idx] + 1