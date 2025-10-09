import torch
import networkx as nx
import math
from tabulate import tabulate

class ObstacleGraph:
    def __init__(self, objects_config: list, device='cuda:0'):
        self.device = device
        self.graph = nx.Graph()
        self.objects_config = objects_config
        self.num_nodes = sum(obj['count'] for obj in self.objects_config)
        self._initialize_nodes()

    def _initialize_nodes(self):
        # Атрибуты узлов
        self.positions = torch.zeros(self.num_nodes, 3, device=self.device)
        self.radii = torch.zeros(self.num_nodes, device=self.device)
        self.sizes = torch.zeros(self.num_nodes, 3, device=self.device)
        self.active = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        
        # Иерархия поверхностей
        self.on_surface_idx = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)  # -1 означает пол
        self.surface_level = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)  # 0 означает пол
        
        # Параметры сетки
        max_per_row = 4  # Максимум объектов в строке
        spacing = 1.0    # Расстояние между объектами (1 метр)
        start_x = 3.0    # Начало сетки по X (за пределами x_max=2.0)
        start_y = 4.0    # Начало сетки по Y (за пределами y_max=3.0)
        
        # Рассчитываем количество строк
        num_rows = (self.num_nodes + max_per_row - 1) // max_per_row
        
        # Инициализация позиций в сетке
        node_idx = 0
        for row in range(num_rows):
            for col in range(min(max_per_row, self.num_nodes - row * max_per_row)):
                self.positions[node_idx, 0] = start_x + col * spacing
                self.positions[node_idx, 1] = start_y + row * spacing
                self.positions[node_idx, 2] = 0.0  # На полу
                node_idx += 1
        
        node_idx = 0
        for obj_cfg in self.objects_config:
            count = obj_cfg['count']
            size_tensor = torch.tensor(obj_cfg['size'], device=self.device)
            # Автоматический расчет радиуса
            radius = math.sqrt((size_tensor[0] / 2)**2 + (size_tensor[1] / 2)**2)

            for i in range(count):
                self.sizes[node_idx] = size_tensor
                self.radii[node_idx] = radius
                
                # Инициализация узла в графе NetworkX
                self.graph.add_node(
                    node_idx,
                    name=obj_cfg['name'],
                    types=set(obj_cfg['type']),
                    size=tuple(size_tensor.tolist()),
                    radius=radius,
                    active=False,
                    on_surface_idx=-1,
                    surface_level=0
                )
                node_idx += 1
        
        self.default_positions = self.positions.clone()
        self.update_graph_state()

    def get_nodes_by_type(self, type_str: str, only_active: bool = False) -> list[int]:
        """Возвращает список индексов узлов с указанным типом."""
        nodes = []
        for i, data in self.graph.nodes(data=True):
            if type_str in data['types']:
                if not only_active or data['active']:
                    nodes.append(i)
        return nodes

    def get_objects_on_surface(self, surface_node_idx: int) -> list[int]:
        """Возвращает объекты, находящиеся на указанной поверхности."""
        return [
            i for i, idx in enumerate(self.on_surface_idx) if idx == surface_node_idx
        ]

    def update_graph_state(self):
        """Синхронизирует данные из тензоров в атрибуты узлов графа NetworkX."""
        for i in range(self.num_nodes):
            node_data = self.graph.nodes[i]
            node_data['position'] = tuple(self.positions[i].tolist())
            node_data['active'] = self.active[i].item()
            node_data['on_surface_idx'] = self.on_surface_idx[i].item()
            node_data['surface_level'] = self.surface_level[i].item()

    def print_graph_info(self, add_info=None):
        """Prints detailed and formatted information about the ObstacleGraph."""
        print("\n=== ObstacleGraph Information ===")
        
        # Graph Summary
        active_nodes = self.active.sum().item()
        print("\nGraph Summary:")
        if add_info is not None:
            print(add_info)
        print(f"  Total Nodes: {self.num_nodes}")
        print(f"  Active Nodes: {active_nodes}")
        print(f"  Device: {self.device}")
        
        # Object Type Counts
        type_counts = {}
        for obj_cfg in self.objects_config:
            type_counts[obj_cfg['name']] = obj_cfg['count']
        print("\nObject Types and Counts:")
        for name, count in type_counts.items():
            print(f"  {name.capitalize()}: {count}")
        
        # Node Details Table
        table_data = []
        for i, data in self.graph.nodes(data=True):
            pos = data['position']
            types = ", ".join(data['types'])
            size = data['size']
            row = [
                i,
                data['name'],
                types,
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                f"{data['radius']:.2f}",
                f"({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})",
                str(data['active']),
                data['on_surface_idx'],
                data['surface_level']
            ]
            table_data.append(row)
        
        headers = ["ID", "Name", "Types", "Position", "Radius", "Size", "Active", "On Surface", "Surface Level"]
        print("\nNode Details:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Surface Hierarchy
        print("\nSurface Hierarchy:")
        print(self.graph.nodes)
        for i in range(self.num_nodes):
            if self.on_surface_idx[i] != -1:
                obj_name = self.graph.nodes[i]['name']
                print(self.graph.nodes[i]['name'], self.graph.nodes[i], i)
                print(self.on_surface_idx)
                surface_name = self.graph.nodes[self.on_surface_idx[i].item()]['name']
                print(f"  Node {i} ({obj_name}) is on surface Node {self.on_surface_idx[i]} ({surface_name})")
            elif self.graph.nodes[i]['active']:
                print(f"  Node {i} ({self.graph.nodes[i]['name']}) is on floor")