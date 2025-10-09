import torch
import networkx as nx
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import combinations
import importlib.util
import random

def import_class_from_path(module_path, class_name):
    print(f"[DEBUG] Importing class '{class_name}' from module: {module_path}")
    spec = importlib.util.spec_from_file_location("custom_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_obj = getattr(module, class_name)
    print(f"[DEBUG] Successfully imported class: {class_obj}")
    return class_obj

module_path = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/scene_manager_for_pg.py"
SceneManager = import_class_from_path(module_path, "SceneManager")

class PathGenerator:
    def __init__(self, num_obstacles=6, config_path="source/isaaclab_tasks/isaaclab_tasks/direct/aloha/scene_items.json", device='cuda:0', ratio=8, room_len_x=10, room_len_y=8, test_mode=False):
        print(f"[DEBUG] Initializing PathGenerator with:")
        print(f"  - num_obstacles: {num_obstacles}")
        print(f"  - device: {device}")
        print(f"  - ratio: {ratio}")
        print(f"  - room_len_x: {room_len_x}")
        print(f"  - room_len_y: {room_len_y}")
        print(f"  - test_mode: {test_mode}")
        
        self.num_obstacles = num_obstacles
        self.device = device
        self.ratio = ratio
        self.room_len_x = room_len_x
        self.room_len_y = room_len_y
        self.test_mode = test_mode
        self.max_start_nodes = 1 if test_mode else float('inf')  # Ограничение на число стартовых узлов в тестовом режиме
        self.shift = [5, 4]  # Смещение для локальных координат
        
        print(f"[DEBUG] Creating Scene_manager with num_envs=1, device={device}, num_obstacles={num_obstacles}")
        self.scene_manager = SceneManager(num_envs=1, config_path=config_path, device=device)  # Новый вызов
        print(f"[DEBUG] Scene_manager created successfully")
        
        self.log_dir = "logs/aloha_data_graphs"#str(Path().resolve()) + "/logs/"
        print(f"[DEBUG] Log directory: {self.log_dir}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.paths_file = os.path.join("data", "all_paths.json")
        self.graphs_dir = os.path.join(self.log_dir, "graphs")
        print(f"[DEBUG] Paths file: {self.paths_file}")
        print(f"[DEBUG] Graphs directory: {self.graphs_dir}")
        
        os.makedirs(self.graphs_dir, exist_ok=True)
        self.all_paths = {}
        print(f"[DEBUG] PathGenerator initialization complete")

    def _create_grid_with_diagonals(self, width, height):
        print(f"[DEBUG] Creating grid with diagonals: width={width}, height={height}")
        graph = nx.grid_2d_graph(width, height)
        initial_nodes = len(graph.nodes)
        initial_edges = len(graph.edges)
        print(f"[DEBUG] Initial grid: {initial_nodes} nodes, {initial_edges} edges")
        
        for u, v in graph.edges():
            graph[u][v]['weight'] = 1.0  # Устанавливаем вес 1.0 для всех рёбер
        
        diagonal_count = 0
        for x in range(width):
            for y in range(height):
                if x + 1 < width and y + 1 < height:
                    graph.add_edge((x, y), (x + 1, y + 1), weight=1)  # Диагональ
                    diagonal_count += 1
                if x + 1 < width and y - 1 >= 0:
                    graph.add_edge((x, y), (x + 1, y - 1), weight=1)
                    diagonal_count += 1
        
        print(f"[DEBUG] Added {diagonal_count} diagonal edges")
        print(f"[DEBUG] Final grid: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph

    def find_boundary_nodes(self, graph):
        print(f"[DEBUG] Finding boundary nodes in graph with {len(graph.nodes)} nodes")
        if not graph.nodes:
            print(f"[DEBUG] Graph has no nodes, returning empty set")
            return set()
        
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 0
        boundary_nodes = {node for node in graph.nodes() if graph.degree(node) < max_degree}
        
        print(f"[DEBUG] Max degree: {max_degree}")
        print(f"[DEBUG] Found {len(boundary_nodes)} boundary nodes")
        print(f"[DEBUG] Boundary nodes sample: {list(boundary_nodes)[:10]}...")
        return boundary_nodes

    def find_expanded_boundary(self, graph, boundary_nodes, excluded_nodes=set()):
        print(f"[DEBUG] Finding expanded boundary from {len(boundary_nodes)} boundary nodes")
        print(f"[DEBUG] Excluded nodes: {len(excluded_nodes)}")
        
        expanded_boundary = set()
        for node in boundary_nodes:
            neighbors = list(graph.neighbors(node))
            valid_neighbors = [n for n in neighbors if n not in excluded_nodes]
            expanded_boundary.update(valid_neighbors)
        
        result = expanded_boundary - boundary_nodes
        print(f"[DEBUG] Expanded boundary: {len(result)} nodes")
        print(f"[DEBUG] Expanded boundary sample: {list(result)[:10]}...")
        return result

    def assign_edge_weights(self, graph, boundary_nodes, expanded_boundary):
        print(f"[DEBUG] Assigning edge weights")
        print(f"[DEBUG] Boundary nodes: {len(boundary_nodes)}")
        print(f"[DEBUG] Expanded boundary: {len(expanded_boundary)}")
        
        weight_changes = {'boundary': 0, 'expanded': 0, 'normal': 0}

        for u, v in graph.edges():
            old_weight = graph[u][v]['weight']
            if u in boundary_nodes or v in boundary_nodes:
                graph[u][v]['weight'] = max(old_weight, 3)
                weight_changes['boundary'] += 1
            elif u in expanded_boundary or v in expanded_boundary:
                graph[u][v]['weight'] = max(old_weight, 2)
                weight_changes['expanded'] += 1
            else:
                graph[u][v]['weight'] = old_weight
                weight_changes['normal'] += 1
        
        print(f"[DEBUG] Weight assignment complete:")
        print(f"  - Boundary edges: {weight_changes['boundary']}")
        print(f"  - Expanded boundary edges: {weight_changes['expanded']}")
        print(f"  - Normal edges: {weight_changes['normal']}")

    def _check_intersection(self, point, obstacle_positions, obstacle_radii, add_r=0.0):
        # print("obstacle_positions ", obstacle_positions)
        if len(obstacle_positions) == 0:
            return torch.tensor([False], device=self.device)
        # point_tensor = self.grid_to_real(point)
        # # for pos in obstacle_positions
        # print("point_tensor is ", point_tensor, point)
        result = False
        for pos in obstacle_positions:
            # print("pos ", pos, point)
            distances = torch.norm(pos - point)
            # print(distances)
            if distances < (0.35 + self.scene_manager.robot_radius + add_r):
                result = True
        # result = torch.any(distances < (obstacle_radii + self.scene_manager.robot_radius + add_r), dim=-1)
        return result

    def get_scene_grid(self, obstacle_positions, obstacle_radii):
        ratio_x = self.ratio * self.room_len_x
        ratio_y = self.ratio * self.room_len_y
        G = self._create_grid_with_diagonals(ratio_x, ratio_y)
        obstacle_positions = torch.tensor(obstacle_positions, device=self.device, dtype=torch.float32)
        obstacle_radii = torch.tensor(obstacle_radii, device=self.device, dtype=torch.float32)
        room_bounds = self.scene_manager.room_bounds
        add_r = 1 / self.ratio

        # print(f"[DEBUG] Initial graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        for node in list(G.nodes):
            x, y = node
            scaled_point = self.grid_to_real(node)
            # print(scaled_point)
            # print(1)
            if (self._check_intersection(scaled_point, obstacle_positions, obstacle_radii, add_r) or
                scaled_point[0] < room_bounds['x_min'] + 0.2 or
                scaled_point[0] > room_bounds['x_max'] - self.scene_manager.robot_radius or
                scaled_point[1] < room_bounds['y_min'] + self.scene_manager.robot_radius or
                scaled_point[1] > room_bounds['y_max'] - self.scene_manager.robot_radius):
                # print("remove node ", node, )
                G.remove_node(node)

        # print(f"[DEBUG] Graph after removing nodes: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        boundary_nodes = self.find_boundary_nodes(G)
        expanded_boundary = [boundary_nodes]
        levels = 2
        for i in range(levels):
            expanded_boundary.append(self.find_expanded_boundary(G, expanded_boundary[-1]))
        expanded_boundary.reverse()

        print(f"[DEBUG] Assigning edge weights for {len(G.edges)} edges")
        self.assign_edge_weights(G, boundary_nodes, expanded_boundary[-1])
        for u, v in G.edges():
            for i in range(1, len(expanded_boundary)):
                set_nodes = expanded_boundary[i]
                prev_set_nodes = expanded_boundary[i - 1]
                if (u in prev_set_nodes and v in set_nodes) or (u in set_nodes and v in set_nodes):
                    G[u][v]['weight'] = 1 + i

        print(f"[DEBUG] Graph has {len(G.nodes)} nodes and {len(G.edges)} edges after weight assignment")
        return G

    def find_nearest_reachable_node(self, graph, target):
        print(f"[DEBUG] Finding nearest reachable node to target: {target}")
        
        if target in graph and len(list(graph.neighbors(target))) > 0:
            print(f"[DEBUG] Target {target} is already reachable")
            return target
        
        min_distance = float('inf')
        nearest_node = None
        target_x, target_y = target
        candidates_checked = 0
        
        for node in graph.nodes:
            if len(list(graph.neighbors(node))) > 0:
                candidates_checked += 1
                x, y = node
                distance = abs(target_x - x) + abs(target_y - y)
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node
        
        print(f"[DEBUG] Checked {candidates_checked} candidate nodes")
        print(f"[DEBUG] Nearest reachable node: {nearest_node} (distance: {min_distance})")
        return nearest_node

    def remove_straight_segments(self, path):
        print(f"[DEBUG] Removing straight segments from path of length {len(path)}")
        
        if len(path) < 3:
            print(f"[DEBUG] Path too short, returning unchanged")
            return path
        
        filtered_path = [path[0]]
        removed_count = 0
        
        for i in range(1, len(path) - 1):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            x3, y3 = path[i + 1]
            
            if (x3 - x1) * (y2 - y1) != (y3 - y1) * (x2 - x1):
                filtered_path.append(path[i])
            else:
                removed_count += 1
        
        filtered_path.append(path[-1])
        print(f"[DEBUG] Removed {removed_count} straight segments, new length: {len(filtered_path)}")
        return filtered_path

    def save_graph_image(self, graph, path, config_key, n_save=1000):
        print(f"[DEBUG] Saving graph image for config: {config_key}")
        if len(path) > 4:
            static_path = self.remove_straight_segments(path)
        else:
            static_path = path
        pos = {node: node for node in graph.nodes}
        
        node_colors = []
        color_counts = {'green': 0, 'red': 0, 'lightblue': 0, 'yellow': 0, 'cyan': 0}
        start_node = path[0] if path else None
        end_node = path[-1] if path else None
        
        for node in graph.nodes:
            if node == end_node:
                node_colors.append('yellow')  # Цель — жёлтый
                color_counts['yellow'] += 1
            elif node == start_node:
                node_colors.append('cyan')   # Старт — голубой
                color_counts['cyan'] += 1
            elif node in static_path:
                node_colors.append('green')
                color_counts['green'] += 1
            elif node in path:
                node_colors.append('red')
                color_counts['red'] += 1
            else:
                node_colors.append('lightblue')
                color_counts['lightblue'] += 1
        
        print(f"[DEBUG] Node colors: {color_counts}")
        
        plt.figure(figsize=(32, 32))
        nx.draw(graph, pos, with_labels=False, node_color=node_colors, node_size=200)
        
        edge_labels = {(u, v): f"{d.get('weight', 100.0):.1f}" for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=12, font_color='red')
        
        # Добавляем точку (0, 0)
        plt.scatter([0], [0], color='black', s=500, marker='*', label='Grid Origin (0, 0)')
        
        # Добавляем оси координат
        arrow_length = min(self.ratio * self.room_len_x, self.ratio * self.room_len_y) * 0.1  # 10% от размера сетки
        plt.arrow(0, 0, arrow_length, 0, head_width=1, head_length=2, fc='blue', ec='blue', label='X-axis')
        plt.arrow(0, 0, 0, arrow_length, head_width=1, head_length=2, fc='red', ec='red', label='Y-axis')
        
        # Добавляем подписи осей
        plt.text(arrow_length + 2, 0, 'X', fontsize=14, color='blue', verticalalignment='center')
        plt.text(0, arrow_length + 2, 'Y', fontsize=14, color='red', horizontalalignment='center')
        
        plt.title(f"Grid Graph for Configuration {config_key}")
        if path:
            start_node = path[0]
            end_node = path[-1]
            graph_img_file = os.path.join(self.graphs_dir, f"grid_graph_{config_key}_start_{start_node[0]}_{start_node[1]}_end_{end_node[0]}_{end_node[1]}.png")
        else:
            graph_img_file = os.path.join(self.graphs_dir, f"grid_graph_{config_key}_no_path.png")
        
        print(f"[DEBUG] Saving graph image to: {graph_img_file}")
        plt.legend(fontsize=12)
        plt.savefig(graph_img_file, dpi=200)
        plt.close()
        print(f"[DEBUG] Graph image saved successfully")

    def generate_paths(self, n_save=1500, targets=[[-4.5, 0]]):
        """
        Генерирует и сохраняет пути для всех возможных комбинаций координат препятствий.
        Логика:
        - Извлекаем grid_coords для movable_obstacle из конфига (self.scene_manager.config).
        - Для каждого k (от 1 до len(grid_coords)): Генерируем combinations(grid_coords, k) — подмножества позиций.
        - Для каждой комбо позиций: 
        - Создаем config_str как ','.join(sorted([f"{x:.1f}_{y:.1f}_{z:.1f}" for x,y,z in combo])).
        - Устанавливаем эти позиции для активных movable_obstacle nodes с помощью set_obstacle_positions (первые k nodes активны, остальным — default).
        - Для каждой цели в targets: Устанавливаем позицию цели с set_goal_position (учитывая surface_only: на surface_provider с margin=0 для фикса).
        - Строим граф, удаляем intersecting nodes, вычисляем пути как раньше.
        - Сохраняем в all_paths.json после n_save конфигураций или в конце.
        """
        print(f"[DEBUG] Starting path generation for {n_save} saves, targets={targets}")
        
        # Шаг 1: Извлекаем grid_coords для movable_obstacle из конфига
        movable_grid_coords = []
        for obj_cfg in self.scene_manager.config:
            if "movable_obstacle" in obj_cfg['type']:
                placement = obj_cfg.get('placement', [])
                if placement and placement[0]['strategy'] == 'grid':
                    movable_grid_coords = placement[0]['grid_coordinates']  # [[x,y,z], ...]
                    num_movable = obj_cfg['count']  # 3 для chairs
                    break
        if not movable_grid_coords:
            raise ValueError("[ERROR] No movable_obstacle with grid strategy found in config")
        
        print(f"[DEBUG] Movable obstacle grid_coords: {movable_grid_coords}, count={num_movable}")
        
        # Шаг 2: Извлекаем возможные цели (targets уже даны, но проверяем possible_goal)
        goal_surfaces = []
        for obj_cfg in self.scene_manager.config:
            if "possible_goal" in obj_cfg['type']:
                placement = obj_cfg.get('placement', [])
                if "surface_only" in obj_cfg['type'] and placement and placement[0]['strategy'] == 'on_surface':
                    # Находим surface_provider grid_coords
                    surface_types = placement[0]['surface_types']
                    for surf_cfg in self.scene_manager.config:
                        if any(t in surf_cfg['type'] for t in surface_types):
                            surf_placement = surf_cfg.get('placement', [])
                            if surf_placement and surf_placement[0]['strategy'] == 'grid':
                                goal_surfaces.extend(surf_placement[0]['grid_coordinates'])
        if goal_surfaces:
            print(f"[DEBUG] Goal possible positions from surfaces: {goal_surfaces}")
            targets = goal_surfaces
            # Но используем targets как фиксированные, можно расширить targets += goal_surfaces если нужно
        
        # Шаг 3: Генерация комбинаций подмножеств координат (subsets)
        from itertools import combinations
        grid = self._create_grid_with_diagonals(self.room_len_x * self.ratio, self.room_len_y * self.ratio)
        print(f"[DEBUG] Base grid created with {len(grid.nodes)} nodes")
        
        num_processed = 0
        for k in range(0, len(movable_grid_coords) + 1):  # k=1 to 3
            for comb in combinations(movable_grid_coords, k):  # comb = tuple of [x,y,z] lists
                # Создаем config_str: sorted string of positions
                sorted_comb = sorted(comb, key=lambda p: (p[0], p[1], p[2]))  # Sort by x,y,z
                config_str = ','.join([f"{p[0]:.1f}_{p[1]:.1f}_{p[2]:.1f}" for p in sorted_comb])
                print(f"[DEBUG] Processing config {config_str} with {k} obstacles")
                
                # Шаг 4: Устанавливаем позиции препятствий детерминировано
                env_ids = torch.tensor([0], device=self.device)
                self.scene_manager.set_obstacle_positions(env_ids, list(comb))  # Новая функция, см. пункт 2
                
                # Извлекаем positions/radii для активных movable_obstacle
                graph = self.scene_manager.graphs[0]
                obstacle_indices = graph.get_nodes_by_type("movable_obstacle", only_active=True)
                obstacle_positions = graph.positions[obstacle_indices, :2]  # [num_active, 2]
                obstacle_radii = graph.radii[obstacle_indices]  # [num_active]
                
                # Шаг 5: Строим граф сцены (удаляем intersections)
                config_graph = self.get_scene_grid(obstacle_positions, obstacle_radii)
                print(f"[DEBUG] Config graph for {config_str}: {len(config_graph.nodes)} nodes")
                
                # Шаг 6: Для каждой цели в targets
                for target in targets:
                    # Устанавливаем позицию цели (с учетом surface_only)
                    self.scene_manager.set_goal_position(env_ids, target)  # Новая функция, см. пункт 2
                    
                    # Конвертируем target в grid (теперь target — реальная, после set)
                    goal_pos = self.scene_manager.goal_positions[0, :3].cpu().tolist()  # После set
                    grid_targets = self.get_grid_targets([goal_pos], obstacle_positions, obstacle_radii, config_graph)
                    
                    for grid_target in grid_targets:
                        print(f"[DEBUG] Computing paths for target {grid_target}")
                        
                        # Вычисляем shortest paths (как раньше)
                        try:
                            paths_from_target = nx.single_source_dijkstra_path(config_graph, grid_target)
                        except nx.NodeNotFound:
                            print(f"[WARNING] Target {grid_target} not in graph, skipping")
                            continue
                        
                        # Обрабатываем пути (reverse, simplify)
                        start_nodes = list(paths_from_target.keys())
                        if self.test_mode:
                            start_nodes = random.sample(start_nodes, min(self.max_start_nodes, len(start_nodes)))
                        
                        for start in start_nodes:
                            path = paths_from_target[start][::-1]  # Reverse
                            simplified_path = _simplify_path(path)
                            num_processed += 1
                            # if num_processed % n_save == 0:
                            #     self.save_graph_image(config_graph, simplified_path, config_str)
                            if len(simplified_path) > 1:
                                self.all_paths.setdefault(config_str, {}).setdefault(str(grid_target), {})[str(start)] = simplified_path
                        # self.save_graph_image(config_graph, path, config_key=15, n_save=500000000)
                        print(f"[DEBUG] Processed {len(start_nodes)} start nodes for target {grid_target}")
                
                
            
        self._save_paths()
        print(f"\n[DEBUG] ===== PATH GENERATION COMPLETE =====")
        print(f"[DEBUG] Total configurations processed: {len(self.all_paths)}")

    def _save_paths(self):
        import re
        json_paths = {}
        print("all paths: ", self.all_paths)
        for config_str, targets_inner in self.all_paths.items():
            print("config_str: ", config_str)
            json_paths[config_str] = {}
            for target, nodes in targets_inner.items():
                print("target: ", target)
                target_str = re.sub(r"[()]", "", target)  # str(grid_target) уже tuple, но на str
                json_paths[config_str][target_str] = {}
                print(config_str, target_str)
                for node, path in nodes.items():
                    print("node", node)
                    print("path", path)
                    node_str = re.sub(r"[()]", "", node)
                    json_paths[config_str][target_str][node_str] = [list(p) for p in path]
        with open(self.paths_file, 'w') as f:
            json.dump(json_paths, f, indent=4)
        print(f"[DEBUG] Saved {len(self.all_paths)} configurations to {self.paths_file}")
        print(f"[DEBUG] JSON file size: {os.path.getsize(self.paths_file)} bytes")

    def get_grid_targets(self, targets, obstacle_positions, obstacle_radii, config_graph):
        print(f"[DEBUG] Converting targets to grid: {targets}")
        grid_targets = []
        for target in targets:
            grid_node = self.real_to_grid(target)
            
            scaled_point = self.grid_to_real(grid_node)
            print(3)
            is_valid = (
                not self._check_intersection(scaled_point, obstacle_positions, obstacle_radii) and
                grid_node in config_graph and
                len(list(config_graph.neighbors(grid_node))) > 0
            )
            
            if is_valid:
                grid_targets.append(grid_node)
            else:
                nearest_node = self.find_nearest_reachable_node(config_graph, grid_node)
                if nearest_node is not None:
                    grid_targets.append(nearest_node)
                else:
                    print(f"Warning: No reachable node found for target {target}. Using original grid node.")
                    grid_targets.append(grid_node)
        
        print(f"[DEBUG] Grid targets: {grid_targets}")
        return grid_targets

    def grid_to_real(self, grid_point):
        """Преобразует сеточные координаты (x, y) в реальные с поворотом на 90 градусов по часовой стрелке."""
        x, y = grid_point
        real_x = x / self.ratio - self.shift[0]  # x_real = y / ratio + shift_x
        real_y = y / self.ratio - self.shift[1]  # y_real = -x / ratio + shift_y
        return torch.tensor([real_x, real_y], device=self.device, dtype=torch.float32)

    def real_to_grid(self, real_point):
        """Преобразует реальные координаты (x, y) в сеточные с поворотом на 90 градусов по часовой стрелке."""
        x, y, _ = real_point
        grid_x = int((x + self.shift[0]) * self.ratio)  # x_grid = (-y + shift_y) * ratio
        grid_y = int((y + self.shift[1]) * self.ratio)   # y_grid = (x - shift_x) * ratio
        return (grid_x, grid_y)

import math

def _simplify_path(points, tol=1e-5):
    """Удаляет точки на прямых сегментах и 'ложные углы'"""
    if len(points) <= 2:
        return points

    simplified = [points[0]]
    for i in range(1, len(points) - 1):
        prev = simplified[-1]  # Последняя оставленная точка
        curr = points[i]
        nxt = points[i + 1]

        # Векторы
        v1 = (curr[0] - prev[0], curr[1] - prev[1])
        v2 = (nxt[0] - curr[0], nxt[1] - curr[1])

        # -------------------------
        # 1. Удаляем точки на прямых
        len1 = math.hypot(*v1)
        len2 = math.hypot(*v2)
        if len1 >= tol and len2 >= tol:
            v1n = (v1[0] / len1, v1[1] / len1)
            v2n = (v2[0] / len2, v2[1] / len2)
            if abs(v1n[0] - v2n[0]) < tol and abs(v1n[1] - v2n[1]) < tol:
                continue

        # -------------------------
        # 2. Удаляем 'ложные углы'
        dx1, dy1 = int(round(v1[0])), int(round(v1[1]))
        dx2, dy2 = int(round(v2[0])), int(round(v2[1]))
        if ((abs(dx1) == 1 and abs(dy1) == 1) and (abs(dx2) + abs(dy2) == 1)) or \
           ((abs(dx2) == 1 and abs(dy2) == 1) and (abs(dx1) + abs(dy1) == 1)):
            continue

        simplified.append(curr)

    simplified.append(points[-1])
    return simplified


if __name__ == "__main__":
    print("[DEBUG] Starting PathGenerator script...")
    generator = PathGenerator(num_obstacles=3, device='cuda:0', test_mode=False)
    generator.generate_paths(n_save=500000, targets=[[-4.5, 0],[-4.5, -1],[-4.5, 1]])    #[[-4.2, -1],[-4.2, 0],[-4.2, 1]])
    print("[DEBUG] PathGenerator script complete!")