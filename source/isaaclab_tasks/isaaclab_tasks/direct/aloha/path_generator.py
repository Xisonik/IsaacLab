
import json
import math
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Iterable, Optional
import networkx as nx
import time
from datetime import datetime
# from .config_codec import ConfigCodec
import os
import importlib.util
def import_class_from_path(module_path, class_name):
    spec = importlib.util.spec_from_file_location("custom_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
current_dir = os.getcwd()
module_path = os.path.join(current_dir, "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/config_codec.py")
ConfigCodec = import_class_from_path(module_path, "ConfigCodec")

class PathGeneratorV2:
    """Standalone generator with optional graph image saving."""

    def __init__(
        self,
        config_path: str,
        ratio: int = 4,
        room_len_x: int = 10,
        room_len_y: int = 10,
        shift_xy: Tuple[float, float] = (0.0, 0.0),
        robot_radius: float = 0.5,
        room_bounds: Dict[str, float] = None,
        obstacle_types: Iterable[str] = ("movable_obstacle", "static_obstacle"),
        goal_grid_types: Iterable[str] = ("goal_possible_positions",),
        max_movable_subset_size: int = None,
        min_movable_subset_size: int = None,
        limit_start_nodes: int = None,
        add_clearance: float = None,
        save_dir: str = "data",
        graphs_dir: str = "logs/aloha_data_graphs/graphs",
        max_start_obstacle_dist: float | None = None,   # макс. допустимая дистанция (м) от старта до препятствий
        start_obst_dist_mode: str = "edge",             # "edge" (по кромкам) или "center" (по центрам)
    ):

        self.config_path = config_path
        self.ratio = ratio
        self.room_len_x = room_len_x
        self.room_len_y = room_len_y
        self.shift = shift_xy
        self.robot_radius = robot_radius
        self.room_bounds = room_bounds or {"x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5}
        self.obstacle_types = tuple(obstacle_types)
        self.goal_grid_types = tuple(goal_grid_types)
        self.max_movable_subset_size = max_movable_subset_size
        self.min_movable_subset_size = min_movable_subset_size
        
        self.limit_start_nodes = limit_start_nodes
        self.add_clearance = (1.0 / ratio) if add_clearance is None else add_clearance
        self.paths_file = str(Path(save_dir) / "all_paths.json")
        self.graphs_dir = str(Path(graphs_dir))
        Path(self.graphs_dir).mkdir(parents=True, exist_ok=True)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.max_start_obstacle_dist = max_start_obstacle_dist
        self.start_obst_dist_mode = start_obst_dist_mode  # "edge"|"center"
        with open(config_path, "r") as f:
            raw = json.load(f)
        self.raw = raw
        self.objects_cfg: List[dict] = raw["objects"]
        self.type_placements: Dict[str, dict] = raw.get("type_placements", {})

        self.type_radii: Dict[str, float] = self._compute_type_radii(self.objects_cfg)
        self.type_grids: Dict[str, List[Tuple[float, float, float]]] = self._collect_type_grids(self.type_placements)
        self.codec = ConfigCodec(self.type_grids,
                         types=self.obstacle_types,   # важно: тот же порядок
                         quant=8,
                         max_radius=70.0)
        self.all_paths: Dict[str, Dict[str, Dict[str, List[Tuple[int, int]]]]] = {}

    # ----------- Visual debug -----------
    def save_graph_image(
        self,
        G: nx.Graph,
        out_path: str,
        obs_xy: Optional[List[Tuple[float, float]]] = None,
        obs_r: Optional[List[float]] = None,
        target: Optional[Tuple[int, int]] = None,
        starts: Optional[List[Tuple[int, int]]] = None,
        path: Optional[List[Tuple[int, int]]] = None,
        dpi: int = 180,
    ) -> str:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        ax.set_aspect("equal")

        pos = {n: (n[0], n[1]) for n in G.nodes()}

        if G.number_of_edges() > 0:
            lines, widths, colors = [], [], []
            for u, v, data in G.edges(data=True):
                x1, y1 = pos[u]; x2, y2 = pos[v]
                lines.append([(x1, y1), (x2, y2)])
                w = float(data.get("weight", 1.0))
                widths.append(0.5 + 0.6 * (w - 1.0))
                shade = min(0.85, 0.35 + 0.1 * (w - 1.0))
                colors.append((shade, shade, shade))
            lc = LineCollection(lines, linewidths=widths, colors=colors, zorder=1)
            ax.add_collection(lc)

        xs = [pos[n][0] for n in G.nodes()]
        ys = [pos[n][1] for n in G.nodes()]
        ax.scatter(xs, ys, s=6, c="#a7d0ff", edgecolors="none", zorder=2)

        if target is not None and target in pos:
            ax.scatter([pos[target][0]], [pos[target][1]], s=40, c="yellow", edgecolors="black", zorder=4)

        if starts:
            sx = [pos[s][0] for s in starts if s in pos]
            sy = [pos[s][1] for s in starts if s in pos]
            if sx:
                ax.scatter(sx, sy, s=20, c="cyan", edgecolors="black", zorder=3)

        if path and len(path) >= 2:
            px = [n[0] for n in path]
            py = [n[1] for n in path]
            ax.plot(px, py, "-", lw=2.0, c="red", zorder=5)

        if obs_xy and obs_r:
            for (x, y), r in zip(obs_xy, obs_r):
                gx = (x + self.shift[0]) * self.ratio
                gy = (y + self.shift[1]) * self.ratio
                gr = r * self.ratio
                circ = Circle((gx, gy), gr, facecolor="none", edgecolor="black", linewidth=1.0, alpha=0.9, zorder=6)
                ax.add_patch(circ)

        ax.set_xlim(-1, self.ratio * self.room_len_x + 1)
        ax.set_ylim(-1, self.ratio * self.room_len_y + 1)
        ax.invert_yaxis()
        ax.set_title("Grid graph (target=yellow, starts=cyan, path=red)")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print("save to ", out_path)
        return out_path

    # ----------- JSON helpers -----------
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

    # ----------- Coord transforms -----------
    def grid_to_real(self, grid_point: Tuple[int, int]) -> Tuple[float, float]:
        x, y = grid_point
        return (x / self.ratio - self.shift[0], y / self.ratio - self.shift[1])

    def real_to_grid(self, real_point: Tuple[float, float, float]) -> Tuple[int, int]:
        x, y, _ = real_point
        return (int((x + self.shift[0]) * self.ratio), int((y + self.shift[1]) * self.ratio))

    # ----------- Graph construction -----------
    @staticmethod
    def _grid8(width: int, height: int) -> nx.Graph:
        G = nx.grid_2d_graph(width, height)
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0
        for i in range(width):
            for j in range(height):
                if i + 1 < width and j + 1 < height:
                    G.add_edge((i, j), (i + 1, j + 1), weight=1.0)
                if i + 1 < width and j - 1 >= 0:
                    G.add_edge((i, j), (i + 1, j - 1), weight=1.0)
        return G

    @staticmethod
    def _boundary_nodes(G: nx.Graph) -> set:
        if not G.nodes:
            return set()
        deg = dict(G.degree())
        md = max(deg.values()) if deg else 0
        return {n for n in G.nodes if G.degree(n) < md}

    @staticmethod
    def _expand_once(G: nx.Graph, S, exclude: set) -> set:
        out = set()
        for n in S:
            for nb in G.neighbors(n):
                if nb not in exclude:
                    out.add(nb)
        return out.difference(set(S))

    @staticmethod
    def _assign_ring_weights(G: nx.Graph, boundary: set, extra: set, rings: List[set]):
        for u, v in G.edges():
            if u in boundary or v in boundary:
                G[u][v]["weight"] = max(G[u][v]["weight"], 3.0)
            elif u in extra or v in extra:
                G[u][v]["weight"] = max(G[u][v]["weight"], 2.0)
        for i in range(1, len(rings)):
            ring = rings[i]
            prev = rings[i - 1]
            for u, v in G.edges():
                if (u in prev and v in ring) or (v in prev and u in ring) or (u in ring and v in ring):
                    G[u][v]["weight"] = max(G[u][v]["weight"], 1.0 + i)

    # ----------- Collision & pruning -----------
    def _collides(self, xy: Tuple[float, float], obs_xy: List[Tuple[float, float]], obs_r: List[float]) -> bool:
        if not obs_xy:
            return False
        for (ox, oy), r in zip(obs_xy, obs_r):
            dx = ox - xy[0]
            dy = oy - xy[1]
            if math.hypot(dx, dy) < (r + self.robot_radius + self.add_clearance):
                return True
        return False

    def _build_obs_cloud(self, config_subset: Dict[str, List[Tuple[float, float, float]]]) -> Tuple[List[Tuple[float, float]], List[float]]:
        pts_xy: List[Tuple[float, float]] = []
        rads: List[float] = []
        for t in self.obstacle_types:
            pts = config_subset.get(t, [])
            r = self.type_radii.get(t, 0.35)
            for (x, y, _z) in pts:
                pts_xy.append((x, y))
                rads.append(r)
        return pts_xy, rads

    def _pruned_graph(self, obs_xy: List[Tuple[float, float]], obs_r: List[float]) -> nx.Graph:
        width = self.ratio * self.room_len_x
        height = self.ratio * self.room_len_y
        G = self._grid8(width, height)

        bounds = self.room_bounds
        x_min = bounds["x_min"] + 0.2
        x_max = bounds["x_max"] - self.robot_radius
        y_min = bounds["y_min"] + self.robot_radius
        y_max = bounds["y_max"] - self.robot_radius

        to_remove = []
        for n in list(G.nodes()):
            xy = self.grid_to_real(n)
            if (
                self._collides(xy, obs_xy, obs_r) or
                xy[0] < x_min or xy[0] > x_max or
                xy[1] < y_min or xy[1] > y_max
            ):
                to_remove.append(n)
        G.remove_nodes_from(to_remove)

        boundary = self._boundary_nodes(G)
        rings = [boundary]
        levels = 2
        for _ in range(levels):
            nxt = self._expand_once(G, rings[-1], exclude=set())
            rings.append(nxt)
        rings = list(reversed(rings))
        extra = rings[-1] if rings else set()
        self._assign_ring_weights(G, boundary, extra, rings)
        return G

    # ----------- Targets & paths -----------
    def _grid_targets_from_types(self) -> List[Tuple[int, int]]:
        targets: List[Tuple[int, int]] = []
        for t in self.goal_grid_types:
            for p in self.type_grids.get(t, []):
                targets.append(self.real_to_grid(p))
        targets = sorted(set(targets))
        return targets

    @staticmethod
    def _nearest_reachable(G: nx.Graph, target: Tuple[int, int]):
        if target in G and any(True for _ in G.neighbors(target)):
            return target
        tx, ty = target
        best = None
        best_d = 10**9
        for n in G.nodes():
            if any(True for _ in G.neighbors(n)):
                d = abs(tx - n[0]) + abs(ty - n[1])
                if d < best_d:
                    best_d = d
                    best = n
        return best

    @staticmethod
    def _simplify(points: List[Tuple[int, int]], tol: float = 1e-5) -> List[Tuple[int, int]]:
        if len(points) <= 2:
            return points
        out = [points[0]]
        for i in range(1, len(points) - 1):
            prev = out[-1]
            cur = points[i]
            nxt = points[i + 1]
            v1 = (cur[0] - prev[0], cur[1] - prev[1])
            v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
            l1 = math.hypot(*v1)
            l2 = math.hypot(*v2)
            if l1 >= tol and l2 >= tol:
                v1n = (v1[0] / l1, v1[1] / l1)
                v2n = (v2[0] / l2, v2[1] / l2)
                if abs(v1n[0] - v2n[0]) < tol and abs(v1n[1] - v2n[1]) < tol:
                    continue
            dx1, dy1 = int(round(v1[0])), int(round(v1[1]))
            dx2, dy2 = int(round(v2[0])), int(round(v2[1]))
            if ((abs(dx1) == 1 and abs(dy1) == 1) and (abs(dx2) + abs(dy2) == 1)) or \
               ((abs(dx2) == 1 and abs(dy2) == 1) and (abs(dx1) + abs(dy1) == 1)):
                continue
            out.append(cur)
        out.append(points[-1])
        return out

    # ----------- Generation helpers -----------
    def _config_key(self, config_subset: Dict[str, List[Tuple[float, float, float]]]) -> str:
        parts = []
        for t in sorted(config_subset.keys()):
            pts = sorted(config_subset[t])
            s = ",".join([f"{x:.1f}_{y:.1f}_{z:.1f}" for x, y, z in pts])
            parts.append(f"{t}:{s}")
        return "|".join(parts) if parts else "empty"

    def _movable_subsets(self) -> List[List[Tuple[float, float, float]]]:
        movable_pts = self.type_grids.get("movable_obstacle", [])
        if not movable_pts:
            return [[]]

        m = len(movable_pts)

        # текущее "потолочное" ограничение оставляем
        lim_by_param = self.max_movable_subset_size if self.max_movable_subset_size is not None else m
        k_min = self.min_movable_subset_size if self.min_movable_subset_size is not None else 0
        # НОВОЕ: читаем мин/макс из dict-параметров
           # по умолчанию прежнее поведение
        k_max_type = m
        # итоговый k_max — минимум из всех «потолков»
        k_max = max(0, min(m, lim_by_param, k_max_type))

        if k_min > k_max:
            return []  # нет допустимых конфигураций

        out = []
        for k in range(k_min, k_max + 1):
            out.extend(list(combinations(movable_pts, k)))
        return [list(x) for x in out]

    # ----------- Main generation -----------
    def _fmt_hms(self, s: float) -> str:
        s = max(0, int(s))
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def generate(self, save_graph_images: bool = True, save_every_n: Optional[int] = None):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Generate start, time: ", current_time)
        static_pts = []
        if "static_obstacle" in self.obstacle_types:
            static_pts = self.type_grids.get("static_obstacle", [])
            if not static_pts and "surface_provider" in self.type_grids:
                static_pts = self.type_grids["surface_provider"]
        targets_grid = self._grid_targets_from_types()
        if not targets_grid:
            raise RuntimeError("No target grid positions found for the requested goal_grid_types.")

        result: Dict[str, Dict[str, Dict[str, List[Tuple[int, int]]]]] = {}
        img_counter = 0
        t0 = time.perf_counter()
        mov_subsets = self._movable_subsets()
        cfg_total = len(mov_subsets)
        print("Total num: ", cfg_total)
        cfg_idx = 0
        for mov_subset in mov_subsets:
            cfg_idx += 1
            if (cfg_idx - 1) % 100 == 0 and cfg_idx >= 1:
                elapsed_s = time.perf_counter() - t0
                frac = cfg_idx / cfg_total
                remaining_s = elapsed_s * (cfg_total-cfg_idx) / cfg_idx
                print(f"[{cfg_idx}/{cfg_total}] {frac*100:5.1f}%  "
                    f"pass={self._fmt_hms(elapsed_s)}  "
                    f"remain={self._fmt_hms(remaining_s)}")
            cfg_subset = {}
            if "static_obstacle" in self.obstacle_types and static_pts:
                cfg_subset["static_obstacle"] = static_pts
            if "movable_obstacle" in self.obstacle_types:
                cfg_subset["movable_obstacle"] = mov_subset

            # cfg_key = self._config_key(cfg_subset)
            cfg_id = self.codec.encode_id_hex(cfg_subset)

            obs_xy, obs_r = self._build_obs_cloud(cfg_subset)
            G = self._pruned_graph(obs_xy, obs_r)

            for tgt in targets_grid:
                tgt_node = tgt
                if tgt_node not in G:
                    alt = self._nearest_reachable(G, tgt_node)
                    if alt is None:
                        continue
                    tgt_node = alt
                try:
                    paths_from_target = nx.single_source_dijkstra_path(G, tgt_node)
                except Exception:
                    continue
                start_nodes = list(paths_from_target.keys())
                if self.limit_start_nodes is not None and self.limit_start_nodes < len(start_nodes):
                    start_nodes = start_nodes[: self.limit_start_nodes]
                # ВСТАВИТЬ ЭТО:
                start_nodes = [s for s in start_nodes if self._start_within_obstacle_range(s, obs_xy, obs_r)]
                if not start_nodes:
                    continue

                for start in start_nodes:
                    path = list(reversed(paths_from_target[start]))
                    if len(path) < 2:
                        continue
                    simp = self._simplify(path)
                    result.setdefault(cfg_id, {}).setdefault(f"{tgt_node[0]},{tgt_node[1]}", {})[f"{start[0]},{start[1]}"] = simp

                    if save_graph_images and (save_every_n is None or (img_counter % max(1, save_every_n) == 0)):
                        out_png = Path(self.graphs_dir) / f"G_{img_counter:06d}.png"
                        self.save_graph_image(
                            G, str(out_png),
                            obs_xy, obs_r,
                            target=tgt_node,
                            starts=[start],
                            path=simp,
                        )
                    img_counter += 1

        self.all_paths = result
        print("write result: start")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current time: ", current_time)
        with open(self.paths_file, "w") as f:
            json.dump(result, f, indent=2)
        print("write result: end")
        return self.paths_file
    
    def _start_within_obstacle_range(
        self,
        start_node: tuple[int, int],
        obs_xy: list[tuple[float, float]],
        obs_r: list[float],
    ) -> bool:
        """True, если старт находится не дальше self.max_start_obstacle_dist
        от хотя бы одного препятствия (с учётом режима измерения)."""
        if self.max_start_obstacle_dist is None or not obs_xy:
            return True

        sx, sy = self.grid_to_real(start_node)  # -> реальные координаты (м)
        best = float("inf")
        for (ox, oy), r in zip(obs_xy, obs_r):
            d = math.hypot(ox - sx, oy - sy)
            if self.start_obst_dist_mode == "edge":
                d = max(0.0, d - (r + self.robot_radius))  # расстояние по кромкам
            # else "center": d как есть
            if d < best:
                best = d
                if best <= self.max_start_obstacle_dist:
                    return True
        return False


if __name__ == "__main__":
    pg = PathGeneratorV2(
        config_path="source/isaaclab_tasks/isaaclab_tasks/direct/aloha/scene_items.json",  # твой JSON
        ratio=4, room_len_x=10, room_len_y=10,
        shift_xy=(5, 5),
        robot_radius=0.5,
        room_bounds={"x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5},
        obstacle_types=("movable_obstacle",),   # можно менять "static_obstacle"
        goal_grid_types=("goal_possible_positions",),                    # сетки для целей
        max_movable_subset_size=10,        # ограничь мощность комбинаций movable
        min_movable_subset_size=7,        # ограничь мощность комбинаций movable
        limit_start_nodes=None,           # можно ограничить число стартов на таргет
        max_start_obstacle_dist=7, # метры
        start_obst_dist_mode="center",
    )

    out_path = pg.generate(save_graph_images = True, save_every_n = 20000)
    print("saved to:", out_path)