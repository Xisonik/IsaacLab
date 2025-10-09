import torch
import math

class Control_module:
    def __init__(self, num_envs: int, device: str = 'cuda:0'):
        """
        Инициализирует модуль управления для векторизованной среды.

        Args:
            num_envs (int): Количество сред.
            ratio (float): Масштабный коэффициент для преобразования координат.
            device (str): Устройство для тензоров ('cuda:0' или 'cpu').
        """
        #print(f"[DEBUG] Initializing Control_module with:")
        print(f"  - num_envs: {num_envs}")
        print(f"  - device: {device}")
        
        self.num_envs = num_envs
        self.device = device
        max_path_length = 72
        self.max_path_length = max_path_length
        self.paths = torch.full((self.num_envs, max_path_length, 2), -666.0, device=self.device, dtype=torch.float32)  # [num_envs, max_path_length, 2]
        self.current_pos = torch.zeros((num_envs, 2), device=device)
        self.target_positions = torch.zeros((num_envs, 2), device=device)
        self.end = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.start = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.first_ep = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.lookahead_distance = 0.35
        self.linear_velocity = 0.6
        self.max_angular_velocity = 3
        
        #print(f"[DEBUG] Control parameters:")
        # print(f"  - lookahead_distance: {self.lookahead_distance}")
        # print(f"  - linear_velocity: {self.linear_velocity}")
        # print(f"  - max_angular_velocity: {self.max_angular_velocity}")
        #print(f"[DEBUG] Control_module initialization complete")

    def update(self, current_positions: torch.Tensor, target_positions: torch.Tensor, paths: torch.Tensor, env_ids: torch.Tensor = None):
        """
        Обновляет текущие позиции, цели и пути для следования.

        Args:
            current_positions (torch.Tensor): Текущие позиции роботов [num_envs, 2].
            target_positions (torch.Tensor): Целевые позиции [num_envs, 2].
            paths (torch.Tensor): Пути для каждой среды [num_envs, max_path_length, 2].
        """
        # print(f"\n[DEBUG] ===== UPDATE METHOD =====")
        #print(f"[DEBUG] Input shapes:")
        # print(f"  - current_positions: {current_positions.shape}")
        # print(f"  - target_positions: {target_positions.shape}")
        # print(f"  - paths: {paths.shape}")
        
        #print(f"[DEBUG] Current positions sample: {current_positions[:3]}")
        #print(f"[DEBUG] Target positions sample: {target_positions[:3]}")
        #print(f"[DEBUG] Paths sample (first env, first 3 points): {paths[0, :3]}")
        # Reset states
        prev_end = self.end.clone()
        prev_start = self.start.clone()
        if env_ids is None:
            print("env_ids is none in control.update")
            self.paths = paths.clone()
            self.current_pos = current_positions.clone()
            self.target_positions = target_positions.clone()
            self.end[:] = False
            self.start[:] = True
            self.first_ep[:] = True
        else:
            self.current_pos[env_ids] = current_positions.clone()
            self.target_positions[env_ids] = target_positions.clone()
            self.paths[env_ids] = paths.clone()
            self.end[env_ids] = False
            self.start[env_ids] = True
            self.first_ep[env_ids] = True
       
        #print(f"[DEBUG] State reset:")
        # print(f"  - end: {prev_end.sum().item()} -> {self.end.sum().item()} (True count)")
        # print(f"  - start: {prev_start.sum().item()} -> {self.start.sum().item()} (True count)")
        # print(f"  - first_ep: {self.first_ep.sum().item()} (True count)")

    def normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Нормализует углы в диапазон [-pi, pi].

        Args:
            angle (torch.Tensor): Углы [num_envs].

        Returns:
            torch.Tensor: Нормализованные углы [num_envs].
        """
        original_angle = angle.clone()
        iterations = 0
        while torch.any(angle > math.pi):
            angle[angle > math.pi] -= 2 * math.pi
            iterations += 1
            if iterations > 100:  # Safety check
                print(f"[DEBUwndbG] Warning: normalize_angle stuck in positive loop")
                break
        while torch.any(angle < -math.pi):
            angle[angle < -math.pi] += 2 * math.pi
            iterations += 1
            if iterations > 100:  # Safety check
                print(f"[DEBUG] Warning: normalize_angle stuck in negative loop")
                break
        # if iterations > 0:
        #     #print(f"[DEBUG] Normalized angles in {iterations} iterations")
        #     print(f"  - Original range: [{original_angle.min():.3f}, {original_angle.max():.3f}]")
        #     print(f"  - Normalized range: [{angle.min():.3f}, {angle.max():.3f}]")
        
        return angle

    def get_lookahead_point(self, current_positions: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Вычисляет точки следования (lookahead points) для всех сред.

        Args:
            current_positions (torch.Tensor): Текущие позиции роботов [num_envs, 2].
            mask (torch.Tensor, optional): Маска активных сред.

        Returns:
            torch.Tensor: Точки следования [num_envs, 2].
        """
        # print(f"\n[DEBUG] ===== GET_LOOKAHEAD_POINT =====")
        
        lookahead_points = torch.zeros((self.num_envs, 2), device=self.device)
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        active_envs = torch.where(mask)[0]
        
        for i in active_envs:
            found_lookahead = False
            for j in range(self.max_path_length-1, -1, -1):  # Поиск с конца, max_path_length=50
                segment_start = self.paths[i, j]
                segment_end = self.paths[i, min(j + 1, self.max_path_length-1)]
                # Проверка близости последней точки к цели
                if j == self.max_path_length-1:
                    # print("self.target_positions ", self.target_positions)
                    dist_to_target = torch.norm(segment_start - self.target_positions[i])
                    # if dist_to_target > 0.1:
                    #     print(f"[DEBUG] Warning: Env {i} last path point {segment_start} is far from target {self.target_positions[i]}")
                # Логика поиска lookahead point
                dist_to_start = torch.norm(segment_start - current_positions[i])
                if dist_to_start <= self.lookahead_distance:
                    # print(f"[ WHERE MY GOOD LP ] dist_to_start: {dist_to_start}\n lookahead_points[{i}]: segment_start")
                    lookahead_points[i] = segment_end
                    found_lookahead = True
                    break
            if not found_lookahead:
                # print(f"[DEBUG] Env {i}: No valid lookahead point, setting end=True")
                # self.end[i] = True
                lookahead_points[i] = self.target_positions[i]
        
        #print(f"[DEBUG] Lookahead points sample: {lookahead_points[active_envs][:3]}")
        return lookahead_points


    def get_quadrant(self, nx: torch.Tensor, ny: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """
        Определяет квадрант для вектора относительно осей nx, ny.

        Args:
            nx (torch.Tensor): Ось X [num_envs, 2].
            ny (torch.Tensor): Ось Y [num_envs, 2].
            vector (torch.Tensor): Вектор к цели [num_envs, 2].

        Returns:
            torch.Tensor: Знак квадранта [num_envs].
        """
        LR = vector[:, 0] * nx[:, 1] - vector[:, 1] * nx[:, 0]
        signs = torch.sign(LR)
        
        #print(f"[DEBUG] get_quadrant:")
        # print(f"  - LR values: {LR[:5]}")
        # print(f"  - Signs: {signs[:5]}")
        
        return signs

    def pure_pursuit_controller(self, current_positions: torch.Tensor, current_orientations: torch.Tensor):
        """
        Реализует Pure Pursuit контроллер для всех сред.

        Args:
            current_positions (torch.Tensor): Текущие позиции роботов [num_envs, 2].
            current_orientations (torch.Tensor): Текущие ориентации (yaw) [num_envs].

        Returns:
            tuple: (linear_velocity [num_envs], angular_velocity [num_envs])
        """
        # print(f"\n[DEBUG] ===== PURE_PURSUIT_CONTROLLER =====")
        #print(f"[DEBUG] Input shapes:")
        # print(f"  - current_positions: {current_positions}")
        # print(f"  - current_orientations: {current_orientations}")
        
        #print(f"[DEBUG] Current orientations sample: {current_orientations[:5]}")
        
        linear_velocity = torch.full((self.num_envs,), self.linear_velocity, device=self.device)
        angular_velocity = torch.zeros(self.num_envs, device=self.device)
        # print("current_orientations", current_orientations)
        current_heading = torch.where(
            current_orientations < 0,
            2*math.pi + current_orientations,
            current_orientations
        )
        distance_to_target = torch.norm(self.target_positions - current_positions, dim=1)
        distance_to_path_end = torch.norm(self.paths[:, -1] - current_positions, dim=1)
        # print(f"  - distance_to_target {distance_to_target}", self.target_positions)
        # print(f"  - distance_to_path_end {distance_to_path_end}", self.paths[:, -1])
        close_to_target = (distance_to_target < 1.0) | (distance_to_path_end < 0.3)
        # print(f"  - close_to_target: {close_to_target}")
        self.end = close_to_target
        
        mask_start_or_end = self.start | self.end
        mask_normal = ~mask_start_or_end
        
        #print(f"[DEBUG] Environment states:")
        # print(f"  - start: {self.start.sum().item()}")
        # print(f"  - end: {self.end.sum().item()}")
        # print(f"  - first_ep: {self.first_ep.sum().item()}")
        # print(f"  - mask_normal: {mask_normal}")
        # print(f"  - mask_start_or_end: {mask_start_or_end}")
        # print(f"  - self.start: {self.start}")
        # print(f"  - self.end: {self.end}")

        # Normal following behavior
        if torch.any(mask_normal):
            #print(f"[DEBUG] Processing {mask_normal.sum().item()} environments with normal following")
            
            lookahead_points = self.get_lookahead_point(current_positions, mask_normal)
            # print(f" [ CONTROL MODULE DEBUG ] lookahead points: {lookahead_points}")
            to_target = lookahead_points - current_positions
            #print(f"[DEBUG] lookahead_points is {lookahead_points} from current_positions {current_positions} and to target is {to_target}")
            target_angle = torch.atan2(to_target[:, 1], to_target[:, 0])
            target_angle = torch.where(target_angle < 0, target_angle + 2 * math.pi, target_angle)
            alpha = self.normalize_angle(target_angle - current_heading)
            
            # #print(f"[DEBUG] Normal following calculations:")
            # print(f"  - target_angle sample: {target_angle[mask_normal][:3]}")
            # print(f"  - alpha sample: {alpha[mask_normal][:3]}")
            
            curvature = 2 * torch.sin(alpha) / self.lookahead_distance
            angular_velocity[mask_normal] = curvature[mask_normal] * self.linear_velocity
            
            # #print(f"[DEBUG] Curvature sample: {curvature[mask_normal][:3]}")
            # #print(f"[DEBUG] Angular velocity before clamp: {angular_velocity[mask_normal][:3]}")
            
            angular_velocity[mask_normal] = torch.clamp(
                angular_velocity[mask_normal],
                -self.max_angular_velocity,
                self.max_angular_velocity
            )
            
            # #print(f"[DEBUG] Angular velocity after clamp: {angular_velocity[mask_normal][:3]}")
            
            # Velocity scaling
            velocity_scale = (self.max_angular_velocity - angular_velocity[mask_normal].abs()) / self.max_angular_velocity
            linear_velocity[mask_normal] *= velocity_scale
            
            # #print(f"[DEBUG] Velocity scale: {velocity_scale[:3]}")
            # #print(f"[DEBUG] Final linear velocity: {linear_velocity[mask_normal][:3]}")

        # Start or end behavior
        if torch.any(mask_start_or_end):
            #print(f"[DEBUG]  {mask_start_or_end.sum().item()} environments with start/end behavior")
            
            if torch.any(self.first_ep):
                #print(f"[DEBUG] First episode: stopping {self.first_ep.sum().item()} environments")
                linear_velocity[self.first_ep] = 0.0
                angular_velocity[self.first_ep] = 0.0
                self.first_ep[:] = False
            else:
                #print(f"[DEBUG] Processing orientation adjustment for {mask_start_or_end.sum().item()} environments")
                
                nx = torch.tensor([[1.0, 0.0]], device=self.device).repeat(self.num_envs, 1)
                ny = torch.tensor([[0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
                # Находим первую валидную точку пути (не -666)
                # valid_mask = ~torch.any(self.paths == -666.0, dim=2)  # [num_envs, 50]
                # print("________________________check")
                # print("paths: ", self.paths)
                valid_mask = ~torch.any(torch.abs(self.paths) > 30, dim=2)
                valid_indices = torch.argmax(valid_mask.to(torch.int32), dim=1)  # Индекс первой валидной точки
                invalid_paths = ~torch.any(valid_mask, dim=1)  # Среды с полностью недействительными путями
                # print("paths: ", invalid_paths)
                # print("paths: ",  self.paths[invalid_paths])
                # Выводим информацию о невалидных путях
                # if torch.any(invalid_paths):
                #     invalid_env_ids = invalid_paths.nonzero(as_tuple=False).squeeze(-1)
                #     print(f"[DEBUG] Invalid paths detected in environments: {invalid_env_ids.tolist()}")
                #     for env_id in invalid_env_ids:
                #         print(f"[DEBUG] Path for env {env_id.item()}: {self.paths[env_id].tolist()}")
                self.end |= invalid_paths
                if torch.any(invalid_paths):
                    print(f"[ CM DEBUG ] Environments with invalid paths: {invalid_paths.sum().item()}")
                
                # Собираем первые валидные точки
                first_valid_points = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
                for i in range(self.num_envs):
                    if invalid_paths[i]:
                        first_valid_points[i] = self.target_positions[i]  # Для пустых путей используем цель
                    else:
                        add = 0
                        if len(self.paths[i]) > 1 and valid_indices[i] < self.max_path_length - 1:
                            add = 1
                        first_valid_points[i] = self.paths[i, valid_indices[i]+add]
                # print(f"[ CM DEBUG ] valid_indices: {valid_indices}")
                 # Вычисляем угол к целевой точке
                to_goal_vec = torch.where(
                    self.end[:, None],
                    self.target_positions - current_positions,
                    first_valid_points - current_positions
                )
                # print(f"[ CM DEBUG ] first_valid_points: {first_valid_points}, cur_position: {current_positions}", to_goal_vec)
                target_angle = torch.atan2(to_goal_vec[:, 1], to_goal_vec[:, 0])
                target_angle = torch.where(target_angle < 0, target_angle + 2 * math.pi, target_angle)
                # current_heading = 2 * math.pi - current_heading
                # current_heading = math.pi - current_heading
                current_heading = torch.where(
                    current_heading < 0,
                    2 * math.pi + current_heading,
                    current_heading
                )
                alpha = self.normalize_angle(target_angle - current_heading)
                # print("current_heading ", current_heading)
                # print("target_angle ", target_angle)
                # print("alpha ", alpha)
                # Устанавливаем угловую скорость для выравнивания
                angular_velocity[mask_start_or_end] = torch.sign(alpha[mask_start_or_end]) * 1.2
                linear_velocity[mask_start_or_end] = 0.0
                
                # Проверяем, достаточно ли выровнена ориентация
                angle_error = torch.abs(alpha)
                orientation_ok = angle_error < math.pi / 80
                self.start &= ~orientation_ok
                
                #print(f"[DEBUG] Orientation adjustment:")
                # print(f"  - angle_error sample: {angle_error[mask_start_or_end][:3]}")
                # print(f"  - threshold: {math.pi / 80:.4f}")
                # print(f"  - orientation_ok: {orientation_ok.sum().item()}")
                # print(f"  - start count: {prev_start_count} -> {new_start_count}")

        #print(f"[DEBUG] Final control outputs:")
        # print(f"  - linear_velocity range: [{linear_velocity.min():.3f}, {linear_velocity.max():.3f}]")
        # print(f"  - angular_velocity range: [{angular_velocity.min():.3f}, {angular_velocity.max():.3f}]")
        # print(f"  - linear_velocity sample: {linear_velocity[:5]}")
        # print(f"  - angular_velocity sample: {angular_velocity[:5]}")

        return linear_velocity, angular_velocity