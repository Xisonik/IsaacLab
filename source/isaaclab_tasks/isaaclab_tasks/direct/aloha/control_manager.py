import torch
import math

class VectorizedPurePursuit:
    def __init__(self, num_envs, device='cuda', max_path_length=15, lookahead_distance=0.35,
                 base_linear_velocity=1.0, max_angular_velocity=1.8, arrival_threshold=0.2):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_path_length = max_path_length
        self.lookahead_distance = lookahead_distance
        self.base_linear_velocity = float(base_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.arrival_threshold = float(arrival_threshold)

        # paths: (num_envs, max_path_length, 2) padded with NaN
        self.paths = torch.full((num_envs, max_path_length, 2),
                                float('nan'), dtype=torch.float32, device=self.device)
        self.path_lengths = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.finished = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.target_positions = torch.full((num_envs, 2), float('nan'),
                                           dtype=torch.float32, device=self.device)

        # new: прогресс по арк-длине (не падает назад)
        self.progress_arclen = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

    def update_paths(self, env_indices, new_paths, target_positions):
        if not isinstance(env_indices, torch.Tensor):
            env_indices = torch.tensor(env_indices, dtype=torch.int64, device=self.device)

        # set target positions (assume target_positions aligns with env_indices)
        self.target_positions[env_indices] = torch.tensor(target_positions, dtype=torch.float32, device=self.device)

        for i, env_id in enumerate(env_indices):
            path = new_paths[i]
            # accept numpy or torch
            if not isinstance(path, torch.Tensor):
                path = torch.tensor(path, dtype=torch.float32, device=self.device)
            length = int(path.shape[0])
            if length > self.max_path_length:
                raise ValueError(f"Path length {length} exceeds max_path_length {self.max_path_length}")
            self.paths[env_id, :length] = path
            # pad remainder with NaN
            if length < self.max_path_length:
                self.paths[env_id, length:] = float('nan')
            self.path_lengths[env_id] = length

            # reset progress counter for this env
            self.progress_arclen[env_id] = 0.0

        # mark these envs as not finished (start following)
        self.finished[env_indices] = False

    def compute_controls(self, positions, orientations):
        linear_vels = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        angular_vels = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # active: есть путь >=2 и не finished
        active = (self.path_lengths >= 2) & (~self.finished)
        if not active.any():
            # Нет активных движений — обрабатываем только выравнивание для пришедших (finished)
            # Но только для тех, у кого есть валидная target_positions
            # print("chachacha")
            valid_target_mask = self.finished & torch.isfinite(self.target_positions).all(dim=1)
            jf_indices = torch.where(valid_target_mask)[0]
            if jf_indices.numel() == 0:
                return linear_vels, angular_vels

            pos_jf = positions[jf_indices]
            ori_jf = orientations[jf_indices]
            target_pos_jf = self.target_positions[jf_indices]

            to_targets_jf = target_pos_jf - pos_jf
            target_angles_jf = torch.atan2(to_targets_jf[:, 1], to_targets_jf[:, 0])
            alphas_jf = target_angles_jf - ori_jf
            alphas_jf = (alphas_jf + math.pi) % (2 * math.pi) - math.pi
            signs = torch.sign(alphas_jf)
            signs[signs == 0] = 1
            
            ang_vels_jf = torch.clamp(signs * 2.0, -self.max_angular_velocity, self.max_angular_velocity)
            # P-controller for orientation
            # kp = 2.0
            # ang_vels_jf = torch.clamp(kp * alphas_jf, -self.max_angular_velocity, self.max_angular_velocity)

            linear_vels[jf_indices] = 0.0
            angular_vels[jf_indices] = ang_vels_jf

            return linear_vels, angular_vels

        # subset active
        active_indices = torch.where(active)[0]
        num_active = active_indices.shape[0]
        pos = positions[active_indices]  # (num_active, 2)
        ori = orientations[active_indices]  # (num_active,)
        paths_active = self.paths[active_indices]  # (num_active, max_path_length, 2)
        path_lens = self.path_lengths[active_indices]  # (num_active,)

        max_segments = self.max_path_length - 1
        segment_starts = paths_active[:, :-1, :]  # (num_active, max_segments, 2)
        segment_ends = paths_active[:, 1:, :]     # (num_active, max_segments, 2)
        segment_vecs = segment_ends - segment_starts
        segment_lengths = torch.norm(segment_vecs, dim=-1)  # (num_active, max_segments)

        # mask valid segments by path length and by non-zero length
        seg_index = torch.arange(max_segments, device=self.device).unsqueeze(0).expand(num_active, max_segments)
        segment_mask = seg_index < (path_lens - 1).unsqueeze(1)
        # also ignore extremely short segments (to avoid div by ~0)
        segment_mask = segment_mask & (segment_lengths > 1e-6)

        # compute projections safely
        pos_exp = pos.unsqueeze(1)  # (num_active, 1, 2)
        to_starts = pos_exp - segment_starts
        denom = (segment_lengths ** 2) + 1e-8
        projs = torch.sum(to_starts * segment_vecs, dim=-1) / denom
        projs_clamped = torch.clamp(projs, min=0.0, max=1.0)
        closest_points = segment_starts + segment_vecs * projs_clamped.unsqueeze(-1)
        dists = torch.norm(pos_exp - closest_points, dim=-1)

        # invalidate non-valid segments
        dists[~segment_mask] = float('inf')

        # choose closest segment per env
        min_dists, min_segments = torch.min(dists, dim=1)  # (num_active,)
        min_projs = projs_clamped[torch.arange(num_active, device=self.device), min_segments]
        min_seg_lengths = segment_lengths[torch.arange(num_active, device=self.device), min_segments]

        # cumulative arclengths along path (start-of-segment positions)
        padded_segment_lengths = segment_lengths.clone()
        padded_segment_lengths[~segment_mask] = 0.0
        cum_lengths = torch.cat([torch.zeros(num_active, 1, device=self.device), padded_segment_lengths], dim=1)  # (num_active, max_segments+1)
        cum_lengths = torch.cumsum(cum_lengths, dim=1)

        cum_at_min_seg = cum_lengths[torch.arange(num_active, device=self.device), min_segments]
        closest_arclen = cum_at_min_seg + min_projs * min_seg_lengths  # arclength to the closest point on path

        # **Главная правка**: не позволяем прогрессу уменьшаться (чтобы не идти назад)
        prev_progress = self.progress_arclen[active_indices]
        new_progress = torch.maximum(prev_progress, closest_arclen)
        self.progress_arclen[active_indices] = new_progress
        curr_arclen = new_progress

        # цель по арк-длине
        target_arclen = curr_arclen + self.lookahead_distance

        # полная длина пути:
        total_lengths = cum_lengths[torch.arange(num_active, device=self.device), (path_lens - 1).clamp(max=max_segments)]

        is_beyond = target_arclen >= total_lengths

        lookahead_points = torch.zeros(num_active, 2, dtype=torch.float32, device=self.device)
        last_point_indices = (path_lens - 1).clamp(max=self.max_path_length - 1)
        last_points = paths_active[torch.arange(num_active, device=self.device), last_point_indices]
        lookahead_points[is_beyond] = last_points[is_beyond]

        not_beyond = ~is_beyond
        if not_beyond.any():
            nb_idx = torch.where(not_beyond)[0]
            target_arclen_nb = target_arclen[not_beyond]
            cum_lengths_nb = cum_lengths[not_beyond]  # (num_nb, max_segments+1)
            # find segment that contains target_arclen
            segs_nb = torch.searchsorted(cum_lengths_nb, target_arclen_nb.unsqueeze(1), right=False).squeeze(1) - 1
            segs_nb = torch.clamp(segs_nb, min=0, max=max_segments - 1)
            cum_at_seg_nb = cum_lengths_nb[torch.arange(segs_nb.shape[0], device=self.device), segs_nb]
            seg_lengths_nb = padded_segment_lengths[not_beyond, segs_nb]
            fracs_nb = (target_arclen_nb - cum_at_seg_nb) / (seg_lengths_nb + 1e-8)
            fracs_nb = torch.clamp(fracs_nb, min=0.0, max=1.0)
            starts_nb = segment_starts[not_beyond, segs_nb]
            vecs_nb = segment_vecs[not_beyond, segs_nb]
            lookahead_points[not_beyond] = starts_nb + vecs_nb * fracs_nb.unsqueeze(-1)

        # управление
        to_targets = lookahead_points - pos
        target_angles = torch.atan2(to_targets[:, 1], to_targets[:, 0])
        alphas = target_angles - ori
        alphas = (alphas + math.pi) % (2 * math.pi) - math.pi
        curvatures = 2 * torch.sin(alphas) / (self.lookahead_distance + 1e-8)
        ang_vels_active = curvatures * self.base_linear_velocity
        ang_vels_active = torch.clamp(ang_vels_active, -self.max_angular_velocity, self.max_angular_velocity)
        lin_vels_active = self.base_linear_velocity * (1 - torch.abs(ang_vels_active) / (self.max_angular_velocity + 1e-8))
        lin_vels_active = torch.clamp(lin_vels_active, min=0.0)

        # пометим те активные среды, которые достигли конца пути
        dists_to_end = torch.norm(pos - last_points, dim=1)
        finished_active = dists_to_end < self.arrival_threshold
        if finished_active.any():
            # объявляем их finished; дальше они будут обрабатываться в блоке finished
            self.finished[active_indices[finished_active]] = True
            lin_vels_active[finished_active] = 0.0
            ang_vels_active[finished_active] = 0.0

        # записываем результаты в глобальные векторы
        linear_vels[active_indices] = lin_vels_active
        angular_vels[active_indices] = ang_vels_active

        # блок выравнивания ориентации для всех finished (и имеющих валидные target_positions)
        valid_target_mask = self.finished & torch.isfinite(self.target_positions).all(dim=1)
        jf_indices = torch.where(valid_target_mask)[0]
        if jf_indices.numel() > 0:
            pos_jf = positions[jf_indices]
            ori_jf = orientations[jf_indices]
            target_pos_jf = self.target_positions[jf_indices]
            to_targets_jf = target_pos_jf - pos_jf
            target_angles_jf = torch.atan2(to_targets_jf[:, 1], to_targets_jf[:, 0])
            alphas_jf = target_angles_jf - ori_jf
            alphas_jf = (alphas_jf + math.pi) % (2 * math.pi) - math.pi
            kp = 2.0
            # ang_vels_jf = torch.clamp(kp * alphas_jf, -self.max_angular_velocity, self.max_angular_velocity)
            signs = torch.sign(alphas_jf)
            signs[signs == 0] = 1
            
            ang_vels_jf = torch.clamp(signs * 2.0, -self.max_angular_velocity, self.max_angular_velocity)
            linear_vels[jf_indices] = 0.0
            angular_vels[jf_indices] = ang_vels_jf

            # если уже выровнялись — обнулим угловую скорость
            aligned_mask = torch.abs(alphas_jf) < 0.1
            if aligned_mask.any():
                aligned_idxs = jf_indices[aligned_mask]
                angular_vels[aligned_idxs] = 0.0
                # (оставляем self.finished=True — внешняя логика может считать эту среду "done")

        return linear_vels, angular_vels
