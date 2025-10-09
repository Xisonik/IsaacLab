# evaluation_manager.py
import torch
import pandas as pd
import os
import json

class EvaluationManager:
    def __init__(self, num_envs, device="cuda"):
        self.num_envs = num_envs
        self.device = device

        # позиции/углы (списки задач)
        self.robot_positions = []   # list of [x,y,(z)]
        self.angle_errors = []      # list (or list of lists) depending on формат

        # указатели задач по env
        self.position_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.angle_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

        # сохранённые стартовые дистанции (на ресете)
        self.start_dists = torch.full((num_envs,), float("nan"), device=device, dtype=torch.float32)

        # результаты по env: список словарей
        self.results = {env: [] for env in range(num_envs)}

        # finished flags
        self.finished_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.all_done = False

    def set_task_lists(self, robot_positions, angle_errors):
        """Одинаковые списки для всех env (по желанию можно сделать per-env)."""
        self.robot_positions = robot_positions
        self.angle_errors = angle_errors
        self.position_idx[:] = 0
        self.angle_idx[:] = 0
        self.finished_envs[:] = False
        self.all_done = False
        self.start_dists[:] = float("nan")
        # очистим результаты
        self.results = {env: [] for env in range(self.num_envs)}

    def set_start_dists(self, env_ids: torch.Tensor, start_dists: torch.Tensor):
        env_ids = env_ids.to(self.device)
        start_dists = start_dists.to(self.device).to(dtype=torch.float32)
        self.start_dists[env_ids] = start_dists

    def get_start_dists(self, env_ids: torch.Tensor):
        env_ids = env_ids.to(self.device)
        return self.start_dists[env_ids]
    
    def get_positions(self):
        current_dir = os.getcwd()
        filepath = os.path.join(current_dir, "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/eval_scenes.json")
        with open(filepath, 'r', encoding='utf-8') as f:
            positions = json.load(f)
        return positions

    def get_current_tasks(self, env_ids: torch.Tensor):
        """
        Вернёт пачку (robot_pos_batch, angle_err_batch) для env_ids.
        Ожидается, что robot_positions и angle_errors — списки, одинаковые для всех env.
        """
        num_envs = len(env_ids)
        robot_pos_batch = []
        angle_batch = []
        for env in env_ids.tolist():
            pidx = int(self.position_idx[env].item())
            aidx = int(self.angle_idx[env].item())
            # clamp
            pidx = min(pidx, max(0, len(self.robot_positions)-1))
            aidx = min(aidx, max(0, len(self.angle_errors)-1))
            robot_pos_batch.append(self.robot_positions[pidx])
            angle_batch.append(self.angle_errors[aidx])
        # возвращаем тензоры на нужном устройстве
        final_yaw = torch.tensor(angle_batch, device=self.device)
        robot_quats = torch.zeros(num_envs, 4, device=self.device)
        robot_quats[:, 0] = torch.cos(final_yaw / 2.0)
        robot_quats[:, 3] = torch.sin(final_yaw / 2.0)
        return torch.tensor(robot_pos_batch, device=self.device), robot_quats

    def log_results(self, env_ids: torch.Tensor, successes, traj_lens, start_dists, durations):
        """
        Логируем пачку результатов. `start_dists`/`durations`/`traj_lens` - батчи длины len(env_ids).
        Важно: вызывать ДО next_episode.
        """
        env_ids = env_ids.to(self.device)
        # allow tensor or list for other inputs
        successes = torch.as_tensor(successes, device=self.device).to(dtype=torch.float32)
        traj_lens = torch.as_tensor(traj_lens, device=self.device).to(dtype=torch.float32)
        durations = torch.as_tensor(durations, device=self.device).to(dtype=torch.float32)
        start_dists = torch.as_tensor(start_dists, device=self.device).to(dtype=torch.float32)

        for i, env in enumerate(env_ids.tolist()):
            env = int(env)
            pos_i = int(self.position_idx[env].item())
            ang_i = int(self.angle_idx[env].item())
            s = float(successes[i].item())
            tl = float(traj_lens[i].item())
            sd = float(start_dists[i].item()) if not torch.isnan(start_dists[i]) else float(self.start_dists[env].item())
            dur = float(durations[i].item())

            spl = 0.0
            if s > 0.5 and tl > 1e-6 and sd > 1e-6:
                spl = (sd / tl)
            # сохраняем запись
            self.results[env].append({
                "position_idx": pos_i,
                "angle_idx": ang_i,
                "success": s,
                "traj_len": tl,
                "start_dist": sd,
                "duration": dur,
                "spl": spl,
            })

    def next_episode(self, env_ids: torch.Tensor):
        """Продвигаем указатели задач для пачки env (после логирования)."""
        env_ids = env_ids.to(self.device)
        for env in env_ids.tolist():
            env = int(env)
            if self.finished_envs[env]:
                continue
            # продвигаем angle
            if (self.angle_idx[env] + 1) < len(self.angle_errors):
                self.angle_idx[env] += 1
            else:
                self.angle_idx[env] = 0
                # продвигаем позицию
                if (self.position_idx[env] + 1) < len(self.robot_positions):
                    self.position_idx[env] += 1
                else:
                    self.finished_envs[env] = True
        if torch.all(self.finished_envs):
            self.all_done = True

    def is_all_done(self):
        return bool(self.all_done)

    def summarize(self):
        all_rows = []
        for env, recs in self.results.items():
            for r in recs:
                row = {"env": env}
                row.update(r)
                all_rows.append(row)
        df = pd.DataFrame(all_rows)
        if df.empty:
            return df, pd.Series(), pd.DataFrame()
        df = df.round(2)
        global_stats = df.mean(numeric_only=True)
        pos_stats = df.groupby("position_idx").mean(numeric_only=True)
        return df, global_stats, pos_stats
