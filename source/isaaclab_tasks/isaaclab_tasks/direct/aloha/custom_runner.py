from rl_games.common import env_configurations, experiment
from rl_games.torch_runner import Runner
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

class TensorBoardRunner(Runner):
    def __init__(self):
        super().__init__()
        self.writer = None

    def reset(self):
        super().reset()
        # Инициализация TensorBoard, если включено
        if self.params['config'].get('enable_tensorboard', False):
            log_dir = self.params['config']['log_dir']
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

    def run(self, args):
        super().run(args)
        # Закрываем writer после завершения обучения
        if self.writer is not None:
            self.writer.close()

    def log_metrics(self, metrics, step):
        # Логирование метрик в TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    self.writer.add_scalar(key, value, step)
                elif isinstance(value, (list, np.ndarray)):
                    # Логируем среднее значение для массивов
                    self.writer.add_scalar(key, np.mean(value), step)

    def play_one_epoch(self):
        # Переопределяем метод для добавления логирования
        metrics = super().play_one_epoch()
        if self.writer is not None:
            # Логируем стандартные метрики rl_games
            self.log_metrics({
                'train/loss': metrics.get('loss', 0.0),
                'train/reward': np.mean(metrics.get('rewards', 0.0)),
                'train/episode_reward': np.mean(metrics.get('episode_rewards', 0.0))
            }, self.global_step)
            # Логируем пользовательские метрики из среды
            if 'infos' in metrics:
                for info in metrics['infos']:
                    if 'custom_metrics' in info:
                        self.log_metrics({
                            f'custom/{key}': value for key, value in info['custom_metrics'].items()
                        }, self.global_step)
        return metrics