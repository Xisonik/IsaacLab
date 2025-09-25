import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import gymnasium as gym
from skrl.utils.spaces.torch import (
    compute_space_size,
    flatten_tensorized_space,
    sample_space,
    unflatten_tensorized_space,
)

# define models (stochastic and deterministic models) using mixins
class CustomActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.observation_space = observation_space
        # Вычисляем размеры
        self.img_dim = observation_space["img"].shape[0]  # 36
        self.num_objects = observation_space["graph"]["node_features"].shape[0]  # num_total_objects, e.g., 10
        self.node_dim = observation_space["graph"]["node_features"].shape[1]  # 14
        self.edge_dim = observation_space["graph"]["edge_features"].shape[1]  # 6
        
        # Общее количество элементов
        self.num_observations = self.img_dim# + (self.num_objects * self.node_dim) + (self.num_objects * self.edge_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),  # Теперь 72
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        ).to(device)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), 0.2, device=device))

    def compute(self, inputs, role):
        # Сглаживаем policy и graph
        inputs_for_flatten = inputs["states"]
        inputs_unflatten = unflatten_tensorized_space(self.observation_space, inputs_for_flatten)
        # print("inputs_unflatten: ", inputs_unflatten)
        x = torch.cat([
            inputs_unflatten["img"],  # (num_envs, 36)
            inputs_unflatten["graph"]["node_features"].reshape(-1, self.num_objects * self.node_dim),  # (num_envs, 10)
            inputs_unflatten["graph"]["edge_features"].reshape(-1, self.num_objects * self.edge_dim),  # (num_envs, 5)
        ], dim=-1)
        inputs_final = x
        return self.net(inputs_final), self.log_std_parameter, {}

class CustomCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.observation_space = observation_space
        # Вычисляем размеры
        self.img_dim = observation_space["img"].shape[0]  # 36
        self.num_objects = observation_space["graph"]["node_features"].shape[0]  # num_total_objects, e.g., 10
        self.node_dim = observation_space["graph"]["node_features"].shape[1]  # 14
        self.edge_dim = observation_space["graph"]["edge_features"].shape[1]  # 6
        
        # Общее количество элементов
        self.num_observations = self.img_dim# + (self.num_objects * self.node_dim) + (self.num_objects * self.edge_dim)
        self.num_actions = self.num_actions
        dropout_p=0.2
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 512),  # 72 + 8 = 80
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 1)
        ).to(device)

    def compute(self, inputs, role):
        inputs_for_flatten = inputs["states"]
        inputs_unflatten = unflatten_tensorized_space(self.observation_space, inputs_for_flatten)
        x = torch.cat([
            inputs_unflatten["img"],  # (num_envs, 36)
            inputs_unflatten["graph"]["node_features"].reshape(-1, self.num_objects * self.node_dim),  # (num_envs, 10)
            inputs_unflatten["graph"]["edge_features"].reshape(-1, self.num_objects * self.edge_dim),  # (num_envs, 5)
        ], dim=-1)
        inputs_final = torch.cat([x, inputs["taken_actions"]], dim=-1)  # (num_envs, 60)
        return self.net(inputs_final), {}