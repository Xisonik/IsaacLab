# env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
 
import gymnasium as gym
import torch
import math
import numpy as np
import os
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from .scene_manager import SceneManager
from .evaluation_manager import EvaluationManager
from .control_manager import VectorizedPurePursuit
from .path_manager import Path_manager
from .memory_manager import Memory_manager, PathTracker
from .asset_manager import AssetManager
import omni.kit.commands
import omni.usd
import datetime
# from torch.utils.tensorboard import SummaryWriter
##
# Pre-defined configs
##
from isaaclab_assets.robots.aloha import ALOHA_CFG
from isaaclab.markers import CUBOID_MARKER_CFG
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import omni.kit.commands  # Уже импортировано в вашем коде
from omni.usd import get_context  # Для доступа к stage
from pxr import Gf

class WheeledRobotEnvWindow(BaseEnvWindow):
    def __init__(self, env: 'WheeledRobotEnv', window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class WheeledRobotEnvCfg(DirectRLEnvCfg):
    episode_length_s = 512.0
    decimation = 8
    action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        shape=(2,)
    )
    # Observation space is now the ResNet18 embedding size (512)
    m = 1  # Например, 3 эмбеддинга и действия
    # observation_space = gym.spaces.Box(
    #     low=-float("inf"),
    #     high=float("inf"),
    #     shape=(m * (512 + 3),),  # m * (embedding_size + action_size) + 2 (скорости)
    #     dtype="float32"
    # )
    # TODO automat compute num_total_objects
    num_total_objects = 10

    observation_space = gym.spaces.Dict({
        "img": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(512 + 3,), dtype=np.float32),
        "graph": gym.spaces.Dict({
            "node_features": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(num_total_objects, 14), dtype=np.float32),
            "edge_features": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(num_total_objects, 6), dtype=np.float32),
        })
    })
    state_space = 0
    debug_vis = False

    ui_window_class_type = WheeledRobotEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=0.2,
            dynamic_friction=0.15,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="min",
        #     restitution_combine_mode="min",
        #     static_friction=0.8,
        #     dynamic_friction=0.6,
        #     restitution=0.0,
        # ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=18, replicate_physics=True)
    robot: ArticulationCfg = ALOHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheel_radius = 0.068
    wheel_distance = 0.34
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/box2_Link/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.35, 0, 1.1), rot=(0.99619469809,0,0.08715574274,0), convention="world"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0, 0.9), rot=(1,0,0,0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0, focus_distance=2.0, horizontal_aperture=36, clipping_range=(0.2, 10.0)
        ),
        width=224,
        height=224,
    )
    current_dir = os.getcwd()
    kitchen = sim_utils.UsdFileCfg(
        usd_path=os.path.join(current_dir, "source/isaaclab_assets/data/aloha_assets", "scenes/scenes_sber_kitchen_for_BBQ/kitchen_new_simple.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
            rigid_body_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.1,
        history_length=1,
        debug_vis=True,
        filter_prim_paths_expr=["/World/envs/env_.*"],
    )

class WheeledRobotEnv(DirectRLEnv):
    cfg: WheeledRobotEnvCfg

    def __init__(self, cfg: WheeledRobotEnvCfg, render_mode: str | None = None, **kwargs):
        self._super_init = True
        self.current_dir = os.getcwd()
        self.config_path=os.path.join(self.current_dir, "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/scene_items.json")
        super().__init__(cfg, render_mode, **kwargs)
        self._super_init = False
        self.eval = False
        self.eval_name = "CI"

        self.eval_printed = False
        self.scene_manager = SceneManager(self.num_envs, self.config_path, self.device)
        self.eval_manager = EvaluationManager(self.num_envs)
        self.eval_manager.set_task_lists(
            robot_positions=[[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                             [1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                             [2.0, -1.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]],  # список стартовых позиций
            angle_errors=[torch.pi, torch.pi*0.9, torch.pi*0.8]                         # список ошибок угла
        )
        self.use_controller = True
        self.imitation = False
        if self.imitation:
            self.use_controller = True
        if self.use_controller:
            self.path_manager = Path_manager(scene_manager=self.scene_manager, ratio=8.0, shift=[5, 4], device=self.device)
            self.control_module = VectorizedPurePursuit(num_envs=self.num_envs, device=self.device)
        self.memory_on = False
        self.tracker = PathTracker(num_envs=self.num_envs, device=self.device)
        if self.memory_on:
            self.memory_manager = Memory_manager(
                num_envs=self.num_envs,
                embedding_size=512,  # Размер эмбеддинга ResNet18
                action_size=2,      # Размер действия (линейная и угловая скорость)
                history_length=25,  # n = 10, можно настроить
                device=self.device
            )

        self._actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._actions[:, 1] = 0.0
        self._left_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._right_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["moves"]
        }
        self._left_wheel_id = self._robot.find_joints("left_wheel")[0]
        self._right_wheel_id = self._robot.find_joints("right_wheel")[0]

        self.set_debug_vis(self.cfg.debug_vis)
        self.Debug = True
        self.event_update_counter = 0
        self.episode_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.possible_goal_position = []
        
        self.delete = 1
        self.count = 0
        self._debug_log_enabled = True
        self._debug_envs_to_log = list(range(min(5, self.num_envs)))
        self._inconsistencies = []
        self._debug_step_counter = 0
        self._debug_log_frequency = 10
        self.turn_on_controller = False #it is not use or not use controller, it is flag for the first step
        self.turn_on_controller_step = 0
        self.my_episode_lenght = 256
        self.turn_off_controller_step = 0
        self.use_obstacles = True
        self.turn_on_obstacles = False
        self.turn_on_obstacles_always = False
        if self.use_obstacles:
            self.use_obstacles = True
        self.previous_distance_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_angle_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_lin_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_ang_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.angular_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Initialize ResNet18 for image embeddings
        # self.resnet18 = models.resnet18(pretrained=True).to(self.device)
        # self.resnet18.eval()  # Set to evaluation mode
        # # Remove the final fully connected layer to get embeddings
        # self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        # # Image preprocessing for ResNet18
        # transforms.ToTensor()
        # self.transform = transforms.Compose([
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.success_rate = 0
        self.sr_stack_capacity = 0
        self.episode_completion_history = torch.zeros((self.num_envs*4, self.num_envs), dtype=torch.bool, device=self.device)
        self.success_history = torch.zeros((self.num_envs*4, self.num_envs), dtype=torch.bool, device=self.device)
        self.history_index = 0
        self.history_len = torch.zeros(self.num_envs, device=self.device)
        self._step_update_counter = 0
        self.mean_radius = 4.3
        self.max_angle_error = torch.pi / 6
        self.cur_angle_error = torch.pi / 12
        self.warm = True
        self.warm_len = 2500
        self.without_imitation = self.warm_len / 2
        self._obstacle_update_counter = 0
        self.has_contact = torch.full((self.num_envs,), True, dtype=torch.bool, device=self.device)
        self.sim = SimulationContext.instance()
        self.obstacle_positions = None
        self.key = None
        self.success_ep_num = 0
        # self.run = wandb.init(project="aloha_direct")
        self.first_ep = [True, True]
        self.first_ep_step = 0
        self.second_ep = True
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
        name = "dev"
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device)
        self.episode_count = 0
        self.total_episode_length = 0.0
        # self.tensorboard_writer = SummaryWriter(log_dir=f"/home/xiso/IsaacLab/logs/tensorboard/navigation_rl_{name}_{timestamp}")
        self.tensorboard_step = 0
        self.cur_step = 0
        self.velocities = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()  # Установить в режим оценки
        self.second_try = 0
        self.foult_ep_num = 0
        # Инициализация стеков для хранения успехов (1 - успех, 0 - неуспех)
        self.success_stacks = [[] for _ in range(self.num_envs)]  # Список списков для каждой среды
        self.max_stack_size = 20  # Максимальный размер стека
        self.sr_stack_full = False
        self.start_mean_radius = 0
        self.min_level_radius = 0
        self.sr_treshhold = 85
        self.LOG = False
        self.text_embeddings = torch.zeros((self.num_envs, 512), device=self.device)
        if self.LOG:
            from comet_ml import start
            from comet_ml.integration.pytorch import log_model
            self.experiment = start(
                api_key="DRYfW6B6VtUQr9llvf3jup57R",
                project_name="general",
                workspace="xisonik"
            )
        self.print_config_info()
        self._setup_scene()
        self.prim_paths = self.asset_manager.all_prim_paths
        # сразу после создания scene_manager
        self._material_cache = {}        # key -> material prim path (строка), key = "r_g_b"
        self._applied_color_map = {}     # obj_index (int) -> color_key (str), чтобы не биндим повторно


    def print_config_info(self):
        print("__________[ CONGIFG INFO ]__________")
        print(f"|")
        print(f"| Start mean radius is: {self.mean_radius}")
        print(f"|")
        print(f"| Start amx angle is: {self.max_angle_error}")
        print(f"|")
        print(f"| Use controller: {self.use_controller}")
        print(f"|")
        print(f"| Full imitation: {self.imitation}")
        print(f"|")
        print(f"| Use memory: {self.memory_on}")
        print(f"|")
        print(f"| Use obstacles: {self.use_obstacles}")
        print(f"|")
        print(f"| Start radius: {self.start_mean_radius}, min: {self.min_level_radius}")
        print(f"|")
        print(f"| Warm len: {self.warm_len}")
        print(f"|")
        print(f"| Turn on obstacles always: {self.turn_on_obstacles_always}")
        print(f"|")
        print(f"_______[ CONGIFG INFO CLOSE ]_______")

    def _setup_scene(self):
        from isaaclab.sensors import ContactSensor
        import time
        from pxr import Usd
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        import random
        if self._super_init:
            self._robot = Articulation(self.cfg.robot)
            self.scene.articulations["robot"] = self._robot
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
            self.scene.clone_environments(copy_from_source=True)
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera
            # Спавн кухни (статический элемент)
            spawn_from_usd(
                prim_path="/World/envs/env_.*/Kitchen",
                cfg=self.cfg.kitchen,
                translation=(5.0, 4.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor
            self.asset_manager = AssetManager(config_path=self.config_path)
            self.scene_objects = self.asset_manager.spawn_assets_in_scene()
            
            # self.scene_manager.update_prims(prim_path)
            

        # light_cfg = sim_utils.DomeLightCfg(intensity=300.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        self.tensorboard_step += 1
        self.cur_step += 1
        self.episode_lengths += 1
        import os
        from PIL import Image, ImageDraw, ImageFont
        # Получение RGB изображений с камеры
        camera_data = self._tiled_camera.data.output["rgb"].clone()  # Shape: (num_envs, 224, 224, 3)
        
        # Преобразование изображений для CLIP
        # CLIP ожидает изображения в формате PIL или тензоры с правильной нормализацией
        images = camera_data.cpu().numpy().astype(np.uint8)  # Конвертация в numpy uint8
        # inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
        images_list = [Image.fromarray(im) for im in images]  # если images shape (N,H,W,3) numpy, это даёт список 2D-arrays
        inputs = self.clip_processor(images=images_list, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        # Получение эмбеддингов изображений
        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(**inputs)  # Shape: (num_envs, 512)
            image_embeddings = image_embeddings / (image_embeddings.norm(dim=1, keepdim=True) + 1e-9)
        
        # Получение скоростей робота
        root_lin_vel_w = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1).unsqueeze(-1)
        root_ang_vel_w = self._robot.data.root_ang_vel_w[:, 2].unsqueeze(-1)
        
        scene_embeddings = self.scene_manager.get_graph_embedding(self._robot._ALL_INDICES.clone())
        scene_embeddings_dict = self.scene_manager.get_graph_obs(self._robot._ALL_INDICES.clone())
        # obs = torch.cat([image_embeddings, scene_embeddings, text_embeddings, root_lin_vel_w*0.1, root_ang_vel_w*0.1, self.previous_ang_vel.unsqueeze(-1)*0.1], dim=-1)
        obs_img = torch.cat([image_embeddings, root_lin_vel_w*0.1, root_ang_vel_w*0.1, self.previous_ang_vel.unsqueeze(-1)*0.1], dim=-1)
        obs = {
            "img": obs_img,
            "graph": scene_embeddings_dict
        }
        self.previous_ang_vel = self.angular_speed
        # log_embedding_stats(image_embeddings)
        
        observations = {"policy": obs}       
        return observations

    # as they are not affected by the observation space change.

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.cur_step % 256 == 0:
            print("[ sr ]: ", round(self.success_rate, 2), self.sr_stack_capacity)
        env_ids = self._robot._ALL_INDICES.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        nan_mask = torch.isnan(self._actions) | torch.isinf(self._actions)
        nan_indices = torch.nonzero(nan_mask.any(dim=1), as_tuple=False).squeeze()  # env_ids где любой action NaN/inf
        if nan_indices.numel() > 0:
            print(f"[WARNING] NaN/Inf in actions for envs: {nan_indices.tolist()}. Attempting recovery...")
        r = self.cfg.wheel_radius
        L = self.cfg.wheel_distance
        self._step_update_counter += 1
        if self.turn_on_controller or self.imitation:
            self.turn_on_controller_step += 1
            # Получаем текущую ориентацию (yaw) из кватерниона
            quat = self._robot.data.root_quat_w
            siny_cosp = 2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
            cosy_cosp = 1 - 2 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3])
            yaw = torch.atan2(siny_cosp, cosy_cosp)
            linear_speed, angular_speed = self.control_module.compute_controls(
                self.to_local(self._robot.data.root_pos_w[:, :2],env_ids),
                yaw
            )
            # angular_speed = -angular_speed 
            self._actions[:, 0] = (linear_speed / 0.6) - 1
            self._actions[:, 1] = angular_speed / 2
            actions.copy_(self._actions.clamp(-1.0, 1.0))
        else:
            self.turn_off_controller_step += 1
            linear_speed = 0.6*(self._actions[:, 0] + 1.0) # [num_envs], всегда > 0
            angular_speed = 2*self._actions[:, 1]  # [num_envs], оставляем как есть от RL
        # linear_speed = torch.tensor([0], device=self.device)
        self.angular_speed = angular_speed
        self.velocities = torch.stack([linear_speed, angular_speed], dim=1)
        # if self.tensorboard_step % 4 ==0:
        # self.delete = -1 * self.delete 
        # angular_speed = torch.tensor([0], device=self.device)
        # print("vel is: ", linear_speed, angular_speed)
        self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
        self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r
        # self._left_wheel_vel = torch.clamp(self._left_wheel_vel, -10, 10)
        # self._right_wheel_vel = torch.clamp(self._right_wheel_vel, -10, 10)

    def _apply_action(self):
        wheel_velocities = torch.stack([self._left_wheel_vel, self._right_wheel_vel], dim=1).unsqueeze(-1).to(dtype=torch.float32)
        self._robot.set_joint_velocity_target(wheel_velocities, joint_ids=[self._left_wheel_id, self._right_wheel_id])

    def _get_rewards(self) -> torch.Tensor:
        # env_ids = self._robot._ALL_INDICES.clone()
        # num_envs = len(env_ids)
        # value = torch.tensor([0, 0], dtype=torch.float32, device=self.device)
        # robot_pos = value.unsqueeze(0).repeat(num_envs, 1)
        # joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        # joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        # default_root_state = self._robot.data.default_root_state[env_ids].clone()
        # default_root_state[:, :2] = self.to_global(robot_pos, env_ids)
        # default_root_state[:, 2] = 0.1
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        lin_vel = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        
        lin_vel_reward = torch.clamp(lin_vel*0.02, min=0, max=0.15)
        ang_vel = self._robot.data.root_ang_vel_w[:, 2]
        ang_vel_reward = torch.abs(self.angular_speed) * 0.1
        a_penalty = 0.1 * torch.abs(self.angular_speed - self.previous_ang_vel) #+ torch.abs(lin_vel - self.previous_lin_vel))
        # print("a_penalty ", -a_penalty, self.angular_speed, self.previous_ang_vel )
        # self.previous_lin_vel = lin_vel

        goal_reached, num_subs, r_error, a_error = self.goal_reached(get_num_subs=True)

        moves = torch.clamp(5 * (self.previous_distance_error - r_error), min=0, max=1) + \
                    torch.clamp(5 * (self.previous_angle_error - a_error), min=0, max=1)
        env_ids = self._robot._ALL_INDICES.clone()
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        self.tracker.add_step(env_ids, self.to_local(root_pos_w, env_ids), self.velocities)
        path_lengths = self.tracker.compute_path_lengths(env_ids)

        moves_reward = moves * 0.1
        
        self.previous_distance_error = r_error
        self.previous_angle_error = a_error

        has_contact = self.get_contact()

        time_out = self.is_time_out(self.my_episode_lenght-1)
        time_out_penalty = -5 * time_out.float()

        vel_penalty = -1 * (ang_vel_reward + lin_vel_reward)
        mask = ~goal_reached
        vel_penalty[mask] = 0
        lin_vel_reward[goal_reached] = 0

        paths = self.tracker.get_paths(env_ids)
        # jerk_counts = self.tracker.compute_jerk(env_ids, threshold=0.2)
        # print(jerk_counts)
        start_dists = self.eval_manager.get_start_dists(env_ids)
        if self.turn_on_controller:
            IL_reward = 0.5
            punish = - 0.05
        else:
            IL_reward = 0
            punish = (
                - 0.1
                - ang_vel_reward / (1 + 2 * self.mean_radius)
                + lin_vel_reward / (1 + 2 * self.mean_radius)
            )
        reward = (
            IL_reward + punish #* r_error
            + torch.clamp(goal_reached.float() * 7, min=0, max=15) #* (1 + start_dists) / (1 + path_lengths)
            - torch.clamp(has_contact.float() * (5 + lin_vel_reward), min=0, max=10)
        )

        if torch.any(has_contact) or torch.any(goal_reached) or torch.any(time_out):
            sr = self.update_success_rate(goal_reached)

        # if torch.any(has_contact) or torch.any(goal_reached) or torch.any(time_out):
        # print("path info")
        # print(path_lengths)
        # print(r_error)
        # print("reward ", reward)
        # print("- 0.1 - 0.05 * r_error ", - 0.1 - 0.05 * r_error)
        # print("IL_reward * r_error ", IL_reward * r_error)
        # print("goal_reached ", goal_reached)
        # print("lin_vel_reward ", lin_vel_reward)
        # print("torch.clamp(goal_reached ", torch.clamp(goal_reached.float() * 7 * (1 + self.mean_radius) / (1 + path_lengths), min=0, max=10))
        # print("torch.clamp(has_contact ", torch.clamp(has_contact.float() * (3 + lin_vel_reward), min=0, max=6))
        # print("ang_vel_reward ", -ang_vel_reward)
        # print("___________")
        check = {
            "moves":moves,
        }
        for key, value in check.items():
            self._episode_sums[key] += value

        # if self.tensorboard_step % 100 == 0:
        #     self.tensorboard_writer.add_scalar("Metrics/reward", torch.sum(reward), self.tensorboard_step)
        return reward
    
    def quat_rotate(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Вращение вектора vec кватернионом quat.
        quat: [N, 4] (w, x, y, z)
        vec: [N, 3]
        Возвращает: [N, 3] - вектор vec, повернутый кватернионом quat
        """
        w, x, y, z = quat.unbind(dim=1)
        vx, vy, vz = vec.unbind(dim=1)

        # Кватернионное умножение q * v
        qw = -x*vx - y*vy - z*vz
        qx = w*vx + y*vz - z*vy
        qy = w*vy + z*vx - x*vz
        qz = w*vz + x*vy - y*vx

        # Обратный кватернион q*
        rw = w
        rx = -x
        ry = -y
        rz = -z

        # Результат (q * v) * q*
        rx_new = qw*rx + qx*rw + qy*rz - qz*ry
        ry_new = qw*ry - qx*rz + qy*rw + qz*rx
        rz_new = qw*rz + qx*ry - qy*rx + qz*rw

        return torch.stack([rx_new, ry_new, rz_new], dim=1)


    def goal_reached(self, angle_threshold: float = 17, radius_threshold: float = 1.3, get_num_subs=False):
        """
        Проверяет достижение цели с учётом расстояния и направления взгляда робота.
        distance_to_goal: [N] расстояния до цели
        angle_threshold: максимально допустимый угол в радианах между направлением взгляда и вектором на цель
        Возвращает: [N] булев тензор, True если цель достигнута
        """
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        # print("root: ", self.to_local(root_pos_w))
        # print("root_pos_w ", root_pos_w)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        # Проверка по расстоянию (например, радиус достижения stored в self.radius)
        close_enough = distance_to_goal <= radius_threshold

        # Получаем ориентацию робота в виде кватерниона (w, x, y, z)
        root_quat_w = self._robot.data.root_quat_w  # shape [N, 4]

        # Локальный вектор взгляда робота (вперёд по оси X)
        local_forward = torch.tensor([1.0, 0.0, 0.0], device=root_quat_w.device, dtype=root_quat_w.dtype)
        local_forward = local_forward.unsqueeze(0).repeat(root_quat_w.shape[0], 1)  # [N, 3]

        # Вектор взгляда в мировых координатах
        forward_w = self.quat_rotate(root_quat_w, local_forward)  # [N, 3]

        # Вектор от робота к цели
        root_pos_w = self._robot.data.root_pos_w  # [N, 3]
        to_goal = self._desired_pos_w - root_pos_w  # [N, 3]

        # Нормализуем векторы
        forward_w_norm = torch.nn.functional.normalize(forward_w[:, :2] , dim=1)
        to_goal_norm = torch.nn.functional.normalize(to_goal[:, :2] , dim=1)

        # Косинус угла между векторами взгляда и направления на цель
        cos_angle = torch.sum(forward_w_norm * to_goal_norm, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # для безопасности
        # direction_to_goal = to_goal
        # yaw_g = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])

        # Вычисляем угол между векторами
        angle = torch.acos(cos_angle)
        angle_degrees = torch.abs(angle) * 180.0 / 3.141592653589793
        # Проверяем, что угол меньше порога
        facing_goal = angle_degrees < angle_threshold

        # Итоговое условие: близко к цели и смотрит в её сторону
        # print(distance_to_goal, angle_degrees)
        
        conditions = torch.stack([close_enough, facing_goal], dim=1)  # shape [N, M]
        num_conditions_met = conditions.sum(dim=1)  # shape [N], количество True в каждой строк

        # self.step_counter += torch.ones_like(self.step_counter) #torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # enouth_steps = self.step_counter > 4
        # returns = torch.logical_and(torch.logical_and(close_enough, facing_goal), enouth_steps)
        # self.step_counter = torch.where(returns, torch.zeros_like(self.step_counter), self.step_counter)
        returns = torch.logical_and(close_enough, facing_goal)
        # if torch.any(returns):
        #     print(close_enough, facing_goal)
        # print("returns", returns)
        if get_num_subs == False:
            return returns
        return returns, num_conditions_met, distance_to_goal+0.1-radius_threshold, angle_degrees

    def get_contact(self):
        force_matrix = self.scene["contact_sensor"].data.net_forces_w
        force_matrix[..., 2] = 0
        forces_magnitude = torch.norm(torch.norm(force_matrix, dim=2), dim=1)  # shape: [batch_size, num_contacts]
        # вычисляем модуль силы для каждого контакта
        if force_matrix is not None and force_matrix.numel() > 0:
            contact_forces = torch.norm(force_matrix, dim=-1)
            if contact_forces.dim() >= 3:
                has_contact = torch.any(contact_forces > 0.1, dim=(1, 2))
            else:
                has_contact = torch.any(contact_forces > 0.1, dim=1) if contact_forces.dim() == 2 else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # print("c ", has_contact)
            num_contacts_per_env = torch.sum(contact_forces > 0.1, dim=1)
            high_contact_envs = num_contacts_per_env >= 1
        else:
            print("force_matrix_w is None or empty")
            high_contact_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # if torch.any(high_contact_envs):
        #     print("high_contact_envs ", high_contact_envs)
        return high_contact_envs

    def update_SR_history(self):
        self.episode_completion_history = torch.zeros((self.num_envs*4, self.num_envs), dtype=torch.bool, device=self.device)
        self.success_history = torch.zeros((self.num_envs*4, self.num_envs), dtype=torch.bool, device=self.device)
        self.history_index = 0
        self.history_len = torch.zeros(self.num_envs, device=self.device)

    def update_success_rate(self, goal_reached):
        if self.turn_on_controller:
            return torch.tensor(self.success_rate, device=self.device)
        
        # Получаем завершенные эпизоды
        died, time_out = self._get_dones(self.my_episode_lenght - 1, inner=True)
        completed = died | time_out
        
        if torch.any(completed):
            # Получаем релевантные среды среди завершенных
            # Фильтруем завершенные среды, оставляя только релевантные
            relevant_completed = self._robot._ALL_INDICES[completed] #relevant_env_ids[(relevant_env_ids.view(1, -1) == self._robot._ALL_INDICES[completed].view(-1, 1)).any(dim=0)]
            success = goal_reached.clone()
            # Обновляем стеки для релевантных завершенных сред
            for env_id in self._robot._ALL_INDICES.clone()[completed]:
                env_id = env_id.item()
                if not success[env_id]:#here idia is colulate all fault and sucess only on relative envs
                    self.success_stacks[env_id].append(0)
                elif env_id in relevant_completed:
                    self.success_stacks[env_id].append(1)
                
                if len(self.success_stacks[env_id]) > self.max_stack_size:
                    self.success_stacks[env_id].pop(0)
            # print("self.success_stacks ", self.success_stacks)
        # Вычисляем процент успеха для всех сред с непустыми стеками
        # Подсчитываем общий процент успеха по всем релевантным средам
        total_successes = 0
        total_elements = 0
        # print(self.success_stacks)
        for env_id in range(self.num_envs):
            stack = self.success_stacks[env_id]
            if len(stack) == 0:
                continue
            total_successes += sum(stack)
            total_elements += len(stack)
        # print("update ", self.success_stacks)
        # Вычисляем процент успеха
        # print("total_successes ", total_successes, total_elements)
        self.sr_stack_capacity = total_elements
        if total_elements > 0:
            self.success_rate = (total_successes / total_elements) * 100.0
        else:
            self.success_rate = 0.0
        # print(total_elements)
        if total_elements >= self.num_envs * self.max_stack_size * 0.9:
            self.sr_stack_full = True
        # print(success_rates, self.success_rate)
        return self.success_rate
    
    def update_sr_stack(self):
        self.success_stacks = [[] for _ in range(self.num_envs)]  # Список списков для каждой среды
        self.sr_stack_full = False

    def _get_dones(self, my_episode_lenght = 256, inner=False) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.is_time_out(my_episode_lenght)
        
        has_contact = self.get_contact()
        self.has_contact = has_contact
        died = torch.logical_or(
            torch.logical_or(self.goal_reached(), has_contact),
            time_out,
        )
        env_ids = self._robot._ALL_INDICES[died]
        if torch.any(died):
            goal_reached = self.goal_reached()
            if self.eval:
                # === EVAL LOGGING ===
                envs_finished = torch.where(died)[0]  # индексы завершившихся эпизодов
                # успехи конкретно для них
                successes = goal_reached[envs_finished].float()
                traj_lens = self.tracker.compute_path_lengths(envs_finished)  # [K]
                durations = self.tracker.lengths[envs_finished].float()       # [K]
                start_dists = self.eval_manager.get_start_dists(envs_finished) # [K]
                # логируем (ВАЖНО: log_results -> ДО next_episode)
                self.eval_manager.log_results(envs_finished, successes, traj_lens, start_dists, durations)
                self.eval_manager.next_episode(envs_finished)

                if self.eval_manager.is_all_done() and not self.eval_printed:
                    import pandas as pd
                    df, global_stats, pos_stats = self.eval_manager.summarize()
                    print("=== FINAL EVAL SUMMARY ===")
                    print(global_stats)
                    self.eval_printed = True

                    # создаём директорию logs если её нет
                    log_dir = os.path.join(self.current_dir, "logs/skrl/results")
                    os.makedirs(log_dir, exist_ok=True)

                    # 1. Сохраняем сырые результаты
                    save_path = os.path.join(log_dir, f"eval_results_{self.eval_name}.csv")
                    df.to_csv(save_path, index=False)

                    # 2. Сохраняем агрегированную информацию
                    summary_path = os.path.join(log_dir, f"eval_summary_{self.eval_name}.csv")

                    # превращаем global_stats и pos_stats в один DataFrame
                    summary_df = pos_stats.copy()
                    summary_df["position_idx"] = summary_df.index
                    summary_df.reset_index(drop=True, inplace=True)

                    # добавляем глобальные метрики как отдельную строку
                    global_row = global_stats.to_dict()
                    global_row["position_idx"] = "ALL"
                    summary_df = pd.concat([summary_df, pd.DataFrame([global_row])], ignore_index=True)

                    summary_df.to_csv(summary_path, index=False)

                    print(f"[EVAL] Results saved to {save_path}")
                    print(f"[EVAL] Summary saved to {summary_path}")

                
        if not inner:
            self.episode_length_buf[died] = 0
        # print("died ", time_out, self.episode_length_buf)
        return died, time_out
    
    def is_time_out(self, max_episode_length=256):
        if self.first_ep[1]:
            self.first_ep[1] = False
            max_episode_length = 2
        time_out = self.episode_length_buf >= max_episode_length
        return time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if self.first_ep[0] or env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES.clone()

        num_envs = len(env_ids)
        if self.eval:
            positions = self.eval_manager.get_positions()
            self.scene_manager.apply_fixed_positions(env_ids, positions)
        else:
            self.scene_manager.randomize_scene(
                env_ids,
                mess=False, # или False, в зависимости от режима
                use_obstacles=self.turn_on_obstacles,
                all_defoult=False
            )
        self.scene_manager.get_graph_embedding(self._robot._ALL_INDICES.clone())
        goal_pos_local  = self.scene_manager.get_active_goal_state(env_ids)
        # BLOCK TEXT_EMB
        # colors = ["red" if x.item() > 0 else "green" for x in goal_pos_local[:, 0]]
        # text_prompts = [f"move to bowl near {c} wall" for c in colors]

        # text_inputs = self.clip_processor(
        #     text=text_prompts, return_tensors="pt", padding=True
        # ).to(self.device)
        # with torch.no_grad():
        #     text_embeddings = self.clip_model.get_text_features(**text_inputs)
        #     text_embeddings = text_embeddings / (text_embeddings.norm(dim=1, keepdim=True) + 1e-9)
        # self.text_embeddings[env_ids] = text_embeddings

        # print("goal_pos_local ", goal_pos_local)
        self._desired_pos_w[env_ids, :3] = goal_pos_local 
        self._desired_pos_w[env_ids, :2] = self.to_global(goal_pos_local , env_ids)

        self.curriculum_learning_module(env_ids) 

        if self.turn_on_controller_step > self.my_episode_lenght and self.turn_on_controller:
            self.turn_on_controller_step = 0
            self.turn_on_controller = False
        
        if not self.eval:
            cond_imitation = (
                not self.warm and
                # self.mean_radius >= 3.3 and
                self.sr_stack_full and
                self.mean_radius != 0 and
                self.use_controller and
                not self.turn_on_controller and
                not self.first_ep[0] and
                self.turn_off_controller_step > self.my_episode_lenght
            )
            if cond_imitation: 
                self.turn_on_controller_step = 0
                self.turn_off_controller_step = 0
                prob = lambda x: torch.rand(1).item() <= x
                self.turn_on_controller = prob(0.01 * max(10, min(40, 100 - self.success_rate)))
                print(f"turn controller: {self.turn_on_controller} with SR {self.success_rate}")
            elif self.cur_step < self.warm_len:
                if self.cur_step < self.without_imitation:
                    self.turn_on_controller = False
                else:
                    self.turn_on_controller = True
                
        
        if (self.mean_radius >= 3.3 and self.use_obstacles) or self.turn_on_obstacles_always or self.warm and not self.first_ep[0]:
        # if self.use_obstacles or self.turn_on_obstacles_always or self.warm and not self.first_ep[0]:
            if self.turn_on_obstacles_always and self.cur_step % 300:
                print("[ WARNING ] ostacles allways turn on")

            self.turn_on_obstacles = True
            if not self.turn_on_obstacles_always and not self.warm and self.min_level_radius < 3.3:
                print("level_up min_level_radius to: ", 3.3)
                self.min_level_radius = 3.3
        else:
            self.turn_on_obstacles = False
        env_ids = env_ids.to(dtype=torch.long)

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids, :2] - self._robot.data.root_pos_w[env_ids, :2], dim=1
        ).mean()
        self._robot.reset(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.zeros_like(self.episode_length_buf) #, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        min_radius = 1.2
        if self.eval:
            robot_pos_local, robot_quats = self.eval_manager.get_current_tasks(env_ids)
            # здесь angle_errors можно применить для ориентации робота
            # вычислим стартовую евклидову дистанцию в локальных координатах
            start_dists_local = torch.linalg.norm(goal_pos_local[:, :2] - robot_pos_local[:, :2], dim=1)
            # сохраним стартовые дистанции в eval_manager
            self.eval_manager.set_start_dists(env_ids, start_dists_local)
        else:
            robot_pos_local, robot_quats = self.scene_manager.place_robot_for_goal(
                env_ids,
                mean_dist=self.mean_radius,
                min_dist=1.2,
                max_dist=4.0,
                angle_error=self.cur_angle_error,
            )
        robot_pos  = robot_pos_local
        # print("robot_pos_local ", robot_pos_local)
        # print("bounds ", self.scene_manager.room_bounds)
        # print("i'm in path_manager")
        if self.turn_on_controller or self.imitation:
            if self.imitation:
                print("[ WARNING ] imitation mode on")
            if self.turn_on_controller_step == 0:
                env_ids_for_control = self._robot._ALL_INDICES.clone()
                robot_pos_for_control = self._robot.data.default_root_state[env_ids_for_control, :2].clone()
                robot_pos_for_control[env_ids, :2] = robot_pos[:, :2]
                goal_pos_for_control = self._desired_pos_w[env_ids_for_control, :2].clone()
                goal_pos_for_control[env_ids, :2] = goal_pos_local[:, :2]
            else:
                env_ids_for_control = env_ids
                robot_pos_for_control = robot_pos
                goal_pos_for_control = goal_pos_local[:, :2]
            paths = None
            possible_try_steps = 3
            obstacle_positions_list = self.scene_manager.get_active_obstacle_positions_for_path_planning(env_ids)

            for i in range(possible_try_steps):
                paths = self.path_manager.get_paths( # Используем старый get_paths
                    env_ids=env_ids_for_control,
                    # Передаем данные для генерации ключа
                    active_obstacle_positions_list=obstacle_positions_list,
                    start_positions=robot_pos_local,
                    target_positions=goal_pos_local[:, :2]
                )
                if paths is None:
                    print(f"[ ERROR ] GET NONE PATH {i + 1} times")
                    self.scene_manager.randomize_scene(
                        env_ids_for_control,
                        mess=False, # или False, в зависимости от режима
                        use_obstacles=self.turn_on_obstacles,
                    )
                    goal_pos_local = self.scene_manager.get_active_goal_state(env_ids_for_control)
                    self._desired_pos_w[env_ids_for_control, :3] = goal_pos_local
                    self._desired_pos_w[env_ids_for_control, :2] = self.to_global(goal_pos_local, env_ids_for_control)
                else:
                    break
            # print("out path_manager, paths: ", paths)
            # print(len(paths), len(env_ids_for_control), len(goal_pos_for_control))
            self.control_module.update_paths(env_ids_for_control, paths, goal_pos_for_control)
        if self.memory_on:
            self.memory_manager.reset()
        # print(f"in reset envs: {env_ids} goals:", goal_pos_local[:, :2])
        
        self._update_scene_objects(env_ids) #self._robot._ALL_INDICES.clone())
        # value = torch.tensor([0, 0], dtype=torch.float32, device=self.device)
        # robot_pos = value.unsqueeze(0).repeat(num_envs, 1)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :2] = self.to_global(robot_pos, env_ids)
        default_root_state[:, 2] = 0.1
        default_root_state[:, 3:7] = robot_quats
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Логируем длину эпизодов для сброшенных сред
        self.total_episode_length += torch.sum(self.episode_lengths[env_ids]).item()
        self.episode_count += len(env_ids)
        mean_episode_length = self.total_episode_length / self.episode_count if self.episode_count > 0 else 0.0
        # self.tensorboard_writer.add_scalar("Metrics/Mean_episode_length", mean_episode_length, self.tensorboard_step)
        # Сбрасываем счетчик длины для сброшенных сред
        self.episode_lengths[env_ids] = 0
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        # print("root_pos_w ", root_pos_w)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        # print("distance_to_goal ", distance_to_goal)
        _, _, r_error, a_error = self.goal_reached(get_num_subs=True)
        self.previous_distance_error[env_ids] = r_error[env_ids]
        self.previous_angle_error[env_ids] = a_error[env_ids]
        self.first_ep[0] = False
        self.tracker.reset(env_ids)
        env_ids_for_scene_embeddings = self._robot._ALL_INDICES.clone()
        # scene_embeddings = self.scene_manager.get_scene_embedding(env_ids)
        # for i in env_ids_for_scene_embeddings:
        #     self.scene_manager.print_graph_info(i)
        if self.LOG and self.sr_stack_full:
            self.experiment.log_metric("success_rate", self.success_rate, step=self.tensorboard_step)
            self.experiment.log_metric("mean_radius", self.mean_radius, step=self.tensorboard_step)
            self.experiment.log_metric("max_angle", self.max_angle_error, step=self.tensorboard_step)
            # self.experiment.log_metric("use obstacles", self.turn_on_obstacles.float(), step=self.tensorboard_step)

    def to_local(self, pos, env_ids=None, env_origins=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES.clone()
        if env_origins is None:
            env_origins = self._terrain.env_origins
        return pos[:, :2] - env_origins[env_ids, :2]
    
    def to_global(self, pos, env_ids=None, env_origins=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES.clone()
        if env_origins is None:
            env_origins = self._terrain.env_origins
        return pos[:, :2] + env_origins[env_ids, :2]

    def curriculum_learning_module(self, env_ids: torch.Tensor):
        # print("self.success_rate ", self.success_rate)
        # if self.mean_radius > 3.3:
        #     max_angle_error = torch.pi * 0.8
        if self.warm and self.cur_step >= self.warm_len:
            self.warm = False
            self.mean_radius = self.start_mean_radius
            self.cur_angle_error = 0
            self._step_update_counter = 0
            print(f"end worm stage r: {round(self.mean_radius, 2)}, a: {round(self.cur_angle_error, 2)}")
        elif not self.warm and not self.turn_on_controller and self.sr_stack_full:
            if self.success_rate >= self.sr_treshhold:
                self.success_ep_num += 1
                self.foult_ep_num = 0
                if self.success_ep_num > self.num_envs:
                    self.second_try = max(self.mean_radius, self.second_try)
                    self.success_ep_num = 0
                    old_mr = self.mean_radius
                    old_a = self.cur_angle_error
                    self.cur_angle_error += self.max_angle_error / 2
                    print("[ sr ]: ", round(self.success_rate, 2), self.sr_stack_capacity)
                    if self.cur_angle_error > self.max_angle_error:
                        self.cur_angle_error = 0
                        if self.mean_radius == 0:
                            self.mean_radius += 0.3
                        else:
                            self.mean_radius += 1
                        print(f"udate [ UP ] r: from {round(old_mr, 2)} to {round(self.mean_radius, 2)}")
                    else:
                        print(f"udate [ UP ] r: {round(self.mean_radius, 2)} a: from {round(old_a, 2)} to {round(self.cur_angle_error, 2)}")
                    self._step_update_counter = 0
                    self.update_sr_stack()
            elif self.success_rate <= 10 or (self._step_update_counter >= 4000 and self.success_rate <= self.sr_treshhold):
                self.foult_ep_num += 1
                if self.foult_ep_num > 2000:
                    self.success_ep_num = 0
                    self.foult_ep_num = 0
                    old_mr = self.mean_radius
                    if self.cur_angle_error == 0:
                        self.mean_radius += -0.1
                        self.mean_radius = max(self.min_level_radius, self.mean_radius)
                    self.cur_angle_error = 0
                   
                    self._step_update_counter = 0
                    print("[ sr ]: ", round(self.success_rate, 2), self.sr_stack_capacity)
                    print(f"udate [ DOWN ] r: from {round(old_mr, 2)} to {round(self.mean_radius, 2)}, a: {round(self.cur_angle_error, 2)}")
                    self.update_sr_stack()

        self._obstacle_update_counter += 1
        return None

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

    def close(self):
        # self.tensorboard_writer.close()
        super().close()

    def _update_scene_objects(self, env_ids: torch.Tensor):
        """Векторизованное обновление позиций всех объектов в симуляторе."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES.clone()
        # Получаем все локальные позиции из scene_manager'а
        all_local_positions = self.scene_manager.positions
        
        # Конвертируем в глобальные координаты
        env_origins_expanded = self._terrain.env_origins.unsqueeze(1).expand_as(all_local_positions)
        all_global_positions = all_local_positions + env_origins_expanded
        
        # Создаем тензор для ориентации (по умолчанию Y-up: w=1)
        all_quats = torch.zeros(self.num_envs, self.scene_manager.num_total_objects, 4, device=self.device)
        all_quats[..., 0] = 1.0
        
        # Собираем полные состояния (поза + ориентация)
        all_root_states = torch.cat([all_global_positions, all_quats], dim=-1)
        
        # Итерируемся по объектам, управляемым симулятором
        for name, object_instances in self.scene_objects.items():
            if name not in self.scene_manager.object_map:
                continue
            
            # Получаем индексы для данного типа объектов
            indices = self.scene_manager.object_map[name]['indices']
            
            # Собираем состояния только для этих объектов
            object_root_states = all_root_states[:, indices, :]
            
            # Обновляем каждый экземпляр этого типа
            for i, instance in enumerate(object_instances):
                # Выбираем срез для i-го экземпляра по всем окружениям
                instance_states = object_root_states[:, i, :]
                # Применяем маску: неактивные объекты берём из default_positions
                active_mask = self.scene_manager.active[:, indices[i]]
                # Используем дефолтные позиции из SceneManager
                inactive_pos = self.scene_manager.default_positions[0, indices[i]]  # (3,)
                inactive_pos = inactive_pos.expand(self.num_envs, -1)  # (num_envs, 3)
                # Конвертируем в глобальные координаты
                inactive_pos_global = inactive_pos + env_origins_expanded[:, indices[i], :]
                # Векторизованное обновление позиций
                final_positions = torch.where(
                    active_mask.unsqueeze(-1),
                    instance_states[:, :3],
                    inactive_pos_global
                )
                instance_states[:, :3] = final_positions
                if name == "bowl":
                    rot = torch.tensor([0.0, 0.0, 0.7071, 0.7071], device=self.device).expand(self.num_envs, -1)
                    instance_states[:, 3:7] = rot
                if name == "cabinet":
                    rot = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.device).expand(self.num_envs, -1)
                    instance_states[:, 3:7] = rot
                # Записываем состояния в симулятор
                instance.write_root_pose_to_sim(instance_states, env_ids=self._robot._ALL_INDICES.clone())
                # changeable_indices = self.scene_manager.type_map.get("changeable_color", torch.tensor([], dtype=torch.long))
                # if len(changeable_indices) > 0:
                #     obj_global_idx = indices[i].item()  # глобальный индекс объекта
                #     if len(changeable_indices) > 0:
                #         obj_global_idx = indices[i].item()
                #         is_changeable = bool(((changeable_indices == obj_global_idx).any()).item())
                #         if is_changeable:
                #             color_vec = self.scene_manager.colors[0, obj_global_idx].cpu().tolist()
                #             prim_path_template = self.asset_manager.all_prim_paths[obj_global_idx]

                #             # Преобразуем шаблон env_.* -> env_{idx}
                #             for env_id in env_ids.tolist():
                #                 prim_path = prim_path_template.replace("env_.*", f"env_{env_id}")
                #                 if prim_path is None:
                #                     continue
                #                 self._set_object_color(prim_path, obj_global_idx, color_vec)

    def _prepare_color_material(self, color_vec: list[float]):
        """Создаёт OmniPBR-материал для цвета и сохраняет в кеш (если ещё нет)."""
        from pxr import Gf
        color_key = f"{color_vec[0]:.3f}_{color_vec[1]:.3f}_{color_vec[2]:.3f}"
        if color_key in self._material_cache:
            return self._material_cache[color_key]

        mtl_created = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name=f"mat_{color_key}",
            mtl_created_list=mtl_created
        )
        if not mtl_created:
            return None

        mtl_path = mtl_created[0]
        stage = omni.usd.get_context().get_stage()
        shader = stage.GetPrimAtPath(mtl_path + "/Shader")
        try:
            shader.GetAttribute("inputs:diffuse_color_constant").Set(Gf.Vec3f(*color_vec))
        except Exception:
            try:
                shader.GetAttribute("inputs:base_color").Set(Gf.Vec3f(*color_vec))
            except Exception:
                pass

        self._material_cache[color_key] = mtl_path
        return mtl_path

    def _set_object_color(self, prim_path: str, obj_idx: int, color_vec: list[float]):
        """Назначает цвет объекту (только если он изменился)."""
        color_key = f"{color_vec[0]:.3f}_{color_vec[1]:.3f}_{color_vec[2]:.3f}"
        prev_key = self._applied_color_map.get(obj_idx)
        if prev_key == color_key:
            return  # цвет уже установлен

        mtl_path = self._prepare_color_material(color_vec)
        if mtl_path is None:
            return

        omni.kit.commands.execute(
            "BindMaterial",
            prim_path=prim_path,
            material_path=mtl_path
        )
        self._applied_color_map[obj_idx] = color_key


    def _apply_color_to_prim(self, prim_path: str, color_vec: list[float]):
        """
        Создаёт (или берёт из кеша) материал OmniPBR для указанного цвета и привязывает его к prim_path.
        color_vec — список/кортеж из 3 чисел [r,g,b] (0..1).
        """
        if prim_path is None:
            return

        color_key = f"{color_vec[0]:.3f}_{color_vec[1]:.3f}_{color_vec[2]:.3f}"
        # если уже применён к этому объекту — пропускаем
        # (в _update_scene_objects мы будем сравнивать с self._applied_color_map[obj_idx])
        print("material cache: ", self._material_cache)
        if color_key not in self._material_cache:
            mtl_created = []
            omni.kit.commands.execute(
                "CreateAndBindMdlMaterialFromLibrary",
                mdl_name="OmniPBR.mdl",
                mtl_name=f"mat_{color_key}",
                mtl_created_list=mtl_created
            )
            if len(mtl_created) == 0:
                # если по какой-то причине не создалось — выходим
                return
            mtl_path = mtl_created[0]
            # попытка поменять параметр цвета внутри шейдера
            stage = omni.usd.get_context().get_stage()
            shader = stage.GetPrimAtPath(mtl_path + "/Shader")
            try:
                shader.GetAttribute("inputs:diffuse_color_constant").Set(Gf.Vec3f(*color_vec))
            except Exception:
                # у разных материалов имя порта может отличаться
                try:
                    shader.GetAttribute("inputs:base_color").Set(Gf.Vec3f(*color_vec))
                except Exception:
                    pass
            self._material_cache[color_key] = mtl_path

        # Привязываем материал (bind) к приму
        omni.kit.commands.execute(
            "BindMaterial",
            prim_path=prim_path,
            material_path=self._material_cache[color_key]
        )


def log_embedding_stats(embedding):
    mean_val = embedding.mean().item()
    std_val = embedding.std().item()
    min_val = embedding.min().item()
    max_val = embedding.max().item()
    print(f"[ EM ] mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
