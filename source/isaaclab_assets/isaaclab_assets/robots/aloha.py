# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Aloha robot with velocity-driven wheels."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import os
# import importlib.util
# def import_class_from_path(module_path, class_name):
#     spec = importlib.util.spec_from_file_location("custom_module", module_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return getattr(module, class_name)
current_dir = os.getcwd()
# module_path = os.path.join(current_dir, "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/asset_manager.py")
# Asset_paths = import_class_from_path(module_path, "Asset_paths")
# Asset_paths_manager = Asset_paths()
##
# Configuration
##
stiffness_arm_const = 500
damping_arm_const = 500
stiffness_wheel_const = 2.6  # Жесткость для управления скоростью колес
damping_wheel_const = 10.6   # Демпфирование для управления скоростью колес

ALOHA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(current_dir, "source/isaaclab_assets/data/aloha_assets", "aloha/aloha.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=0.3,
            enable_gyroscopic_forces=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=20,
            solver_velocity_iteration_count=20,
            sleep_threshold=0.0005,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1), joint_pos={".*": 0.0}
    ),
    actuators={
        # Актуатор для колес (управление скоростью)
        "wheel_actuators": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel", "right_wheel"],
            effort_limit=100,  # Оставляем, но не используется для скоростей
            velocity_limit=100.0,  # Ограничение скорости в рад/с
            stiffness={
                "left_wheel": stiffness_wheel_const,
                "right_wheel": stiffness_wheel_const,
            },
            damping={
                "left_wheel": damping_wheel_const,
                "right_wheel": damping_wheel_const,
            },
        ),
        # Актуатор для остальных суставов (управление усилием)
        "arm_actuators": ImplicitActuatorCfg(
            joint_names_expr=["fl_joint.*", "fr_joint.*", "lr_joint.*", "rr_joint.*"],
            effort_limit=40,
            velocity_limit=1000.0,
            stiffness={
                "fl_joint.*": stiffness_arm_const,
                "fr_joint.*": stiffness_arm_const,
                "lr_joint.*": stiffness_arm_const,
                "rr_joint.*": stiffness_arm_const,
            },
            damping={
                "fl_joint.*": damping_arm_const,
                "fr_joint.*": damping_arm_const,
                "lr_joint.*": damping_arm_const,
                "rr_joint.*": damping_arm_const,
            },
        ),
    },
)
"""Configuration for the Aloha robot with velocity-driven wheels."""