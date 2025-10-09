# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Aloha robot with velocity-driven wheels."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##
stiffness_arm_const = 500
damping_arm_const = 500
stiffness_wheel_const = 3  # Жесткость для управления скоростью колес
damping_wheel_const = 20    # Демпфирование для управления скоростью колес

ALOHA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/xiso/Downloads/assets/assets/aloha/ALOHA_with_sensor_02.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=10,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(3.0, 0.0, 0.1), joint_pos={".*": 0.0}
    ),
    actuators={
        # Актуатор для колес (управление скоростью)
        "wheel_actuators": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel", "right_wheel"],
            effort_limit=40,  # Оставляем, но не используется для скоростей
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
            velocity_limit=10000.0,
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