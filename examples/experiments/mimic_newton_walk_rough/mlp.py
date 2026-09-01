# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Newton Walk Rough Terrain
=========================

walk 모션 단독으로 비평지(discrete obstacles) 지형 적응 학습.
v11_squat_suit / v10_squat_skeleton 에서 warm_start.

지형 구성: flat 50% / discrete obstacles 50%  (slope·stairs 없음)
장애물 높이: 2.5cm (난이도 0) ~ 17.5cm (난이도 9), 커리큘럼 자동 조정
terrain observation 없음 → v10/v11 warm_start 네트워크 호환

--- skeleton 학습 명령어 ---
    rm -rf results/mimic_newton_walk_rough_skeleton/
    /home/user/miniforge3/envs/env_isaaclab/bin/python protomotions/train_agent.py \\
        --robot-name skeleton_torque \\
        --simulator newton \\
        --experiment-path examples/experiments/mimic_newton_walk_rough/mlp.py \\
        --experiment-name mimic_newton_walk_rough_skeleton \\
        --motion-file data/motion_for_trackers/skeleton_torque_walk.pt \\
        --checkpoint checkpoints/v10_squat_skeleton/score_based.ckpt \\
        --num-envs 4096 \\
        --batch-size 16384 \\
        --training-max-steps 50000000

--- suit 학습 명령어 ---
    rm -rf results/mimic_newton_walk_rough_suit/
    /home/user/miniforge3/envs/env_isaaclab/bin/python protomotions/train_agent.py \\
        --robot-name skeleton_torque_suit \\
        --simulator newton \\
        --experiment-path examples/experiments/mimic_newton_walk_rough/mlp.py \\
        --experiment-name mimic_newton_walk_rough_suit \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt \\
        --checkpoint checkpoints/v11_squat_suit/score_based.ckpt \\
        --num-envs 4096 \\
        --batch-size 16384 \\
        --training-max-steps 50000000

--- 지형 시각화 (4 envs, Newton 뷰어 창 열림) ---
    /home/user/miniforge3/envs/env_isaaclab/bin/python protomotions/train_agent.py \\
        --robot-name skeleton_torque_suit \\
        --simulator newton \\
        --experiment-path examples/experiments/mimic_newton_walk_rough/mlp.py \\
        --experiment-name mimic_newton_walk_rough_preview \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt \\
        --checkpoint checkpoints/v11_squat_suit/score_based.ckpt \\
        --num-envs 4 \\
        --batch-size 32 \\
        --training-max-steps 100000

--- 추론 명령어 (학습 완료 후) ---
    /home/user/miniforge3/envs/env_isaaclab/bin/python protomotions/inference_agent.py \\
        --checkpoint results/mimic_newton_walk_rough_suit/score_based.ckpt \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_walk.pt \\
        --simulator newton \\
        --num-envs 1 \\
        --cycle-seconds 20 \\
        --overrides "robot.asset.asset_file_name=mjcf/skeleton_torque_suit_mesh.xml"
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse


def terrain_config(args: argparse.Namespace):
    # terrain_proportions 순서:
    # [smooth_slope, rough_slope, stairs_up, stairs_down, discrete, stepping_stones, poles, flat]
    # discrete obstacles 높이: 2.5cm(난이도 0) ~ 17.5cm(난이도 9), 커리큘럼으로 자동 조정
    return TerrainConfig(
        terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5],
        num_levels=10,
        num_terrains=10,
        discrete_obstacles_min_height=0.025,
        discrete_obstacles_max_height=0.175,
    )


def scene_lib_config(args: argparse.Namespace):
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        action_smoothness_factory,
        mimic_tracking_rewards_factory,
        pow_rew_factory,
        contact_match_rew_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.action import make_pd_action_config

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
    }

    termination_components = {
        "tracking_error": tracking_error_term_factory(threshold=0.5),
    }

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gv_weight=0.1,
            gav_weight=0.2,
            rh_weight=0.2,
            gt_coef=-25.0,
            gr_coef=-5.0,
            gv_coef=-0.5,
            gav_coef=-0.1,
            rh_coef=-100.0,
        ),
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        "contact_match_rew": contact_match_rew_factory(
            weight=-0.1, zero_during_grace_period=True
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "previous_actions",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_cfg: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "previous_actions",
            ],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )
    return agent_cfg


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
