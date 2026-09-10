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
Mimic Environment Configuration
================================

Full-body motion tracking environment with pose and velocity tracking.
Uses early termination on tracking error and bootstrapping at episode end.
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
    """HS-ROUGH 복합지형 **단계 D** — discrete 상한을 실제로 20 cm 로

    단계 C(경사 13°) 대비 **딱 한 가지만** 바꾼다: discrete_reach_max=True.
    경사 13°·수직벽·러프·평지·종료 임계 0.5·num_envs 16384 는 C 와 동일하다.

    왜 필요한가 (사용자 지시 2026-09-02: "20cm까지 학습을 할 수 있도록 반영")
    ------------------------------------------------------------------------
    upstream 커리큘럼은 difficulty = level_idx / num_levels 이므로 최고 레벨이
    9/10 = 0.9 다. 따라서 discrete 상한을 0.20 으로 설정해도 실제 도달값은
    0.01 + 0.9 × (0.20-0.01) = **0.181 m** 에 그친다(단계 A~C 실측 18.0 cm).
    discrete_reach_max=True 는 discrete 에도 정규화된 난이도를 넘겨
    최고 레벨에서 **정확히 20 cm** 가 되게 한다.
    rough 는 지시대로 손대지 않는다(raw difficulty 유지), flat 은 난이도 무관.

    왜 13° 인가 (사용자 지시 2026-09-02: "예전 시험에서 13도 통과했으니 13도로")
    ------------------------------------------------------------------------
    v1 의 slope_scale=0.227 은 tan(12.8°) 이지만, upstream 커리큘럼이
    difficulty = level_idx/num_levels 로 최고 0.9 이므로 실제 도달값은
    0.9 × 0.227 = 0.204 → **11.6°** 였다. 설정한 13° 에 닿은 적이 없다.
    근거는 ondi(10.254.74.144) 의 v27 = mixed_slope13_discrete15 실험이다.

    ※ 건축법 조사 결과(2026-09-02): 국내 보행 경사로 상한은 1:8 = 7.1°
      (피난·방화구조 규칙 제15조⑤), 주차장 경사로 17% = 9.6°, 임도 18% = 10.2°.
      13° 는 이들보다 급하고 계단(26~35°)보다 완만한, 규정 사이의 값이다.
      당초 계획했던 20° 는 건축법상 경사로 근거가 없어 13° 로 조정했다.

    열 배치는 v1/A/B 와 동일하게 유지한다. upstream 의 terrain_proportions 경로는
    choice<0.05 조건으로 **col 0 에 음의 경사(골짜기)** 를 준다(terrain.py:476).
    C 는 EtriSlopeCapTerrain 을 쓰려고 terrain_sequence 로 명시하되
    SMOOTH_SLOPE_VALLEY(100) 로 그 골짜기를 재현한다.
    proportions 도 남겨 둔다 — Terrain.is_flat() 이 그 값을 본다.
    """
    from tasks_for_smpl.mimic_smpl.terrains.etri_terrain import (
        EtriSlopeCapTerrainConfig,
        SMOOTH_SLOPE,
        SMOOTH_SLOPE_VALLEY,
        ROUGH_SLOPE,
    )

    DISCRETE, FLAT = 4, 7
    return EtriSlopeCapTerrainConfig(
        # is_flat() 판정용으로만 남는다 (배치는 terrain_sequence 가 결정).
        terrain_proportions=[0.25, 0.25, 0.0, 0.0, 0.30, 0.0, 0.0, 0.20],
        # v1/A/B 의 proportions 경로와 같은 배치를 명시한다.
        #   col 0 은 골짜기(음의 경사) — terrain.py:476 의 choice<0.05 재현
        terrain_sequence=[
            SMOOTH_SLOPE_VALLEY, SMOOTH_SLOPE, SMOOTH_SLOPE,   # col 0-2
            ROUGH_SLOPE, ROUGH_SLOPE,                          # col 3-4
            DISCRETE, DISCRETE, DISCRETE,                      # col 5-7
            FLAT, FLAT,                                        # col 8-9
        ],
        num_levels=10,
        num_terrains=10,
        # discrete: 1~20cm, 직각 모서리 — 단계 B 와 동일 (raw difficulty 로 위임)
        discrete_obstacles_min_height=0.01,
        discrete_obstacles_max_height=0.20,
        discrete_obstacles_bevel_size=0.0,
        # rough: 유지 (지시). raw difficulty × slope_scale 그대로 쓴다.
        slope_scale=0.227,
        rough_terrain_amplitude=0.10,
        smooth_slope_max_deg=13.0,          # 단계 C 에서 확보 (실측 12.85°)
        # ★ 단계 D 의 유일한 변경: discrete 도 최고 레벨에서 상한에 도달
        #   18.0 cm → 20.0 cm. rough 는 영향 없음(raw difficulty 유지).
        discrete_reach_max=True,
    )


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults)."""
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
    """Build agent configuration."""
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

    agent_config: PPOAgentConfig = PPOAgentConfig(
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
                # ★ 균등 샘플링 (2026-07-31, 27 모션 대응)
                #   원래 이 조합은 **실패 모션의 가중치를 올리는 커리큘럼**이다:
                #     성공 *= 0.999^N (감소) / 실패는 1.0 으로 리셋
                #   → 10회 eval(2000 ep) 후 실패 클립이 성공 클립의 약 7배 샘플링.
                #   "어렵지만 학습 가능한" 클립 전제인데, 27 모션 중 HF 11개는
                #   foot-skate 로 **물리적 재현이 불가능**하다(ORDER.md §4).
                #   그 클립들이 학습을 점령하면 전체 품질이 떨어진다.
                #   → success_discount=1.0 으로 감쇠를 끄면 모든 가중치가 1.0 로
                #     유지되어 **균등 샘플링**이 된다. 어느 쪽으로도 편향 없음.
                motion_weights_update_success_discount=1.0,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot to add contact sensors for foot contact tracking."""
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
    """Apply evaluation-specific overrides."""
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
