# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for the base environment.

This module defines the configuration dataclasses for environment settings,
rewards, terminations, and observation components.
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field

from protomotions.envs.obs.scene_obs import SceneObsConfig
from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.control.base import ControlComponentConfig

if TYPE_CHECKING:
    from protomotions.envs.mdp_component import MdpComponent


@dataclass
class RecoveryResetConfig:
    """Reset-time sampling from a cache of physically settled fall poses."""

    recovery_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability that a reset uses a fall pose.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    fall_sim_steps: int = field(
        default=150,
        metadata={
            "help": "Physics steps used to generate settled fall poses.",
            "min": 0,
        },
    )


@dataclass
class EnvConfig:
    """Main environment configuration."""

    max_episode_length: int = field(
        default=300,
        metadata={"help": "Maximum steps per episode before automatic reset.", "min": 1}
    )
    reset_grace_period: int = field(
        default=5,
        metadata={"help": "Steps after reset where grace period applies (for zeroing unreliable rewards).", "min": 0}
    )
    num_state_history_steps: int = field(
        default=0,
        metadata={"help": "Number of historical state steps to store. 0 = no history.", "min": 0}
    )

    _target_: str = "protomotions.envs.base_env.env.BaseEnv"

    scene_obs: SceneObsConfig = field(
        default_factory=SceneObsConfig,
        metadata={"help": "Scene observation configuration."}
    )

    motion_manager: MotionManagerConfig = field(
        default_factory=MotionManagerConfig,
        metadata={"help": "Motion manager for reference motion handling."}
    )

    ref_respawn_offset: float = field(
        default=0.05,
        metadata={"help": "Height offset for respawning relative to reference.", "min": 0.0}
    )
    ref_object_respawn_offset: float = field(
        default=0.0,
        metadata={"help": "Height offset for object respawning."}
    )
    ref_contact_smooth_window: int = field(
        default=0,
        metadata={"help": "Window length for smoothing contact labels. 0 = no smoothing.", "min": 0}
    )
    recovery_reset: RecoveryResetConfig = field(
        default_factory=RecoveryResetConfig,
        metadata={"help": "Optional reset sampling from generated fall poses."},
    )
    skip_correct_terrain_height_on_flat: bool = field(
        default=True,
        metadata={"help": "Skip terrain height correction when terrain is flat (optimization)."}
    )

    show_terrain_markers: bool = field(
        default=False,
        metadata={"help": "Show terrain markers during evaluation. Uses significant memory in IsaacGym."}
    )
    save_dir: str = field(
        default="",
        metadata={"help": "Directory for saving evaluation outputs."}
    )

    reward_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of named reward components. Each is a MdpComponent."}
    )
    
    control_components: Dict[str, ControlComponentConfig] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of stateful task/control managers."}
    )
    
    termination_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of termination functions. Each is a MdpComponent."}
    )
    
    observation_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of observation functions. Each is a MdpComponent."}
    )

    action_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Single action processing config dict with 'fn' key. Use make_pd_action_config() helper."}
    )

    # ── [ETRI 2026-08-25] 미참조 DOF 의 PD 목표를 고정한다 ──────────────────
    # 리타깃된 SMPL 참조 모션에서 **전 구간 std=0** 인 DOF 가 12개 있다
    # (L_Toe/R_Toe/L_Hand/R_Hand 각 x,y,z = 인덱스
    #  [9,10,11,21,22,23,51,52,53,66,67,68]). mocap 리타깃이 그 관절을 구동하지
    # 않기 때문이다. 참조가 상수면 트래킹 보상이 그 DOF 를 제약하지 못하므로
    # 정책이 자유롭게 흔든다 — 실측(2026-08-25, walk_cmu_103_07 exosuitHS S1):
    # 발가락 진폭 최대 23.5°, 손 최대 6.1°.
    #
    # 액션 EMA(alpha=0.4)는 전체 jerk 를 -30% 줄이지만 이 항목은 오히려 늘렸다
    # (22.0° -> 27.3°) — 저역통과로는 "유인 없는 자유도"를 없앨 수 없다.
    #
    # 여기서는 _process_action 이후의 processed_action(= PD 목표, rad)을 지정
    # 인덱스에서 상수 0 으로 덮어쓴다. 참조값이 정확히 0 rad 이므로 참조를
    # 완벽히 추종하는 것과 같다. None 이면 기존 동작 불변.
    etri_zero_dof_targets: Optional[List[int]] = field(
        default=None,
        metadata={"help": "PD 목표를 0 rad 로 고정할 DOF 인덱스 목록 (미참조 DOF)."}
    )

    # Odometer corruption parameters.  Used by odom_offset_factory to
    # simulate per-session calibration error in the G1 leg-kinematics odometer.
    # The env samples odom_scale and a yaw_bias angle once per episode at reset.
    # Identity defaults mean no corruption when the factory is not used.
    odom_scale_range: tuple = field(
        default=(1.0, 1.0),
        metadata={
            "help": (
                "Per-episode odometer scale factor range (lo, hi) drawn from Uniform. "
                "Default (1.0, 1.0) = no scale corruption. "
                "Recommended for odom experiments: (0.7, 1.3)."
            )
        },
    )
    odom_yaw_range_deg: float = field(
        default=0.0,
        metadata={
            "help": (
                "Per-episode odometer yaw bias magnitude in degrees. "
                "The actual bias is drawn from Uniform(-deg, +deg). "
                "Default 0.0 = no yaw corruption. "
                "Recommended for odom experiments: 6.0."
            )
        },
    )
    odom_log_noise_std: float = field(
        default=0.0,
        metadata={
            "help": (
                "Per-step odometer noise std in log(1+mag) space. "
                "Default 0.0 = no per-step noise. "
                "Recommended for odom experiments: 0.12."
            )
        },
    )
    odom_soft_threshold: float = field(
        default=0.15,
        metadata={
            "help": (
                "Smooth noise ramp characteristic length in metres. "
                "Noise weight = mag / (mag + threshold). "
                "Default 0.15."
            )
        },
    )
