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
from protomotions.robot_configs.base import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
    SimulatorParams,
)
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import IsaacLabSimParams
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.simulator.mujoco.config import MujocoSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class SkeletonTorque27DofSmplGainsRobotConfig(RobotConfig):
    """
    Ablation variant of SkeletonTorque27DofRobotConfig (skeleton_torque_27dof.py):
    SAME 27-DOF OpenSim skeleton body/DOF structure, but with joint ranges and
    PD gains (stiffness/damping/effort_limit/velocity_limit) replaced by SMPL
    humanoid's values (protomotions/robot_configs/smpl.py), group-mapped by
    joint role:

        SMPL group                          -> skeleton_torque joints
        Hip/Knee/Ankle (800/80/500/100)     -> hip_*, knee_angle_*, ankle_angle_*
        Torso/Spine/Chest (1000/100/500/100)-> lumbar_*
        Shoulder/Elbow (500/50/500/100)     -> arm_*, elbow_flex_*, pro_sup_*
        Wrist/Hand (300/30/500/100)         -> wrist_flex_*, wrist_dev_*

    Joint ranges widened to SMPL's -180/180 deg (unconstrained) in the asset
    XML itself (range isn't part of ControlInfo, must be set in the MJCF).

    Motivation (2026-08-03): a remote system training a SMPL-body policy on
    the identical 27 motions showed visually more natural arm motion than our
    skeleton_torque policy (which has a persistent elbow_flex plateau band and
    an arm_flex_r asymmetry -- see project_mimic_skeleton_motions15_27dof
    memory). The most salient difference in SMPL's config is velocity_limit=100
    vs skeleton_torque's velocity_limit=10 (10x) -- a plausible bottleneck for
    fast arm-swing tracking. This config isolates "same body, SMPL's PD/range"
    to test whether that's the actual cause, independent of body-model choice.

    See tasks/mimic_smpl_to_skeleton_motions48/data/assets/
    mjcf_skeleton_torque_newton_smplgains/skeleton_torque_27dof_for_train.xml.
    """

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["calcn_l", "toes_l"],
            "all_right_foot_bodies": ["calcn_r", "toes_r"],
            "all_left_hand_bodies": ["hand_l"],
            "all_right_hand_bodies": ["hand_r"],
            "head_body_name": ["torso"],
            "torso_body_name": ["torso"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso",
            "calcn_r",
            "calcn_l",
            "hand_r",
            "hand_l",
        ]
    )

    contact_bodies: List[str] = field(
        default_factory=lambda: [
            "calcn_r",
            "toes_r",
            "calcn_l",
            "toes_l",
        ]
    )

    default_root_height: float = 0.975

    # [ETRI patch] upstream(2026-08-13 릴리즈)이 필수화한 필드. 해부학적 정면 = +X.
    # 근거: MJCF 실측(toes 가 calcn 기준 X +0.19, 좌우 분리는 Y축) + OpenSim
    # (x,y,z)->(x,-z,y) 치환이 X-forward 보존 + base.py 의 레거시 폴백 규약(+X).
    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_root="tasks_for_skeleton/mimic_smpl_to_skeleton_motions48/data/assets/mjcf_skeleton_torque_newton_smplgains",
            asset_file_name="skeleton_torque_27dof_for_train.xml",
            # [ETRI patch] upstream 이 RobotAssetConfig 에서 제거한 필드
            # usd_asset_file_name="usd/skeleton_torque/skeleton_torque.usda",
            # [ETRI patch] upstream 이 RobotAssetConfig 에서 제거한 필드
            # usd_bodies_root_prim_path="/World/envs/env_.*/Robot/pelvis/",
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                "hip_(flexion|adduction)_[rl]": ControlInfo(
                    stiffness=800.0, damping=80.0, effort_limit=500, velocity_limit=100
                ),
                "hip_rotation_[rl]": ControlInfo(
                    stiffness=800.0, damping=80.0, effort_limit=500, velocity_limit=100
                ),
                "knee_angle_[rl]": ControlInfo(
                    stiffness=800.0, damping=80.0, effort_limit=500, velocity_limit=100
                ),
                "ankle_angle_[rl]": ControlInfo(
                    stiffness=800.0, damping=80.0, effort_limit=500, velocity_limit=100
                ),
                "lumbar_(extension|bending|rotation)": ControlInfo(
                    stiffness=1000.0, damping=100.0, effort_limit=500, velocity_limit=100
                ),
                "arm_(flex|add|rot)_[rl]": ControlInfo(
                    stiffness=500.0, damping=50.0, effort_limit=500, velocity_limit=100
                ),
                "(elbow_flex|pro_sup)_[rl]": ControlInfo(
                    stiffness=500.0, damping=50.0, effort_limit=500, velocity_limit=100
                ),
                "wrist_(flex|dev)_[rl]": ControlInfo(
                    stiffness=300.0, damping=30.0, effort_limit=500, velocity_limit=100
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=40,
                decimation=2,
                substeps=2,
            ),
            isaaclab=IsaacLabSimParams(
                fps=120,
                decimation=6,
            ),
            genesis=GenesisSimParams(
                fps=60,
                decimation=3,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=120,
                decimation=6,
            ),
            mujoco=MujocoSimParams(
                fps=120,
                decimation=6,
            ),
        )
    )
