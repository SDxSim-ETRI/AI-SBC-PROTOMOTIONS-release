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
class SkeletonTorquePaper1RobotConfig(RobotConfig):
    """
    skeleton_torque (23-DOF) with PAPER-1 fix: SMPL-style PD gains only.

    IDENTICAL body/DOF structure and joint ranges to the baseline
    `skeleton_torque` config (same anatomical-range asset XML,
    `mjcf/skeleton_torque_for_train.xml` — UNCHANGED). The ONLY difference is
    the BUILT_IN_PD `override_control_info` gains: stiffness/damping raised from
    the original anatomical values (elbow 50/5, arm/lumbar/ankle 100/10,
    hip/knee 200/20; velocity_limit 10) to SMPL-style values
    (elbow/arm 500/50, lumbar 1000/100, hip/knee/ankle 800/80; velocity_limit 100,
    effort_limit 500).

    Rationale (Paper 1, contribution C5 — constraint-learnability trade-off):
    the persistent elbow_flex plateau band (~35-38 deg) and arm_flex_r asymmetry
    seen with the baseline `skeleton_torque` config is NOT a reward or joint-range
    problem — it is a PD-authority (stiffness/damping) problem. A 2x2 ablation on
    the 27-DOF sibling model showed the PD main effect (~-1.12 rad on elbow
    tracking RMS) dominates the joint-range main effect (~-0.35 rad), and that
    velocity_limit alone had no effect. Raising PD stiffness/damping while KEEPING
    the anatomical joint ranges (paper cell (4): anat range + SMPL PD) already
    resolves the pathology (elbow RMS 1.5 -> 0.46 @ep3000 -> 0.21 @ep6000, within
    the normal band). Range widening to +-180 (paper cell (2)) only marginally
    improves further (0.14) and is intentionally NOT applied here.

    This config is the transfer of that finding to the 23-DOF motions11/14 task.
    Swap in via `--robot-name skeleton_torque_paper1`; no reward/motion/XML change.
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
            "calcn_r",   # 오른발 뒤꿈치+발바닥 (calcaneus)
            "toes_r",    # 오른발 발가락
            "calcn_l",   # 왼발 뒤꿈치+발바닥
            "toes_l",    # 왼발 발가락
        ]
    )

    default_root_height: float = 0.975

    # [ETRI patch] upstream(2026-08-13 릴리즈)이 필수화한 필드. 해부학적 정면 = +X.
    # 근거: MJCF 실측(toes 가 calcn 기준 X +0.19, 좌우 분리는 Y축) + OpenSim
    # (x,y,z)->(x,-z,y) 치환이 X-forward 보존 + base.py 의 레거시 폴백 규약(+X).
    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            # UNCHANGED anatomical-range XML (range is not the lever; see docstring).
    # [ETRI 2026-08-26] ETRI 고유 자산을 코드 패키지에서 빼내 계열 폴더로 옮겼다.
    #   여러 mimic 실험이 공유하므로 특정 실험 폴더가 아니라 계열 루트에 둔다.
            asset_root="tasks_for_skeleton/data/assets",
            asset_file_name="mjcf/skeleton_torque_for_train.xml",
            # [ETRI patch] upstream 이 RobotAssetConfig 에서 제거한 필드
            # usd_asset_file_name="usd/skeleton_torque/skeleton_torque.usda",
            # [ETRI patch] upstream 이 RobotAssetConfig 에서 제거한 필드
            # usd_bodies_root_prim_path="/World/envs/env_.*/Robot/pelvis/",
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            # PAPER-1 fix: SMPL-style PD gains (cf. skeleton_torque_27dof_smplgains).
            # Baseline (pathology) values kept in comments for reference.
            override_control_info={
                "hip_(flexion|adduction)_[rl]": ControlInfo(
                    stiffness=800.0, damping=80.0, effort_limit=500, velocity_limit=100
                    # baseline: stiffness=200, damping=20, effort=600, vel=10
                ),
                "hip_rotation_[rl]": ControlInfo(
                    stiffness=800.0, damping=80.0, effort_limit=500, velocity_limit=100
                    # baseline: stiffness=200, damping=20, effort=600, vel=10
                ),
                "knee_angle_[rl]": ControlInfo(
                    stiffness=800.0, damping=80.0, effort_limit=500, velocity_limit=100
                    # baseline: stiffness=200, damping=20, effort=600, vel=10
                ),
                "ankle_angle_[rl]": ControlInfo(
                    stiffness=800.0, damping=80.0, effort_limit=500, velocity_limit=100
                    # baseline: stiffness=100, damping=10, effort=500, vel=10
                ),
                "lumbar_(extension|bending|rotation)": ControlInfo(
                    stiffness=1000.0, damping=100.0, effort_limit=500, velocity_limit=100
                    # baseline: stiffness=100, damping=10, effort=160, vel=10
                ),
                "arm_(flex|add|rot)_[rl]": ControlInfo(
                    stiffness=500.0, damping=50.0, effort_limit=500, velocity_limit=100
                    # baseline: stiffness=100, damping=10, effort=250, vel=10
                ),
                "(elbow_flex|pro_sup)_[rl]": ControlInfo(
                    stiffness=500.0, damping=50.0, effort_limit=500, velocity_limit=100
                    # baseline: stiffness=50, damping=5, effort=250, vel=10  <- weakest joint, plateau source
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
