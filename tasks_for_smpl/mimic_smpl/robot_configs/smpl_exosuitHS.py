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
SMPL 인체 + 힙 전용 외골격(`exosuitHS`) 로봇 설정.

**골격은 맨몸 SMPL 과 완전히 동일하다**(24 body / 69 DOF). 슈트는 body 에 얹힌
geom 일 뿐이고 새 DOF 를 만들지 않는다. 따라서:

  · `SmplRobotConfig` 를 상속하고 **asset 경로만** 바꾼다
    → control_info / body 매핑 / trackable / simulation_params 가 자동 동기화되고
      맨몸 설정이 바뀌면 따라온다
  · 맨몸 모션·체크포인트를 **리타깃·패딩 없이** 재사용한다

슈트가 물리에 주는 영향은 **질량·관성뿐**이다 (총 **4.700 kg**):
  Torso +1.370 (가방) / Pelvis +1.690 (허리밴드 ㄷ 프레임)
  각 Hip +0.820 (힙모터 0.530 + 허벅지박스·링 0.290)
  무릎 이하 파트는 **없다** — 힙 전용 슈트.

슈트 geom 은 `contype=0 conaffinity=0` (충돌 제외, 질량만). MuJoCo·Newton 모두
질량·관성은 그대로 계산된다. 평지 보행에서 슈트가 지면·환경과 부딪힐 일이 없고,
몸과의 겹침은 착용 구조상 정상이라 충돌로 처리하면 레퍼런스 모션과 싸우게 된다.

보조 모터는 **힙 2개**(`L_Hip_y` / `R_Hip_y`, Unitree GO-M8010-6 최대 23.7 Nm).

XML 생성: `tasks_for_smpl/script/make_exosuit_train.py hs`
명세: `tasks_for_smpl/script/exosuit_spec.py` (SUITS["hs"])
"""

from typing import Tuple
from dataclasses import dataclass, field

from protomotions.robot_configs.base import RobotAssetConfig
from protomotions.robot_configs.smpl import SmplRobotConfig

_ASSET_ROOT = "tasks_for_smpl/mimic_smpl/data/assets"


@dataclass
class SmplExosuitHSRobotConfig(SmplRobotConfig):
    """맨몸 SMPL 설정 + exosuitHS 자산 경로."""

    # [ETRI patch] upstream(2026-08-13 릴리즈)이 필수화한 필드. 해부학적 정면 = +X.
    # 근거: MJCF 실측 + OpenSim (x,y,z)->(x,-z,y) 치환의 X-forward 보존
    #       + base.py 레거시 폴백 규약(+X). 기존 체크포인트 동작과 동일.
    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_root=_ASSET_ROOT,
            asset_file_name="mjcf_newton_exosuitHS/smpl_humanoid_exosuitHS_for_train.xml",
            # IsaacLab 학습·렌더용. Newton 학습에는 불필요(MJCF 직접 사용).
            # 변환: IsaacLab convert_mjcf.py (usd_isaaclab_exosuitHS/ 는 아직 비어 있음)
            # [ETRI patch] upstream 이 RobotAssetConfig 에서 제거한 필드
            # usd_asset_file_name="usd_isaaclab_exosuitHS/smpl_humanoid_exosuitHS_for_train.usda",
            # [ETRI patch] upstream 이 RobotAssetConfig 에서 제거한 필드
            # usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )
