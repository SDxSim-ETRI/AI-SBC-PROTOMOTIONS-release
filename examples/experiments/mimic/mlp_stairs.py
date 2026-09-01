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
직선 계단 Mimic 학습 설정 (LinearStairsTerrain)
==================================================

LinearStairsTerrain 사용:
  - 평지 접근로(2m) + 한 방향 직선 계단(10단) 구성
  - scene_stairs_isaaclab.py 와 동일한 느낌 (계단 바로 앞 스폰)
  - 계단 높이 17cm, 폭 30cm → 최대 높이 1.7m

지형 레이아웃 (x 방향):
  [─── 평지 2m ───][─── 계단 10단 × 0.30m = 3m ───]
  캐릭터 스폰 ↑       한 발 내딛으면 닿는 첫 계단 ↑

학습 명령어 (skeleton, IsaacLab):
    cd /home/user/ProtoMotions
    /home/user/miniforge3/envs/env_isaaclab/bin/python protomotions/train_agent.py \\
        --robot-name skeleton_torque \\
        --simulator isaaclab \\
        --experiment-path examples/experiments/mimic/mlp_stairs.py \\
        --experiment-name mimic_isaaclab_stairs_skeleton \\
        --motion-file data/motion_for_trackers/skeleton_torque_stairs_koo.pt \\
        --checkpoint checkpoints/v10_squat_skeleton/score_based.ckpt \\
        --num-envs 2048 --batch-size 8192

학습 명령어 (suit, IsaacLab):
    cd /home/user/ProtoMotions
    /home/user/miniforge3/envs/env_isaaclab/bin/python protomotions/train_agent.py \\
        --robot-name skeleton_torque_suit \\
        --simulator isaaclab \\
        --experiment-path examples/experiments/mimic/mlp_stairs.py \\
        --experiment-name mimic_isaaclab_stairs_suit \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_stairs_koo.pt \\
        --checkpoint checkpoints/v11_squat_suit/score_based.ckpt \\
        --num-envs 2048 --batch-size 8192
"""
import argparse
import importlib.util
from pathlib import Path

from protomotions.components.terrains.config import TerrainConfig

# mlp.py를 경로 기반으로 직접 로드 (sys.path에 examples가 없어도 동작)
_mlp_path = Path(__file__).parent / "mlp.py"
_spec = importlib.util.spec_from_file_location("mlp_base", _mlp_path)
_mlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mlp)

scene_lib_config = _mlp.scene_lib_config          # noqa: F401
motion_lib_config = _mlp.motion_lib_config        # noqa: F401
env_config = _mlp.env_config                      # noqa: F401
agent_config = _mlp.agent_config                  # noqa: F401
configure_robot_and_simulator = _mlp.configure_robot_and_simulator  # noqa: F401
apply_inference_overrides = _mlp.apply_inference_overrides          # noqa: F401

# 계단 파라미터 상수
_STEP_HEIGHT = 0.17    # m — 계단 높이 (riser)
_STEP_WIDTH  = 0.30    # m — 계단 폭 (tread depth)
_NUM_STEPS   = 10      # 단수
_APPROACH    = 2.0     # m — 계단 앞 평지 길이 (캐릭터 스폰 구간)
_CORRIDOR    = 5.0     # m — 복도 폭 (y 방향, 2048 envs spacing 충족)

# map_length = 평지(2m) + 계단(10 × 0.30m = 3m) = 5m
_MAP_LENGTH = _APPROACH + _NUM_STEPS * _STEP_WIDTH  # 5.0 m

# horizontal_scale 0.05m → 6 픽셀/단
_H_SCALE = 0.05


def terrain_config(args: argparse.Namespace) -> TerrainConfig:
    """
    직선 계단 terrain config (LinearStairsTerrain).

    지형: 평지 2m + 오름 계단 10단 (한 방향, +x)
    스폰: 평지 중앙 → 첫 계단까지 약 1m → 한 발 내딛으면 닿음
    """
    return TerrainConfig(
        _target_="protomotions.components.terrains.terrain_linear_stairs.LinearStairsTerrain",
        map_length=_MAP_LENGTH,   # 5.0 m  (x: 평지 + 계단)
        map_width=_CORRIDOR,      # 5.0 m  (y: 복도 폭)
        num_levels=10,
        num_terrains=10,
        terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        pyramid_stairs_step_height=_STEP_HEIGHT,
        pyramid_stairs_step_width=_STEP_WIDTH,
        pyramid_stairs_platform_size=_APPROACH,
        horizontal_scale=_H_SCALE,
        vertical_scale=0.005,
    )
