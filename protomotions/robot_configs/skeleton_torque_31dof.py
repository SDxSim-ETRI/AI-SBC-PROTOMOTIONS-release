# SPDX-FileCopyrightText: Copyright (c) 2026 ETRI. All rights reserved.
#
# Backward-compatibility shim — 정본은 `protomotions.robot_configs.etri_skeleton_torque_31dof` 이다.
# 옛 resolved_configs.pt 피클과 task-local 코드가 이 모듈 경로를 기록/import 하고 있어
# 로드되게 남겨둔다. 신규 작업은 정본 모듈을 직접 import 할 것.
from protomotions.robot_configs.etri_skeleton_torque_31dof import (  # noqa: F401
    SkeletonRobotConfig,
    SkeletonTorque31DofRobotConfig,
)

__all__ = ["SkeletonRobotConfig", "SkeletonTorque31DofRobotConfig"]
