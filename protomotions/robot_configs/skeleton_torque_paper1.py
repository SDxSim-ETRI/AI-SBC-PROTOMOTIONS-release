# SPDX-FileCopyrightText: Copyright (c) 2026 ETRI. All rights reserved.
#
# Backward-compatibility shim — 정본은 `protomotions.robot_configs.etri_skeleton_torque_paper1` 이다.
# 옛 resolved_configs.pt 피클이 이 모듈 경로를 기록하고 있어(실측 46개) 로드되게 남겨둔다.
# 신규 작업은 정본 모듈을 직접 import 할 것.
from protomotions.robot_configs.etri_skeleton_torque_paper1 import (  # noqa: F401
    SkeletonTorquePaper1RobotConfig,
)

__all__ = ["SkeletonTorquePaper1RobotConfig"]
