# SPDX-FileCopyrightText: Copyright (c) 2026 ETRI. All rights reserved.
#
# Backward-compatibility shim — 정본은 `protomotions.robot_configs.etri_etrisuit` 이다.
# 옛 resolved_configs.pt 피클과 task-local 코드가 이 모듈 경로를 기록/import 하고 있어
# 로드되게 남겨둔다. 신규 작업은 정본 모듈을 직접 import 할 것.
from protomotions.robot_configs.etri_etrisuit import (  # noqa: F401
    EtriSuitRobotConfig,
)

__all__ = ["EtriSuitRobotConfig"]
