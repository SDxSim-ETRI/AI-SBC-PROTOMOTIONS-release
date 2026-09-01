# SPDX-FileCopyrightText: Copyright (c) 2026 ETRI. All rights reserved.
#
# Backward-compatibility shim — 정본은 `protomotions.envs.base_env.etri_active_cable_v2_3_tension_env` 이다.
# 옛 resolved_configs.pt 피클과 설정이 이 모듈 경로를 문자열로 기록하고 있어
# (실측: 피클 43개, 텍스트 42개) 로드되게 남겨둔다.
# 신규 작업은 정본 모듈을 직접 import 할 것.
from protomotions.envs.base_env.etri_active_cable_v2_3_tension_env import (  # noqa: F401
    ActiveCableV2TensionEnv,
)

__all__ = ["ActiveCableV2TensionEnv"]
