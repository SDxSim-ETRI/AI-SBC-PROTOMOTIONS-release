# SPDX-FileCopyrightText: Copyright (c) 2026 ETRI. All rights reserved.
#
# Backward-compatibility shim — 정본은 `protomotions.envs.obs.etri_cable` 이다.
# 옛 resolved_configs.pt 피클과 설정이 이 모듈 경로를 문자열로 기록하고 있어
# (실측: 피클 43개, 텍스트 42개) 로드되게 남겨둔다.
# 신규 작업은 정본 모듈을 직접 import 할 것.
from protomotions.envs.obs.etri_cable import (  # noqa: F401
    compute_cable_path_lengths,
    compute_cable_waypoints,
    compute_cable_velocities,
    compute_cable_obs,
    compute_cable_full_obs,
    compute_cable_tension_penalty,
    compute_cable_body_wrench,
    cable_length_obs_factory,
    cable_full_obs_factory,
    cable_tension_penalty_factory,
    make_cable_render_hook,
)
