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
"""
ETRI IsaacLab 뷰어 커스텀 키 — 카메라 시점 프리셋·추적 토글
==========================================================

이전 fork 는 `IsaacLabSimulator` 에서 `keyboard_interface.add_callback(...)` 을
직접 호출해 키를 붙였다. upstream 이 2026-07 에 `UserInterface` 추상화로 바꾸고
IsaacLab 키보드를 그 위에 브릿지하면서(167행 충돌) ETRI 키도 등록 API 로 옮긴다.

CLAUDE.md §2 규칙: 원본 폴더 안에 둬야 하므로 **파일명 앞에 `etri_`**.
`simulator.py` 에는 등록 호출 몇 줄만 남는다.

## 키 목록

| 키 | 동작 |
|----|------|
| `C` | 측면 시점 |
| `V` | 사선 뒤 45° 시점 |
| `B` | 정면 시점 |
| `F` | 카메라 자동 추적 on/off |
| `[` | 방위각 −30° |
| `]` | 방위각 +30° |

핸들러가 호출하는 `_set_camera_offset` / `_toggle_camera_follow` /
`_rotate_camera_azimuth` 는 `simulator.py` 안의 ETRI 추가분이며 자동 병합됐다.

upstream 이 등록하는 키(Q/J/L/;/O/M/R 등)는 여기서 다루지 않는다.
Newton 백엔드용은 `protomotions/simulator/newton/etri_viewer_keys.py`.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

ETRI_KEY_OWNER = "etri.isaaclab.camera"

AZIMUTH_STEP_DEG = 30.0


def register_etri_camera_keys(simulator) -> None:
    """ETRI 카메라 키를 `simulator.user_interface` 에 등록한다.

    upstream 의 `_setup_keyboard()` 마지막(또는 UserInterface 준비 이후)에서 호출한다.

    Args:
        simulator: IsaacLabSimulator 인스턴스.
    """
    ui = getattr(simulator, "user_interface", None)
    if ui is None:
        log.warning("[ETRI] user_interface 가 없어 카메라 키 등록을 건너뜁니다.")
        return

    def _preset(**kwargs):
        label = next(iter(kwargs))

        def handler():
            fn = getattr(simulator, "_set_camera_offset", None)
            if fn is None:
                log.warning("[ETRI] _set_camera_offset 이 없습니다.")
                return
            fn(**kwargs)
            log.info(f"[Camera] {label} 시점")

        return handler

    def _rotate(delta: float):
        def handler():
            fn = getattr(simulator, "_rotate_camera_azimuth", None)
            if fn is None:
                log.warning("[ETRI] _rotate_camera_azimuth 이 없습니다.")
                return
            fn(delta)

        return handler

    def _follow():
        fn = getattr(simulator, "_toggle_camera_follow", None)
        if fn is None:
            log.warning("[ETRI] _toggle_camera_follow 이 없습니다.")
            return
        fn()

    bindings = (
        ("C", "Camera side view", _preset(side=True)),
        ("V", "Camera diagonal-rear 45 view", _preset(diagonal=True)),
        ("B", "Camera front view", _preset(front=True)),
        ("F", "Toggle camera follow", _follow),
        ("LEFT_BRACKET", "Orbit camera -30", _rotate(-AZIMUTH_STEP_DEG)),
        ("RIGHT_BRACKET", "Orbit camera +30", _rotate(+AZIMUTH_STEP_DEG)),
    )

    for key, description, handler in bindings:
        try:
            ui.register_key(
                key,
                owner=ETRI_KEY_OWNER,
                description=description,
                on_press=handler,
            )
        except Exception as e:
            # upstream 이 같은 키를 선점했을 수 있다 — 치명적이지 않으므로 경고만.
            log.warning(f"[ETRI] 키 '{key}' 등록 실패: {e}")
