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
ETRI Newton 뷰어 커스텀 키 — 카메라 방위각 제어
==============================================

이전 fork 는 `NewtonSimulator._handle_user_input()` 의 `if/elif is_key_down(...)`
체인에 키를 직접 끼워 넣었다. upstream 이 2026-07 에 그 체인을 전부 없애고
`UserInterface.register_key(owner=...)` 등록 방식으로 바꿨기 때문에(113행 충돌)
ETRI 키도 그 API 로 옮긴다.

CLAUDE.md §2 규칙: 원본 폴더 안에 둬야 하므로 **파일명 앞에 `etri_`**.
`simulator.py` 에는 등록 호출 2행만 남는다.

## 키 목록

| 키 | 동작 |
|----|------|
| `[` | 방위각 −30° (반시계 궤도) |
| `]` | 방위각 +30° (시계 궤도) |
| `B` | 후면 시점 (180°) |
| `N` | 정면 시점 (0°) |

`_camera_azimuth` 는 `NewtonSimulator.__init__` 에서 초기화되고
`_update_camera()` 가 읽는다 (둘 다 ETRI 추가분, 자동 병합됨).

upstream 이 이미 등록하는 키(Q/J/L/;/O/M/R 등)는 여기서 다루지 않는다 —
`base_simulator/simulator.py` 가 `owner="simulator"` 로 등록한다.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

ETRI_KEY_OWNER = "etri.newton.camera"

# 궤도 회전 1회당 각도
AZIMUTH_STEP_DEG = 30.0


def register_etri_camera_keys(simulator) -> None:
    """ETRI 카메라 키를 `simulator.user_interface` 에 등록한다.

    `NewtonSimulator.__init__` 에서 upstream 의
    `_register_custom_user_interface_keys()` 직후에 호출한다.

    Args:
        simulator: NewtonSimulator 인스턴스. `user_interface` 와
            `_camera_azimuth` 를 가지고 있어야 한다.
    """
    ui = getattr(simulator, "user_interface", None)
    if ui is None:
        log.warning("[ETRI] user_interface 가 없어 카메라 키 등록을 건너뜁니다.")
        return

    if not hasattr(simulator, "_camera_azimuth"):
        # __init__ 순서가 바뀌어 아직 없을 수 있다 — 기본값으로 만들어 둔다.
        simulator._camera_azimuth = 0.0

    def _orbit(delta: float):
        def handler():
            simulator._camera_azimuth = (simulator._camera_azimuth + delta) % 360.0
            log.info(f"[Camera] 방위각 {simulator._camera_azimuth:.0f}°")

        return handler

    def _set_azimuth(value: float, label: str):
        def handler():
            simulator._camera_azimuth = value
            log.info(f"[Camera] {label} ({value:.0f}°)")

        return handler

    bindings = (
        ("[", "Orbit camera counter-clockwise (-30°)", _orbit(-AZIMUTH_STEP_DEG)),
        ("]", "Orbit camera clockwise (+30°)", _orbit(+AZIMUTH_STEP_DEG)),
        ("B", "Camera rear view (180°)", _set_azimuth(180.0, "후면 시점")),
        ("N", "Camera front view (0°)", _set_azimuth(0.0, "정면 시점")),
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
            # 키 충돌(upstream 이 같은 키를 선점) 등은 치명적이지 않다 — 경고만.
            log.warning(f"[ETRI] 키 '{key}' 등록 실패: {e}")
