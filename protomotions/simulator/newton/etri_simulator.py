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
"""ETRI 확장 Newton 시뮬레이터 — **upstream 파일을 수정하지 않는 서브클래스**.

CLAUDE.md §3 의 방침(“기존 클래스 동작 변경 → 서브클래스를 `etri_*` 에 만들고
`_target_` 로 지정”)을 따른다. `protomotions/simulator/newton/simulator.py` 는
그대로 두므로 upstream merge 에서 충돌하지 않는다.

사용법 — 학습·추론 시 `_target_` 만 바꾼다
------------------------------------------
```bash
python protomotions/train_agent.py ... \
    --overrides simulator._target_=protomotions.simulator.newton.etri_simulator.EtriNewtonSimulator
```
`resolved_configs.pt` 에 이 경로가 기록되므로 이후 resume·추론도 자동으로 이 클래스를 쓴다.

추가 기능 1 — 창 없는 오프스크린 렌더 (`PM_OFFSCREEN_VIEWER=1`)
---------------------------------------------------------------
왜 필요한가: 살(LBS) 영상 배치 녹화는 2단계다. ① 추론으로 물리 롤아웃(`.motion`) 기록
② 그 롤아웃을 살로 다시 렌더. ①은 **학습 XML = 도형**을 그리는데, 그 창이 최종 산출물로
오인돼 반복해서 혼란을 일으켰다(2026-08-05).

upstream 은 `headless` 면 뷰어를 **아예 만들지 않아**(`_setup_sim`) `get_frame()` 이
`None` 이 되고 `--headless --auto-record` 조합이 `AttributeError` 로 죽는다.
`newton.viewer.ViewerGL(headless=True)` 는 창 없이 오프스크린 렌더가 되므로, 그것으로
뷰어를 만들어 프레임 캡처는 살리고 창만 없앤다.

학습에는 영향이 없다 — 환경변수가 없으면 뷰어를 만들지 않는다(GL 컨텍스트 비용 0).

추가 기능 2 — 숨겨진 콜라이더 복원 (`PM_FORCE_VISIBLE=1`)
---------------------------------------------------------
Newton 은 `contype=0`(시각 전용) geom 이 든 body 의 **콜라이더** geom 에서 VISIBLE 을
자동으로 지운다. 외골격 슈트처럼 질량만 붙이는 자산에서는 인체 캡슐이 화면에서 사라진다.
이 변수로 되돌린다. 렌더 전용이며 물리·질량·접촉에는 영향이 없다.
★ `set_model()` 이 플래그를 스냅샷하므로 **그 호출 전에** 되살려야 한다.
"""

import os

import newton

from protomotions.simulator.newton.simulator import NewtonSimulator


def _want_offscreen() -> bool:
    return os.environ.get("PM_OFFSCREEN_VIEWER") == "1"


def _want_force_visible() -> bool:
    return os.environ.get("PM_FORCE_VISIBLE") == "1"


class EtriNewtonSimulator(NewtonSimulator):
    """오프스크린 렌더 + 콜라이더 가시성 복원을 더한 Newton 시뮬레이터."""

    # 서브클래스에서 처음 접근할 수 있으므로 기본값을 클래스 속성으로 둔다
    _etri_offscreen = False

    # ── 셋업: headless 여도 오프스크린 뷰어를 만든다 ────────────────────
    def _setup_sim(self) -> None:
        super()._setup_sim()

        self._etri_offscreen = bool(self.headless and _want_offscreen())
        if not self._etri_offscreen:
            # 창 모드(headless=False)에서는 upstream 이 이미 뷰어를 만들었다.
            # 그때도 PM_FORCE_VISIBLE 이 필요하면 upstream 이 처리하지 않으므로
            # 여기서는 손대지 않는다 — set_model 이 이미 끝났기 때문이다.
            return

        # PyOpenGL 을 GLX 모드로 초기화해야 ViewerGL 이 컨텍스트를 만든다.
        os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
        self.viewer = newton.viewer.ViewerGL(headless=True)
        self.viewer.show_ui = False

        if _want_force_visible():
            flags = getattr(self.model, "shape_flags", None)
            if flags is not None:
                visible = int(newton.ShapeFlags.VISIBLE)
                sf = flags.numpy()
                hidden = int((sf & visible == 0).sum())
                if hidden:
                    flags.assign(sf | visible)
                    print(f"[ETRI] PM_FORCE_VISIBLE: restored {hidden} hidden shape(s)")

        self.viewer.set_model(self.model)      # ★ 플래그 스냅샷 — 반드시 복원 뒤에
        self.viewer.vsync = False              # 창이 없으니 의미 없다

        # upstream 은 스텝마다 `self.viewer.apply_forces(...)` 를 **인라인**으로 부른다
        # (simulator.py:987). 마우스 드래그로 힘을 주는 기능이라 창이 없으면 의미가 없고,
        # 메서드가 아니어서 오버라이드할 수 없다 → 인스턴스에서 no-op 으로 갈아끼운다.
        self.viewer.apply_forces = lambda *_a, **_k: None
        setup_hook = getattr(self, "_setup_scene_box_render_hook", None)
        if callable(setup_hook):
            setup_hook()
        print("[ETRI] offscreen viewer ready (no window)")

    # ── 렌더: 카메라·프레임은 돌리고 키보드 폴링은 건너뛴다 ─────────────
    def render(self) -> None:
        if not self._etri_offscreen:
            super().render()
            return

        # 창이 없으므로 키 입력을 받을 수 없다. upstream `render()` 의 키보드 블록을
        # 타지 않도록 프레임 경로만 직접 돈다.
        if not self._camera_initialized:
            self._init_camera()
            self._camera_initialized = True
        else:
            self._update_camera()

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        if self._render_hook is not None:
            self._render_hook()
        self.viewer.end_frame()

        # 상위(Simulator.render)의 녹화·프레임 저장 처리
        super(NewtonSimulator, self).render()
