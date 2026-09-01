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
"""IsaacLab 렌더 재질을 **USD `displayColor` 에서** 만들어 붙인다.

왜 필요한가 (2026-08-06)
------------------------
ProtoMotions 는 IsaacLab 렌더 색을
`isaaclab_tasks.direct.visual_material_utils.apply_humanoid_visual_materials()` 에
맡긴다. 그 함수는 **prim 경로 문자열에 특정 키워드가 있는지**로 색을 고른다:

    exo_main / hip_ring / hat_*        → suit_pink
    slider1 / rh_dump                  → suit_red
    slider2 / rh_dump2                 → suit_yellow
    slider3 / lh_dump                  → suit_cyan
    slider4 / lh_dump2                 → suit_green
    그 밖의 전부                        → skin (아이보리)

23dof 케이블 슈트에 맞춘 규칙이라 **exosuitHS 파트 대부분이 걸리지 않는다.** 실측:
가방(`exo_main_col`)과 허벅지 링(`hip_ring`)만 분홍이 되고, 허리 ㄷ바·ㄷ암·힙 모터·
허벅지 박스는 인체와 똑같은 아이보리로 렌더돼 **슈트를 눈으로 구분할 수 없었다.**

또한 RTX 에서는 **Material 바인딩이 `displayColor` 보다 우선**이므로, USD 에
displayColor 를 올바르게 써 넣어도 그 함수가 바인딩을 덮어쓰면 무시된다.

이 모듈의 방식 — 이름 규칙을 쓰지 않는다
----------------------------------------
각 visual Gprim 의 **`primvars:displayColor` 값을 그대로 읽어** 그 색의
`UsdPreviewSurface` 를 만들고 바인딩한다. 색의 출처가 MJCF 의 `rgba` 이므로
(`usd_convert/etri_mjcf_to_flat_usda.py` 가 옮겨 적는다) **MJCF 를 고치면 렌더 색이
따라온다.** 슈트가 늘어나도 이 파일은 손대지 않는다.

  · 인체 primitive   MJCF `rgba="0.8 0.6 0.4 1"`   → 살색
  · 슈트 파트        MJCF `rgba="0 0 0.55 …"`      → 파랑
  · 힙 모터          MJCF `rgba="0 0.45 0.2 …"`    → 녹색

`displayColor` 가 없는 prim 은 `fallback` 색(살색)으로 둔다.

★ 바인딩 순서가 중요하다
   IsaacLab spawner 가 로봇 루트에 `/Robot/material` 을 걸어 두는데 그것이
   자손보다 강하다(`strongerThanDescendants`). 그래서 자손에 붙이기 전에 루트 바인딩을
   **먼저 푼다** — upstream 함수도 같은 순서로 한다.
"""

from __future__ import annotations

import os

# ── 인체 재질 프리셋 (2026-08-11) ────────────────────────────────────────
# `ETRI_SKIN_MAT=porcelain` 이면 **인체 색(fallback 살색)인 prim 만** 무광 도자기로
# 바꾼다. 슈트 파트는 MJCF rgba 를 그대로 유지하므로 색 구분이 살아 있다.
#   무광 도자기 = 높은 알베도 + 중간 거칠기 + 유약 광택 없음(PreviewSurface 는
#   clearcoat 를 안 쓰므로 거칠기만으로 무광을 만든다).
SKIN_MAT = os.environ.get("ETRI_SKIN_MAT", "")
SKIN_PRESETS = {"porcelain": {"rgb": (0.92, 0.90, 0.87), "rough": 0.42}}


def _is_skin(color, fallback, tol=0.06) -> bool:
    return all(abs(float(c) - float(f)) <= tol for c, f in zip(color, fallback))


def apply_visual_materials_from_display_color(
    stage, robot_root_path: str, fallback=(0.80, 0.60, 0.40), roughness=0.5
) -> int:
    """`displayColor` 를 읽어 PreviewSurface 를 만들어 바인딩한다. 바인딩한 prim 수를 반환."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(robot_root_path)
    if not root.IsValid():
        return 0

    # ── 1) 루트의 강한 바인딩을 먼저 푼다 (자손 바인딩이 먹히도록) ──────
    root_binding = UsdShade.MaterialBindingAPI(root)
    if root_binding:
        root_binding.UnbindAllBindings()
    for child in root.GetChildren():
        if child.IsA(UsdShade.Material):
            UsdShade.MaterialBindingAPI(child).UnbindAllBindings()

    cache: dict[tuple, object] = {}

    def material_for(color, rough):
        """같은 (색, 거칠기)는 재질 하나를 공유한다."""
        key = tuple(round(float(c), 4) for c in color) + (round(float(rough), 3),)
        if key in cache:
            return cache[key]
        tag = "_".join(f"{int(round(c * 255)):03d}" for c in key[:3])
        path = f"/World/Looks/etri_rgb_{tag}_r{int(round(rough * 100)):03d}"
        if not stage.GetPrimAtPath(path).IsValid():
            cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=key[:3], roughness=rough)
            cfg.func(path, cfg)
        mat = UsdShade.Material(stage.GetPrimAtPath(path))
        cache[key] = mat
        return mat

    n = 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.GetPath().pathString.startswith(robot_root_path):
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        # 콜라이더(`collisions`, purpose=guide)는 렌더 대상이 아니다.
        path_l = prim.GetPath().pathString.lower()
        if "/collisions/" in path_l:
            continue
        gprim = UsdGeom.Gprim(prim)
        color = fallback
        attr = gprim.GetDisplayColorAttr()
        if attr and attr.HasAuthoredValue():
            vals = attr.Get()
            if vals is not None and len(vals) > 0:
                color = tuple(vals[0])
        rough = roughness
        if SKIN_MAT in SKIN_PRESETS and _is_skin(color, fallback):
            color = SKIN_PRESETS[SKIN_MAT]["rgb"]
            rough = SKIN_PRESETS[SKIN_MAT]["rough"]
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material_for(color, rough))
        n += 1
    if SKIN_MAT in SKIN_PRESETS:
        print(f"[INFO] 인체 재질 = {SKIN_MAT} {SKIN_PRESETS[SKIN_MAT]}")
    return n
