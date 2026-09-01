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
외골격 슈트를 IsaacSim(IsaacLab 환경)에서 '추론시각화 룩'으로 렌더 — 단일 모델.

  mode=train : smpl_humanoid_<label>_for_train.xml 의 primitive 를 USD 로 세워 렌더
               (몸 capsule/box + 외골격 사선캡슐 ㄷ·구·box·링, 전부 표시)
  mode=eval  : SMPL LBS 스킨(이음새 없음) + 외골격 CAD mesh(mesh/exosuit_hs)
               몸/외골격 primitive 는 아예 만들지 않으므로 가릴 것이 없다.

render_seamless_isaacsim.py 의 룩(타일 바닥 + 하늘 돔 + 측면 추적 카메라 + HUD,
PathTracing 1920×1080)과 캡처·mp4 경로를 그대로 쓴다.

물리는 쓰지 않는다. 모션의 body 글로벌 회전/이동(rigid_body_rot/pos)을 body Xform 에
프레임마다 직접 넣는 kinematic replay 이므로, Newton compare 스크립트와 같은 그림이 된다.
(변환된 로봇 USD 는 IsaacLab **학습**용으로 usd_isaaclab_<label>/ 에 따로 있다. 렌더에는
 prim 구조 추측이 필요 없는 이 방식이 안전해서 MJCF 에서 직접 세운다.)

다리 CAD mesh 는 Newton 쪽과 동일하게 **힙 굴곡(y)만** 따른다 — 모터 구가 반구 2개로
나뉘어 있어 외전·회전까지 따르면 두 반구가 최대 36mm 어긋난다(exosuit_leg_mesh.py 참고).

스킨 살 축소(THIGH_SLIM/TORSO_SLIM)는 Newton 쪽 확정값 0.85 를 기본으로 쓴다.

Usage:
  cd /home/user/ProtoMotions
  /home/user/miniforge3/envs/env_isaaclab/bin/python \
      tasks_for_smpl/script/render_exosuit_isaacsim.py <hs|cr> <train|eval> <out.mp4> [nframes]

  env: CAM=turntable|front_rel|back_rel|side_rel|front|side  (기본 turntable=360° 회전)
       ★ *_rel = **진행방향 기준** 상대 카메라(2026-08-28 추가). 모션의 월드 방향이
         달라도 항상 같은 앵글이 나온다. front_rel = 사람을 정면에서 바라봄.
         CAM_YAW(도) 로 각도 미세조정, CAM_SMOOTH(프레임) 로 평활 창 조절(기본 45).
         turntable/front/side 는 월드 고정이라 "+x 로 걷는다"를 가정한다 — 기존
         산출물 재현용으로 남겨둔 것이며 신규 영상에는 *_rel 을 쓸 것.
       MOTION_NAME=<표시할 모션명>  ★ HUD 에 찍힌다. 미지정이면 녹화 폴더명
         'YYYY-MM-DD-HH-MM-SS-<내용>' 에서 <내용> 을 쓴다. 둘 다 없으면 롤아웃
         파일명(타임스탬프)이 나가므로 경고를 찍는다 — HUD 는 날짜가 아니라 모션명이다.
       CAM_R=반경(m),
       THIGH_SLIM / TORSO_SLIM (eval 스킨 살 축소, 기본 0.85)
       SKIN_MAT=flat|skin_rubber|rubber|porcelain   (기본 flat = 기존 displayColor 살색 룩)
         ★ 릴리즈 룩은 skin_rubber(갈색 무광 고무). 직접 지정하지 말고
           `source tasks_for_smpl/script/render_look.env` 로 룩 전체를 불러올 것.
         스킨 메시에 UsdPreviewSurface 재질을 바인딩한다(2026-08-11 추가).
         프리셋 값은 MAT_RGB="r,g,b" / MAT_ROUGH / MAT_CLEARCOAT / MAT_CC_ROUGH 로 덮어쓴다.
       SMOOTH=1|0 (기본 1) — 스킨에 프레임별 부드러운 정점 노멀을 넣는다.
         0 이면 노멀을 안 쓴다 = 삼각형 단위 플랫 셰이딩(2026-08-11 이전 렌더의 룩).
         플랫이면 피부에 삼각형 줄무늬가 보인다 — 특히 광택 재질에서 심하다.
       SKY_INTENSITY / KEY_INTENSITY (기본 500 / 3200)
         ★ 기본값은 살색 스킨이 흰색으로 클리핑될 만큼 밝다. 클리핑되면 재질을 바꿔도
           확산색만 보이고 광택 차이가 안 나타난다. 재질을 보여주려면 낮출 것
           (예: SKY_INTENSITY=200 KEY_INTENSITY=1100).
"""

import sys
import datetime as _dt

from isaacsim import SimulationApp

SUIT = sys.argv[1] if len(sys.argv) > 1 else "hs"
MODE = sys.argv[2] if len(sys.argv) > 2 else "train"
OUT = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/exosuit_{SUIT}_{MODE}.mp4"
NFR = int(sys.argv[4]) if len(sys.argv) > 4 else 240
assert MODE in ("train", "eval"), MODE

sim = SimulationApp({"headless": True})

# 고품질 렌더: RTX 패스트레이싱 + 다중 샘플 (render_seamless_isaacsim.py 와 동일)
import carb  # noqa: E402

_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing")
_s.set("/rtx/pathtracing/spp", 32)
_s.set("/rtx/pathtracing/totalSpp", 128)
_s.set("/rtx/pathtracing/maxBounces", 6)
_s.set("/rtx/post/aa/op", 3)
_s.set("/rtx/post/dlss/execMode", 2)

import glob  # noqa: E402
import os  # noqa: E402
import struct  # noqa: E402
import tempfile  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

# [ETRI 2026-08-25] 저장소 통합으로 /home/user/ProtoMotions 는 제거됐다.
#   코드=ProtoMotions_etri (data/smpl), 실험 산출물=PM_Tasks (tasks_for_smpl).
CODE_ROOT = os.environ.get("PM_CODE_ROOT", "/home/user/ProtoMotions_etri")
ROOT = os.environ.get("PM_TASKS_ROOT", "/home/user/PM_Tasks")
TASK = f"{ROOT}/tasks_for_smpl"
sys.path.insert(0, f"{TASK}/script")
from etri_smpl_model_path import smpl_npz  # noqa: E402
import exosuit_spec  # noqa: E402

SPEC = exosuit_spec.get(SUIT)
_P = exosuit_spec.paths(SUIT)
# MJCF 를 직접 지정하면 슈트가 아닌 모델도 그린다(맨몸 smpl_humanoid.xml 등).
#   train 모드는 MJCF primitive 만 쓰므로 그대로 동작한다. eval 모드는 슈트 CAD 전용.
MJCF = os.environ.get("MJCF", str(_P["train"]))
MESH_DIR = str(_P["mesh_abs"])
# 5번째 인자로 추론 물리 롤아웃(.motion)을 넘길 수 있다. 없으면 기존 레퍼런스.
#   (2026-08-11 추가 — rollout_isaaclab_headless.py 산출물을 그리기 위해)
MOTION = sys.argv[5] if len(sys.argv) > 5 else \
    f"{TASK}/mimic_smpl/motions/test-walk-7/walk_cmu_07_04_30s_aligned.motion"
# SMPL npz 위치는 저장소마다 다르다(학습 fork: data/, 릴리즈: etri_라이선스동의_다운로드/).
# 해석은 etri_smpl_model_path 에 모아 두었다 — 스크립트를 그대로 옮겨도 동작한다.
NPZ = smpl_npz()
sys.path.insert(0, f"{CODE_ROOT}/data/smpl")
from smpl_joint_names import SMPL_BONE_ORDER_NAMES as BONE  # noqa: E402
from smpl_joint_names import SMPL_MUJOCO_NAMES as MUJ       # noqa: E402

# 카메라: turntable = 360° 회전(기본, 전방위 확인용) / side = 측면 추적
CAM_MODE = os.environ.get("CAM", "turntable")
CAM_R = float(os.environ.get("CAM_R", "3.2"))
# 다리 CAD 회전 (2026-08-11 기본값 변경)
#   기본 0 = 허벅지 body 전체 회전. USD(추론 뷰포트)·Newton(eval_fk) 과 같은 방식이다.
#   1  = 구 동작(힙 굴곡 y 만). 모터 반구는 맞물리지만 스트럿·커프가 실제 허벅지에서
#        평균 3.9~5.2°(최대 11°, 커프 위치로 ~67mm) 벗어나 **측면·큰 보폭에서 벌어져 보인다.**
LEG_HIP_FLEX_ONLY = os.environ.get("LEG_HIP_FLEX_ONLY") == "1"
# 진단/보정용 프레임 오프셋: 슈트만 t+SUIT_DT 프레임으로 그린다 (살은 t 그대로).
SUIT_DT = int(os.environ.get("SUIT_DT", "0"))
THIGH_SLIM = float(os.environ.get("THIGH_SLIM", "0.85"))
TORSO_SLIM = float(os.environ.get("TORSO_SLIM", "0.85"))
SKIN_RGB = (0.85, 0.68, 0.55)
SKIN_RGB_MJCF = (0.8, 0.6, 0.4)      # MJCF 인체 geom rgba — train 모드에서 인체 판별용
EXO_RGB = (0.35, 0.38, 0.42)

# ── 보조력 색상 표시 (2026-09-01) ────────────────────────────────────────────
# 왜 필요한가: S1(모터 끔)과 S2(모터 켬) 영상이 겉보기로 구분되지 않는다. 슈트가
#   "지금 얼마나 돕고 있는지"를 색으로 보이면 한 장면만 봐도 갈린다.
# 입력: EXO_TORQUE_LOG 사이드카(.pt) — FrozenHumanExoEnv 가 추론 중에 남긴다.
#   {"tau_exo": [T, M], "motor_peak": [M], "motor_names": [...]}
#   지정하지 않으면 기존 회색 단색 그대로다(S1 영상은 그대로 두면 된다).
# 색 단계: |τ|/정격 비율로 회색→노랑→주황→빨강. 구간 경계에서 튀지 않게 선형 보간한다.
EXO_TORQUE_LOG = os.environ.get("EXO_TORQUE_LOG", "")
# 색 표시 — **절대 토크 |τ| 를 모터 정격으로 3등분**한 지점을 앵커로 잡고
#   그 사이를 **선형 보간**한다(사용자 지정, 2026-09-01).
#   정격 23.7 N·m 기준 앵커:  0 회색 → 7.9 노랑 → 15.8 주황 → 23.7 빨강
#   보간이라 힘이 오르내리는 과정이 색으로 매끄럽게 이어진다. 이산 구간이 필요하면
#   EXO_COLOR_MODE=band 로 바꾼다(경계가 뚜렷해 "지금 몇 단계"를 읽기 쉽다).
_NO_TORQUE_NM = 0.5          # 이 미만은 "보조 없음"(회색). 모터가 사실상 쉬는 상태
_ANCHOR_RGB = [
    EXO_RGB,                 # 0/3 — 기본 회색
    (0.95, 0.85, 0.25),      # 1/3 — 노랑
    (0.95, 0.55, 0.15),      # 2/3 — 주황
    (0.90, 0.15, 0.12),      # 3/3 — 빨강 (정격)
]
EXO_COLOR_MODE = os.environ.get("EXO_COLOR_MODE", "ramp")   # ramp(기본) | band
# 모터 → 색칠할 CAD 메시. 좌우 허벅지 스트럿이 각 힙 모터에 대응한다.
_MOTOR_MESH = {"exo_hip_l": "exo_thigh_l", "exo_hip_r": "exo_thigh_r"}


def force_color(tau_abs, peak):
    """|τ|[N·m] 와 정격[N·m] → RGB. 정격 3등분 지점을 앵커로 선형 보간."""
    if tau_abs < _NO_TORQUE_NM:
        return EXO_RGB                       # 무보조 = 기본 회색
    x = max(0.0, min(1.0, tau_abs / peak)) * 3.0     # 0..3 (앵커 인덱스 공간)
    if EXO_COLOR_MODE == "band":             # 이산 구간
        return _ANCHOR_RGB[min(3, int(x) + 1)]
    i = min(2, int(x))                       # 보간 구간 [i, i+1]
    f = x - i
    ca, cb = _ANCHOR_RGB[i], _ANCHOR_RGB[i + 1]
    return tuple(ca[j] + (cb[j] - ca[j]) * f for j in range(3))


def load_exo_torque(path, nframes):
    """사이드카를 읽어 프레임별 모터 비율 [T, M] 과 이름을 돌려준다. 없으면 None."""
    if not path or not os.path.exists(path):
        return None, None
    d = torch.load(path, map_location="cpu", weights_only=False)
    tau = d["tau_exo"].abs().float()                  # [T, M] |τ| N·m
    peak = d["motor_peak"].float().clamp(min=1e-6)    # [M] 정격 N·m
    names = list(d.get("motor_names", []))
    if len(tau) < nframes:     # 롤아웃보다 짧으면 마지막 값으로 채운다
        tau = torch.cat([tau, tau[-1:].repeat(nframes - len(tau), 1)], 0)
    t3 = float(peak[0]) / 3.0
    n_gray = int((tau < _NO_TORQUE_NM).sum())
    n_y = int(((tau >= _NO_TORQUE_NM) & (tau < t3)).sum())
    n_o = int(((tau >= t3) & (tau < 2 * t3)).sum())
    n_r = int((tau >= 2 * t3).sum())
    print(f"[보조색] {path} — {tuple(tau.shape)} 모터 {names}  정격 {float(peak[0]):.1f} N·m")
    print(f"[보조색] 구간 경계 {t3:.1f} / {2*t3:.1f} N·m  →  "
          f"회색 {n_gray} · 노랑 {n_y} · 주황 {n_o} · 빨강 {n_r} (모터-프레임)")
    return tau, (names, peak)
#: eval 렌더에서 primitive 로 그리는 슈트 파트 (CAD STL 이 없다)
# [ETRI 2026-08-26] 힙링 제거 — 허벅지 CAD 스트럿(left/right_leg.stl)과 겹쳐 보여
#   사용자 판단으로 감추기로 했다. 가방만 primitive 로 남긴다(CAD 없음).
KEEP_PRIMS = ("exo_main_col",)
# 스킨 재질: flat = 재질 없이 displayColor(기존 룩) / 그 외는 UsdPreviewSurface 프리셋
#   skin_rubber ★릴리즈 룩 — 갈색 무광 고무. 클리어코트 0 이라 광택층이 없다.
#               (2026-08-28 프리셋으로 승격. 그전까지는 rubber + MAT_RGB/ROUGH/CLEARCOAT
#                오버라이드 3개를 매번 붙여 썼고, 하나라도 빠지면 검정 고무가 됐다.)
#   rubber      검정 고무 — 거친 확산 + 얕은 클리어코트(넓고 부드러운 광택)
#   porcelain   무광 도자기 — 높은 알베도 + 중간 거칠기, 클리어코트 없음(유약 광택 없음)
SKIN_MAT = os.environ.get("SKIN_MAT", "flat")
MAT_PRESETS = {
    "skin_rubber": {"rgb": (0.70, 0.50, 0.40), "rough": 0.75, "cc": 0.00, "ccr": 0.30},
    "rubber":      {"rgb": (0.12, 0.12, 0.13), "rough": 0.45, "cc": 0.35, "ccr": 0.25},
    "porcelain":   {"rgb": (0.92, 0.90, 0.87), "rough": 0.42, "cc": 0.00, "ccr": 0.30},
}
SMOOTH = os.environ.get("SMOOTH", "1") != "0"

# CAD → MJCF 는 exosuit_spec 에서 (CAD 없는 슈트면 eval 은 스킨만 그린다)
R_CAD2MJCF = exosuit_spec.cad_rot_np(SUIT)
CAD_ANCHOR_MM = np.array(SPEC["cad_anchor_mm"]) if SPEC["cad_anchor_mm"] else None
PELVIS_ANCHOR = np.array(SPEC["pelvis_anchor"])
EXO_MESHES = [(n, fn, b) for n, fn, b, _flex in SPEC["cad_meshes"]]
exo_color_attr = {}          # 메시명 → displayColor 속성 (보조색 갱신용)
EXO_TAU, _exo_meta = load_exo_torque(EXO_TORQUE_LOG, NFR)
EXO_NAMES, EXO_PEAK = _exo_meta if _exo_meta else ([], None)


# ── 공통 유틸 ────────────────────────────────────────────────────────
def quats_to_mats(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    M = np.empty(q.shape[:-1] + (3, 3), float)
    M[..., 0, 0] = 1 - 2 * (y * y + z * z); M[..., 0, 1] = 2 * (x * y - z * w); M[..., 0, 2] = 2 * (x * z + y * w)
    M[..., 1, 0] = 2 * (x * y + z * w); M[..., 1, 1] = 1 - 2 * (x * x + z * z); M[..., 1, 2] = 2 * (y * z - x * w)
    M[..., 2, 0] = 2 * (x * z - y * w); M[..., 2, 1] = 2 * (y * z + x * w); M[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return M


def wxyz_to_mat(q):
    w, x, y, z = q
    return quats_to_mats(np.array([x, y, z, w]))


def mat4(R, t):
    M = Gf.Matrix4d(1.0)
    for r in range(3):
        for c in range(3):
            M[r, c] = float(R[c, r])          # USD 는 행벡터 규약
    for c in range(3):
        M[3, c] = float(t[c])
    return M


_skin_mat_cache = {}


def bind_surface(prim, name):
    """UsdPreviewSurface 유전체 재질(metallic 0, ior 1.5)을 만들어 바인딩한다.

    재질이 없으면 RTX 가 displayColor 를 확산색으로 쓰는 무광 룩이 된다(기존 flat).
    바인딩하면 displayColor 는 무시되므로 색도 이 재질이 지배한다.
    ★ 광택 차이는 표면이 클리핑되지 않을 때만 보인다 — SKY/KEY_INTENSITY 참고.
    """
    if name in _skin_mat_cache:                    # train 모드는 geom 마다 호출된다
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(_skin_mat_cache[name])
        return
    p = MAT_PRESETS[name]
    rgb = tuple(float(v) for v in os.environ["MAT_RGB"].split(",")) if "MAT_RGB" in os.environ else p["rgb"]
    rough = float(os.environ.get("MAT_ROUGH", p["rough"]))
    cc = float(os.environ.get("MAT_CLEARCOAT", p["cc"]))
    ccr = float(os.environ.get("MAT_CC_ROUGH", p["ccr"]))
    mat = UsdShade.Material.Define(stage, "/World/Looks/Skin")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/Skin/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(cc)
    sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(ccr)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    _skin_mat_cache[name] = mat
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(mat)
    print(f"[인체] 재질 = {name}  rgb={rgb} roughness={rough} clearcoat={cc}/{ccr}")


def vertex_normals(pts, faces):
    """면적 가중 정점 노멀. 노멀을 안 넣으면 RTX 가 삼각형 단위로 플랫 셰이딩해
    피부에 6890정점 메시의 삼각형 줄무늬가 그대로 드러난다."""
    fn = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])
    vn = np.zeros_like(pts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def prim_from_mjcf_geom(path, g, rgb):
    """MJCF geom(box / fromto 원통·캡슐) 하나를 USD prim 으로 만든다.

    CAD STL 이 없는 슈트 파트(가방 박스, 허벅지 원통)를 eval 렌더에 넣기 위한 것.
    Newton LBS 렌더러의 `SPEC["eval_hidden"]` 정책과 같다 — STL 이 대신 보여주는
    파트만 숨기고, 대체물이 없는 파트는 primitive 를 그대로 보여준다.
    """
    gt = g.get("type", "sphere")
    size = [float(v) for v in (g.get("size") or "0.05").split()]
    pos = np.array([float(v) for v in (g.get("pos") or "0 0 0").split()])
    Rl = wxyz_to_mat([float(v) for v in (g.get("quat") or "1 0 0 0").split()])
    if g.get("fromto"):
        ft = np.array([float(v) for v in g.get("fromto").split()]).reshape(2, 3)
        pos = ft.mean(0); ax = ft[1] - ft[0]; h = float(np.linalg.norm(ax))
        z = ax / (h + 1e-12)
        a = np.array([0, 0, 1.0])
        v = np.cross(a, z); c = float(np.dot(a, z))
        if np.linalg.norm(v) < 1e-9:
            Rl = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            Rl = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
        cls = UsdGeom.Cylinder if gt == "cylinder" else UsdGeom.Capsule
        prim = cls.Define(stage, path)
        prim.CreateRadiusAttr(size[0]); prim.CreateHeightAttr(h); prim.CreateAxisAttr("Z")
    elif gt == "box":
        prim = UsdGeom.Cube.Define(stage, path)
        prim.CreateSizeAttr(2.0)
        Rl = Rl @ np.diag([float(v) for v in size[:3]])     # half-extent 를 변환에 곱한다
    else:
        return None
    UsdGeom.Xformable(prim).AddTransformOp(
        UsdGeom.XformOp.PrecisionDouble, "local").Set(mat4(Rl, pos))
    prim.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    return prim


def load_stl(path):
    d = open(path, "rb").read()
    n = struct.unpack("<I", d[80:84])[0]
    V = np.frombuffer(bytearray(b"".join(d[84 + i * 50 + 12: 84 + i * 50 + 48] for i in range(n))),
                      dtype="<f4").reshape(-1, 3).astype(float)
    return V


# ── 모션 ─────────────────────────────────────────────────────────────
mot = torch.load(MOTION, weights_only=False)
rbr = mot["rigid_body_rot"].numpy(); rbp = mot["rigid_body_pos"].numpy()
dof = mot["dof_pos"].numpy()
muj = [MUJ.index(BONE[j]) for j in range(24)]
gq = rbr[:, muj, :]                      # (T,24,4) xyzw, BONE 순서
gp = rbp[:, muj, :].copy()
gp[:, :, 0] -= gp[0, 0, 0]; gp[:, :, 1] -= gp[0, 0, 1]      # env origin 제거
T = len(gp)
print(f"[모션] frames={T}  mode={MODE}")

# ── 진행방향(heading) 사전계산 ────────────────────────────────────────────
# [ETRI 2026-08-28] 기존 CAM 모드(turntable/front/side)는 전부 **월드 고정** 오프셋이라
#   "+x 방향으로 걷는다"를 암묵 가정한다. loop 모션마다 월드 상 진행방향이 달라서
#   같은 CAM 설정인데 모션마다 정면/측면이 달라 보였다.
#   → *_rel 모드는 루트 궤적에서 진행방향을 구해 그 기준으로 카메라를 놓는다.
_v = np.gradient(gp[:, 0, :2], axis=0)
_sp = np.linalg.norm(_v, axis=1)
_hd = np.zeros(T)
_last = float(np.arctan2(_v[0, 1], _v[0, 0])) if _sp[0] > 1e-6 else 0.0
for _i in range(T):
    if _sp[_i] > 1e-3:                      # 거의 멈춘 프레임은 방향이 튀므로 직전 값 유지
        _last = float(np.arctan2(_v[_i, 1], _v[_i, 0]))
    _hd[_i] = _last
_hd = np.unwrap(_hd)                        # ±π 경계에서 카메라가 튀지 않게
_win = max(3, int(os.environ.get("CAM_SMOOTH", "45")))   # 1.5초 이동평균 = 카메라 흔들림 억제
_k = np.ones(_win) / _win
HEADING = np.convolve(np.pad(_hd, (_win, _win), mode="edge"), _k, mode="same")[_win:-_win]
print(f"[카메라] heading 사전계산 완료 (평활 {_win}f, 총회전 {np.degrees(HEADING[-1]-HEADING[0]):+.0f}도)")

# MJCF hinge 순서 = dof 순서 → 힙 굴곡 인덱스
mroot = ET.parse(MJCF).getroot()
hinge = [j.get("name") for j in mroot.iter("joint") if j.get("type") == "hinge"]
IDOF = {"L_Hip": hinge.index("L_Hip_y"), "R_Hip": hinge.index("R_Hip_y")}

# ── 스테이지 ─────────────────────────────────────────────────────────
ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

body_ops = {}          # body name → transform op (프레임마다 갱신)
mesh_ops = {}          # exo mesh name → (transform op, body)
pts_attr = None


def body_xform(name):
    if name not in body_ops:
        x = UsdGeom.Xform.Define(stage, f"/World/Model/{name}")
        body_ops[name] = UsdGeom.Xformable(x).AddTransformOp()
    return body_ops[name]


if MODE == "train":
    # ── MJCF primitive → USD prim ────────────────────────────────────
    bodies = {}

    def walk(el):
        for b in el.findall("body"):
            bodies[b.get("name")] = b
            walk(b)

    walk(mroot.find("worldbody"))
    n_geom = 0
    for bname, bel in bodies.items():
        body_xform(bname)
        for gi, g in enumerate(bel.findall("geom")):
            gt = g.get("type", "sphere")
            rgba = [float(v) for v in (g.get("rgba") or "0.8 0.6 0.4 1").split()]
            path = f"/World/Model/{bname}/g{gi}"
            size = [float(v) for v in (g.get("size") or "0.05").split()]
            pos = np.array([float(v) for v in (g.get("pos") or "0 0 0").split()])
            quat = [float(v) for v in (g.get("quat") or "1 0 0 0").split()]
            Rl = wxyz_to_mat(quat)
            if g.get("fromto"):                     # capsule fromto
                ft = np.array([float(v) for v in g.get("fromto").split()]).reshape(2, 3)
                pos = ft.mean(0); ax = ft[1] - ft[0]; h = float(np.linalg.norm(ax))
                z = ax / (h + 1e-12)
                a = np.array([0, 0, 1.0])
                v = np.cross(a, z); c = float(np.dot(a, z))
                if np.linalg.norm(v) < 1e-9:
                    Rl = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
                else:
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    Rl = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
                prim = UsdGeom.Capsule.Define(stage, path)
                prim.CreateRadiusAttr(size[0]); prim.CreateHeightAttr(h); prim.CreateAxisAttr("Z")
            elif gt == "capsule":
                prim = UsdGeom.Capsule.Define(stage, path)
                prim.CreateRadiusAttr(size[0]); prim.CreateHeightAttr(2 * size[1]); prim.CreateAxisAttr("Z")
            elif gt == "sphere":
                prim = UsdGeom.Sphere.Define(stage, path)
                prim.CreateRadiusAttr(size[0])
            elif gt == "box":
                # Cube(size=2, 즉 ±1) 에 half-extent 를 **변환행렬에 직접 곱한다**.
                # 별도 ScaleOp 를 쓰면 xformOpOrder 합성 순서에 의존하게 되므로 피한다.
                prim = UsdGeom.Cube.Define(stage, path)
                prim.CreateSizeAttr(2.0)
                Rl = Rl @ np.diag([float(v) for v in size[:3]])
            else:
                continue
            UsdGeom.Xformable(prim).AddTransformOp(
                UsdGeom.XformOp.PrecisionDouble, "local").Set(mat4(Rl, pos))
            prim.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in rgba[:3]])])
            # 인체색(MJCF rgba 0.8 0.6 0.4) geom 만 재질 교체 — 슈트 색은 유지한다.
            if SKIN_MAT in MAT_PRESETS and max(abs(a - b) for a, b in zip(rgba[:3], SKIN_RGB_MJCF)) <= 0.06:
                bind_surface(prim.GetPrim(), SKIN_MAT)
            n_geom += 1
    print(f"[train] body {len(bodies)}개 / geom {n_geom}개 생성")

else:
    # ── LBS 스킨 (rest + 웨이트를 npz 에서 직접) ──────────────────────
    d = np.load(NPZ, allow_pickle=True)
    v0 = np.asarray(d["v_template"], float)
    W = np.asarray(d["weights"], float)
    faces = np.asarray(d["f"], int)
    J = np.asarray(d["J_regressor"], float) @ v0
    C = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)      # Y-up → Z-up
    off = np.einsum("ij,bvj->bvi", C, (v0[None] - J[:, None]))   # (24,N,3)
    for grp, fac, specs in (("허벅지", THIGH_SLIM, (("L_Hip", "L_Knee"), ("R_Hip", "R_Knee"))),
                            ("상체", TORSO_SLIM, (("Pelvis", "Torso"), ("Torso", "Spine"),
                                                  ("Spine", "Chest"), ("Chest", "Neck")))):
        if abs(fac - 1.0) <= 1e-6:
            continue
        for nm, ch in specs:
            b, kc = BONE.index(nm), BONE.index(ch)
            ax = C @ (J[kc] - J[b]); ax /= np.linalg.norm(ax)
            o = off[b]; t = o @ ax; axl = np.outer(t, ax)
            off[b] = axl + fac * (o - axl)
        print(f"[eval] {grp} 살 반경 ×{fac}")
    Wt = W.T
    mesh = UsdGeom.Mesh.Define(stage, "/World/Skin")
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in (C @ v0.T).T])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(x) for x in faces.reshape(-1)])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*SKIN_RGB)])
    if SKIN_MAT in MAT_PRESETS:
        bind_surface(mesh.GetPrim(), SKIN_MAT)
    elif SKIN_MAT != "flat":
        raise SystemExit(f"SKIN_MAT 는 flat|{'|'.join(MAT_PRESETS)} 중 하나여야 한다: {SKIN_MAT}")
    pts_attr = mesh.GetPointsAttr()
    if SMOOTH:
        mesh.CreateNormalsAttr([])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        nrm_attr = mesh.GetNormalsAttr()
        print("[eval] 스킨 노멀 = 부드러운 정점 노멀(프레임마다 재계산)")

    # ── 외골격 CAD mesh ──────────────────────────────────────────────
    hip_local = {}
    for bn in ("L_Hip", "R_Hip"):
        el = next(e for e in mroot.iter("body") if e.get("name") == bn)
        hip_local[bn] = np.array([float(v) for v in el.get("pos").split()])
    t_pelvis = PELVIS_ANCHOR - R_CAD2MJCF @ (CAD_ANCHOR_MM * 0.001)
    exo_verts = {}
    for name, fn, bodyname in EXO_MESHES:
        V = (R_CAD2MJCF @ (load_stl(f"{MESH_DIR}/{fn}") * 0.001).T).T + t_pelvis
        if bodyname != "Pelvis":
            V = V - hip_local[bodyname]          # body 로컬
        m = UsdGeom.Mesh.Define(stage, f"/World/{name}")
        m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in V])
        m.CreateFaceVertexCountsAttr([3] * (len(V) // 3))
        m.CreateFaceVertexIndicesAttr(list(range(len(V))))
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateDisplayColorAttr([Gf.Vec3f(*EXO_RGB)])
        exo_color_attr[name] = m.GetDisplayColorAttr()   # 보조색 갱신용 핸들
        mesh_ops[name] = (UsdGeom.Xformable(m).AddTransformOp(), bodyname)
        exo_verts[name] = V

    # ── CAD STL 이 없는 슈트 파트: 가방 박스 + 허벅지 원통 ────────────────
    # 치수는 eval XML 을 따른다(원통이 capsule→cylinder 로 바뀌고 반경도 다르다).
    # [ETRI 2026-08-26] 기본 경로를 `_for_eval_lbskin.xml` 로 바꿨다.
    #   구 `_for_eval.xml` 은 `_for_eval_{skeleton,lbskin}` 분리 이전 잔존물이라
    #   현행 생성기(make_exosuit_for_eval.py)가 갱신하지 않는다. 2026-08-25 에
    #   미사용으로 판단해 _deprecated_20260825/ 로 옮겼더니, 이 경로가 조용히
    #   실패하며 **가방이 통째로 렌더에서 사라졌다**(로그: "eval XML 없음").
    #   lbskin 판은 매번 재생성되므로 가방 위치 수정 등이 자동 반영된다.
    _cands = [_P["dir"] / f"smpl_humanoid_{SPEC['label']}_for_eval_lbskin.xml",
              _P["dir"] / f"smpl_humanoid_{SPEC['label']}_for_eval.xml"]
    _eval_xml = os.environ.get("EVAL_XML") or str(next((c for c in _cands if c.exists()), _cands[0]))
    keep_bodies = set()
    if os.path.exists(_eval_xml):
        eroot = ET.parse(_eval_xml).getroot()
        for b in eroot.iter("body"):
            for g in b.findall("geom"):
                if g.get("name") in KEEP_PRIMS:
                    bn = b.get("name")
                    body_xform(bn)                      # 프레임마다 갱신되는 부모 Xform
                    prim_from_mjcf_geom(f"/World/Model/{bn}/{g.get('name')}", g, EXO_RGB)
                    keep_bodies.add(bn)
        print(f"[eval] 슈트 primitive {len(keep_bodies)}개 body 에 {', '.join(sorted(KEEP_PRIMS))}")
    else:
        print(f"[eval] ⚠️  eval XML 없음 — 가방이 렌더에서 빠진다: {_eval_xml}")
        print("[eval] ⚠️  EVAL_XML 환경변수로 경로를 지정하거나 자산을 재생성할 것")
    print(f"[eval] 스킨 {len(v0)}정점 + CAD mesh {len(EXO_MESHES)}개")


# ── 바닥 / 조명 / 카메라 (render_seamless_isaacsim.py 와 동일 룩) ─────
# 바닥 범위는 모션이 지나가는 xy 에서 잡는다 — 900스텝(30초) 보행이면 예전 고정 범위
# (x −4..34)를 넘어가 타일이 끊긴다.
_MG = 12
X0 = min(-4, int(np.floor(gp[:, 0, 0].min())) - _MG); X1 = max(34, int(np.ceil(gp[:, 0, 0].max())) + _MG)
Y0 = min(-8, int(np.floor(gp[:, 0, 1].min())) - _MG); Y1 = max(10, int(np.ceil(gp[:, 0, 1].max())) + _MG)
FLOOR_A = float(os.environ.get("FLOOR_A", "0.30"))     # 밝은 타일 / 어두운 타일 반사율
FLOOR_B = float(os.environ.get("FLOOR_B", "0.24"))
pts, cnts, idx, col = [], [], [], []
vi = 0
for ix in range(X0, X1):
    for iy in range(Y0, Y1):
        pts += [(ix, iy, 0), (ix + 1, iy, 0), (ix + 1, iy + 1, 0), (ix, iy + 1, 0)]
        cnts.append(4); idx += [vi, vi + 1, vi + 2, vi + 3]; vi += 4
        shade = FLOOR_A if (ix + iy) % 2 == 0 else FLOOR_B
        col.append((shade, shade, shade + 0.015))
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
ground.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
ground.CreateFaceVertexCountsAttr(cnts); ground.CreateFaceVertexIndicesAttr(idx)
ground.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set([Gf.Vec3f(*c) for c in col])

dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome.CreateIntensityAttr(float(os.environ.get("SKY_INTENSITY", "500")))
dome.CreateColorAttr(Gf.Vec3f(0.50, 0.68, 0.93))
key = UsdLux.DistantLight.Define(stage, "/World/Key")
key.CreateIntensityAttr(float(os.environ.get("KEY_INTENSITY", "3200")))
key.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.90))
# DistantLight 는 로컬 −Z 로 빛을 보낸다. 기본 (-42,0,35) 의 진행 방향은
# (+0.38,−0.55,−0.74) 로 **카메라(+x)에서 멀어지는** 쪽이라, CAM=front 에서는
# 보이는 면이 전부 음영이 된다. Z 를 +180 하면(=215) 카메라 쪽에서 비춘다.
UsdGeom.Xformable(key).AddRotateXYZOp().Set(
    tuple(float(v) for v in os.environ.get("KEY_ROT", "-42,0,35").split(","))
)

cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateFocalLengthAttr(22.0); cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
cam_op = UsdGeom.Xformable(cam).AddTransformOp()


def lookat_mat(eye, tgt):
    eye = np.asarray(eye, float); tgt = np.asarray(tgt, float)
    fwd = tgt - eye; fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, [0, 0, 1.0]); right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, fwd)
    M = Gf.Matrix4d(1.0)
    for r, v in enumerate([right, up, -fwd, eye]):
        for c in range(3):
            M[r, c] = float(v[c])
    return M


OUTDIR = tempfile.mkdtemp(prefix=f"exosuit_{SUIT}_{MODE}_")
_RW, _RH = (1080, 1920) if os.environ.get("PORTRAIT") == "1" else (1920, 1080)  # [ETRI] 세로모드
render_prod = rep.create.render_product("/World/Cam", (_RW, _RH))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=OUTDIR, rgb=True)
writer.attach([render_prod])
for _ in range(20):
    sim.update()

# ── 프레임 루프 ──────────────────────────────────────────────────────
BIDX = {BONE[j]: j for j in range(24)}
for i in range(NFR):
    t = min(i, T - 1)
    # 진단용: 슈트(CAD + 가방·원통 primitive)만 프레임 인덱스를 옮긴다.
    # 살은 mesh points, 슈트는 xform op 으로 갱신되므로 렌더러 전파 시점이
    # 다를 수 있다. SUIT_DT 로 어긋남의 부호·크기를 실측한다.
    ts = int(np.clip(t + SUIT_DT, 0, T - 1))
    Rall = quats_to_mats(gq[t])                    # (24,3,3)  살
    Rsuit = Rall if ts == t else quats_to_mats(gq[ts])
    if MODE == "train":
        for bname, op in body_ops.items():
            j = BIDX[bname]
            op.Set(mat4(Rall[j], gp[t, j]))
    else:
        rotated = np.einsum("bij,bvj->bvi", Rall, off) + gp[t][:, None, :]
        pv = np.einsum("bv,bvi->vi", Wt, rotated)
        pts_attr.Set([Gf.Vec3f(*map(float, p)) for p in pv])
        if SMOOTH:
            nrm_attr.Set([Gf.Vec3f(*map(float, n)) for n in vertex_normals(pv, faces)])
        for bn in keep_bodies:                  # 가방·허벅지 원통은 body 자체를 따라간다
            j = BIDX[bn]
            body_ops[bn].Set(mat4(Rsuit[j], gp[ts, j]))
        Rp, Tp = Rsuit[BIDX["Pelvis"]], gp[ts, BIDX["Pelvis"]]
        for name, (op, bodyname) in mesh_ops.items():
            if bodyname == "Pelvis":
                op.Set(mat4(Rp, Tp))
            elif LEG_HIP_FLEX_ONLY:                 # 힙 굴곡(y)만 반영 (구 동작)
                th = float(dof[ts, IDOF[bodyname]]); c, s = np.cos(th), np.sin(th)
                Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                op.Set(mat4(Rp @ Ry, Tp + Rp @ hip_local[bodyname]))
            elif os.environ.get("EXO_LEG_MODE", "femur_flex") == "femur_flex":
                # [ETRI 2026-08-27] 대퇴골구(힙 관절) 기준 강체 고정.
                #   스트럿을 대퇴골구 글로벌 위치 gp[hip] 에 앵커하고 힙 굴곡(y)만 반영 →
                #   thigh 3-DOF 전체 회전이 스트럿을 비틀어 어긋나 보이던 문제 방지.
                th = float(dof[ts, IDOF[bodyname]]); c, s2 = np.cos(th), np.sin(th)
                Ry = np.array([[c, 0, s2], [0, 1, 0], [-s2, 0, c]])
                op.Set(mat4(Rp @ Ry, gp[ts, BIDX[bodyname]]))
            else:                                   # 허벅지 body 전체 회전 (구 기본, 2026-08-11)
                j = BIDX[bodyname]
                op.Set(mat4(Rsuit[j], gp[ts, j]))
        # ── 보조력 색상 (EXO_TORQUE_LOG 지정 시에만) ──────────────────────
        #   좌우 허벅지 스트럿을 각 힙 모터의 |τ|/정격 으로 칠한다.
        #   프레임 인덱스는 롤아웃과 1:1 — 사이드카가 같은 순서로 기록됐다.
        if EXO_TAU is not None:
            fi = min(ts, len(EXO_TAU) - 1)
            for mi, mname in enumerate(EXO_NAMES):
                mesh = _MOTOR_MESH.get(mname)
                if mesh and mesh in exo_color_attr:
                    exo_color_attr[mesh].Set([Gf.Vec3f(*force_color(
                        float(EXO_TAU[fi, mi]), float(EXO_PEAK[mi])))])
    tgt = gp[t, 0].copy(); tgt[2] = 0.9
    if CAM_MODE == "turntable":
        a = 2.0 * np.pi * (i / max(NFR - 1, 1))          # 1회전 = 전체 길이
        eye = tgt + np.array([CAM_R * np.cos(a), CAM_R * np.sin(a), 0.55])
    elif CAM_MODE in ("front_rel", "back_rel", "side_rel"):
        # [ETRI 2026-08-28] 진행방향 기준 상대 카메라. 모션의 월드 방향과 무관하게
        #   항상 같은 앵글이 나온다(front_rel = 사람을 정면에서 바라봄).
        #   CAM_YAW(도) 로 미세조정 가능 — 0 이면 정확히 정면.
        _base = {"front_rel": 0.0, "back_rel": np.pi, "side_rel": -np.pi / 2}[CAM_MODE]
        _a = HEADING[t] + _base + np.radians(float(os.environ.get("CAM_YAW", "0")))
        eye = tgt + np.array([CAM_R * np.cos(_a), CAM_R * np.sin(_a), 0.20 * CAM_R])
    elif CAM_MODE == "front":
        # 진행 방향(+x) 앞에서 뒤를 본다 = 정면. IsaacLab 기본 뷰포트에 가까운 구도
        # (조금 높은 곳에서 내려다봐 바닥이 넓게 깔린다).
        # ※ 월드 고정이라 +x 로 걷는 모션에서만 정면이다 → 일반적으로는 front_rel 을 쓸 것.
        eye = tgt + np.array([CAM_R, -0.20 * CAM_R, 0.20 * CAM_R])
    else:
        eye = tgt + np.array([1.0, -3.6, 0.45])
    cam_op.Set(lookat_mat(eye, tgt))
    sim.update()
    rep.orchestrator.step(rt_subframes=48, delta_time=0.0, pause_timeline=True)
    if i % 30 == 0:
        print(f"  frame {i+1}/{NFR}")
rep.orchestrator.wait_until_complete()

# ── HUD + mp4 ────────────────────────────────────────────────────────
pngs = sorted(glob.glob(os.path.join(OUTDIR, "rgb", "*.png")) + glob.glob(os.path.join(OUTDIR, "*.png")))
print(f"캡처 PNG {len(pngs)}개")
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
except Exception:
    font = ImageFont.load_default()
label = (f"{SPEC['label']} for_train (primitive)" if MODE == "train"
         else f"{SPEC['label']} for_eval (LBS skin + exosuit CAD, thigh/torso x{THIGH_SLIM}/{TORSO_SLIM})")
if MODE == "eval" and SKIN_MAT != "flat":
    label += f"  [skin={SKIN_MAT}{'' if SMOOTH else ', flat-shaded'}]"
label = os.environ.get("LABEL", label)      # MJCF 를 바꿨으면 라벨도 직접 준다

# ── HUD 모션명 결정 ──────────────────────────────────────────────────
# [ETRI 2026-08-28] **HUD 에는 모션명을 쓴다. 날짜를 쓰지 않는다.**
#   깨졌던 이유: 2단계 워크플로우(추론 녹화 → 오프라인 렌더)에서 argv[5] 로 넘어오는
#   것은 원본 모션이 아니라 **롤아웃 파일**이고, 그 파일명은 타임스탬프다
#   (예: 2026-08-28-15-15-33.motion). basename 을 그대로 쓰면 날짜가 찍힌다.
#   → 우선순위: ① MOTION_NAME 환경변수(호출측이 명시)
#              ② 녹화 폴더명에서 앞의 일시 접두어를 떼어낸 것
#                 (recordings/YYYY-MM-DD-HH-MM-SS-<내용>/ 규칙)
#              ③ 파일명. 날짜꼴이면 경고를 찍는다(조용히 날짜가 나가는 것 방지).
def _resolve_hud_name():
    import re
    _date_re = re.compile(r"^\d{4}-\d{2}-\d{2}([-_]\d{2}){3}$")
    env = os.environ.get("MOTION_NAME", "").strip()
    if env:
        return env
    # 녹화 폴더: <out.mp4 의 부모> 가 recordings/<일시>-<내용>
    parent = os.path.basename(os.path.dirname(os.path.abspath(OUT)))
    m = re.match(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-(.+)$", parent)
    if m:
        return m.group(1)
    stem = os.path.splitext(os.path.basename(MOTION))[0]
    for _suf in ("_newton", "_rollout"):
        if stem.endswith(_suf):
            stem = stem[: -len(_suf)]
    if _date_re.match(stem):
        print(f"[경고] HUD 이름이 날짜꼴이다({stem}). MOTION_NAME 을 지정하거나 "
              f"녹화 폴더명을 'YYYY-MM-DD-HH-MM-SS-<내용>' 규칙으로 둘 것.")
    return stem

_HUD_NAME = _resolve_hud_name()
print(f"[HUD] 모션명 = {_HUD_NAME}")
frames = []
for i, p in enumerate(pngs):
    im = Image.fromarray(np.asarray(imageio.imread(p))[..., :3]).convert("RGB")
    dr = ImageDraw.Draw(im)
    _mot_name = _HUD_NAME
    # [ETRI 2026-08-27] "[1/1] <모션명> step: i/total", 날짜 없음, 좌상단(중앙 인물 안 가림)
    txt = f"[1/1] {_mot_name} step: {i+1}/{len(pngs)}"
    tb = dr.textbbox((0, 0), txt, font=font); tw, th = tb[2] - tb[0], tb[3] - tb[1]
    mx, my = 24, 20
    dr.rectangle([mx - 10, my - 6, mx + tw + 10, my + th + 12], fill=(40, 40, 40))
    dr.text((mx, my), txt, font=font, fill=(255, 255, 255))
    frames.append(np.asarray(im))
if frames:
    imageio.mimsave(OUT, frames, fps=30, quality=9, bitrate="12M", macro_block_size=1)
    print(f"완료! {OUT} ({len(frames)}프레임)")

    # ── 룩 각인 ─────────────────────────────────────────────────────────
    # 산출물 옆에 실제 적용된 룩을 남긴다. 스크립트 기본값이 나중에 바뀌어도
    # "이 영상이 어떤 룩이었는지"를 사후에 항상 특정할 수 있게 하기 위함이다.
    # (INFO.md §7 에 "이 영상은 인자를 특정하지 못했다"고 남은 사태 방지)
    try:
        _mat = MAT_PRESETS.get(SKIN_MAT, {})
        _look = {
            "rendered_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "script": os.path.basename(__file__),
            "argv": " ".join(sys.argv[1:]),
            "frames": len(frames),
            "SKIN_MAT": SKIN_MAT,
            "resolved_rgb": os.environ.get("MAT_RGB", ",".join(str(v) for v in _mat.get("rgb", ()))),
            "resolved_roughness": os.environ.get("MAT_ROUGH", _mat.get("rough", "")),
            "resolved_clearcoat": os.environ.get("MAT_CLEARCOAT", _mat.get("cc", "")),
            "resolved_cc_rough": os.environ.get("MAT_CC_ROUGH", _mat.get("ccr", "")),
            "SMOOTH": os.environ.get("SMOOTH", "1"),
            "SKY_INTENSITY": os.environ.get("SKY_INTENSITY", "(기본)"),
            "KEY_INTENSITY": os.environ.get("KEY_INTENSITY", "(기본)"),
            "KEY_ROT": os.environ.get("KEY_ROT", "(기본)"),
            "FLOOR_A": os.environ.get("FLOOR_A", "(기본)"),
            "FLOOR_B": os.environ.get("FLOOR_B", "(기본)"),
            "CAM": os.environ.get("CAM", "(기본)"),
            "CAM_R": os.environ.get("CAM_R", "(기본)"),
            "THIGH_SLIM": os.environ.get("THIGH_SLIM", "0.85"),
            "TORSO_SLIM": os.environ.get("TORSO_SLIM", "0.85"),
        }
        _stamp = os.path.join(os.path.dirname(os.path.abspath(OUT)) or ".", "RENDER_LOOK.txt")
        with open(_stamp, "w") as _f:
            _f.write(f"# 렌더 룩 각인 — {os.path.basename(OUT)}\n")
            for _k, _v in _look.items():
                _f.write(f"{_k}: {_v}\n")
            if SKIN_MAT == "flat":
                _f.write("\n⚠️ SKIN_MAT=flat 은 릴리즈 룩이 아니다"
                         "(릴리즈=skin_rubber, 갈색 무광 고무).\n"
                         "   render_look.env 를 source 하지 않고 렌더한 것으로 보인다.\n")
        print(f"룩 각인: {_stamp}")
    except Exception as _e:
        print(f"[경고] 룩 각인 실패(렌더 자체는 성공): {_e}")
else:
    print("ERROR: 캡처 PNG 없음")
sim.close()
