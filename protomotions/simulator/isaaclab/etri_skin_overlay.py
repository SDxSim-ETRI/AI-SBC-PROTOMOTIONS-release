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
"""IsaacLab **기본 씬 안에서** 무이음새 SMPL 살을 그린다 (매 프레임 LBS).

풀려는 문제
-----------
IsaacLab 추론 렌더는 물리 강체를 그대로 그리므로, 살을 보여주려면 body 마다 STL 을
붙여야 하고 어깨·엉덩이·무릎에서 **이음새가 갈라진다**(INFO.md §6-C).

무이음새 방법(UsdSkel, §6-D)은 **물리 강체에 바인딩되지 않는다.** 그래서 지금까지는
§7 의 2단계(추론 롤아웃 → 별도 렌더러)로만 무이음새를 얻었고, 그 렌더러가 자기 씬을
갖고 있어 **배경이 IsaacLab 기본 씬과 달랐다.**

이 모듈은 그 간극을 메운다 — IsaacLab 기본 씬을 그대로 두고, 스테이지에 **단일 SMPL
메시**를 하나 추가해 **매 렌더 프레임 body 자세로 LBS 를 계산해 정점을 갱신**한다.
Newton 렌더러(`log_mesh`)가 하는 일과 같은 수식이다:

    posed_v = Σ_b W[v,b] · ( R_b · C·(v_template[v] − J[b]) + T_b )
    C = 캐노니컬(Y-up) → Z-up 순환회전

로봇의 원래 시각 prim 은 숨긴다(콜라이더는 물리 그대로). 슈트 CAD 처럼 **살이 아닌**
파트는 그대로 보인다 — 이름으로 구분한다.

켜는 법
    PM_SKIN_OVERLAY=1  … 추론/학습 어느 쪽이든 렌더가 도는 경우에만 동작한다
    SKIN_BETAS="0,0,…" 체형(기본 neutral)
    SKIN_RGB / SKIN_ROUGHNESS / SKIN_SPECULAR   살 재질

★ 이 모듈은 **렌더 전용**이다. 물리·질량·접촉에 손대지 않는다.
"""

from __future__ import annotations

import os

import numpy as np

#: 살에 해당하는 시각 prim 을 가릴 때 **제외**할 이름 조각 (슈트는 계속 보인다)
SUIT_KEYWORDS = ("exo", "mesh_", "hip_ring", "thigh_box", "waist_hip", "motor", "backpack")


def _enabled() -> bool:
    return os.environ.get("PM_SKIN_OVERLAY") == "1"


def _rgb(env: str, default):
    v = os.environ.get(env)
    return tuple(float(x) for x in v.split(",")) if v else default


def _quat_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2], q_xyzw[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q_xyzw.shape[:-1], 3, 3)


class SkinOverlay:
    """스테이지에 단일 SMPL 메시를 얹고 매 프레임 LBS 로 갱신한다."""

    PRIM_PATH = "/World/EtriSkinOverlay"

    def __init__(self, stage, robot_root_path: str, body_names: list[str]):
        import sys
        from pathlib import Path
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        repo = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo / "tasks_for_smpl/script"))
        sys.path.insert(0, str(repo / "data/smpl"))
        from etri_smpl_model_path import smpl_npz
        from smpl_joint_names import SMPL_BONE_ORDER_NAMES as BONE

        betas = np.zeros(10)
        if os.environ.get("SKIN_BETAS"):
            vals = [float(x) for x in os.environ["SKIN_BETAS"].split(",")]
            betas[:len(vals)] = vals

        d = np.load(str(smpl_npz()), allow_pickle=True)
        v0 = np.asarray(d["v_template"], np.float64) \
            + np.asarray(d["shapedirs"], np.float64)[:, :, :10] @ betas
        W = np.asarray(d["weights"], np.float64)              # (6890,24) BONE 순서
        J = np.asarray(d["J_regressor"], np.float64) @ v0
        faces = np.asarray(d["f"], np.int64)
        C = _quat_to_mat(np.array([0.5, 0.5, 0.5, 0.5]))      # Y-up → Z-up
        self._off = np.einsum("ij,bvj->bvi", C, (v0[None] - J[:, None]))   # (24,6890,3)
        self._Wt = W.T                                        # (24,6890)

        # ★ 시뮬레이터의 body 순서 → SMPL BONE 순서 매핑.
        #   빼먹으면 스킨 조각이 흩어진다(INFO.md §2 경고).
        self._idx = np.array([body_names.index(BONE[j]) for j in range(24)])

        mesh = UsdGeom.Mesh.Define(stage, self.PRIM_PATH)
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in v0])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
        mesh.CreateSubdivisionSchemeAttr("none")
        self._points_attr = mesh.GetPointsAttr()
        self._Gf = Gf

        # ★ 살은 **스테이지에 실제로 기록된 body 변환**에서 계산한다.
        #   `robot.data.body_pos_w` 는 물리 스텝 직후 값이라, 화면에 그려지는 prim 이
        #   아직 이전 자세를 들고 있으면 한 프레임(≈3.7 cm @1.2 m/s)씩 어긋난다.
        #   실측(2026-08-07): 살이 도형보다 뒤로 밀려 보였다. 같은 소스를 쓰면 그 차이가
        #   원리적으로 사라진다.
        from pxr import Usd as _Usd
        self._Usd = _Usd
        self._body_prims = []
        for j in range(24):
            bp = stage.GetPrimAtPath(f"{robot_root_path}/bodies/{BONE[j]}")
            self._body_prims.append(bp if bp and bp.IsValid() else None)
        # ★ prim 변환은 쓰지 않는다. IsaacLab 은 물리를 **Fabric(USDRT)** 로만 동기화하고
        #   USD 스테이지의 body 변환은 자산의 정적 bind 자세 그대로다.
        #   실측(2026-08-07): 그 경로로 계산하면 살이 눌리고(z 0.294~1.218, 정상 0~1.67)
        #   프레임이 지나도 값이 변하지 않았다. → `robot.data.body_*_w` 를 쓴다.
        self._use_prims = False

        # 재질 — RTX 는 Material 이 displayColor 보다 우선이라 명시 바인딩해야 한다.
        skin = _rgb("SKIN_RGB", (0.85, 0.68, 0.55))
        mat = UsdShade.Material.Define(stage, self.PRIM_PATH + "/Mat")
        sh = UsdShade.Shader.Define(stage, self.PRIM_PATH + "/Mat/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*skin))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(os.environ.get("SKIN_ROUGHNESS", "0.35")))
        sh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(
            float(os.environ.get("SKIN_SPECULAR", "0.30")))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

        # 진단용: PM_SKIN_KEEP_BODY=1 이면 인체 도형을 숨기지 않는다.
        #   살과 도형이 겹쳐 보이면 정합 정상, 어긋나면 매핑/규약 버그다.
        self._hidden = (0 if os.environ.get("PM_SKIN_KEEP_BODY") == "1"
                        else self._hide_body_visuals(stage, robot_root_path))
        print(f"[ETRI] skin overlay: 정점 {len(v0)} 면 {len(faces)}  "
              f"인체 시각 prim {self._hidden}개 숨김 (슈트는 유지)")

    @staticmethod
    def _hide_body_visuals(stage, robot_root_path: str) -> int:
        """살이 대신 보여주므로 **인체** 시각 prim 만 감춘다. 콜라이더·슈트는 그대로."""
        from pxr import Usd, UsdGeom

        n = 0
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            path = prim.GetPath().pathString
            if not path.startswith(robot_root_path) or not prim.IsA(UsdGeom.Gprim):
                continue
            low = path.lower()
            if "/collisions/" in low:
                continue                       # 물리 전용, 이미 purpose=guide
            if any(k in low for k in SUIT_KEYWORDS):
                continue                       # 슈트는 계속 보여준다
            UsdGeom.Imageable(prim).MakeInvisible()
            n += 1
        return n

    def _pose_from_prims(self):
        """스테이지에 기록된 body 변환 → (R(24,3,3), T(24,3)). 렌더와 완전 동기."""
        from pxr import UsdGeom

        R = np.empty((24, 3, 3)); T = np.empty((24, 3))
        for j, prim in enumerate(self._body_prims):
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                self._Usd.TimeCode.Default())
            M = np.array([[m[r][c] for c in range(4)] for r in range(4)])
            R[j] = M[:3, :3].T          # USD 는 행벡터 규약 → 전치해야 열벡터 R
            T[j] = M[3, :3]
        return R, T

    def update(self, body_pos_w: np.ndarray, body_quat_w: np.ndarray) -> None:
        """한 프레임 갱신.

        ★ `PM_SKIN_LAG=N` 이면 **N 프레임 전 자세**로 그린다.
          살(USD 스테이지)과 로봇 도형(Fabric/USDRT)은 반영 시점이 달라, 같은 프레임
          데이터를 써도 화면에서는 살이 뒤처져 보인다(2026-08-07 실측).
          근본 해결은 Fabric 에 직접 쓰는 것이지만, 지연 보정으로 화면 정합을 맞춘다.
        """
        lag = int(os.environ.get("PM_SKIN_LAG", "0"))
        if lag > 0:
            buf = getattr(self, "_pose_buf", None)
            if buf is None:
                from collections import deque
                buf = self._pose_buf = deque(maxlen=lag + 1)
            buf.append((np.array(body_pos_w), np.array(body_quat_w)))
            body_pos_w, body_quat_w = buf[0]      # 가장 오래된 = N 프레임 전
        if self._use_prims:
            Rb, bp = self._pose_from_prims()
        else:
            bp = np.asarray(body_pos_w, np.float64)[self._idx]       # (24,3)
            bq = np.asarray(body_quat_w, np.float64)[self._idx]      # (24,4) xyzw
            Rb = _quat_to_mat(bq)
        posed = np.einsum("bv,bvi->vi", self._Wt,
                          np.einsum("bij,bvj->bvi", Rb, self._off) + bp[:, None, :])
        if os.environ.get("PM_SKIN_DEBUG") == "1" and getattr(self, "_dbg", 0) < 3:
            self._dbg = getattr(self, "_dbg", 0) + 1
            print(f"[ETRI dbg] pelvis(body) {bp[0].round(4)}  "
                  f"skin centroid {posed.mean(0).round(4)}  "
                  f"skin z범위 [{posed[:,2].min():.3f},{posed[:,2].max():.3f}]")
        self._points_attr.Set([self._Gf.Vec3f(*map(float, p)) for p in posed])


def maybe_create(simulator) -> "SkinOverlay | None":
    """`PM_SKIN_OVERLAY=1` 이면 오버레이를 만든다. 실패해도 렌더는 계속되게 한다."""
    if not _enabled():
        return None
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        names = list(simulator._robot.data.body_names)
        return SkinOverlay(stage, "/World/envs/env_0/Robot", names)
    except Exception as e:      # 살이 없어도 렌더 자체는 살려 둔다
        print(f"[ETRI] skin overlay 비활성: {type(e).__name__}: {e}")
        return None


def maybe_update(simulator) -> None:
    """렌더 직전에 호출. 필요하면 **여기서 생성**한다.

    ★ 재질 적용 시점(`_apply_visual_materials`)에는 아직 `_robot` 이 없다
      (`AttributeError: 'IsaacLabSimulator' object has no attribute '_robot'`).
      articulation 이 만들어진 뒤 첫 렌더에서 지연 생성한다.
    """
    if not _enabled():
        return
    ov = getattr(simulator, "_etri_skin_overlay", "unset")
    if ov == "unset":
        simulator._etri_skin_overlay = ov = maybe_create(simulator)
    if ov is None:
        return
    try:
        # ★ `robot.data.body_*_w` 는 **캐시 버퍼**라 렌더 시점보다 오래된 값일 수 있다.
        #   실측(2026-08-07): 그 값으로 그리면 살이 도형보다 약 15 cm 뒤에 나타났고,
        #   지연 보정으로는 방향이 반대라 고칠 수 없었다(미래 자세가 필요).
        #   PhysX 뷰(`root_physx_view.get_link_transforms()`)는 현재 상태를 직접 준다.
        #   PM_SKIN_SRC=data 로 옛 경로를 강제할 수 있다.
        src = os.environ.get("PM_SKIN_SRC", "physx")
        view = getattr(simulator._robot, "root_physx_view", None)
        if src == "physx" and view is not None:
            tf = view.get_link_transforms()          # (num_env, nbody, 7) pos + quat xyzw
            tf = tf[0].detach().cpu().numpy() if hasattr(tf, "detach") else np.asarray(tf[0])
            pos, quat_xyzw = tf[:, :3], tf[:, 3:7]
        else:
            # IsaacLab 은 내부적으로 wxyz 를 쓴다 → 공통 xyzw 로 바꿔 넘긴다.
            pos = simulator._robot.data.body_pos_w[0].detach().cpu().numpy()
            quat_wxyz = simulator._robot.data.body_quat_w[0].detach().cpu().numpy()
            quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        ov.update(pos, quat_xyzw)
    except Exception as e:
        print(f"[ETRI] skin overlay 갱신 실패, 비활성화: {type(e).__name__}: {e}")
        simulator._etri_skin_overlay = None
