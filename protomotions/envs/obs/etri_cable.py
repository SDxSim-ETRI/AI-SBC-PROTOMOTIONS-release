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
"""Cable path length observation for v2 exoskeleton suit.

Newton / IsaacLab 호환 설계
============================
Newton  : spatial tendon 물리가 실제 케이블 힘을 계산. 이 모듈은 동일한
          waypoint 공식으로 관측값만 독립 계산 (텐돈 물리와 중복되지 않음).
IsaacLab: spatial tendon 미지원 → 이 모듈의 계산값을 외부힘 인가에도 사용.

케이블 구조 (v2 suit)
=====================
3-waypoint 공간 텐돈:
  cable_exit (exo_main_body, torso 고정체)
    → pelvis_anchor (pelvis)
    → cable_insertion (대퇴골 dump body)

경로 길이 = |exit_world - anchor_world| + |anchor_world - insertion_world|

텐돈 range="0 1.0":
  경로 길이 < 1.0m → slack (힘 없음)
  경로 길이 ≥ 1.0m → taut (수동 저항 발생)

body 인덱스 (ProtoMotions 기준 = MuJoCo body ID - 1, world 제외)
=================================================================
  pelvis       : 0   (MuJoCo 1)
  RH_dump      : 2   (MuJoCo 3)  ← cable2 insertion
  RH_dump2     : 3   (MuJoCo 4)  ← cable4 insertion
  femur_l      : 8   (MuJoCo 9)
  LH_dump      : 9   (MuJoCo 10) ← cable1 insertion
  LH_dump2     : 10  (MuJoCo 11) ← cable3 insertion
  exo_main_body: 24  (MuJoCo 25) ← cable1-4 exit (모두 동일 body)

사이트 로컬 좌표 (Newton XML, exo_main_body 프레임)
===================================================
  cable1_exit: (-0.06, -0.05, -0.12)
  cable2_exit: (-0.06, -0.05,  0.12)
  cable3_exit: (-0.06,  0.05, -0.12)
  cable4_exit: (-0.06,  0.05,  0.12)

  cable1_posterior_pelvis_anchor: (-0.192669,  0.08525766, -0.01749136)
  cable2_posterior_pelvis_anchor: (-0.192669, -0.08525764, -0.01749146)
  cable3_anterior_pelvis_anchor:  ( 0.0,       0.07726005,  0.0)
  cable4_anterior_pelvis_anchor:  ( 0.0,      -0.07726005,  0.0)

  cable1/2/3/4_insertion: (0, 0, 0) — body 중심
"""

import numpy as np
import torch
from torch import Tensor

# ── 텐돈 파라미터 ────────────────────────────────────────────────────────────
TENDON_REST_LENGTH = 1.0  # m (range="0 1" → 1.0m 초과 시 taut)

# ── ProtoMotions body 인덱스 (world body 제외, MuJoCo ID - 1) ────────────────
_IDX_PELVIS = 0
_IDX_RH_DUMP = 2    # cable2 insertion (right hip posterior)
_IDX_RH_DUMP2 = 3   # cable4 insertion (right hip anterior)
_IDX_LH_DUMP = 9    # cable1 insertion (left hip posterior)
_IDX_LH_DUMP2 = 10  # cable3 insertion (left hip anterior)
_IDX_EXO_MAIN = 24  # cable1~4 exit (exo_main_body)

# ── 케이블 exit 로컬 좌표 [4, 3] (exo_main_body 프레임) ─────────────────────
_EXIT_LOCAL = torch.tensor([
    [-0.06, -0.05, -0.12],  # cable1_exit
    [-0.06, -0.05,  0.12],  # cable2_exit
    [-0.06,  0.05, -0.12],  # cable3_exit
    [-0.06,  0.05,  0.12],  # cable4_exit
], dtype=torch.float32)

# ── pelvis anchor 로컬 좌표 [4, 3] (pelvis 프레임, Newton XML 기준) ──────────
_ANCHOR_LOCAL = torch.tensor([
    [-0.192669,  0.08525766, -0.01749136],  # cable1_posterior
    [-0.192669, -0.08525764, -0.01749146],  # cable2_posterior
    [ 0.0,       0.07726005,  0.0       ],  # cable3_anterior
    [ 0.0,      -0.07726005,  0.0       ],  # cable4_anterior
], dtype=torch.float32)

# ── insertion body 인덱스 [4] ────────────────────────────────────────────────
_INSERTION_IDX = [_IDX_LH_DUMP, _IDX_RH_DUMP, _IDX_LH_DUMP2, _IDX_RH_DUMP2]


def _rotate_local_to_world(local_pos: Tensor, body_rot_xyzw: Tensor) -> Tensor:
    """로컬 좌표를 body 회전으로 월드 좌표 offset으로 변환.

    Args:
        local_pos: [..., 3] 로컬 좌표
        body_rot_xyzw: [..., 4] xyzw 쿼터니언 (ProtoMotions 공통 포맷)

    Returns:
        [..., 3] 회전된 오프셋
    """
    x, y, z, w = body_rot_xyzw.unbind(-1)
    # 쿼터니언 회전 행렬 없이 직접 계산 (qvq*)
    lx, ly, lz = local_pos.unbind(-1)
    tx = 2.0 * (y * lz - z * ly)
    ty = 2.0 * (z * lx - x * lz)
    tz = 2.0 * (x * ly - y * lx)
    rx = lx + w * tx + (y * tz - z * ty)
    ry = ly + w * ty + (z * tx - x * tz)
    rz = lz + w * tz + (x * ty - y * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def compute_cable_path_lengths(
    body_pos: Tensor,
    body_rot: Tensor,
) -> Tensor:
    """케이블 4개의 3-waypoint 경로 길이를 body 위치/회전으로 계산.

    Newton과 IsaacLab 모두에서 동일한 방식으로 동작.
    Newton은 spatial tendon 물리가 실제 힘을 처리하므로 이 값은 obs 전용.
    IsaacLab은 이 값을 외부힘 인가 계산에도 사용.

    Args:
        body_pos: [num_envs, num_bodies, 3]  월드 body 위치
        body_rot: [num_envs, num_bodies, 4]  월드 body 회전 (xyzw)

    Returns:
        cable_lengths: [num_envs, 4]  각 케이블의 경로 길이 (m)
    """
    device = body_pos.device
    exit_local = _EXIT_LOCAL.to(device)    # [4, 3]
    anchor_local = _ANCHOR_LOCAL.to(device)  # [4, 3]

    # exo_main_body 월드 위치/회전
    exo_pos = body_pos[:, _IDX_EXO_MAIN, :]   # [E, 3]
    exo_rot = body_rot[:, _IDX_EXO_MAIN, :]   # [E, 4]

    # pelvis 월드 위치/회전
    pel_pos = body_pos[:, _IDX_PELVIS, :]     # [E, 3]
    pel_rot = body_rot[:, _IDX_PELVIS, :]     # [E, 4]

    lengths = []
    for i in range(4):
        # exit 월드 위치: exo_main_body pos + rotated local exit
        exit_offset = _rotate_local_to_world(
            exit_local[i].expand(exo_pos.shape[0], -1), exo_rot
        )  # [E, 3]
        exit_world = exo_pos + exit_offset    # [E, 3]

        # anchor 월드 위치: pelvis pos + rotated local anchor
        anc_offset = _rotate_local_to_world(
            anchor_local[i].expand(pel_pos.shape[0], -1), pel_rot
        )  # [E, 3]
        anchor_world = pel_pos + anc_offset   # [E, 3]

        # insertion 월드 위치: dump body 중심 (local pos = 0)
        ins_world = body_pos[:, _INSERTION_IDX[i], :]  # [E, 3]

        seg1 = (exit_world - anchor_world).norm(dim=-1)    # [E]
        seg2 = (anchor_world - ins_world).norm(dim=-1)     # [E]
        lengths.append(seg1 + seg2)

    return torch.stack(lengths, dim=-1)  # [E, 4]


def compute_cable_waypoints(
    body_pos: Tensor,
    body_rot: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """케이블 4개의 3-waypoint 월드 좌표 (exit/anchor/insertion) 계산.

    compute_cable_path_lengths()와 동일한 기하학이지만 길이 대신 좌표 자체를
    반환한다 — 시각화(케이블 캡슐 렌더링)용. 물리/관측 계산에는 영향 없음.

    Args:
        body_pos: [num_envs, num_bodies, 3]  월드 body 위치
        body_rot: [num_envs, num_bodies, 4]  월드 body 회전 (xyzw)

    Returns:
        exit_world, anchor_world, insertion_world: 각 [num_envs, 4, 3]
    """
    device = body_pos.device
    exit_local = _EXIT_LOCAL.to(device)      # [4, 3]
    anchor_local = _ANCHOR_LOCAL.to(device)  # [4, 3]

    exo_pos = body_pos[:, _IDX_EXO_MAIN, :]  # [E, 3]
    exo_rot = body_rot[:, _IDX_EXO_MAIN, :]  # [E, 4]
    pel_pos = body_pos[:, _IDX_PELVIS, :]    # [E, 3]
    pel_rot = body_rot[:, _IDX_PELVIS, :]    # [E, 4]

    exits, anchors, insertions = [], [], []
    for i in range(4):
        exit_offset = _rotate_local_to_world(
            exit_local[i].expand(exo_pos.shape[0], -1), exo_rot
        )
        exits.append(exo_pos + exit_offset)

        anc_offset = _rotate_local_to_world(
            anchor_local[i].expand(pel_pos.shape[0], -1), pel_rot
        )
        anchors.append(pel_pos + anc_offset)

        insertions.append(body_pos[:, _INSERTION_IDX[i], :])

    return (
        torch.stack(exits, dim=1),       # [E, 4, 3]
        torch.stack(anchors, dim=1),     # [E, 4, 3]
        torch.stack(insertions, dim=1),  # [E, 4, 3]
    )


def compute_cable_velocities(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
) -> Tensor:
    """케이블 4개의 경로 길이 시간 미분 (m/s).

    d|p_a - p_b|/dt = (p_a - p_b)·(v_a - v_b) / (|p_a - p_b| + eps)

    근사: body 중심 선속도만 사용, 사이트 오프셋의 회전 기여분 무시.
    보행 중 병진 속도가 지배적이므로 obs 정밀도에 충분.

    Args:
        body_pos: [num_envs, num_bodies, 3]
        body_rot: [num_envs, num_bodies, 4]  xyzw
        body_vel: [num_envs, num_bodies, 3]  선속도 (월드 프레임)

    Returns:
        [num_envs, 4]  각 케이블의 경로 길이 변화율 (m/s)
    """
    device = body_pos.device
    exit_local = _EXIT_LOCAL.to(device)
    anchor_local = _ANCHOR_LOCAL.to(device)

    exo_pos = body_pos[:, _IDX_EXO_MAIN, :]
    exo_rot = body_rot[:, _IDX_EXO_MAIN, :]
    exo_vel = body_vel[:, _IDX_EXO_MAIN, :]   # 근사: 중심 선속도

    pel_pos = body_pos[:, _IDX_PELVIS, :]
    pel_rot = body_rot[:, _IDX_PELVIS, :]
    pel_vel = body_vel[:, _IDX_PELVIS, :]

    _EPS = 1e-6
    vel_list = []
    for i in range(4):
        exit_offset = _rotate_local_to_world(
            exit_local[i].expand(exo_pos.shape[0], -1), exo_rot
        )
        exit_world = exo_pos + exit_offset

        anc_offset = _rotate_local_to_world(
            anchor_local[i].expand(pel_pos.shape[0], -1), pel_rot
        )
        anchor_world = pel_pos + anc_offset

        ins_world = body_pos[:, _INSERTION_IDX[i], :]
        ins_vel   = body_vel[:, _INSERTION_IDX[i], :]

        seg1_vec = exit_world - anchor_world                                 # [E, 3]
        seg1_len = seg1_vec.norm(dim=-1).clamp(min=_EPS)                    # [E]
        dseg1 = (seg1_vec * (exo_vel - pel_vel)).sum(dim=-1) / seg1_len     # [E]

        seg2_vec = anchor_world - ins_world
        seg2_len = seg2_vec.norm(dim=-1).clamp(min=_EPS)
        dseg2 = (seg2_vec * (pel_vel - ins_vel)).sum(dim=-1) / seg2_len

        vel_list.append(dseg1 + dseg2)

    return torch.stack(vel_list, dim=-1)  # [E, 4]


def compute_cable_obs(
    body_pos: Tensor,
    body_rot: Tensor,
) -> Tensor:
    """케이블 관측값: [normalized_length × 4].

    정규화: length / TENDON_REST_LENGTH
      < 1.0 → slack (0~1 범위)
      = 1.0 → 팽팽 임계
      > 1.0 → taut (드문 경우)

    Args:
        body_pos: [num_envs, num_bodies, 3]
        body_rot: [num_envs, num_bodies, 4]  xyzw

    Returns:
        [num_envs, 4]  정규화된 케이블 길이
    """
    lengths = compute_cable_path_lengths(body_pos, body_rot)
    return lengths / TENDON_REST_LENGTH


def compute_cable_full_obs(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
) -> Tensor:
    """케이블 전체 관측값: [normalized_length×4 + velocity×4] = 8 dim.

    velocity는 경로 길이 변화율(m/s). 보행 중 일반적으로 ±2m/s 범위.
    policy가 케이블 상태 변화에 즉각 반응 가능 (길이만으로는 지연 발생).

    Args:
        body_pos: [num_envs, num_bodies, 3]
        body_rot: [num_envs, num_bodies, 4]  xyzw
        body_vel: [num_envs, num_bodies, 3]  선속도

    Returns:
        [num_envs, 8]  [length_norm×4, velocity×4]
    """
    lengths = compute_cable_path_lengths(body_pos, body_rot)     # [E, 4]
    velocities = compute_cable_velocities(body_pos, body_rot, body_vel)  # [E, 4]
    return torch.cat([lengths / TENDON_REST_LENGTH, velocities], dim=-1)  # [E, 8]


def compute_cable_tension_penalty(
    body_pos: Tensor,
    body_rot: Tensor,
) -> Tensor:
    """케이블 taut 상태 패널티: -(excess)² per env.

    rest_length(1.0m) 초과분에만 패널티 적용. Newton의 수동 텐돈 폭발 방지.
    reward로 사용: 값이 항상 ≤ 0.

    Args:
        body_pos: [num_envs, num_bodies, 3]
        body_rot: [num_envs, num_bodies, 4]

    Returns:
        [num_envs]  패널티 값 (≤ 0)
    """
    lengths = compute_cable_path_lengths(body_pos, body_rot)  # [E, 4]
    excess = (lengths - TENDON_REST_LENGTH).clamp(min=0.0)    # taut 초과분만
    return -excess.pow(2).mean(dim=-1)                         # [E]


def compute_cable_body_wrench(
    body_pos: Tensor,
    body_rot: Tensor,
    num_bodies: int,
    tension: Tensor,
) -> Tensor:
    """Cable pulling force applied directly to bodies, as a replacement for the
    original MuJoCo `<motor tendon="cableN_tendon" ctrlrange="-140 0">` actuator.

    Neither Newton nor IsaacLab support tendon actuators (see
    tasks/mimic_suit_active_cable_walk_23dof_v2/scripts/convert_mujoco_to_newton.py
    step 5 for why the original design had to be dropped for Newton). Both engines
    do support applying an arbitrary world-frame wrench (force+torque) at a body's
    center of mass, so this reproduces the tendon-actuator's physics manually from
    the same exit/anchor/insertion waypoint geometry already used by
    compute_cable_path_lengths(): a positive tension pulls exit<->anchor and
    anchor<->insertion together, exactly like a winch shortening the cable.

    Args:
        body_pos: [num_envs, num_bodies, 3] world body positions
        body_rot: [num_envs, num_bodies, 4] world body rotations (xyzw)
        num_bodies: total body count (for sizing the zero-elsewhere output)
        tension: [num_envs, 4] pulling force per cable in N, >=0 (0=slack/no pull)

    Returns:
        [num_envs, num_bodies, 6] world-frame wrench (force xyz, torque xyz)
        applied at each body's COM, zero for bodies not involved in any cable.
        Pass directly to Simulator.set_external_body_wrench().
    """
    device = body_pos.device
    num_envs = body_pos.shape[0]
    exit_local = _EXIT_LOCAL.to(device)
    anchor_local = _ANCHOR_LOCAL.to(device)

    exo_pos = body_pos[:, _IDX_EXO_MAIN, :]
    exo_rot = body_rot[:, _IDX_EXO_MAIN, :]
    pel_pos = body_pos[:, _IDX_PELVIS, :]
    pel_rot = body_rot[:, _IDX_PELVIS, :]

    wrench = torch.zeros(num_envs, num_bodies, 6, device=device, dtype=body_pos.dtype)
    _EPS = 1e-6

    for i in range(4):
        t = tension[:, i].clamp(min=0.0)  # [E], pull-only

        exit_offset = _rotate_local_to_world(
            exit_local[i].expand(exo_pos.shape[0], -1), exo_rot
        )
        exit_world = exo_pos + exit_offset

        anc_offset = _rotate_local_to_world(
            anchor_local[i].expand(pel_pos.shape[0], -1), pel_rot
        )
        anchor_world = pel_pos + anc_offset

        ins_idx = _INSERTION_IDX[i]
        ins_world = body_pos[:, ins_idx, :]

        # Segment 1: exit <-> anchor. Tension pulls them together.
        seg1_vec = anchor_world - exit_world
        seg1_dir = seg1_vec / seg1_vec.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        force1 = t.unsqueeze(-1) * seg1_dir  # [E, 3], on exit body (toward anchor)

        # Segment 2: anchor <-> insertion. Tension pulls them together.
        seg2_vec = ins_world - anchor_world
        seg2_dir = seg2_vec / seg2_vec.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        force2 = t.unsqueeze(-1) * seg2_dir  # [E, 3], on anchor body (toward insertion)

        # exo_main_body: pulled toward anchor along segment 1.
        torque_exit = torch.cross(exit_offset, force1, dim=-1)
        wrench[:, _IDX_EXO_MAIN, :3] += force1
        wrench[:, _IDX_EXO_MAIN, 3:] += torque_exit

        # pelvis: pulled toward exit (reaction of seg1) and toward insertion (seg2).
        force_anchor = -force1 + force2
        torque_anchor = torch.cross(anc_offset, force_anchor, dim=-1)
        wrench[:, _IDX_PELVIS, :3] += force_anchor
        wrench[:, _IDX_PELVIS, 3:] += torque_anchor

        # insertion body: pulled toward anchor (reaction of seg2). Site is the
        # body's own center (local pos (0,0,0)), so no torque contribution.
        wrench[:, ins_idx, :3] += -force2

    return wrench


def cable_length_obs_factory():
    """케이블 경로 길이만 (4 dim). cable_full_obs_factory 권장."""
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.context_views import EnvContext

    return MdpComponent(
        compute_func=compute_cable_obs,
        dynamic_vars={
            "body_pos": EnvContext.current.rigid_body_pos,
            "body_rot": EnvContext.current.rigid_body_rot,
        },
        static_params={},
    )


def cable_full_obs_factory():
    """케이블 경로 길이 + 변화율 (8 dim).

    반환: [normalized_length×4, velocity×4]
      - normalized_length: length / 1.0m (slack <1, taut >1)
      - velocity: d(length)/dt in m/s (단축 = 음수, 신장 = 양수)

    Newton/IsaacLab 공통 사용 가능.
    """
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.context_views import EnvContext

    return MdpComponent(
        compute_func=compute_cable_full_obs,
        dynamic_vars={
            "body_pos": EnvContext.current.rigid_body_pos,
            "body_rot": EnvContext.current.rigid_body_rot,
            "body_vel": EnvContext.current.rigid_body_vel,
        },
        static_params={},
    )


def cable_tension_penalty_factory(weight: float = -0.01):
    """케이블 taut 패널티 reward factory.

    taut 상태(경로 길이 > 1.0m) 초과분²에 패널티.
    텐돈 폭발 방지 및 자연스러운 케이블 slack 유지 유도.

    Args:
        weight: 패널티 강도 (음수). 기본 -0.01.
                너무 강하면 policy가 케이블 완전 회피 → -0.05 이하 비권장.
    """
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.context_views import EnvContext

    return MdpComponent(
        compute_func=compute_cable_tension_penalty,
        dynamic_vars={
            "body_pos": EnvContext.current.rigid_body_pos,
            "body_rot": EnvContext.current.rigid_body_rot,
        },
        static_params={"weight": weight},
    )


# ── Newton live render hook (2026-07-10) ────────────────────────────────────
#
# Newton's viewer never renders <tendon> paths (only <geom> shapes -- see
# NewtonSimulator._create_envs / add_mjcf, which only tracks tendon names for
# actuator resolution). So for BOTH active-cable (real body-wrench force) and
# passive-cable (native spatial-tendon physics) suits, the 4 cables are
# invisible during --auto-record / interactive play unless something manually
# draws them each frame from live body positions -- exactly what this hook
# does, reusing the same exit/anchor/insertion waypoint geometry as
# compute_cable_path_lengths()/compute_cable_body_wrench(). Shared by
# ActiveCableV2TensionEnv and the passive-suit cable-render env so the capsule
# math lives in one place.

_CABLE_CAPSULE_RADIUS = 0.007
_CABLE_CAPSULE_COLOR = (0.0, 0.25, 1.0)


def _segment_to_capsule(start, end, radius: float):
    """세그먼트(start->end, numpy [3]) -> Newton capsule transform/scale."""
    import warp as wp

    center = (start + end) / 2.0
    d = end - start
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return None, None
    d_n = d / length
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, d_n)
    dot = float(np.dot(z, d_n))
    if np.linalg.norm(cross) < 1e-8:
        q = [0.0, 0.0, 0.0, 1.0] if dot > 0 else [1.0, 0.0, 0.0, 0.0]
    else:
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        cn = cross / np.linalg.norm(cross)
        s = np.sin(angle / 2.0)
        q = [cn[0] * s, cn[1] * s, cn[2] * s, np.cos(angle / 2.0)]
    xform = wp.transform(center.tolist(), wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    return xform, wp.vec3(radius, radius, length / 2.0)


def make_cable_render_hook(simulator):
    """Newton simulator._render_hook용 no-arg 클로저 생성 -- env 0의 케이블
    4개를 매 프레임 실제 body 위치로부터 파란 캡슐로 그린다.

    사용법 (env __init__에서):
        if hasattr(self.simulator, "_render_hook"):
            self.simulator._render_hook = make_cable_render_hook(self.simulator)
    """
    def _hook() -> None:
        import warp as wp

        viewer = simulator.viewer
        if viewer is None:
            return
        robot_state = simulator.get_robot_state()
        body_pos = robot_state.rigid_body_pos[:1].detach().cpu().numpy()  # env 0
        body_rot = robot_state.rigid_body_rot[:1].detach().cpu().numpy()
        exit_w, anchor_w, insertion_w = compute_cable_waypoints(
            torch.from_numpy(body_pos), torch.from_numpy(body_rot)
        )
        exit_w, anchor_w, insertion_w = (
            exit_w[0].numpy(), anchor_w[0].numpy(), insertion_w[0].numpy()
        )

        xforms, scales = [], []
        for i in range(exit_w.shape[0]):
            for a, b in ((exit_w[i], anchor_w[i]), (anchor_w[i], insertion_w[i])):
                xf, sc = _segment_to_capsule(a, b, _CABLE_CAPSULE_RADIUS)
                if xf is not None:
                    xforms.append(xf)
                    scales.append(sc)
        if not xforms:
            return
        viewer.log_capsules(
            "cables", "",
            wp.array(xforms, dtype=wp.transform),
            wp.array(scales, dtype=wp.vec3),
            wp.array([_CABLE_CAPSULE_COLOR] * len(xforms), dtype=wp.vec3),
            None,
        )

    return _hook
