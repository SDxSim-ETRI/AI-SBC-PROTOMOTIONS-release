#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""AMASS SMPL-H npz(원본 mocap 파라미터)에서 보행 1주기를 자동으로 찾아, 루트 궤적이
끊기지 않게 반복해 1200프레임을 살짝 넘는 긴 루프 npz를 만든다.

★ 핵심 원칙: **FK 역산 없음.** SkeletonTorque .motion 루프(make_looped_skeletontorque.py)는
   FK된 rigid_body 위치/회전을 변환하고 필요시 FK 재계산했지만, SMPL은 npz의 네이티브
   파라미터(trans, poses=[root_orient(3)+body(63)+hands(90)])에 루프 변환을 **직접** 적용한다.
   - trans(xy), root_orient(=poses[:, :3])만 world-Z(yaw) SE(2) 변환·누적
   - body/hand pose(poses[:, 3:]), trans-z, betas, dmpls는 **그대로 타일링** (변형 없음)
   이렇게 하면 관절 포즈가 원본과 100% 동일하고, 루트만 이어붙어 FK 일관성이 자동 보장된다
   (SMPL은 root가 world pose이므로 FK 재계산 불필요).

좌표계(로컬 실측): AMASS CMU npz는 **Z-up**. trans의 분산 최소축=Z(높이), XY=지면.
   heading(yaw)=world Z축 회전. framerate는 클립별로 다름(120 또는 60fps) → mocap_framerate 사용.

자동 탐색(HANDOFF_loop1200.md 2c 이식):
  1. 주기 P*: body pose 자기유사도 최소 지연  argmin_P mean|pose_body[P:]-pose_body[:-P]|
     (탐색범위 [0.8·fps, 1.9·fps] = 보행주기 0.8~1.9s)
  2. 길이 P 창 [s,s+P] 중  ① |yaw 드리프트|(루트, 주기당) ≤ gate(기본 0.5°)  ② 이음매
     min(mean|pose_body[s]-pose_body[s+P]|) 순으로 선택. (facing 기준은 head FK가 필요해
     '역산 금지' 원칙상 생략 — drift+seam만으로 선택)

루프 규칙: Δyaw,Δxy = (frame0→frameN-1) 루트 변환. loop k는 Rz(k·Δyaw)+누적이동 적용,
  [1:N]만 이어붙여 중복 프레임 제거. 총 프레임 = N + (K-1)(N-1).

사용:
  ~/miniforge3/envs/env_newton/bin/python OUR_MOTION_DATA/scripts/make_looped_smpl_npz.py \
    <src_poses.npz> <out_loop.npz> [목표프레임=1200] [--range s:e] [--drift-gate-deg 0.5]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

UP = 2  # AMASS CMU npz는 Z-up (로컬 실측). heading=Z축 회전, 지면=XY.
GX, GY = 0, 1  # ground-plane 축


def yaw_of_rotvec(rv):
    """axis-angle(3,) → world Z축 heading(yaw). Z-up 기준."""
    m = R.from_rotvec(rv).as_matrix()
    return float(np.arctan2(m[GY, GX], m[GX, GX]))


def yaw_arr_rotvec(rvs):
    m = R.from_rotvec(rvs).as_matrix()  # (T,3,3)
    return np.arctan2(m[:, GY, GX], m[:, GX, GX])


def ang_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    m = np.eye(3)
    m[GX, GX] = c; m[GX, GY] = -s
    m[GY, GX] = s; m[GY, GY] = c
    return m


def find_period(pose_body, pmin, pmax):
    T = len(pose_body)
    scores = []
    for P in range(pmin, min(pmax, T - 2) + 1):
        d = np.mean(np.abs(pose_body[P:] - pose_body[:-P]))
        scores.append((P, float(d)))
    scores.sort(key=lambda x: x[1])
    return scores[0][0], scores


def search_window(pose_body, root_yaw, P, gate_rad):
    """길이 P 창 [s,s+P] 전수 평가 → drift 게이트 통과분 중 이음매 최소 선택."""
    T = len(pose_body)
    rows = []
    for s in range(0, T - P):
        e = s + P
        drift = abs(ang_diff(root_yaw[e], root_yaw[s]))
        seam = float(np.mean(np.abs(pose_body[s] - pose_body[e])))
        rows.append((s, drift, seam))
    gated = [r for r in rows if r[1] <= gate_rad]
    pool = gated if gated else rows
    pool.sort(key=lambda r: r[2])  # 이음매 최소
    # drift 게이트 통과분이 없으면 drift 최소부터
    if not gated:
        pool = sorted(rows, key=lambda r: (r[1], r[2]))
    best = pool[0]
    return best[0], best, len(gated)


def loop_smpl(trans, ao, body_hands, dmpls, a, b, target):
    """[a:b] 구간(양끝 포함, N=b-a+1)을 루트 연속으로 K회 반복. ao=root_orient(rotvec)."""
    N = b - a + 1
    K = max(1, int(np.ceil((target - N) / (N - 1))) + 1)

    seg_tr = trans[a:b + 1].astype(np.float64)       # (N,3)
    seg_ao = ao[a:b + 1].astype(np.float64)          # (N,3) rotvec
    seg_bh = body_hands[a:b + 1]                      # (N,153) 불변
    seg_dm = dmpls[a:b + 1] if dmpls is not None else None

    dyaw = yaw_of_rotvec(seg_ao[N - 1]) - yaw_of_rotvec(seg_ao[0])
    # 한 주기 순수 xy 이동(첫 프레임 heading 정렬 후): loop_motion과 동일한 누적식
    R1 = rotz(dyaw)
    t1 = seg_tr[N - 1, [GX, GY]] - (R1 @ seg_tr[0])[[GX, GY]]

    seg_ao_mat = R.from_rotvec(seg_ao).as_matrix()   # (N,3,3)

    out_tr, out_ao = [], []
    for k in range(K):
        sl = slice(0, N) if k == 0 else slice(1, N)
        Rk = rotz(k * dyaw)
        # 누적 xy 이동 t_k
        t = np.zeros(2)
        for _ in range(k):
            t = (rotz(dyaw) @ np.array([t[0], t[1], 0.0]))[[GX, GY]] + t1
        # trans 변환: xy 회전+이동, z 불변
        tr = (seg_tr[sl] @ Rk.T)
        tr[:, GX] += t[0]; tr[:, GY] += t[1]
        tr[:, UP] = seg_tr[sl][:, UP]  # 높이 원본 유지
        out_tr.append(tr)
        # root_orient 변환: Rz(kΔyaw) 좌승 후 rotvec 복원
        ao_k = R.from_matrix(Rk @ seg_ao_mat[sl]).as_rotvec()
        out_ao.append(ao_k)

    trans_out = np.concatenate(out_tr, 0)
    ao_out = np.concatenate(out_ao, 0)
    # body/hand pose + dmpls: 동일 프레임 슬라이스로 타일링(불변)
    bh_out = np.concatenate([seg_bh[0:N]] + [seg_bh[1:N]] * (K - 1), 0)
    dm_out = None
    if seg_dm is not None:
        dm_out = np.concatenate([seg_dm[0:N]] + [seg_dm[1:N]] * (K - 1), 0)
    return trans_out, ao_out, bh_out, dm_out, N, K, dyaw


def seam_blend_smpl(poses, trans, N, K, h):
    """각 이음매(±h 프레임)에서 poses(전 156채널)·trans를 표준 3차 Hermite로 대체.
    경계(lo, hi) 위치·속도(프레임간 차분)를 정확히 이어받아 위치·속도 C1 연속.
    SMPL은 파라미터가 곧 모션이라 FK 재계산 불필요(역산 없음)."""
    poses = poses.astype(np.float64).copy()
    trans = trans.astype(np.float64).copy()
    T = len(poses)
    seams = [N - 1 + i * (N - 1) for i in range(K - 1)]
    for sidx in seams:
        lo, hi = sidx - h, sidx + h
        if lo < 1 or hi > T - 2:
            continue
        Lw = hi - lo
        for arr in (poses, trans):
            a0, a1 = arr[lo].copy(), arr[hi].copy()
            m0 = arr[lo] - arr[lo - 1]     # 경계 접선(프레임당 속도)
            m1 = arr[hi + 1] - arr[hi]
            for j, idx in enumerate(range(lo, hi + 1)):
                t = j / Lw
                h00 = 2 * t**3 - 3 * t**2 + 1
                h10 = t**3 - 2 * t**2 + t
                h01 = -2 * t**3 + 3 * t**2
                h11 = t**3 - t**2
                arr[idx] = h00 * a0 + h10 * (Lw * m0) + h01 * a1 + h11 * (Lw * m1)
    return poses, trans


def apply_arm_open(poses, dL, dR):
    """양팔 벌림(abduction): 어깨 z성분에 상수 오프셋. FK 검증 확정 —
    L_shoulder(SMPL joint16)=col50 += dL, R_shoulder(joint17)=col53 += dR.
    dL>0·dR<0이 양팔을 바깥으로 (원본 대비 대칭 벌림 권장)."""
    p = poses.copy()
    p[:, 50] += dL
    p[:, 53] += dR
    return p


def _seam_jump_ratio(poses, N, K):
    body = poses[:, 3:66]
    fr = np.mean(np.abs(np.diff(body, axis=0)), axis=1)
    seams = [N - 1 + i * (N - 1) for i in range(K - 1)]
    if not seams:
        return 0.0, float(np.median(fr))
    sj = float(np.mean([fr[min(s, len(fr) - 1)] for s in seams]))
    return sj, float(np.median(fr))


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    pos = [a for a in sys.argv[3:] if not a.startswith("--")]
    target = int(pos[0]) if pos else 1200
    gate = np.radians(float(sys.argv[sys.argv.index("--drift-gate-deg") + 1])
                      if "--drift-gate-deg" in sys.argv else 0.5)
    rng = None
    if "--range" in sys.argv:
        s_, e_ = sys.argv[sys.argv.index("--range") + 1].split(":")
        rng = (int(s_), int(e_))
    seam_h = int(sys.argv[sys.argv.index("--seam-blend") + 1]) if "--seam-blend" in sys.argv else 0
    arm_open = None
    if "--arm-open" in sys.argv:
        dl_, dr_ = sys.argv[sys.argv.index("--arm-open") + 1].split(",")
        arm_open = (float(dl_), float(dr_))

    d = np.load(str(src), allow_pickle=True)
    trans = d["trans"].astype(np.float64)            # (T,3)
    poses = d["poses"].astype(np.float64)            # (T,156)
    ao = poses[:, :3]                                # root_orient rotvec
    body_hands = poses[:, 3:]                        # (T,153) body(63)+hands(90)
    dmpls = d["dmpls"] if "dmpls" in d else None
    fps = float(d["mocap_framerate"])
    T = trans.shape[0]
    print(f"■ {src.name}: {T} frames @ {fps:.0f}fps = {T/fps:.2f}s  (Z-up, poses={poses.shape[1]})")

    root_yaw = yaw_arr_rotvec(ao)
    if rng is None:
        pmin = max(6, int(round(0.8 * fps)))
        pmax = int(round(1.9 * fps))
        Pstar, scores = find_period(body_hands[:, :63], pmin, pmax)  # body 63만으로 주기
        s, best, ngate = search_window(body_hands[:, :63], root_yaw, Pstar, gate)
        a, b = s, s + Pstar
        print(f"   주기 P*={Pstar} ({Pstar/fps:.3f}s) [상위후보 {[p for p,_ in scores[:4]]}]  "
              f"drift게이트 통과창 {ngate}개")
        print(f"   구간 [{a}:{b}] N={b-a+1}  드리프트 {np.degrees(best[1]):+.3f}°  이음매 {best[2]:.4f}")
    else:
        a, b = rng
        print(f"   구간(수동) [{a}:{b}] N={b-a+1}")

    trans_out, ao_out, bh_out, dm_out, N, K, dyaw = loop_smpl(
        trans, ao, body_hands, dmpls, a, b, target)
    poses_out = np.concatenate([ao_out, bh_out], axis=1).astype(np.float64)

    if arm_open is not None:
        poses_out = apply_arm_open(poses_out, arm_open[0], arm_open[1])
        print(f"   양팔 벌림(arm-open): L_sh z+{arm_open[0]:+.3f}  R_sh z{arm_open[1]:+.3f} rad "
              f"(={np.degrees(arm_open[0]):+.1f}°/{np.degrees(arm_open[1]):+.1f}°)")

    if seam_h > 0:
        sj0, med0 = _seam_jump_ratio(poses_out, N, K)
        poses_out, trans_out = seam_blend_smpl(poses_out, trans_out, N, K, seam_h)
        sj1, med1 = _seam_jump_ratio(poses_out, N, K)
        print(f"   seam-blend ±{seam_h}f: 이음매 관절점프 {sj0/max(med0,1e-9):.2f}× → "
              f"{sj1/max(med1,1e-9):.2f}× (정상=1.0×)")

    poses_out = poses_out.astype(np.float32)

    save = dict(trans=trans_out.astype(np.float32), poses=poses_out,
                betas=d["betas"], gender=d["gender"], mocap_framerate=d["mocap_framerate"])
    if dm_out is not None:
        save["dmpls"] = dm_out.astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(dst), **save)

    # ---- 검증 ----
    Tout = trans_out.shape[0]
    xy = trans_out[:, [GX, GY]]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    path = float(step.sum())
    z = trans_out[:, UP]
    ry = yaw_arr_rotvec(ao_out)
    seams = [N - 1 + i * (N - 1) for i in range(K - 1)]
    seam_steps = [step[min(sq, len(step) - 1)] for sq in seams]
    print(f"   → {dst.name}: 주기 {N}프레임 × {K}회 = {Tout}프레임 / {Tout/fps:.2f}s")
    print(f"   yaw/주기 {np.degrees(dyaw):+.3f}°  순이동 {net:.2f}m  경로 {path:.2f}m  "
          f"직선도 {100*net/max(path,1e-9):.1f}%")
    print(f"   높이 z [{z.min():.3f},{z.max():.3f}]m  루트 프레임간이동 중앙 {np.median(step)*1000:.1f}mm "
          f"최대 {step.max()*1000:.1f}mm ({step.max()/max(np.median(step),1e-9):.2f}×)")
    if seam_steps:
        print(f"   이음매 이동량: {[f'{s*1000:.1f}' for s in seam_steps[:6]]}... mm")
    ok_frames = Tout >= 1200
    ok_cycle = (Tout - N) % (N - 1) == 0
    ok_drift = abs(np.degrees(dyaw)) < 0.5
    ok_line = 100 * net / max(path, 1e-9) > 95
    ok_seam = step.max() / max(np.median(step), 1e-9) < 1.5
    print(f"   [체크] 프레임≥1200:{ok_frames}  주기정수배:{ok_cycle}  드리프트<0.5°:{ok_drift}  "
          f"직선도>95%:{ok_line}  이음매비<1.5:{ok_seam}")


if __name__ == "__main__":
    main()
