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
"""FrozenHumanExoEnv — 동결 HumanNet + 외골격 모터(ActionNet) 학습 env.

모터 수·부착 body·피크 토크는 `config.exo_motors` 로 주입된다(슈트마다 다름).
힙 전용 슈트는 힙 2개, 전신 슈트는 힙+무릎 4개.

Stage 2. 계획: ../학습계획.md §4 / 이력: ../학습구현.md

왜 사람을 동결하는가
--------------------
원격 케이블 태스크(`tasks/mimic_assist_suit_actionnet_walk_23dof_v2`)에서 단일
정책(사람 PD + 슈트 출력을 한 정책이 모두 냄)이 effort 저감 −60~70% 를 기록했지만
보조를 0 으로 끄는 ablation 에서 **슈트 기여가 ~0** 으로 판명됐다. 정책이 슈트를
쓰는 대신 "사람 관절이 저토크로 걷는 스타일"로 보상을 우회한 것이다.
사람을 동결하면 hip/knee 토크 감소는 **정의상 전부 모터 기여**가 된다.

모터 토크를 넣는 통로 (검증됨)
------------------------------
ProtoMotions 의 액션은 MJCF `<actuator>` 가 아니라 **DOF 단위 PD 목표**이고,
`BUILT_IN_PD` 에서는 솔버가 내부에서 토크를 만든다. 외부 토크를 더할 통로는
`Simulator.set_external_body_wrench([E, B, 6])` 하나뿐이므로 모터 토크를
**순수 짝힘**으로 넣는다:

    자식 body(대퇴골/정강이)에  +τ·n̂ ,  부모 body(골반/대퇴골)에  −τ·n̂

짝힘은 작용점과 무관하므로 관절 토크와 정확히 동등하다. `scripts/verify_exo_axis.py`
가 MuJoCo 로 `qacc` 오차 **0.000e+00**(4개 모터 전부)을 확인했다.

★ 축 공식: SMPL 은 관절마다 x→y→z 힌지 3개를 쌓으므로 y(굴곡) 힌지의 월드 축은

    n̂ = R_parent · Rx(q_x) · ŷ

이다. 흔히 쓰는 `R_child · ŷ` 를 쓰면 z 회전만큼 틀려 같은 관절의 x/z DOF 로
토크가 새어나간다(검증 스크립트에서 qacc 오차 최대 103).

실물 대응: 모터는 굴곡 1축만 구동한다(`exo_actuators` 가 `*_y` 만 지정). 슈트
geom 은 팔다리에 강체 용접돼 있어 사람 관절의 3축을 전부 따라가는데, 실제 1축
힌지 하드웨어는 스트랩·프레임 유연성이 그 차이를 흡수한다 — 이상화임을 명시.

obs 규약 (원격의 27=23+4 수법을 69+4 로)
----------------------------------------
`_current_raw_action` 과 state_history 의 actions 버퍼를 **71 = 사람 69 + 모터 2**
로 재할당한다. 따라서

  · ActionNet 의 `previous_actions` = 71 → 사람의 직전 의도 + 자기 직전 토크를 함께 본다
  · 동결 사람에게는 `prev[:, :69]` 만 잘라서 준다 → Stage 1 학습 때와 **완전히 동일**한
    69 차원 규약 (원격이 장력 4 차원을 −10 으로 패치한 것과 같은 자리)

보상 (원격 자산 이식 + 우리 수정 2건)
-------------------------------------
    r = 기존 mimic 트래킹 보상 (부모)
        + λ_assist  · Σ_i sqrt(min(P_i, cap))     ① 보조 극대화 (양수, 주항)
        − λ_anti    · mean(relu(−P_i))            ② 모션 방해 억제
        − λ_hipknee · Σ_j w_j · τ_human,j²        ③ 사람 hip/knee 힘 저감 (목적 직결)
        − λ_smooth  · mean(Δτ_exo²)               ④ 토크 진동 억제
        − λ_dead    · Σ_i relu(floor − EMA_i)     ⑤ 죽은 모터 방지
        − λ_track   · relu(gt_err − base)         ⑥ 전체 모션 근접 (제약)
        − λ_joint   · relu(joint_err − base)      ⑦ 부분 관절 근접 (제약)

P_i = τ_exo,i · ω_i (모터 기계적 일률 [W]). 양수 = 관절에 에너지를 넣음(보조),
음수 = 관절을 제동(방해). 관절 직결이라 케이블처럼 경로 기하를 풀 필요가 없다.

★ 원격과 다른 점 ① — **좌우대칭 페널티(λ_sym)를 쓰지 않는다** (2026-07-29 사용자 지시)
원격은 대칭 모션만 골라 학습해 `cable_symmetry_cost` 가 유효한 정칙화였다. 우리
7 클립에는 `38_04`(선회), `walk_and_stop`(정지)처럼 **좌우가 다르게 일해야 정상인**
모션이 섞여 있다. 대칭을 강제하면 정책이 레퍼런스 모션을 거스른다. 좌우 편차는
페널티가 아니라 로깅 지표(`env/assist_lr_ratio`)로만 감시한다.
"한쪽 모터만 일하는" 퇴화는 대칭항 대신 **⑤ 죽은 모터 페널티**가 막는다 — 각
모터가 *언젠가는* 쓰이길 요구하되 매 순간 좌우 같기를 요구하지는 않으므로
비대칭 모션과 충돌하지 않는다.

★ 원격과 다른 점 ② — **트래킹 항(⑥⑦)을 넣는다**
원격은 "사람이 알아서 추종하니 불필요"로 정리했지만 그건 대칭·잘 추종되는 walk
전제였다. 사람이 동결이라 보조를 스스로 보정할 수 없어 보조가 나쁘면 트래킹은
**악화만** 한다. 절대값이 아니라 **보조 0 기준선 대비 초과분**(hinge)에만 벌점을
주어 "보조가 트래킹을 해쳤는가"만 잡는다. 기준선은 `assist_scale=0` 실측값
(`config.tracking_baseline`)이며 미지정이면 두 항은 **자동 비활성**된다.

보조 ON/OFF 비교 (학습 후 필수 산출물)
--------------------------------------
`config.assist_scale` 하나로 모터 출력을 껐다 켠다 — **코드 경로가 갈리지 않으므로**
같은 env·같은 동결 사람·같은 ActionNet 으로 정직한 ablation 이 된다.
`scripts/ablation_assist.py` 가 이 스위치만 바꿔 3-way 비교를 만든다:
맨몸 / 슈트+보조0 / 슈트+ActionNet.
"""

import re

import os
import torch
from tensordict import TensorDict
from torch import Tensor

from protomotions.envs.base_env.env import BaseEnv
from protomotions.utils.rotations import quat_rotate

# ── 기본 계수 (원격 E8 = 최선 기록 −5.05% 설정을 출발점으로) ──────────────
_DEFAULT_LAMBDA_ASSIST = 1.0
_DEFAULT_LAMBDA_ANTI = 0.12          # E8: 0.04→0.12 는 부작용 없는 순수 이득
_DEFAULT_LAMBDA_HIPKNEE = 0.003      # E6: 저감 크기의 주 지렛대 (0.001→0.003)
_DEFAULT_LAMBDA_SMOOTH = 0.02
_DEFAULT_LAMBDA_DEAD = 0.1           # E: 0.3 은 과했음 (v2.6.2 에서 0.1 로)
# ★ 트래킹 λ 는 **반드시 측정에서 역산**한다. 아래 기본값은 자리표시자일 뿐이고,
#   실제 스케일은 O(100) / O(50) 이다 (2026-07-29 보조 0 실측: hip/knee 20.92 Nm,
#   gt_err 0.0220 m, joint_err 0.0509 m → λ_track≈116, λ_joint≈50).
#   근거: "트래킹 5% 악화"와 "토크 5% 저감"이 상쇄되도록 맞춘다.
#     토크 이득 = λ_hipknee·hk²·(1−0.95²) = 0.003·20.92²·0.0975 ≈ 0.128
#     λ_track = 0.128 / (0.05·gt_err),  λ_joint = 0.128 / (0.05·joint_err)
#   `scripts/ablation_assist.py --assist off` 이 이 값을 계산해 출력한다.
#   tracking_baseline 이 None 이면 두 항은 비활성이므로 λ 값 자체는 무해하다.
_DEFAULT_LAMBDA_TRACK = 116.0        # hinge 라 기준선 이내면 0
_DEFAULT_LAMBDA_JOINT = 50.0

_ASSIST_POWER_CAP_W = 6.0            # E8. 상한 위에서는 펌핑 기울기 0
_DEAD_ASSIST_FLOOR_W = 0.3           # 활성 수준의 ~20%
_DEAD_PENALTY_GT_ERR_GATE = 0.25     # 트래킹 건강할 때만 압력 (v2.6.3)
_ASSIST_EMA_ALPHA = 0.02             # ~2.5s / 약 2 보행주기 (20fps)

# hip/knee τ² 가중치 — 시상면(굴곡) 집중. E5(시상면 가중)+E6(3배) 결합인 E7 의 배분.
# 모터가 굴곡축만 구동하므로 굴곡을 무겁게, 외전/회전은 곁가지로 둔다.
_HIPKNEE_WEIGHTS = {"Hip_y": 1.0, "Knee_y": 0.5, "Hip_x": 0.2, "Hip_z": 0.2,
                    "Knee_x": 0.2, "Knee_z": 0.2}
_HIPKNEE_DOF_RE = re.compile(r"^[LR]_(Hip|Knee)_[xyz]$")
_ARM_DOF_RE = re.compile(r"^[LR]_(Shoulder|Elbow|Wrist|Thorax)_[xyz]$")

_HUMAN_NUM_ACTIONS = 69


class FrozenHumanExoEnv(BaseEnv):
    """동결 HumanNet 이 69 DOF PD 를 채우고, 학습 정책은 모터 토크 4개만 낸다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config

        # ── 모터 정의: (이름, 구동 DOF, 자식 body, 부모 body, 피크 토크) ──
        motors = getattr(cfg, "exo_motors", None)
        if not motors:
            raise ValueError(
                "FrozenHumanExoEnv 는 config.exo_motors 가 필요합니다 "
                "(mlp env_config() 에서 setattr 주입 — exosuit_spec 의 exo_actuators 유래)"
            )
        dof_names = list(self.robot_config.kinematic_info.dof_names)
        body_names = list(self.robot_config.kinematic_info.body_names)
        self._motor_names = [m[0] for m in motors]
        self.num_motors = len(motors)

        y_idx, x_idx, child_idx, parent_idx, peaks = [], [], [], [], []
        for name, dof, child, parent, peak in motors:
            if not dof.endswith("_y"):
                raise ValueError(f"{name}: 모터는 굴곡 1축(_y)만 구동합니다 — got {dof}")
            j = dof_names.index(dof)
            # 같은 body 의 직전 힌지가 x축 (SMPL 은 x→y→z 순서로 쌓음)
            if dof_names[j - 1] != dof.replace("_y", "_x"):
                raise ValueError(
                    f"{name}: {dof} 직전 DOF 가 {dof_names[j-1]} — x→y→z 힌지 순서 가정 위반"
                )
            y_idx.append(j)
            x_idx.append(j - 1)
            child_idx.append(body_names.index(child))
            parent_idx.append(body_names.index(parent))
            peaks.append(float(peak))

        dev = self.device
        self._m_y = torch.tensor(y_idx, device=dev, dtype=torch.long)
        self._m_x = torch.tensor(x_idx, device=dev, dtype=torch.long)
        self._m_child = torch.tensor(child_idx, device=dev, dtype=torch.long)
        self._m_parent = torch.tensor(parent_idx, device=dev, dtype=torch.long)
        self._m_peak = torch.tensor(peaks, device=dev, dtype=torch.float)
        # 좌/우 모터 그룹 — 구동 DOF 이름의 L_/R_ 접두로 판정(모터 나열 순서 무관).
        # 로깅 전용(비대칭 모션이 있어 대칭은 페널티가 아니다 — 모듈 docstring)
        sides = [dof_names[j][:2] for j in y_idx]
        self._left_m = torch.tensor(
            [i for i, s in enumerate(sides) if s == "L_"], device=dev, dtype=torch.long
        )
        self._right_m = torch.tensor(
            [i for i, s in enumerate(sides) if s == "R_"], device=dev, dtype=torch.long
        )

        # ── hip/knee τ² 가중 (합=1 정규화 → λ 스케일이 가중치와 무관) ──
        hk = [i for i, n in enumerate(dof_names) if _HIPKNEE_DOF_RE.match(n)]
        if len(hk) != 12:
            raise ValueError(f"hip/knee DOF 12개를 기대했으나 {len(hk)}개: "
                             f"{[dof_names[i] for i in hk]}")
        self._hk_idx = torch.tensor(hk, device=dev, dtype=torch.long)
        wmap = getattr(cfg, "hipknee_weight_map", None) or _HIPKNEE_WEIGHTS
        w = torch.tensor(
            [next((float(v) for k, v in wmap.items() if dof_names[i].endswith(k)), 1.0)
             for i in hk], device=dev, dtype=torch.float,
        )
        self._hk_w = w / w.sum()

        # [ETRI 2026-08-28] **힙 전용** 지표 인덱스.
        #   exosuitHS 는 힙 모터 2개뿐이고 무릎 하드웨어가 없다. 그런데 판정 지표가
        #   hip+knee 가중합(hipknee_torque_abs_mean)뿐이라 "모터가 힙 부담을 얼마나
        #   덜어줬는가"를 직접 볼 수 없었다(무릎이 섞여 희석). 모터 정격이 23.7 Nm 인데
        #   hip+knee 합산값은 35 Nm 대라 비교 기준으로도 오해를 부른다.
        #   → 힙 전체(6 DOF)와 **모터가 실제로 구동하는 Hip_y 좌우**를 따로 기록한다.
        _hip = [i for i, n in enumerate(dof_names) if re.match(r"^[LR]_Hip_[xyz]$", n)]
        _hipy = [i for i, n in enumerate(dof_names) if n in ("L_Hip_y", "R_Hip_y")]
        if len(_hip) != 6 or len(_hipy) != 2:
            raise ValueError(f"힙 DOF 6개/Hip_y 2개를 기대했으나 {len(_hip)}/{len(_hipy)}개")
        self._hip_idx = torch.tensor(_hip, device=dev, dtype=torch.long)
        self._hipy_idx = torch.tensor(_hipy, device=dev, dtype=torch.long)
        self._arm_idx = torch.tensor(
            [i for i, n in enumerate(dof_names) if _ARM_DOF_RE.match(n)],
            device=dev, dtype=torch.long,
        )

        # ── 계수 (config 로 개별 오버라이드 가능) ──
        self.assist_scale = float(getattr(cfg, "assist_scale", 1.0))
        # "joint"(기본, 학습에 쓴 이상적 정렬) | "frame"(실물형 부모 고정축)
        self.motor_axis_mode = str(getattr(cfg, "motor_axis_mode", "joint"))
        if self.motor_axis_mode not in ("joint", "frame"):
            raise ValueError(f"motor_axis_mode 는 joint|frame — got {self.motor_axis_mode}")
        self.lambda_assist = _DEFAULT_LAMBDA_ASSIST
        self.lambda_anti = _DEFAULT_LAMBDA_ANTI
        self.lambda_hipknee = _DEFAULT_LAMBDA_HIPKNEE
        self.lambda_smooth = _DEFAULT_LAMBDA_SMOOTH
        self.lambda_dead = _DEFAULT_LAMBDA_DEAD
        self.lambda_track = _DEFAULT_LAMBDA_TRACK
        self.lambda_joint = _DEFAULT_LAMBDA_JOINT
        self.assist_power_cap_w = _ASSIST_POWER_CAP_W
        self.dead_assist_floor_w = _DEAD_ASSIST_FLOOR_W
        self.dead_penalty_gt_err_gate = _DEAD_PENALTY_GT_ERR_GATE
        self.assist_ema_alpha = _ASSIST_EMA_ALPHA
        # 트래킹 기준선: assist_scale=0 실측값 {"gt_err":…, "joint_err":…}.
        # 미지정이면 ⑥⑦ 비활성 (절대 트래킹 벌점은 목적이 아니므로 추측하지 않는다)
        self.tracking_baseline = getattr(cfg, "tracking_baseline", None)

        for name, value in (getattr(cfg, "exo_env_overrides", None) or {}).items():
            if not hasattr(self, name):
                raise AttributeError(f"exo_env_overrides: unknown attr '{name}'")
            setattr(self, name, value)

        # ── raw action 버퍼를 69 + 모터수 로 재할당 (모듈 docstring) ──
        self._composed_dim = _HUMAN_NUM_ACTIONS + self.num_motors
        self._current_raw_action = torch.zeros(
            self.num_envs, self._composed_dim, dtype=torch.float, device=dev
        )
        if self.state_history is not None:
            h = self.state_history.actions.shape[1]
            self.state_history.actions = torch.zeros(
                self.num_envs, h, self._composed_dim, dtype=torch.float, device=dev
            )
        # obs 버퍼도 함께 넓힌다 — super().__init__() 이 이미 69 기준으로 할당했다
        buf = getattr(self, "_observation_buffer", None)
        if buf is not None and "previous_actions" in buf:
            old = buf["previous_actions"]
            steps = old.shape[-1] // _HUMAN_NUM_ACTIONS
            if old.shape[-1] != steps * _HUMAN_NUM_ACTIONS:
                raise ValueError(
                    f"previous_actions obs 차원 {old.shape[-1]} 이 사람 DOF "
                    f"{_HUMAN_NUM_ACTIONS} 의 배수가 아닙니다 — 규약 확인 필요"
                )
            buf["previous_actions"] = torch.zeros(
                *old.shape[:-1], steps * self._composed_dim,
                dtype=old.dtype, device=old.device,
            )

        self._frozen_actor = self._load_frozen_actor(
            getattr(cfg, "frozen_human_ckpt", None)
        )
        self._prev_tau_exo = None
        self._assist_ema = None
        # [2026-09-01] 보조 토크 사이드카 로그.
        #   추론 롤아웃 .motion 에는 사람 관절 69개(dof_forces)만 담기고 슈트 모터 2개는
        #   빠진다. 렌더에서 "지금 얼마나 보조 중인가"를 색으로 보이려면 그 값이 필요하다.
        #   기록기(protomotions/…/record.py)는 수정 대상이 아니므로 env 가 직접 남긴다.
        #   EXO_TORQUE_LOG=<경로> 일 때만 동작한다(학습에는 영향 없음).
        self._exo_log_path = os.environ.get("EXO_TORQUE_LOG") or None
        self._exo_log = []
        if self._exo_log_path is not None:
            # 주기 저장만으로는 마지막 몇 프레임이 빠진다(10스텝 주기 → 최대 9개).
            # 렌더가 롤아웃과 프레임을 1:1 로 맞추므로 종료 시 반드시 마무리한다.
            import atexit
            atexit.register(self._flush_exo_log)
        self._wrench = torch.zeros(
            self.num_envs, len(body_names), 6, dtype=torch.float, device=dev
        )
        # PD 게인 (τ_human 추정용 — BUILT_IN_PD 는 get_dof_forces() 가 0 을 준다)
        info = self.robot_config.control.control_info
        self._pd_kp = torch.tensor([info[n].stiffness or 0.0 for n in dof_names], device=dev)
        self._pd_kd = torch.tensor([info[n].damping or 0.0 for n in dof_names], device=dev)
        self._pd_effort = torch.tensor(
            [info[n].effort_limit or torch.inf for n in dof_names], device=dev
        )

        self._restore_collider_visibility()

        print(
            f"[FrozenHumanExoEnv] 모터 {self.num_motors}개 "
            f"{list(zip(self._motor_names, peaks))} Nm, assist_scale={self.assist_scale}, "
            f"raw action {self._composed_dim} = 사람 {_HUMAN_NUM_ACTIONS} + 모터 {self.num_motors}, "
            f"axis={self.motor_axis_mode}, "
            f"트래킹 기준선={'없음(⑥⑦ 비활성)' if not self.tracking_baseline else self.tracking_baseline}"
        )

    def _restore_collider_visibility(self) -> None:
        """★ Newton 은 **visual geom(contype=0)이 있는 body 의 콜라이더에서 VISIBLE
        플래그를 자동으로 지운다.** 슈트를 질량 전용(`suit_collision=False` →
        `contype=0 conaffinity=0`)으로 만들었으므로 슈트가 붙은 body —
        Pelvis / L_Hip / R_Hip / L_Knee / R_Knee / L_Ankle / R_Ankle / Torso —
        의 **인체 캡슐이 렌더에서 사라진다**. 슈트 없는 팔·머리만 보이는 영상이 된다
        (2026-07-30 추론 녹화에서 확인: "하체 plain 이 숨겨져 있네요").

        `shape_flags` 는 **렌더 전용**이라 물리·질량·접촉에 아무 영향이 없다.
        전 shape 에 VISIBLE 을 켜서 몸과 슈트를 함께 보이게 되돌린다.
        compare 스크립트(`compare_smpl_exosuit_train_eval_newton.py`)는 같은 이유로
        보일/감출 파트를 비트로 직접 지정한다 — 여기서는 전부 표시가 목적이다.

        헤드리스 학습에서도 무해하므로 조건 없이 실행하고, Newton 이 아닌 백엔드나
        API 변경 시에는 조용히 건너뛴다(시각화 문제로 학습을 죽이지 않는다).
        """
        model = getattr(self.simulator, "model", None)
        flags = getattr(model, "shape_flags", None)
        if flags is None:
            return
        try:
            import newton as nw

            visible = int(nw.ShapeFlags.VISIBLE)
            sf = flags.numpy()
            n = int((sf & visible == 0).sum())
            if n:
                flags.assign(sf | visible)
                # ★ 이것만으로는 화면이 안 바뀐다. `viewer.set_model(model)` 이
                #   **시뮬레이터 초기화 중**(newton/simulator.py:722)에 이미 호출돼
                #   viewer 가 플래그를 스냅샷했기 때문이다. 다시 불러 갱신한다.
                viewer = getattr(self.simulator, "viewer", None)
                refreshed = False
                if viewer is not None and hasattr(viewer, "set_model"):
                    viewer.set_model(model)
                    refreshed = True
                print(f"[FrozenHumanExoEnv] 렌더 복원: VISIBLE 해제된 shape {n}개 재표시"
                      f"{' + viewer 갱신' if refreshed else ' (viewer 없음 — headless)'}")
        except Exception as e:  # noqa: BLE001 — 시각화 실패가 학습을 막지 않는다
            print(f"[FrozenHumanExoEnv] VISIBLE 복원 건너뜀: {e}")

    # ── 동결 사람 ────────────────────────────────────────────────────────
    def _load_frozen_actor(self, ckpt_path):
        """Stage 1 체크포인트에서 actor 의 **mu 네트워크만** 복원한다.

        이 저장소에는 `load_pretrained_model_module`(신버전 API)이 없어 직접 만든다.
        저장된 설정으로 실제 클래스를 세우므로 레이어 인덱스를 하드코딩하지 않는다
        (원격은 `_ManualFrozenActor` 로 원시 텐서를 직접 곱했는데, 구조가 바뀌면
        조용히 틀리는 방식이라 채택하지 않았다).

        mu 만 쓰는 이유: 우리는 `mean_action` 만 필요하다. PPOActor 전체를 세우면
        매 스텝 Normal 분포에서 표본을 뽑는데, 동결 사람은 **결정적**이어야 한다
        (ablation 에서 보조 ON/OFF 두 롤아웃이 같은 사람이어야 비교가 성립).
        """
        import pathlib

        if not ckpt_path:
            raise ValueError(
                "FrozenHumanExoEnv 는 config.frozen_human_ckpt 가 필요합니다 "
                "(S1 산출 mimic_smpl/output_newton_s1_exosuitHS_motions36/last.ckpt)"
            )
        from protomotions.utils.hydra_replacement import get_class

        ckpt_path = pathlib.Path(ckpt_path)
        cfg_path = ckpt_path.parent / "resolved_configs.pt"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{cfg_path} 없음 — actor 구조를 복원할 수 없습니다. "
                "동결 체크포인트는 학습 산출 폴더째로 두어야 합니다."
            )
        actor_cfg = torch.load(cfg_path, map_location="cpu", weights_only=False)[
            "agent"
        ].model.actor
        mu = get_class(actor_cfg.mu_model._target_)(config=actor_cfg.mu_model)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
        pre = "_actor.mu."
        sub = {k[len(pre):]: v for k, v in state.items() if k.startswith(pre)}
        if not sub:
            raise ValueError(f"{ckpt_path} 에 '{pre}*' 가중치가 없습니다")

        # obs normalizer 는 지연 초기화(RunningMeanStd(shape=None))라 **버퍼가 아직
        # 없다** → 그대로 load_state_dict 하면 mean/var/count 가 unexpected 로 버려지고
        # 정규화가 항등이 되어 동결 정책이 조용히 망가진다. 체크포인트의 shape 로
        # 미리 버퍼를 만든다.
        norm_key = "norm.running_obs_norm.mean"
        if norm_key in sub:
            rms = mu.norm.running_obs_norm
            rms._create_buffers(tuple(sub[norm_key].shape), "cpu")
            rms._initialized = True

        # LazyLinear 는 state_dict 로부터 shape 를 확정한다(_lazy_load_hook)
        missing, unexpected = mu.load_state_dict(sub, strict=False)
        # count 같은 버퍼는 없어도 되지만 weight/bias 누락은 치명적
        bad = [k for k in missing if k.endswith(("weight", "bias"))]
        if bad or unexpected:
            raise ValueError(f"동결 actor 로드 불일치 — missing={bad} unexpected={unexpected}")

        mu = mu.to(self.device).eval().requires_grad_(False)
        self._frozen_mu_key = actor_cfg.mu_key
        n_obs = sub["norm.running_obs_norm.mean"].numel()
        print(f"[FrozenHumanExoEnv] 동결 HumanNet 로드: {ckpt_path} (obs {n_obs} 차원)")
        return mu

    def _frozen_human_raw_action(self) -> Tensor:
        """직전 obs 로 동결 정책의 raw action [E, 69] (결정적, 탐색 노이즈 없음)."""
        obs = dict(self.get_obs())
        prev = obs["previous_actions"]
        # [E, H*71] → 사람이 Stage 1 에서 본 규약대로 앞 69 만 잘라 [E, H*69] 로
        prev = prev.view(prev.shape[0], -1, self._composed_dim)[..., :_HUMAN_NUM_ACTIONS]
        obs["previous_actions"] = prev.reshape(prev.shape[0], -1)
        td = TensorDict(obs, batch_size=[prev.shape[0]], device=self.device)
        with torch.no_grad():
            return self._frozen_actor(td)[self._frozen_mu_key]

    # ── 모터 토크 → 짝힘 ─────────────────────────────────────────────────
    def _motor_axis_world(self) -> Tensor:
        """모터 토크를 넣을 월드 축 n̂ [E, M, 3].

        `motor_axis_mode` (config, 기본 `"joint"`):

        · **`"joint"`** — 사람 관절의 순간 굴곡축 `R_parent · Rx(q_x) · ŷ`.
          body 오프셋에 quat 이 없으므로 자식 body 의 관절 전 프레임 = 부모 회전.
          `Rx(q_x)·ŷ = (0, cos q_x, sin q_x)` 이라 회전행렬이 필요 없다.
          (검증: `scripts/verify_exo_axis.py` — MuJoCo `data.xaxis` 와 오차 ~1e-16)
          **완벽히 정렬된 이상적 작동기** — 학습에 쓴 설정.

        · **`"frame"`** — 부모 body 에 **고정된** 축 `R_parent · ŷ` (Rx(q_x) 없음).
          실물에 가깝다: 힙 모터는 골반의 ㄷ 프레임에, 무릎 모터는 허벅지 박스에
          달려 있어 축이 부모 기준으로 고정이고 사람의 외전(q_x)을 따라가지 못한다.
          어긋남 각이 정확히 `q_x` 이므로 유효 굴곡 성분은 `τ·cos(q_x)`, 나머지
          `τ·sin(q_x)` 는 외전축으로 새는 **기생 토크**가 된다.
          검증 4 (학습계획 §8) 에서 이 모드로 성능 저하를 직접 측정한다.
        """
        body_rot = self.context.current.rigid_body_rot          # [E, B, 4] xyzw
        parent_rot = body_rot[:, self._m_parent]                 # [E, M, 4]
        if getattr(self, "motor_axis_mode", "joint") == "frame":
            local = torch.zeros(
                *parent_rot.shape[:-1], 3, device=parent_rot.device, dtype=parent_rot.dtype
            )
            local[..., 1] = 1.0                                  # ŷ 고정
        else:
            q_x = self.context.current.dof_pos[:, self._m_x]     # [E, M]
            local = torch.stack(
                [torch.zeros_like(q_x), torch.cos(q_x), torch.sin(q_x)], dim=-1
            )                                                    # [E, M, 3]
        return quat_rotate(parent_rot, local, w_last=True)

    def _flush_exo_log(self) -> None:
        """보조 토크 로그를 사이드카 파일로 저장. 렌더(render_exosuit_isaacsim.py)가 읽는다."""
        if not self._exo_log:
            return
        torch.save(
            {
                "tau_exo": torch.stack(self._exo_log),          # [T, M] N·m (부호 있음)
                "motor_peak": self._m_peak.detach().float().cpu(),   # [M] 정격
                "motor_names": list(self._motor_names),
            },
            self._exo_log_path,
        )

    def _apply_motor_wrench(self, tau: Tensor) -> Tensor:
        """자식 +τ·n̂ / 부모 −τ·n̂ 짝힘을 시뮬레이터에 건다. 축 n̂ 반환."""
        axis = self._motor_axis_world()                          # [E, M, 3]
        torque = tau.unsqueeze(-1) * axis                        # [E, M, 3]
        self._wrench.zero_()
        # 여러 모터가 같은 body 를 공유한다(골반은 힙 2개의 반력, 대퇴골은 힙 작용 +
        # 무릎 반력). index_add_ 로 누적해야 한다 — 인덱싱 대입은 마지막 것만 남는다.
        self._wrench[..., 3:6].index_add_(1, self._m_child, torque)
        self._wrench[..., 3:6].index_add_(1, self._m_parent, -torque)
        # detach() 필수 — 학습 rollout 은 @torch.no_grad() 안이라 문제가 없지만
        # inference_agent.py 는 no_grad 로 감싸지 않는다. 그러면 tau → torque →
        # _wrench 가 grad 를 달고, IsaacLab 의 wrench_composer 가 warp 커널에 넘길 때
        # __cuda_array_interface__ 를 못 얻어 죽는다(추론 렌더 전멸, 2026-08-31).
        # 시뮬레이터쪽 형제 경로(_apply_simulator_pd_targets/_torques)도 같은 이유로
        # .detach() 한다. detach 는 저장소를 공유하므로 매 substep 재주입도 그대로 동작.
        self.simulator.set_external_body_wrench(self._wrench.detach())
        return axis

    def _human_torque_abs(self, pd_targets: Tensor) -> Tensor:
        """관절별 사람 PD 토크 |τ| [E, 69]. BUILT_IN_PD 라 명시 공식으로 추정."""
        s = self.simulator.get_dof_state()
        tau = self._pd_kp * (pd_targets - s.dof_pos) - self._pd_kd * s.dof_vel
        return tau.clamp(-self._pd_effort, self._pd_effort).abs()

    # ── step ─────────────────────────────────────────────────────────────
    def step(self, action: Tensor):
        if action.shape[-1] != self.num_motors:
            raise ValueError(
                f"FrozenHumanExoEnv 는 [{self.num_motors}]차 모터 액션을 기대합니다 — "
                f"got {tuple(action.shape)} (ActionNet num_out 확인)"
            )
        self.extras = {}
        self._current_context = None
        self._current_noisy_obs = None

        # 1) 동결 사람의 raw 69 (직전 obs 기준 — 이 스텝의 obs 는 아직 갱신 전)
        human_raw = self._frozen_human_raw_action()

        # 2) 모터 토크: tanh → ±피크. assist_scale=0 이면 정확히 0 (ablation)
        tau_exo = torch.tanh(action) * self._m_peak * self.assist_scale   # [E, M]

        # 2-b) 사이드카 로그 — 카메라 타깃(env 0)의 모터 토크를 프레임마다 남긴다.
        #      렌더가 프레임을 1:1 로 맞춰 읽으므로 롤아웃과 같은 순서·길이여야 한다.
        if self._exo_log_path is not None:
            self._exo_log.append(tau_exo[0].detach().float().cpu().clone())
            if len(self._exo_log) % 10 == 0:      # 중간에 끊겨도 쓸 수 있게 주기 저장
                self._flush_exo_log()

        # 3) previous_actions 용 합성 raw 71 기록
        self._current_raw_action[:, :_HUMAN_NUM_ACTIONS] = human_raw
        self._current_raw_action[:, _HUMAN_NUM_ACTIONS:] = action

        # 4) 사람 액션 → PD 목표
        action_dict = self._process_action(human_raw, self.context)
        processed = action_dict["processed_action"].clone()
        self._current_processed_action[:] = processed

        # 5) 짝힘 주입 후 물리 (Newton 은 substep 마다 body_f 를 지우므로 시뮬레이터가
        #    _push_external_body_wrench() 로 매 substep 재주입한다)
        self._apply_motor_wrench(tau_exo)
        self.simulator.step(processed, markers_callback=self.get_markers_state)
        self.post_physics_step()

        # ── 보조 지표 ────────────────────────────────────────────────────
        omega = self.simulator.get_dof_state().dof_vel[:, self._m_y]      # [E, M] rad/s
        power = tau_exo * omega                                          # [E, M] W
        assist_p = power.clamp(min=0.0)          # 관절에 에너지 투입 = 보조
        anti_p = (-power).clamp(min=0.0)         # 관절 제동 = 모션 방해

        tau_h = self._human_torque_abs(processed)                        # [E, 69]
        hk_abs = tau_h[:, self._hk_idx]                                  # [E, 12]
        hk_cost = (hk_abs.pow(2) * self._hk_w).sum(dim=-1)               # [E]

        # ① 보조: 모터별 concave(sqrt) + 상한. sqrt 합은 같은 총출력이면 여러 모터에
        #    고르게 퍼질 때 최대 → 한 모터가 독식하고 나머지가 무임승차하는 퇴화를 막는다.
        assist_rew = torch.sqrt(
            assist_p.clamp(max=self.assist_power_cap_w) + 1e-6
        ).sum(dim=-1)

        # ⑤ 죽은 모터: EMA 가 floor 미달인 모터마다 상시 압력. 트래킹이 건강할 때만
        #    (보행이 무너지는 중이면 회복 기울기와 싸우지 않도록)
        if self._assist_ema is None:
            self._assist_ema = assist_p.detach().clone()
        else:
            a = self.assist_ema_alpha
            self._assist_ema = a * assist_p.detach() + (1.0 - a) * self._assist_ema
        ref_pos = self.context.mimic.ref_state.rigid_body_pos
        gt_err = (self.context.current.rigid_body_pos - ref_pos).norm(dim=-1).mean(dim=-1)
        joint_err = (self.context.current.rigid_body_pos - ref_pos).norm(dim=-1).max(dim=-1).values
        healthy = (gt_err < self.dead_penalty_gt_err_gate).float()
        dead_pen = (self.dead_assist_floor_w - self._assist_ema).clamp(min=0.0).sum(-1) * healthy

        # ④ 진동
        if self._prev_tau_exo is None:
            self._prev_tau_exo = tau_exo.detach().clone()
        smooth_cost = (tau_exo - self._prev_tau_exo).pow(2).mean(dim=-1)
        self._prev_tau_exo = tau_exo.detach().clone()

        # ⑥⑦ 트래킹 근접 — 보조 0 기준선 초과분만 (기준선 없으면 비활성)
        track_pen = torch.zeros_like(gt_err)
        joint_pen = torch.zeros_like(gt_err)
        if self.tracking_baseline:
            track_pen = (gt_err - float(self.tracking_baseline["gt_err"])).clamp(min=0.0)
            joint_pen = (joint_err - float(self.tracking_baseline["joint_err"])).clamp(min=0.0)

        self.rew_buf += (
            self.lambda_assist * assist_rew
            - self.lambda_anti * anti_p.mean(dim=-1)
            - self.lambda_hipknee * hk_cost
            - self.lambda_smooth * smooth_cost
            - self.lambda_dead * dead_pen
            - self.lambda_track * track_pen
            - self.lambda_joint * joint_pen
        )

        # ── 로깅 (TensorBoard: env/<key>_mean) ──────────────────────────
        ex = self.extras
        for i, name in enumerate(self._motor_names):
            ex[f"tau_{name}"] = tau_exo[:, i].abs()
            ex[f"power_{name}"] = assist_p[:, i]
        ex["exo_torque_abs_mean"] = tau_exo.abs().mean(dim=-1)
        ex["exo_torque_abs_max"] = tau_exo.abs().max(dim=-1).values
        ex["assist_power"] = assist_p.mean(dim=-1)
        ex["anti_assist_power"] = anti_p.mean(dim=-1)
        ex["assist_reward"] = assist_rew
        ex["dead_penalty"] = dead_pen
        ex["smoothness_cost"] = smooth_cost
        # 판정 지표: 이 두 개의 보조 0 대비 변화가 게이트 2 의 본체
        ex["hipknee_torque_abs_mean"] = hk_abs.mean(dim=-1)
        ex["hipknee_torque_cost"] = hk_cost
        # [ETRI 2026-08-28] 힙 전용 — 모터가 직접 부담을 더는 대상. 모터 정격 23.7 Nm 과
        #   같은 축에서 비교되는 값은 hip_y 쪽이다(hipknee 합산은 무릎이 섞여 희석된다).
        _tau_abs = tau_h.abs()
        ex["hip_torque_abs_mean"] = _tau_abs[:, self._hip_idx].mean(dim=-1)    # 힙 6 DOF
        ex["hip_y_torque_abs_mean"] = _tau_abs[:, self._hipy_idx].mean(dim=-1)  # 모터 구동축
        ex["arm_torque_abs_mean"] = tau_h[:, self._arm_idx].mean(dim=-1)  # 우회 감시
        ex["gt_err"] = gt_err
        ex["joint_err_max"] = joint_err
        ex["track_penalty"] = track_pen
        ex["joint_penalty"] = joint_pen
        # 좌우 편차 — 페널티 아님, 감시만 (비대칭 모션에서는 비대칭이 정상)
        lr_l = assist_p[:, self._left_m].sum(-1)
        lr_r = assist_p[:, self._right_m].sum(-1)
        ex["assist_lr_ratio"] = (lr_l - lr_r) / (lr_l + lr_r + 1e-6)

        return self.get_obs(), self.rew_buf, self.reset_buf, self.terminate_buf, ex

    def reset(self, env_ids=None, **kwargs):
        # ★ **kwargs 필수 — evaluator 는 reset(env_ids, sample_flat=…) 처럼 부가
        #   인자를 넘긴다(mimic_evaluator.py `_get_reset_kwargs()`). 이를 받지 않으면
        #   첫 전체 eval 시점에 TypeError 로 죽는다 (2026-07-30 epoch 199 크래시).
        out = super().reset(env_ids, **kwargs)
        # 에피소드 경계에서 EMA/진동 이력을 남기면 리셋 직후 가짜 페널티가 생긴다
        if env_ids is None:
            self._prev_tau_exo = None
            self._assist_ema = None
        else:
            if self._prev_tau_exo is not None:
                self._prev_tau_exo[env_ids] = 0.0
            if self._assist_ema is not None:
                self._assist_ema[env_ids] = self.dead_assist_floor_w
        return out
