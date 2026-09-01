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
"""Stage 2' — 36모션 · 죽은 모터 페널티 제거 (exosuitHS, IsaacLab).

[ETRI 2026-08-28] `mlp_actionnet.py`(3클립 loop3) 에서 파생. **원본은 그대로 둔다.**
차이는 두 가지뿐:

  (a) **lambda_dead = 0** — 죽은 모터 페널티 해제.
      이 항은 각 모터의 보조일률 EMA 가 바닥(0.3W) 위에 있기를 **모터별로**
      요구한다(`(floor - ema).clamp(min=0).sum(-1)`). env 문서가 밝히듯 이것이
      좌우대칭 페널티(λ_sym, 2026-07-29 제거됨)를 대신해 "한쪽 모터 독식"을 막는
      장치다. 즉 **현재 코드에서 좌우 균형을 강제하는 유일한 항**이다.
      36모션에는 38_04(제자리 선회, 총회전 5,349°)처럼 좌우가 크게 다르게 일해야
      정상인 클립이 다수 섞여 있다. EMA 창이 ~2.5초(약 2 보행주기)라 지속 선회
      구간에서는 한쪽 모터가 정당하게 놀 수 있는데, 그 상태에 벌점이 붙는다.
      → 36모션에서는 해제한다. 대신 `env/assist_lr_ratio` 로 감시만 한다.
      ※ 부작용 감시: 모터가 아예 0 으로 죽을 수 있다. `env/exo_torque_abs_mean_mean`
        이 0 에 붙으면 이 해제가 원인이므로 되돌릴 것.

  (b) **TRACKING_BASELINE 을 36모션 실측값으로** — 3클립 값(0.02941/0.06308)을
      그대로 쓰면 hinge 가 항상 켜지거나 꺼져 트래킹 보호가 무력화된다.

학습량은 상한 없이(무제한) 돌린다 — 실행 스크립트 `run_IL_s2p_exosuitHS_motions36.sh` 참조.

──────────────────────────────────────────────────────────────────────────────
(이하 원본 설명)

Stage 2 — 동결 HumanNet + 외골격 ActionNet 학습 설정 (exosuitHS, IsaacLab).

[ETRI 2026-08-28] `mlp_newton_actionnet.py` 에서 파생. 원본과의 차이는 **두 가지뿐**이다 —
이 설정 파일 자체에는 시뮬레이터 특이 코드가 없다(물리 파라미터는 robot/simulator config 소관):
  (a) 동결 인체 체크포인트 기본값을 IL(IsaacLab) S1 산출물로 교체.
  (b) 환경변수 `FROZEN_HUMAN_CKPT` 로 덮어쓸 수 있게 함(실행 스크립트가 best 를 넘긴다).
원본 Newton 판은 그대로 두어 Newton 계보 재현성을 유지한다.

계획 ../학습계획.md §4 / env ../envs/frozen_human_exo_env.py

Stage 1 (`mimic/mlp.py`) 대비 변경점
------------------------------------
1. `EnvConfig._target_` = `FrozenHumanExoEnv` — 69 DOF PD 는 env 안의 **동결**
   Stage 1 정책이 채우고, 학습 정책은 **모터 토크 4차원만** 낸다.
2. actor `num_out` 69→2, 1024×6 → **512×3** (출력 2개뿐인 소형 ActionNet).
   `NUM_MOTOR_ACTIONS = len(EXO_MOTORS)` 이므로 모터 구성을 바꾸면 함께 따라간다.
   critic 1024×4 → 512×3. `in_keys` 는 Stage 1 과 동일 3종 — LazyLinear 라
   입력 크기는 자동 추론된다(`previous_actions` 가 69→71 로 커지는 것도 흡수).
3. evaluator `_target_` = `ExoActionNetMimicEvaluator` — eval actions 버퍼 69→모터수
   (원격이 이 지점에서 epoch 200 에 크래시했다).
   `failure_discount` 0.99 — 0.3^N 언더플로 → inf → multinomial CUDA assert 방지.
4. env 에 `frozen_human_ckpt` / `exo_motors` / `assist_scale` / `exo_env_overrides`
   주입 (EnvConfig 는 plain dataclass 라 커스텀 속성이 resolved_configs.pt 에 함께
   pickle 된다).

원격 9개 실험에서 확인된 출발점 (E8 = 최선 −5.05%)
--------------------------------------------------
시상면 가중 + `assist_power_cap_w` 6.0 + `λ_hipknee` 0.003 + `λ_anti` 0.12.
`λ_hipknee` 0.006(E9)은 **선을 넘는다** — 저감은 정체하고 출력·진동만 악화.

우리 수정 2건 (../학습계획.md §4.2)
-----------------------------------
· **좌우대칭 페널티 없음** — 7 클립에 `38_04`(선회)·`walk_and_stop` 처럼 좌우가
  다르게 일해야 정상인 모션이 섞여 있다. 한쪽 모터 독식은 대칭항이 아니라
  죽은 모터 페널티(EMA hinge)가 막는다.
· **트래킹 항 추가** — 사람이 동결이라 보조가 나쁘면 트래킹은 악화만 한다.
  `tracking_baseline`(보조 0 실측)의 **초과분만** 벌점. 미측정이면 자동 비활성이라
  1차 실행 전에 `scripts/ablation_assist.py --stage baseline` 으로 채워야 한다.

모션 선택
---------
1차는 **단일 클립** `walk_cmu_07_04_30s_aligned.motion`. 저감률이 판정 지표인데
7 클립을 섞으면 클립별 난이도 차이가 지표를 흐려 λ 조정의 인과를 못 본다
(원격 9개 실험도 전부 단일 클립). 보상에 대칭 가정이 없으므로 **게이트 2 통과 후
7 클립 확장 시 설정 변경이 필요 없다** (학습계획 §5.5).
"""

import argparse
import os

from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

_TASK = "tasks_for_smpl.mimic_smpl_exosuitHS"

# 동결 HumanNet: Stage 1 (슈트 4.700 kg 무게 적응) 산출물.
# 같은 폴더의 resolved_configs.pt 로 actor 구조를 복원하므로 폴더째 유지해야 한다.
# [ETRI 2026-08-28] IsaacLab 계보. A 계통(사전학습→S1) 승격본의 릴리즈 체크포인트.
#   · B 계통(S0→S1, output_IL_s0_s1_exosuitHS_motions36)이 준비되면 그쪽으로 바꿔도 된다.
#     현재 스케줄은 S2 를 B 보다 먼저 돌리므로 A 의 S1 을 쓴다.
#   · last.ckpt 가 아니라 **epoch_200.ckpt** 인 이유: success_rate 가 epoch 1 부터 포화해
#     score_based/last 는 드리프트된 늦은 epoch 을 담는다. gt_error 최저 저장본을 쓴다
#     (RETRAIN_GUIDE.md §보상 진단, pick_best_ckpt.py).
#   · 실행 스크립트가 `--overrides env.frozen_human_ckpt=<...>` 로 덮어쓴다.
FROZEN_HUMAN_CKPT = os.environ.get(
    "FROZEN_HUMAN_CKPT",
    "tasks_for_smpl/mimic_smpl/output_IL_s1_exosuitHS_motions36/epoch_200.ckpt",
)

# (모터, 구동 DOF, 자식 body=토크 +, 부모 body=반력 −, 피크 토크 Nm)
# **힙 2개만** — 무릎 이하 하드웨어가 없는 힙 전용 슈트.
# 반력 경로: 힙 모터는 허리밴드 ㄷ 프레임을 통해 골반으로 간다.
# 피크 = 실제 스펙 Unitree GO-M8010-6 최대 23.7 Nm
#   (~/Downloads/Unitree_GO-M8010-6_Specification.md, 30 rad/s, 530 g, 감속 6.33:1)
#   ※ 스펙에 **연속 정격이 없다** — 보행은 지속 동작이라 발열 한계 확인 필요.
_GO_M8010_6_MAX_TORQUE = 23.7
EXO_MOTORS = [
    ("exo_hip_l", "L_Hip_y", "L_Hip", "Pelvis", _GO_M8010_6_MAX_TORQUE),
    ("exo_hip_r", "R_Hip_y", "R_Hip", "Pelvis", _GO_M8010_6_MAX_TORQUE),
]

NUM_MOTOR_ACTIONS = len(EXO_MOTORS)

# 보조 스위치. 1.0 = 학습/추론, 0.0 = ablation 기준선 (코드 경로가 갈리지 않는다).
# 실행 시점 값이 resolved_configs.pt 에 구워지므로 각 run 은 자기 설정으로 재현된다.
#   EXO_ASSIST_SCALE=0 ... → 보조 0 기준선 측정
ASSIST_SCALE = float(os.environ.get("EXO_ASSIST_SCALE", "1.0"))

# 보상 계수. env 의 기본값을 덮어쓴다.
#
# ★ λ_assist = 0 / λ_anti = 0 — **둘 다 CR 실험에서 제거가 결정적이었다.**
#   · λ_assist(보조 일률 보상)는 사람 부담을 줄이는 대신 **와트 펌핑**을 유도했다
#     (hip/knee |τ| +17.3% 악화). 일률 τ·ω 는 "사람을 돕는다"와 다르다 — 모터가
#     관절에 에너지를 넣으면서 PD 목표에서 멀어지게 밀면 사람 토크가 오른다.
#     척도는 **사람 부담의 감소** 그 자체여야 한다. 일률은 로깅만.
#   · λ_anti(음의 일률 페널티)는 케이블 슈트에서 온 기준인데, **관절 직결 모터의
#     음의 일률은 정당한 보조 모드**다(무릎 접지 초기 eccentric 흡수). 제거하자
#     저감이 −6.2% → −17.6% 로 뛰었다.
#   모터가 0 으로 죽는 것은 `lambda_dead`(EMA hinge)가 막는다.
#
# ★ λ_track / λ_joint 는 **기준선 run 의 학습 스칼라에서 역산**한다 (추측 금지).
#   ablation 롤아웃으로 재면 짧은 클립이 롤아웃 내내 돌아 gt_err 이 7배 부풀고,
#   그 값으로 역산한 hinge 가 항상 0 이 되어 **트래킹 보호가 통째로 꺼진다**.
#   기준선 측정 후 파이프라인이 자동 기입한다.
EXO_ENV_OVERRIDES: dict = {
    # [ETRI 2026-08-28 · S2'] 죽은 모터 페널티 해제. 36모션에는 좌우가 다르게 일해야
    #   정상인 선회 클립이 섞여 있어, 모터별 EMA 바닥 요구가 그런 구간을 벌한다.
    #   (이 항이 좌우 균형을 강제하는 유일한 장치 — λ_sym 은 2026-07-29 에 이미 제거됨)
    "lambda_dead": 0.0,
    "lambda_assist": 0.0,
    "assist_power_cap_w": 30.0,   # λ_assist=0 이라 무의미하지만 로깅 상한으로 유지
    "lambda_hipknee": 0.003,
    "lambda_anti": 0.0,
    "lambda_track": 57.9,
    "lambda_joint": 27.8,
}

# 시상면(굴곡) 집중 배분 — 모터가 굴곡축만 구동하므로. E7 의 배분을 SMPL 축 이름으로.
#
# ★ 모터는 힙뿐인데 무릎 항이 남아 있는 이유 (2026-08-03, 사용자 승인)
#   지우지 말 것. 이 맵은 "모터가 구동하는 축"이 아니라 **줄이려는 사람 부하**의
#   범위다. 무릎을 빼면 힙 토크만 줄이고 무릎으로 부하를 떠넘기는 자세 변화가
#   순수한 개선으로 보상된다 — "힙 −20%, 무릎 +15%" 가 성공으로 집계된다.
#   무릎을 0.5 로 남겨 두면 그 떠넘기기가 목적함수에서 상쇄되므로, 힙 보조만으로
#   달성한 저감이 하지 전체의 실제 저감임을 보장한다.
HIPKNEE_WEIGHT_MAP: dict = {
    "Hip_y": 1.0,
    "Knee_y": 0.5,
    "Hip_x": 0.2,
    "Hip_z": 0.2,
    "Knee_x": 0.2,
    "Knee_z": 0.2,
}

# 보조 0 실측 기준선. None 이면 트래킹 항 ⑥⑦ 이 비활성된다.
#
# ★★★★ v5 (2026-08-03): 기준선을 **학습 스칼라**로 교체. v4 의 결함 수정.
#   v4 는 파이프라인이 `ablation_assist.py` 롤아웃으로 기준선을 자동 측정해
#   gt_err 0.353 / joint_err 0.447 을 넣었다. 그런데 **학습 중 실제값은 0.049 / 0.097**
#   로 7배 작다 — hinge 가 relu(0.049 − 0.353) = **항상 0** 이라 트래킹 보호 항이
#   6013 epoch 내내 꺼져 있었다. 그 결과 eval gt_error 가 +48.9% 악화됐다.
#   원인: ablation 은 종료조건 없이 600 스텝을 돌리는데 27 모션 중 다수가 2.5~7초라
#   모션 종료 후에도 계속 돌아 오차가 누적된다. 기준선은 **학습과 같은 방식**으로
#   재야 한다 → <S2 baseline run>/env/gt_err_mean 을 쓴다(IL 계보는 새로 측정 필요).
# 측정 2026-07-29: 동결 HumanNet v2, 512 env × 600 step, 리셋 0 (ablation_B_off.json)
#   hip/knee |τ| 18.479 Nm  (Hip_y 30.030 / Knee_y 24.312)   팔 |τ| 2.512 Nm
# 재측정:
#   scripts/ablation_assist.py --assist off --config <run>/resolved_configs_inference.pt
# [ETRI 2026-08-28] IL 계보 실측값으로 교체. 위 Newton 값(0.02950/0.06148)은 다른 시뮬레이터·
#   다른 S1·단일 클립 기준이라 그대로 쓰면 hinge 가 항상 켜져(실측 0.0314 > 0.0295) 트래킹
#   페널티가 상시 작동한다 — v4 의 '항상 꺼짐' 실패와 반대 방향의 같은 실수다.
#   출처: output_IL_s1_exosuitHS_motions36 epoch_200 의 eval 지표.
#   **학습과 동일한 평가 방식**(36모션 전부, 최대 600스텝, 종료조건 있음)으로 측정된 값이라
#   docstring §4.2 가 요구하는 조건을 만족한다(ablation 600스텝 무종료 측정과 다름).
#     eval/gt_error/mean 0.03140 · eval/max_joint_error/mean 0.06464 · success 36/36
#   환경변수로 덮어쓸 수 있다: TRACKING_BASELINE_GT / TRACKING_BASELINE_JOINT
TRACKING_BASELINE = {
    # [2026-08-28 2차] loop3 팩(02_01+39_03+103_07) 에서 **보조 0 으로 S2 env 를 직접 돌려**
    #   측정한 값. output_IL_s2_baseline_loop3 (EXO_ASSIST_SCALE=0, 200 epoch) 의
    #   env/gt_err_mean 0.02941 · env/joint_err_max_mean 0.06308.
    #   앞서 쓰던 36모션 S1 eval 값(0.03140/0.06464)은 이 3클립에는 과대 — 그대로 두면
    #   relu(0.0294-0.0314)=0 으로 트래킹 보호항이 상시 꺼진다(v4 실패 재현).
    # ★ 36모션 기준선. 실행 스크립트가 보조 0 런에서 실측해 환경변수로 넘긴다.
    #   아래 기본값은 IL S1 의 36모션 eval(epoch_200) 값으로, S2 env 실측 전의 잠정치다.
    #   실측 없이 돌리면 hinge 가 늘 켜지거나 꺼져 트래킹 보호가 무력화될 수 있다.
    "gt_err": float(os.environ.get("TRACKING_BASELINE_GT", "0.03140")),
    "joint_err": float(os.environ.get("TRACKING_BASELINE_JOINT", "0.06464")),
}


def terrain_config(args: argparse.Namespace):
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    return SceneLibConfig(scene_file=getattr(args, "scenes_file", None))


def motion_lib_config(args: argparse.Namespace):
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        contact_match_rew_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        pow_rew_factory,
        previous_actions_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    cfg = EnvConfig(
        _target_=f"{_TASK}.envs.frozen_human_exo_env.FrozenHumanExoEnv",
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        control_components={"mimic": MimicControlConfig(bootstrap_on_episode_end=True)},
        observation_components={
            "max_coords_obs": max_coords_obs_factory(),
            # history_steps=1 → [E, 71]. 동결 사람에게는 env 가 앞 69 만 잘라 준다
            # (Stage 1 학습 때와 동일 규약). ActionNet 은 71 전부 = 사람의 직전
            # 의도 + 자기 직전 토크.
            "previous_actions": previous_actions_factory(history_steps=1),
            "mimic_target_poses": mimic_target_poses_max_coords_factory(
                with_velocities=True
            ),
        },
        termination_components={
            "tracking_error": tracking_error_term_factory(threshold=0.5),
        },
        # Stage 1 과 동일한 mimic 트래킹 보상. 보조 관련 항(assist/anti/hipknee/
        # smooth/dead/track)은 FrozenHumanExoEnv.step() 이 rew_buf 에 직접 더한다.
        reward_components={
            # ★ composed raw 71 에 걸리므로 동결 사람의 action 변화분이 상수
            # 배경으로 섞인다 — ActionNet 이 제어할 수 없는 항이지만 advantage
            # 정규화가 baseline 으로 흡수한다(원격도 동일 판단). 모터 자체의
            # 진동은 env 의 smoothness_cost 가 따로 잡는다.
            "action_smoothness": action_smoothness_factory(weight=-0.02),
            **mimic_tracking_rewards_factory(
                gt_weight=0.5,
                gr_weight=0.3,
                gv_weight=0.1,
                gav_weight=0.2,
                rh_weight=0.2,
                gt_coef=-25.0,
                gr_coef=-5.0,
                gv_coef=-0.5,
                gav_coef=-0.1,
                rh_coef=-100.0,
            ),
            "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
            # HF 11 클립은 foot-skate 로 접촉 라벨이 무효(2.8%) → 클립마다 다른
            # 보상이 되는 것을 막기 위해 가중치 0. (학습구현.md 2026-07-31)
            "contact_match_rew": contact_match_rew_factory(
                weight=0.0, zero_during_grace_period=True
            ),
        },
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2, resample_on_reset=True
        ),
    )
    # plain dataclass — 커스텀 속성은 resolved_configs.pt 와 함께 pickle 된다
    cfg.frozen_human_ckpt = FROZEN_HUMAN_CKPT
    cfg.exo_motors = [tuple(m) for m in EXO_MOTORS]
    cfg.assist_scale = ASSIST_SCALE
    cfg.exo_env_overrides = dict(EXO_ENV_OVERRIDES)
    cfg.hipknee_weight_map = dict(HIPKNEE_WEIGHT_MAP)
    cfg.tracking_baseline = TRACKING_BASELINE
    return cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.agents.ppo.config import (
        AdvantageNormalizationConfig,
        PPOActorConfig,
        PPOModelConfig,
    )
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )

    in_keys = ["max_coords_obs", "mimic_target_poses", "previous_actions"]

    # ActionNet: 모터 토크 raw 4차원만. agent 는 action 차원을 env/robot 에서
    # 확인하지 않고 actor 출력을 그대로 env.step 에 넘기므로, env 가 4 를 받아
    # 사람 69 와 합성한다.
    actor_config = PPOActorConfig(
        num_out=NUM_MOTOR_ACTIONS,
        actor_logstd=-2.9,
        in_keys=in_keys,
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=in_keys,
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=NUM_MOTOR_ACTIONS,
            layers=[MLPLayerConfig(units=512, activation="relu") for _ in range(3)],
        ),
    )

    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=in_keys,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=MLPWithConcatConfig(
                in_keys=in_keys,
                out_keys=["value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[MLPLayerConfig(units=512, activation="relu") for _ in range(3)],
            ),
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            _target_=f"{_TASK}.envs.exo_evaluator.ExoActionNetMimicEvaluator",
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                # ★ 커리큘럼 활성 (2026-08-03, 16 모션 기준) — 실패 클립을 더 뽑는다.
                #   동작: 성공 *= 0.999^N (감소) / 실패는 1.0 으로 리셋
                #   → eval 을 거듭하면 성공 클립 가중치가 줄어 **어려운 클립에 집중**된다.
                #     (10회 eval 후 실패 클립이 성공 클립의 약 7배)
                #
                #   전제는 "어렵지만 **학습 가능한**" 클립이다. 27 모션 때는 이 전제가
                #   깨져서 균등(success=1.0)으로 껐다 — HF 11개가 관절 규약 불일치
                #   (`Torso_y` −114°)와 foot-skate 로 **재현 불가능**했고, 커리큘럼이
                #   그 클립들에 학습을 집중시켰다(ORDER.md, CONTAMINATED.md).
                #
                #   16 모션은 전부 검증됐다 — 몸통 피치 0~13°, 접지 정상, S0 에서
                #   epoch 200 이후 **eval 실패율 0**. 따라서 전제가 성립하고 커리큘럼이
                #   유익하다. 실패가 없으면 모든 가중치가 함께 줄어 균등과 동일하게
                #   동작하므로, 난이도가 올라가는 S1·S2 에서만 실제로 작용한다.
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    if getattr(env_cfg, "termination_components", None):
        env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
