#!/usr/bin/env bash
# run_release.sh — 릴리즈 체크포인트 추론/녹화 (자기완결, 링크 없음)
#   반드시 릴리즈 저장소 루트에서 실행: cd <clone>; bash run_release.sh
#
# 원리: 체크포인트에 박힌 절대경로·모듈경로를 다음으로 무력화한다.
#   · cwd = 릴리즈 루트  → tasks_for_smpl/... 상대경로가 그대로 풀림
#   · --overrides 로 자산 경로를 이 트리 기준 상대경로로 명시 (박힌 절대경로 override)
#   · 모듈경로 etri_tasks_for_smpl.* 는 etri_registry.py 별칭 파인더가 처리(링크 불필요)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
export OMNI_KIT_ACCEPT_EULA=YES     # 없으면 대화형 EULA 프롬프트에서 멈춘다
# GPU 를 골라 쓰려면: CUDA_VISIBLE_DEVICES=3 bash run_release.sh S2_flat

STAGE="${1:-S1}"     # S1 | S2_flat | S1_rough | S2_rough   (S0 는 릴리즈하지 않는다)
CK=tasks_for_smpl/mimic_smpl/checkpoints
USDDIR=tasks_for_smpl/mimic_smpl/data/assets/usd_isaaclab_exosuitHS
# 체형 평가용 자산은 BODY 로 고른다 — 기본은 학습 체형(v2 = std)
#   BODY=short|std|tall|tall15|tall20  (betas −1.0 / 0.0 / +1.0 / +1.5 / +2.0)
BODY="${BODY:-}"
USD=$USDDIR/smpl_humanoid_exosuitHS_for_train_v2.usda
[ -n "$BODY" ] && USD=$USDDIR/smpl_humanoid_exosuitHS_for_train_${BODY}.usda
# 지형 단계는 loop1200 3클립(속도 3종)으로 평가한다 — 학습이 그 세트였다.
MOT36=tasks_for_smpl/mimic_smpl/motions/test_motion_36_foot.pt
MOTLOOP=tasks_for_smpl/mimic_smpl/motions/test-motion-loop1200/loop1200_3motions.yaml
[ -n "$BODY" ] && MOTLOOP=tasks_for_smpl/mimic_smpl/motions/test-motion-loop1200-${BODY}/loop1200_3motions.yaml
case "$STAGE" in
  S1)       CKPT=$CK/S1/last.ckpt;       DEFMOT=$MOT36;   FROZEN_OF="" ;;
  S2_flat)  CKPT=$CK/S2_flat/last.ckpt;  DEFMOT=$MOT36;   FROZEN_OF=$CK/S1/last.ckpt ;;
  S1_rough) CKPT=$CK/S1_rough/last.ckpt; DEFMOT=$MOTLOOP; FROZEN_OF="" ;;
  S2_rough) CKPT=$CK/S2_rough/last.ckpt; DEFMOT=$MOTLOOP; FROZEN_OF=$CK/S1_rough/last.ckpt ;;
  *) echo "usage: run_release.sh [S1|S2_flat|S1_rough|S2_rough]   (env: BODY=short|std|tall|tall15|tall20)"; exit 1 ;;
esac
MOTION="${MOTION:-$DEFMOT}"
OUTDIR="${OUT:-recordings/$STAGE}"

# S2 는 보조 토크를 사이드카로 남긴다 — ②렌더가 이걸 읽어 슈트를 힘 세기 색으로 칠한다.
#   (S1 은 모터가 없으므로 남기지 않는다 → 렌더에서 기본 회색)
# 출력 폴더는 단계와 무관하게 미리 만든다 — recordings/ 는 저장소에 없다(.gitignore).
mkdir -p "$OUTDIR"
EXO_LOG=""
case "$STAGE" in S2_flat|S2_rough) EXO_LOG="$ROOT/$OUTDIR/exo_torque.pt" ;; esac

# ★ S2 는 frozen_human_ckpt 를 덮어써야 한다 — 체크포인트의 resolved config 에
#   학습 당시 절대경로(예: /home/user/PM_Tasks/...)가 박혀 있어 받는 쪽에서는
#   그대로 쓸 수 없다. 이 트리의 S1 을 가리키도록 바꿈. (2026-09-01 서버 실행에서 발견)
#   S1 폴더에 resolved_configs.pt 가 함께 있어야 actor 구조를 복원할 수 있다.
FROZEN_OVERRIDE=""
[ -n "$FROZEN_OF" ] && FROZEN_OVERRIDE="env.frozen_human_ckpt=$ROOT/$FROZEN_OF"

# ★ border_size 는 **평지 전용**이다. 지형 단계에 주면 terrain_sequence 배치가
#   달라져 학습과 다른 지형을 깔게 된다 — 지형에서는 체크포인트 설정을 그대로 쓴다.
BORDER=""
case "$STAGE" in S1|S2_flat) BORDER="terrain.border_size=120.0" ;; esac

EXO_TORQUE_LOG="$EXO_LOG" "$PYTHON" protomotions/inference_agent.py \
  --checkpoint "$CKPT" \
  --motion-file "$MOTION" \
  --simulator isaaclab --num-envs 1 --headless --auto-record \
  --record-steps "${STEPS:-300}" --recording-path "$OUTDIR" \
  --overrides "robot.asset.etri_prebuilt_usd=$ROOT/$USD" \
              "env.ref_respawn_offset=0.0" $BORDER $FROZEN_OVERRIDE
# ★ terrain.border_size=120.0 (기본 40.0) — 긴 모션에서 캐릭터가 지면 밖으로
#   걸어 나가는 것을 막는다. 기본값이면 지면이 280 m 이고 격자가 [40,240] 인데,
#   스폰이 격자 끝(x≈238)에 잡히면 남은 거리가 42 m 뿐이다. 02_01 loop1200 은
#   47 m 를 걸으므로 t≈1079 에서 지면을 벗어나 자유낙하한다(2026-09-01 S1 영상에서 발견,
#   같은 실행의 S2 는 스폰이 안쪽이라 멀쩡했다 — 스폰 위치에 따라 갈리는 경계 문제).
#   120 이면 지면 440 m, 격자 [120,320] 로 최악의 스폰에서도 120 m 여유가 있다.
#   지형이 100% 평지라 물리는 동일하고 걸을 땅만 넓어진다(heightfield 8.7M→20.7M vertices).
# ★ env.ref_respawn_offset=0.0 은 빼면 안 된다.
#   프레임워크 기본값 0.05(50mm)는 리셋 때 캐릭터를 공중에 띄워 착지 동작을 만든다.
#   이 릴리즈의 정책은 0.0 으로 학습했으므로 추론도 0.0 이어야 학습과 같은 조건이다.
#   (2026-09-01: 이 줄이 빠져 있어 랜딩이 재현되던 것을 고쳤다)

echo
echo "롤아웃 저장 위치: $OUTDIR"
echo "영상으로 렌더하려면:  bash render_release.sh $OUTDIR/*/*.motion"
