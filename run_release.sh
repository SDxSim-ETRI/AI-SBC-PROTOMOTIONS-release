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

STAGE="${1:-S1}"     # S1 | S2_flat   (S0 는 릴리즈하지 않는다 — 아래 참조)
case "$STAGE" in
  S1)      CKPT=tasks_for_smpl/mimic_smpl/checkpoints/S1/last.ckpt;      USD=tasks_for_smpl/mimic_smpl/data/assets/usd_isaaclab_exosuitHS/smpl_humanoid_exosuitHS_for_train.usda ;;
  S2_flat) CKPT=tasks_for_smpl/mimic_smpl/checkpoints/S2_flat/last.ckpt; USD=tasks_for_smpl/mimic_smpl/data/assets/usd_isaaclab_exosuitHS/smpl_humanoid_exosuitHS_for_train.usda ;;
  *) echo "usage: run_release.sh [S1|S2_flat]"; exit 1 ;;
esac
MOTION="${MOTION:-tasks_for_smpl/mimic_smpl/motions/test_motion_36_foot.pt}"
OUTDIR="${OUT:-recordings/$STAGE}"

# S2 는 보조 토크를 사이드카로 남긴다 — ②렌더가 이걸 읽어 슈트를 힘 세기 색으로 칠한다.
#   (S1 은 모터가 없으므로 남기지 않는다 → 렌더에서 기본 회색)
EXO_LOG=""
[ "$STAGE" = "S2_flat" ] && { mkdir -p "$OUTDIR"; EXO_LOG="$ROOT/$OUTDIR/exo_torque.pt"; }

EXO_TORQUE_LOG="$EXO_LOG" "$PYTHON" protomotions/inference_agent.py \
  --checkpoint "$CKPT" \
  --motion-file "$MOTION" \
  --simulator isaaclab --num-envs 1 --headless --auto-record \
  --record-steps "${STEPS:-300}" --recording-path "$OUTDIR" \
  --overrides "robot.asset.etri_prebuilt_usd=$ROOT/$USD" \
              "env.ref_respawn_offset=0.0"
# ★ env.ref_respawn_offset=0.0 은 빼면 안 된다.
#   프레임워크 기본값 0.05(50mm)는 리셋 때 캐릭터를 공중에 띄워 착지 동작을 만든다.
#   이 릴리즈의 정책은 0.0 으로 학습했으므로 추론도 0.0 이어야 학습과 같은 조건이다.
#   (2026-09-01: 이 줄이 빠져 있어 랜딩이 재현되던 것을 고쳤다)

echo
echo "롤아웃 저장 위치: $OUTDIR"
echo "영상으로 렌더하려면:  bash render_release.sh $OUTDIR/*/*.motion"
