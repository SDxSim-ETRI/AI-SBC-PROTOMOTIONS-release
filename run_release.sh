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
# 출력 폴더는 단계와 무관하게 미리 만든다 — recordings/ 는 저장소에 없다(.gitignore).
mkdir -p "$OUTDIR"
EXO_LOG=""
[ "$STAGE" = "S2_flat" ] && EXO_LOG="$ROOT/$OUTDIR/exo_torque.pt"

# ★ S2 는 frozen_human_ckpt 를 덮어써야 한다 — 체크포인트의 resolved config 에
#   학습 당시 절대경로(예: /home/user/PM_Tasks/...)가 박혀 있어 받는 쪽에서는
#   그대로 쓸 수 없다. 이 트리의 S1 을 가리키도록 바꿈. (2026-09-01 서버 실행에서 발견)
#   S1 폴더에 resolved_configs.pt 가 함께 있어야 actor 구조를 복원할 수 있다.
FROZEN_OVERRIDE=""
[ "$STAGE" = "S2_flat" ] && \
  FROZEN_OVERRIDE="env.frozen_human_ckpt=$ROOT/tasks_for_smpl/mimic_smpl/checkpoints/S1/last.ckpt"

EXO_TORQUE_LOG="$EXO_LOG" "$PYTHON" protomotions/inference_agent.py \
  --checkpoint "$CKPT" \
  --motion-file "$MOTION" \
  --simulator isaaclab --num-envs 1 --headless --auto-record \
  --record-steps "${STEPS:-300}" --recording-path "$OUTDIR" \
  --overrides "robot.asset.etri_prebuilt_usd=$ROOT/$USD" \
              "env.ref_respawn_offset=0.0" "terrain.border_size=120.0" $FROZEN_OVERRIDE
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
