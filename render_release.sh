#!/usr/bin/env bash
# render_release.sh — 추론 롤아웃(.motion)을 고품질 영상으로 렌더한다.
#
# 왜 2단계인가
#   run_release.sh 가 만드는 mp4 는 IsaacLab **뷰포트 캡처**다(물리 확인용).
#   발표·보고용 영상은 여기서 따로 렌더한다 — 갈색 무광 고무 스킨, 슈트 CAD,
#   측면 고정 카메라, HUD. 입력은 1단계가 남긴 물리 롤아웃 `.motion` 이다.
#
# 실행
#   bash run_release.sh S2_flat                       # ① 추론 → .motion
#   bash render_release.sh recordings/S2_flat/*/*.motion   # ② 렌더 → .mp4
#
# 필요한 것
#   · venv 파이썬 하나면 된다(isaacsim 이 pip 패키지). standalone IsaacSim 불필요.
#   · ★ SMPL 바디모델 `SMPL_NEUTRAL.npz` — 라이선스 자산이라 릴리즈에 없다.
#     RELEASE.md "SMPL 바디모델 받는 법" 을 보고 직접 받아
#     etri_라이선스동의_다운로드/smpl_models/smpl/ 에 두어야 한다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
MOTION="${1:?사용법: render_release.sh <rollout.motion> [out.mp4] [프레임수]}"
OUT="${2:-${MOTION%.motion}.mp4}"
NFRAMES="${3:-}"
# ★★ 렌더러는 **standalone IsaacSim** 의 kit 파이썬으로 돌아야 한다.
#   pip `isaacsim` 패키지만으로는 안 된다 — 기본 experience 가 요구하는 확장이
#   빠져 있어 앱이 뜨지 않고(No versions of isaacsim.anim.robot.schema …),
#   대체 experience 를 쓰면 writer 가 프레임당 PNG 를 40~100장 써서 mimsave 가
#   OOM 으로 죽는다(2026-09-01 실측). 자세한 근거는 RELEASE.md ★★ 항목.
#   추론(run_release.sh)은 venv 로 정상 동작한다 — 이 제약은 렌더에만 해당한다.
PYTHON="${PYTHON:-python}"
export OMNI_KIT_ACCEPT_EULA=YES     # 없으면 대화형 EULA 프롬프트에서 멈춘다
[ -n "${ISAACSIM:-}" ] || {
  cat <<'MSG'
✗ ISAACSIM 이 설정되지 않았다 — 렌더에는 standalone IsaacSim 6.0.1 이 필요하다.

    ISAACSIM=/path/to/IsaacSim-6.0.1/python.sh bash render_release.sh <rollout>.motion

  pip isaacsim(venv)으로는 렌더가 되지 않는다. 근거는 RELEASE.md 의 ★★ 항목.
  IsaacSim 6.0.1 은 NVIDIA 에서 내려받는다(약 28 GB).
MSG
  exit 1; }
RUNNER="$ISAACSIM"
[ -x "$RUNNER" ] || { echo "✗ 실행할 수 없다: $RUNNER"; exit 1; }
"$RUNNER" -c "import isaacsim" >/dev/null 2>&1 || {
  echo "✗ isaacsim 을 import 할 수 없다: $RUNNER"
  echo "  standalone IsaacSim 의 python.sh 를 가리키는지 확인할 것"; exit 1; }

NPZ="etri_라이선스동의_다운로드/smpl_models/smpl/SMPL_NEUTRAL.npz"
[ -f "$NPZ" ] || { echo "✗ SMPL 바디모델 없음: $NPZ"; echo "  RELEASE.md 의 'SMPL 바디모델 받는 법' 참조"; exit 1; }

# 프레임수 미지정이면 롤아웃 길이를 읽는다
if [ -z "$NFRAMES" ]; then
  NFRAMES=$("${PYTHON:-python}" -c "
import torch,sys; print(len(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['rigid_body_pos']))" "$MOTION")
fi

# 룩은 render_look.env 하나가 단일 진실 원천이다(색·조명·바닥을 전부 명시 지정).
# 카메라만 용도에 따라 고른다:
#   CAM=side       월드 고정 측면 — 선회가 섞인 동작에 안정적 (기본)
#   CAM=front_rel  진행방향 정면 — 직선 보행 데모용. 선회가 많으면 화면이 돈다.
#   CAM=turntable  360° 회전 — 슈트 전방위 확인
# ── 보조력 색상 ─────────────────────────────────────────────────────────────
# S2 추론(run_release.sh S2_flat)이 롤아웃 옆에 exo_torque.pt 를 남긴다. 있으면
# 슈트 허벅지 스트럿을 **모터가 내는 힘 세기**로 칠한다 — 회색(무보조) → 노랑 →
# 주황 → 빨강(정격 23.7 N·m). 정격을 3등분한 지점을 앵커로 선형 보간한다.
#   이게 없으면 S1 과 S2 영상이 겉보기로 구분되지 않는다.
# S1 은 사이드카가 없으므로 자동으로 기본 회색이 된다.
EXO_LOG="${EXO_TORQUE_LOG:-}"
if [ -z "$EXO_LOG" ]; then
  for c in "$(dirname "$MOTION")/exo_torque.pt" "$(dirname "$(dirname "$MOTION")")/exo_torque.pt"; do
    [ -f "$c" ] && { EXO_LOG="$c"; break; }
  done
fi
[ -n "$EXO_LOG" ] && echo "보조력 색상: $EXO_LOG" || echo "보조력 색상: 없음(회색 단색)"

source tasks_for_smpl/script/render_look.env
CAM="${CAM:-side}" MOTION_NAME="${MOTION_NAME:-$(basename "${MOTION%.motion}")}" \
HUD=isaaclab EXO_LEG_MODE=femur_flex \
EXO_TORQUE_LOG="$EXO_LOG" EXO_COLOR_MODE="${EXO_COLOR_MODE:-ramp}" \
PM_CODE_ROOT="$ROOT" PM_TASKS_ROOT="$ROOT" PYTHONPATH="$ROOT" \
  "$RUNNER" tasks_for_smpl/script/render_exosuit_isaacsim.py hs eval "$OUT" "$NFRAMES" "$MOTION"

echo "✅ $OUT"
echo "   실제 적용된 룩은 옆의 RENDER_LOOK.txt 에 기록된다."
