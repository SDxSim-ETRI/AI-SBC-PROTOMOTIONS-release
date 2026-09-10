# test-motion-loop1200-tall20 — betas +2.0 레퍼런스 모션

> **2026-09-10 개명**: 이 체형의 라벨은 `xtall` → **`tall20`** 으로 통일됐다.
> 규칙: 사전등록 3수준(short/std/tall)은 의미 이름, 나중에 추가한 탐색점은 betas×10 숫자.
> 파일 내부의 옛 이름은 실행 기록이라 고치지 않는다 (cell36_20260909/NAMING_NOTE.md).


생성 2026-09-10. **§5.4 "병목은 형상이다" 주장의 반증 시험용.**

## 왜 만들었나

betas ±1.0 만으로는 주장이 **한쪽 끝에서만** 증명된다.
아래쪽(−1.0)은 하드웨어가 막고(무릎 여유 0.1 mm) 제어는 멀쩡하다.
**위쪽(+1.0)은 둘 다 여유**라 어느 쪽이 먼저인지 모른다.

무릎 여유는 betas 가 클수록 **넓어지므로**(대퇴 +20.4 mm/betas, 캡슐 고정) 위쪽에서
하드웨어는 영원히 안 막는다 → **제어만 남는다.** 따라서 +2.0 에서 제어가 무너지면
결론이 *"아래는 하드웨어, 위는 제어"* 인 **양측 창**으로 바뀐다. 안 무너지면 현재 주장이
*"하드웨어가 허용하는 전 범위에서 제어가 작동한다"* 로 강해진다. **어느 쪽이든 논문이 좋아진다.**

## 체형

| 항목 | 값 |
|---|---|
| betas[0] | **+2.0** |
| 골격 XML | `mjcf_newton_exosuitHS/smpl_humanoid_exosuitHS_for_train_tall20.xml` |
| 총질량 (슈트 4.70 kg 포함) | **90.8362 kg** — 맨몸 **86.14 kg** |
| 대퇴 / 정강 / 다리합 | 415.4 / 451.3 / **866.7 mm** |
| 무릎 여유 | **+60.4 mm** (std 20.2 · tall 40.3) |
| 발바닥 부유 중앙 | +10.6 / +6.1 / +8.4 mm, frame0 ±0.0 |

> 맨몸 86.14 kg 은 `docs/PAPER_BASIS.md` 의 예측값 **86.14 kg** 과 소수점까지 일치한다(교차 검증).
> OpenSim(86.63 kg)과 질량이 거의 같아 **질량 효과와 골격 효과를 가르는 대조군**이 된다.

## 생성 — std/short/tall 과 **같은 레시피**

```bash
PY=/home/user/venv_newton/bin/python
A=$PWD/tasks_for_smpl/mimic_smpl/data/assets
X=$A/mjcf_newton_exosuitHS/smpl_humanoid_exosuitHS_for_train_tall20.xml
SC=$PWD/tasks_for_smpl/script

# 1) 체형 XML
SRC_XML=$A/mjcf_newton_exosuitHS/smpl_humanoid_exosuitHS_for_train_v2.xml \
  $PY $SC/make_exosuit_betas.py hs tall20 2.0

# 2) 루프 npz (체형 무관 — 정본 명령 그대로 재생성)
C=tasks_for_smpl/mimic_smpl/motions/_backup/amass_cmu/CMU ; O=/tmp/tall20_loopnpz
$PY $SC/make_looped_smpl_npz.py $C/02/02_01_stageii.npz   $O/02_01_loop1200step_poses.npz  4800 --seam-blend 8 --wrap-blend 8
$PY $SC/make_looped_smpl_npz.py $C/39/39_03_stageii.npz   $O/39_03_loop1200step_poses.npz  4800 --seam-blend 8 --wrap-blend 8
$PY $SC/make_looped_smpl_npz.py $C/103/103_07_stageii.npz $O/103_07_loop1200step_poses.npz 2400 --seam-blend 6 --wrap-blend 6 --arm-open 0.19,-0.19

# 3) 변환
SMPL_XML=$X bash $SC/convert_loop_npz_batch.sh $O /tmp/tall20_conv

# 4) 접지 보정 — 반드시 FOOT_XML 로 다시
export PYTHONPATH=/home/user/ProtoMotions_etri/data/smpl:$SC     # ★ 아래 함정 참조
for f in /tmp/tall20_conv/_raw_before_footfix/*.motion; do
  FOOT_XML=$X $PY $SC/fix_foot_float.py "$f" "$(basename $f)"
done
```

### 함정 둘 (2026-09-10 실제로 걸림)

1. **3단계의 배치 내부 접지 보정은 표준 SMPL 발로 잰다.** 그대로 두면 발 부유가
   **−100 mm** 대이고 검증이 `★ 발z이상` 을 띄운다. 4단계를 `FOOT_XML` 로 **반드시** 다시 돌린다
   (−102.0 → +10.6 mm 로 교정됐다).
2. `fix_foot_float.py` 가 `ROOT/data/smpl` 에서 `smpl_joint_names` 를 찾는데
   `ROOT = parents[2] = ~/PM_Tasks` 라 없다. 실제 위치는
   `/home/user/ProtoMotions_etri/data/smpl` → **PYTHONPATH 로 잡아준다.**

## 검증

- `dof_pos` 가 std·short·tall 과 **최대차 0.000000°** — P2 전제 성립
- `rigid_body_pos` 최대차 std 대비 **177.8 mm** = tall(88.4 mm)의 정확히 2배 (betas 2.0 = 1.0 의 2배)
- USDA 변환 `ALL PASS` (질량 90.8362 kg 일치)
