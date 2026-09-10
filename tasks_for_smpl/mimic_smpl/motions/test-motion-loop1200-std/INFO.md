# test-motion-loop1200-std — 체형별 레퍼런스 모션

생성 2026-09-07. 논문(체형×속도×지형)의 **체형 축** 자산. 범위는 `docs/PAPER_BASIS.md`.

## 체형 — SMPL 표준

| 항목 | 값 |
|---|---|
| betas[0] | 0.0 |
| 골격 XML | `/home/user/PM_Tasks/tasks_for_smpl/mimic_smpl/data/assets/mjcf_newton_exosuitHS/smpl_humanoid_exosuitHS_for_train_std.xml` |
| 총질량 (슈트 4.700 kg 포함) | 68.0140 kg |
| 대퇴 / 정강 / 다리합 | 376.8 / 400.6 / 777.4 mm |
| 발바닥 중앙 (자기 XML 기준) | **+9.2 mm**, frame0 접지 ±0.0 mm |

## 무엇이 체형별로 다른가

`dof_pos`(관절각)는 네 체형이 **완전히 동일**하다(최대차 0.000000°). 다른 것은
`rigid_body_pos`(레퍼런스 body 위치)뿐이다. 같은 보행 동작을 각 체형의 뼈 길이로
FK 를 다시 풀어 **그 체형이 도달 가능한 목표**로 만든 것이다(프로토콜 P2).

## 생성 — 훅 두 개가 핵심

체형이 들어가는 곳이 **두 군데**다. 하나만 바꾸면 틀린다.

| 단계 | 훅 | 없으면 |
|---|---|---|
| `retarget_smplx_to_smpl.py` — SMPL FK 로 24 body world pos 산출(= `gts`) | `SMPL_XML` | 모든 체형이 표준 골격 레퍼런스를 받는다 |
| `fix_foot_float.py` — 발바닥 최저 z 를 충돌 geom 코너로 계산해 접지 보정 | `FOOT_XML` | **표준 SMPL 발**로 재서 엉뚱한 평면에 맞춘다 |

두 번째를 빠뜨렸을 때 OpenSim 이 발바닥 −83 mm 로 지면을 뚫는 것처럼 보였다.
두 훅을 같은 XML 로 주면 네 체형의 접지가 +9~16 mm 로 정렬된다.

```bash
cd ~/PM_Tasks
PY=/home/user/venv_newton/bin/python
A=$PWD/tasks_for_smpl/mimic_smpl/data/assets
X=/home/user/PM_Tasks/tasks_for_smpl/mimic_smpl/data/assets/mjcf_newton_exosuitHS/smpl_humanoid_exosuitHS_for_train_std.xml

# 1) 체형 XML (betas 계열만. SRC_XML 로 hs2 소스 지정 — spec 의 train 은 제거된 v1 이름이다)
SRC_XML=$A/mjcf_newton_exosuitHS/smpl_humanoid_exosuitHS_for_train_v2.xml \
  $PY tasks_for_smpl/script/make_exosuit_betas.py hs std 0.0

# 2) 루프 npz — 체형 무관, 한 번만 (9/4 정본 명령)
S=tasks_for_smpl/script/make_looped_smpl_npz.py
C=tasks_for_smpl/mimic_smpl/motions/_backup/amass_cmu/CMU
$PY $S $C/02/02_01_stageii.npz   <o>/02_01_loop1200step_poses.npz  4800 --seam-blend 8 --wrap-blend 8
$PY $S $C/39/39_03_stageii.npz   <o>/39_03_loop1200step_poses.npz  4800 --seam-blend 8 --wrap-blend 8
$PY $S $C/103/103_07_stageii.npz <o>/103_07_loop1200step_poses.npz 2400 --seam-blend 6 --wrap-blend 6 --arm-open 0.19,-0.19

# 3) 변환 → RAW (retarget 까지)
SMPL_XML=$X bash tasks_for_smpl/script/convert_loop_npz_batch.sh <o> <tmp>

# 4) 접지 보정 — **프레임별(기본)**. `--f0-only` 는 프레임 0 만 맞춰 중간이 어긋난다
for f in <tmp>/_raw_before_footfix/*.motion; do
  FOOT_XML=$X $PY tasks_for_smpl/script/fix_foot_float.py "$f" "$(basename $f)"
done
```

## 검증 (2026-09-07)

- 생성기 neutral 재현: body pos 최대이동 **0.05 mm**, std 자산이 소스 `_for_train_v2.xml` 과
  질량·분절길이 **소수점까지 일치**
- betas +1.0 맨몸 73.999 kg = 기존 실측 74.00 kg (교차 검증)
- 네 체형 접지: std +9.2 / short +10.7 / tall +9.4 / opensim +16.2 mm (frame0 전부 ±0.0)

## 기존 `test-motion-loop1200/` 과의 관계

그 폴더는 **hs1 시대 자산이며 손대지 않았다** — 지형 학습이 그것을 쓴다.
접지 보정 방식이 다르다(`--f0-only`) 어서 여기 std 와 body pos 기준 9.7~15.4 mm 차이가
난다. 관절각은 동일하다. 체형 비교는 이 폴더 4종이 **같은 레시피**라 성립한다.

## 사용

```bash
MOTION=.../motions/test-motion-loop1200-std/loop1200_3motions.yaml
```
