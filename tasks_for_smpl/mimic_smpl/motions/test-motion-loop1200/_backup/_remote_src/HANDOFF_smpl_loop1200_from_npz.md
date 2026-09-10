# SMPL loop1200 생성 — npz에서 직접(FK 역산 없이)

SkeletonTorque loop1200 노하우를 SMPL에 이식하되, **FK 역산 없이 AMASS npz의 네이티브
파라미터에 루프 변환을 직접 적용**해 만든다. 로컬에서 이미 4개 클립을 생성·검증 완료했다.
원격 시스템은 이 문서대로 재현/대체하면 된다.

---

## 0. 결론 먼저 (로컬 생성물)

| 파일 | 위치 |
|---|---|
| 생성 스크립트 | `/home/kimje/OUR_MOTION_DATA/scripts/make_looped_smpl_npz.py` |
| 출력 loop npz(4개) | `/home/kimje/OUR_MOTION_DATA/amass_conv/SMPL_loop1200/{02_01,07_04,103_07,39_03}_loop1200_poses.npz` |
| 소스 AMASS npz | `/home/kimje/OUR_MOTION_DATA/amass/CMU/{02,07,103,39}/*_poses.npz` |

생성물은 **AMASS 원본과 동일한 npz 포맷**(keys: `trans, poses, betas, gender, mocap_framerate, dmpls`)
이라, 원격의 기존 npz→학습/모션 변환 파이프라인에 **그대로 투입**하면 된다.

---

## 1. 핵심 원칙 — 역산(FK/IK) 금지

SkeletonTorque .motion 루프(`make_looped_skeletontorque.py`)는 FK된 rigid_body 위치·회전을
변환하고 필요시 FK를 재계산했다. **SMPL은 그럴 필요가 없다.** npz가 이미 kinematic
파라미터를 담고 있으므로 다음만 하면 된다:

- **변환**: `trans`(지면 xy), `root_orient`(=`poses[:, :3]`) → world **Z축(yaw) SE(2)** 회전+누적이동
- **불변(그대로 타일링)**: `body pose`+`hand pose`(=`poses[:, 3:]`), `trans` 높이(z), `betas`, `dmpls`

→ 관절 포즈가 원본과 **바이트 단위로 동일**(로컬 검증: 값범위 일치), 루트만 이어붙으므로
FK 재계산·IK 피팅이 전혀 필요 없다. 결과적으로 SkeletonTorque 버전보다 더 깨끗하다.

## 2. 좌표계 (로컬 실측, 중요)

- AMASS CMU npz는 **Z-up**. `trans` 분산 최소축 = idx2(높이), 지면 = xy(idx0,1).
- heading(yaw) = **world Z축 회전**. `root_orient`(axis-angle)→회전행렬 후
  `yaw = atan2(R[1,0], R[0,0])`로 추출.
- **framerate가 클립마다 다르다**: 02_01/07_04/39_03 = **120fps**, 103_07 = **60fps**.
  반드시 `mocap_framerate`를 읽어 주기 탐색범위(0.8~1.9s)를 프레임으로 환산할 것.
- (주의) up-axis를 잘못 잡거나 Y-up↔Z-up을 이중 적용하면 루트 높이·지면관통이 터진다.
  이 데이터는 Z-up이 확인됐다.

## 3. 자동 구간 탐색 (HANDOFF_loop1200.md 2c 이식)

1. **주기 P\***: body pose 자기유사도 최소 지연
   `P* = argmin_P mean(|pose_body[P:] − pose_body[:−P]|)`, 범위 `[0.8·fps, 1.9·fps]`.
   (`pose_body` = `poses[:, 3:66]` = 21 body joints. 손은 제외.)
2. 길이 P 창 `[s, s+P]` 중 우선순위대로 선택:
   1. **yaw 드리프트 ≤ 게이트(0.5°)** — `|yaw(root[s+P]) − yaw(root[s])|`. 남으면 원을 그린다.
   2. **이음매 최소** — `mean(|pose_body[s] − pose_body[s+P]|)`
   - facing(머리 정면성) 기준은 head FK가 필요 → **역산 금지 원칙상 생략**. drift+seam만으로 충분히 안정적이었다.

## 4. 루프 변환 규칙

```
N = 선택구간 프레임수(양끝 포함),  Δyaw = yaw(root[N-1]) − yaw(root[0])
K = ceil((목표 − N)/(N−1)) + 1
loop 0 : 프레임 [0..N-1]
loop k : 프레임 [1..N-1] 에 Rz(k·Δyaw) 적용 + 누적 xy 이동 t_k, 이어붙임(중복1 제거)
```
- `trans`: `xy' = Rz(k·Δyaw)·xy + t_k`, `z' = z(불변)`
- `root_orient`: `R' = Rz(k·Δyaw) · R(root_orient)` → rotvec 복원
- `body/hand pose`, `dmpls`: 슬라이스 그대로 타일링(불변)

## 5. 로컬 검증 결과 (전 클립 모든 게이트 통과)

| 클립 | fps | 주기 | 총프레임/길이 | yaw/주기 | 직선도 | 이음매비 |
|---|---|---|---|---|---|---|
| 02_01 | 120 | 133f(1.10s) ×10 | 1321f / 11.01s | +0.048° | 99.8% | 1.25× |
| 07_04 | 120 | 174f(1.44s) ×7 | 1212f / 10.10s | −0.273° | 99.3% | 1.38× |
| 103_07 | 60 | 68f(1.12s) ×18 | 1207f / 20.12s | +0.119° | 99.8% | 1.23× |
| 39_03 | 120 | 128f(1.06s) ×10 | 1271f / 10.59s | −0.345° | 100.0% | 1.22× |

체크: 프레임≥1200 ✓ · 주기정수배 ✓ · 드리프트<0.5° ✓ · 직선도>95% ✓ · 이음매비<1.5 ✓ · z 높이 일관(관통 없음).

## 6. framerate/길이 주의 (원격에서 조정 가능)

**native fps를 유지**했다(resample 없이 npz 직접). 그래서 1200프레임이 120fps 클립은 ~10s,
60fps인 103_07은 ~20s다. 기존 로컬 loop1200(30fps 변환본)은 ~40s였다.
원격이 특정 fps/길이를 원하면:
- 목표프레임 인자를 키우거나(예: 40s면 120fps에서 4800), 또는
- 학습 파이프라인의 기존 npz 리샘플 단계를 그대로 태우면 된다(권장 — 이 loop npz는 소스와
  동일 포맷이라 리샘플·리타게팅이 원본과 똑같이 적용됨).

## 7. 실행 (최종본 — seam-blend + 103_07 arm-open 포함)

```bash
PY=~/miniforge3/envs/env_newton/bin/python
S=make_looped_smpl_npz.py
# 일반 3클립: seam-blend h=8 (120fps 클립, ~0.07s 창)
$PY $S amass/CMU/02/02_01_poses.npz  amass_conv/SMPL_loop1200/02_01_loop1200_poses.npz  1200 --seam-blend 8
$PY $S amass/CMU/07/07_04_poses.npz  amass_conv/SMPL_loop1200/07_04_loop1200_poses.npz  1200 --seam-blend 8
$PY $S amass/CMU/39/39_03_poses.npz  amass_conv/SMPL_loop1200/39_03_loop1200_poses.npz  1200 --seam-blend 8
# 103_07: 60fps라 h=6, + 양팔 벌림(arm-open) 필수
$PY $S amass/CMU/103/103_07_poses.npz amass_conv/SMPL_loop1200/103_07_loop1200_poses.npz 1200 --seam-blend 6 --arm-open 0.19,-0.19
# 옵션: [목표프레임] [--range s:e] [--drift-gate-deg 0.5] [--seam-blend h] [--arm-open dL,dR]
```

### seam-blend (매끄러움)
각 이음매 ±h 프레임을 표준 3차 Hermite(경계 위치·속도 일치)로 대체 → 속도 C1 연속.
**SMPL은 파라미터가 곧 모션이라 FK 재계산 불필요.** h는 시간창 ~0.07s 기준(120fps→8, 60fps→6).
효과(실측): 이음매 관절점프 2.8~5.3× → **0.59~1.23×**(정상 프레임 수준). h를 더 키우면 오히려
지표가 나빠질 수 있어(비단조) 8/6 권장.

### 103_07 양팔 벌림 (arm-open) — 필수
SkeletonTorque loop의 arm-open 결정과 동일하게 **103_07 loop은 양팔 벌린 버전**이어야 한다.
SMPL엔 명명된 arm_add 힌지가 없어 **어깨 관절 axis-angle의 z성분**을 조정한다(FK로 축·부호 확정):
- `poses[:, 50]`(L_shoulder=SMPL joint16 z) `+= dL`,  `poses[:, 53]`(R_shoulder=joint17 z) `+= dR`
- **dL>0, dR<0** 가 양팔을 바깥으로. 원본이 좌우 대칭(손측면 L218/R220mm)이라 대칭 `0.19,-0.19`(±10.9°).
- FK 검증: 최종 loop에서 손 측면 **L 311mm / R 300mm**(대칭 ~305mm, SkeletonTorque 클리어런스와 동일 급).
- 검증 도구: `smplx`(venv_loco), 모델 `OUR_HUMAN_DATA/smpl/SMPLH_NEUTRAL.pkl`
  (`smplmodels/smplh/SMPLH_NEUTRAL.pkl` 심링크로 `smplx.create(..., model_type="smplh", gender="neutral")`).

## 8. 미리보기(선택) — 역시 역산 없이

`OUR_MOTION_DATA/scripts/retarget_and_render.py --source native` 로 이 loop npz를 원본 파이프라인
그대로 렌더해 육안 확인 가능(별도 렌더러 만들지 말 것).

## 9. 원격 시스템 수정 요약

- 기존 `tasks_for_smpl/script/make_looped_aligned.py`(FK/.motion 기반, 수동 구간)를 이 **npz 직접
  방식**으로 대체하거나 병행한다. 위 §3~4 로직을 그대로 옮기면 된다(엔진·모델 비의존, numpy+scipy만).
- 입력은 AMASS `*_poses.npz`, 출력도 동일 포맷 `*_loop1200_poses.npz` → 원격 학습/변환 파이프라인
  입력으로 바로 사용.
- 좌표계(Z-up)·클립별 fps·drift 게이트만 주의하면 재현된다.

---

## 2026-09-04 개정 — `--wrap-blend` 추가 (되돌기 보간)

§7 의 명령에 **`--wrap-blend h` 를 함께 주어야 한다.** `--seam-blend` 만으로는 부족하다.

### 왜

`seam_blend_smpl` 의 `seams = [N-1 + i*(N-1) for i in range(K-1)]` 은 **K−1개**만 보간한다.
구간은 K개이므로 **마지막 되돌기(마지막 프레임 → 첫 프레임)가 빠진다.** §7 의 "효과(실측)
이음매 관절점프 2.8~5.3× → 0.59~1.23×" 는 **내부 이음매만** 잰 값이고, 되돌기는 그대로
3.8× 로 남아 있었다.

`.motion` 실측 (구 산출물):

| 클립 | 내부 이음매 | **되돌기** | 최악 DOF |
|---|---:|---:|---|
| 02_01 | 4.26° | **7.67°** | **Head_y** (고개) |
| 39_03 | — | 5.72° | — |
| 103_07 | 5.26° | **16.15°** | **L_Elbow_y** |

`--wrap-blend` 적용 후: 1.01° / 6.73° / 4.97° → 비율 0.13× / 0.70× / 0.61× (전부 합격).

### 무엇을 건드리지 않는가

되돌기 보간은 **관절각(`poses[:, 3:]`)만** 이어 붙인다.
**`root_orient`(`poses[:, :3]`)와 `trans` 는 보간하지 않는다** — 40초 전진 클립이라 루트
위치·헤딩은 되돌기에서 원래 크게 불연속이고(설계상 48~61 m 전진), 그것은 학습 env 의
리스폰이 처리한다. 순환 보간하면 루트가 순간이동한다.

구현은 배열을 `h+2` 롤해 되돌기 접합을 인덱스 `h+1` 에 놓고 §7 과 **동일한 3차 Hermite**
(경계 위치·속도 일치, C1 연속)를 적용한 뒤 되돌린다 — 알고리즘은 새로 만들지 않았다.

### 정본 명령

```bash
$PY $S $A/02/02_01_stageii.npz   <out>/02_01_loop1200step_poses.npz  4800 --seam-blend 8 --wrap-blend 8
$PY $S $A/39/39_03_stageii.npz   <out>/39_03_loop1200step_poses.npz  4800 --seam-blend 8 --wrap-blend 8
$PY $S $A/103/103_07_stageii.npz <out>/103_07_loop1200step_poses.npz 2400 \
     --seam-blend 6 --wrap-blend 6 --arm-open 0.19,-0.19
```

**목표프레임이 §7 과 다르다** — §7 은 `1200` 이라 120 fps 클립이 ~10 s 밖에 안 된다.
30 fps 출력에서 1200 스텝(≈40 s)을 얻으려면 `native_fps ÷ 30 × 1200` 이 필요하다:
**120 fps → 4800, 60 fps → 2400.** 실제 산출물(1222 / 1207 / 1207 스텝)이 이 값으로 나온다.

**소스는 `_backup/amass_cmu/CMU/*_stageii.npz` (165열)** 다. §6 이 이미 지적한 대로
SMPL+H(156열)를 쓰면 변환기가 `66 + 81 = 147` 로 reshape 실패한다.
