# test-motion-loop1200 — 추론시각화용 루프 걷기 모션

**용도:** 추론시각화(inference viz) 전용 모션. 평가(eval)는 학습 모션(`test_motion_36_foot`)을
쓰고, **추론시각화만 이 loop1200 을 쓴다** (연구 규칙). 대표 클립: `walk_cmu_103_07_loop1200.motion`
(orchestrate/360 영상 스크립트 기본값).

## 남긴 파일 (릴리즈/런타임)

| 파일 | 소스 CMU 클립 | native fps |
|---|---|---|
| `walk_cmu_02_01_loop1200.motion`  | CMU 02/02_01  | 120 |
| `walk_cmu_07_04_loop1200.motion`  | CMU 07/07_04  | 120 |
| `walk_cmu_39_03_loop1200.motion`  | CMU 39/39_03  | 120 |
| `walk_cmu_103_07_loop1200.motion` | CMU 103/103_07 | 60 |

모두 30 fps · 1200 스텝(제어주기 30 Hz 기준 약 40초) 걷기 루프. 프레임별 수직 접지 보정 완료.

## 어떻게 만들었나 (생성 계보)

원본 AMASS CMU npz → (A) 루프 npz 생성 → (B) `.motion` 변환. **FK 역산 없음.**

### (A) 루프 npz 생성 — `make_looped_smpl_npz.py`
- 소스: `OUR_MOTION_DATA/amass/CMU/{02,07,103,39}/*_poses.npz` (AMASS SMPL-H, Z-up)
- 방법: AMASS npz의 **네이티브 파라미터에 루프 변환을 직접 적용**(FK/IK 역산 안 함).
  - 보행 1주기 P* 자동 탐색: body pose 자기유사도 최소 지연(탐색범위 0.8~1.9s).
  - 창 선택: 주기당 |yaw 드리프트| ≤ 0.5° + 이음매(seam) 최소.
  - 루프: `trans(xy)`·`root_orient(=poses[:,:3])` 만 world-Z(yaw) SE(2) 누적변환,
    body/hand pose·trans-z·betas·dmpls 는 **그대로 타일링**(관절 포즈 원본과 100% 동일).
  - framerate 클립별로 다름(120/60) → `mocap_framerate` 읽어 처리. 목표 = `fps÷30×1200` 프레임.
- 출력: `<클립>_loop1200step_poses.npz` (AMASS 포맷: trans/poses/betas/gender/mocap_framerate/dmpls)

### (B) `.motion` 변환 — `convert_loop_npz_batch.sh`
입력: 위 loop npz. 단계(원본 CMU 변환과 동일):
1. 스테이징(피험자 하위폴더 구조)
2. `convert_amass_to_proto.py --humanoid-type smplx --output-fps 30`  (Z-up 출력, 120÷4·60÷2 정수배 데시메이션)
3. `retarget_smplx_to_smpl.py`  (SMPL-X 153 → SMPL 69 DOF)
4. `fix_foot_float.py`  (프레임별 수직 접지 보정)
5. 검증(스텝 수·루트 z·발 최저 z)

> ⚠️ `convert_amass_to_proto.py` 가 이미 Z-up 을 낸다 — 뒤에 `convert_motion_yup_to_zup.py` 붙이면 두 번 돌아 누움.

## `_backup/` 격리물 (2026-08-27)
walk*.motion 만 남기고 중간·소스 산출물을 격리:
- `*_loop1200step_poses.npz` — (A) 루프 npz(=`_smplh_from_remote/` 와 동일 내용의 상단 사본)
- `_raw_before_footfix/` — (B)4 접지보정 **전** .motion (디버그용)
- `_smplh_from_remote/` — 리모트가 넘긴 loop npz 원본
- `_remote_src/` — **생성 스크립트 원본**(`make_looped_smpl_npz.py`) + `HANDOFF_smpl_loop1200_from_npz.md`

재생성이 필요하면 `_backup/_remote_src/make_looped_smpl_npz.py` 로 (A), `convert_loop_npz_batch.sh` 로 (B).

---

## 2026-09-04 재생성 — 되돌기(wrap-around) 보간 추가

**세 클립 모두 교체했다.** 백업: `_backup/pre_wrapblend_20260904_194003/`

### 무엇이 문제였나

`make_looped_smpl_npz.py` 의 `seam_blend_smpl` 이
`seams = [N-1 + i*(N-1) for i in range(K-1)]` 로 **K−1개**만 보간했다. 구간은 K개이므로
**마지막 되돌기(마지막 프레임 → 첫 프레임)가 보간에서 빠져 있었다.**

학습은 무작위 시점에서 시작해(`init_start_prob=0.2`) 클립을 되돌며 재생하므로, 일부
에피소드가 이 불연속을 한 프레임에 그대로 받는다. **평가로는 절대 안 잡힌다** —
평가기가 `motion_times=0` 에서 300프레임만 돌려(`mimic_evaluator.py:267`) 1,200프레임
클립의 되돌기에 도달하지 않는다.

### `.motion` 실측 (마지막↔첫 `dof_pos` 최대차 / 내부 인접프레임 최대차 중앙값)

| 클립 | 기존 되돌기 | **신규** | 비율 기존 → 신규 | 최악 DOF |
|---|---:|---:|---|---|
| `walk_cmu_02_01_loop1200`  | 7.67° | **1.01°** | 1.02× → **0.13×** | Head_y → (해소) |
| `walk_cmu_103_07_loop1200` | **16.15°** | **4.97°** | 1.98× → **0.61×** | L_Elbow_y |
| `walk_cmu_39_03_loop1200`  | 5.72° | 6.73° | 0.60× → **0.70×** | (변화 없음) |

**세 클립 모두 합격 기준(≤ 1.0×) 통과.** 103_07 은 1.98× → 0.61× 로, 기존 최良이던
39_03(0.60×)과 같은 수준이 됐다. 39_03 은 0.60 → 0.70 으로 미세하게 나빠졌으나(보간 대가)
합격 범위다.

**02_01 의 결함은 고개였다** — 되돌기 최악 채널이 `Head_y 7.67°` / `Neck_y 5.77°` 였다.
고개는 보행 중 거의 안 움직이므로(내부 0.22°/프레임) 되돌기 비율이 **32배**였고,
주기마다 고개가 한 프레임에 끄덕이는 형태였다. 지금은 1.01° 로 사라졌다.

### 재생성 명령 (정본)

```bash
cd ~/PM_Tasks
PY=~/venv_newton/bin/python
S=tasks_for_smpl/script/make_looped_smpl_npz.py
A=tasks_for_smpl/mimic_smpl/motions/_backup/amass_cmu/CMU      # ★ stageii (165열)

$PY $S $A/02/02_01_stageii.npz   <out>/02_01_loop1200step_poses.npz  4800 --seam-blend 8 --wrap-blend 8
$PY $S $A/39/39_03_stageii.npz   <out>/39_03_loop1200step_poses.npz  4800 --seam-blend 8 --wrap-blend 8
$PY $S $A/103/103_07_stageii.npz <out>/103_07_loop1200step_poses.npz 2400 \
     --seam-blend 6 --wrap-blend 6 --arm-open 0.19,-0.19

bash tasks_for_smpl/script/convert_loop_npz_batch.sh <out> <motion_dir>
```

**★ 소스는 반드시 `_backup/amass_cmu/CMU/*_stageii.npz` (165열)** 다.
`~/OUR_MOTION_DATA/amass/CMU/` 의 SMPL+H(156열)를 쓰면 `convert_amass_to_proto.py` 가
`66 + 81 = 147` 로 reshape 실패한다(그 스크립트 docstring 의 경고 그대로).

**목표프레임**: `native_fps ÷ 30 × 1200` — 120 fps → 4800, 60 fps → 2400.
`--seam-blend` / `--wrap-blend` h 는 시간창 ~0.07 s 기준(120 fps → 8, 60 fps → 6).

### 같이 고친 것 (저장소 이전으로 깨져 있던 경로 3건)

`~/ProtoMotions` → `~/PM_Tasks` 이전 후 상대경로 가정이 깨져 있었다. 전부 후보 탐색으로
바꿨으므로 저장소가 또 옮겨져도 견딘다. 각 파일에 `.bak_before_*` 백업이 있다.

| 파일 | 증상 | 수정 |
|---|---|---|
| `script/convert_loop_npz_batch.sh` | `data/scripts/convert_amass_to_proto.py` 없음 | `PROTO_DATA` 후보 탐색 + `PYTHONPATH` 에 `data/smpl` 추가 |
| 〃 | `data/yaml_files/motion_fps_amassx.yaml` 없음 | 변환기를 **저장소 루트에서** 실행(subshell `cd`) |
| `script/retarget_smplx_to_smpl.py` | `smplx_humanoid.xml` 없음 | `_find_mjcf()` 후보 탐색 (릴리즈 저장소 포함) |

### 영향

3모션이 모두 바뀌었으므로 **단계 E 재학습이 필요하다**
(`output_IL_s1_E_motions3_env16k_DONE` 는 구 모션 산출물). v2 슈트로 지형을 다시 돌릴
예정이므로 그때 새 모션을 쓰면 추가 비용이 없다.
