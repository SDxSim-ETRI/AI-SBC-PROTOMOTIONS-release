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
