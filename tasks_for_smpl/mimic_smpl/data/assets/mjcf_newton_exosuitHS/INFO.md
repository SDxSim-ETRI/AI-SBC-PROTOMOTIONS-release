# mjcf_exosuitHS/ — HyperShell 외골격(exosuitHS) MJCF

SMPL 인체 + 힙 전용 외골격(exosuitHS, HyperShell) 의 MJCF 자산. **S1/S2 학습·추론·렌더의 슈트 원본**.
(2026-08-27 정리 반영.)

> ⚠️ **폴더 이름**: 현재 물리 폴더명은 `mjcf_newton_exosuitHS` 지만, cmu38 영상 렌더가 끝나면
> `finish_videos_and_rename.sh` 가 **`mjcf_exosuitHS`** 로 rename + 참조 일괄 갱신한다
> (`_newton` 제거, `../mjcf` 와 대칭). 이 문서는 rename 후 이름 기준으로 쓴다.

---

## 슈트 물리 사양 (exosuitHS)

**제조사** HyperShell · **형식** 힙 전용 외골격(무릎 이하 없음, CR 전신형에서 종아리 이하 제거) ·
**총 질량 4.700 kg** (실측 확정 2026-08-03). 질량은 density 계산이 아니라 **kg 직접 지정**
(속 찬 도형×density 로 하면 실물과 크게 어긋남).

### 질량 배분 (부착 body별)

| 부착 body | +질량 | 구성 |
|---|---|---|
| Torso | **+1.370 kg** | 가방(backpack): 배터리 404 + 임베디드 + wifi 모듈 |
| Pelvis | **+1.690 kg** | 허리밴드 ㄷ자 프레임 (bar 0.9389 + arm 0.3756 ×2) |
| L_Hip / R_Hip | **+0.820 kg ×2** | 힙 모터 0.530 + 허벅지 box 0.1977 + 링 0.0923 |
| (무릎 이하) | 없음 | knee/shank/ankle 파트 미포함 — 힙 전용 |

합 = 1.370 + 1.690 + 1.640 = **4.700 kg**. (CR 전신형 6.300 kg 의 "종아리 위쪽"과 일치.)

### 보조 모터 (2개, 힙 굴곡)

| 항목 | 값 |
|---|---|
| 위치 | **`L_Hip_y` / `R_Hip_y`** — 힙 관절의 **Y축(굴곡/flexion) hinge** (`axis="0 1 0"`) |
| 모델 | **Unitree GO-M8010-6** |
| 최대 토크 | **23.7 Nm** |
| 모터 질량 | 530 g (위 힙 질량에 포함) |

> 힙 관절 자체는 x/y/z 3축 hinge(stiffness 800 / damping 80 / armature 0.02). 그 중 **Y축(굴곡)에만**
> 보조 모터가 붙는다. S2(ActionNet)가 이 모터 토크를 학습하고, S1 에서는 모터 토크 0(질량 적응만).

### 골격·충돌 관계 (중요)

- **골격은 맨몸 SMPL 과 완전히 동일**(24 body / 69 DOF). 슈트는 body 에 얹힌 geom·질량일 뿐,
  **새 DOF 를 만들지 않는다** → 맨몸 모션·체크포인트를 리타깃·패딩 없이 재사용.
- 슈트 geom 은 **`contype=0 conaffinity=0`**(충돌 제외, **질량·관성만**). 평지 보행에서 슈트가
  지면·환경과 부딪힐 일이 없고, 몸과의 겹침은 착용 구조상 정상이라 충돌 처리하면 레퍼런스와 싸움.
  (MuJoCo·Newton 둘 다 질량·관성은 그대로 계산 — 검증 2026-07-29 71.314 kg 동일.)

### CAD (시각화)

`../mesh/exosuit_hs/` STL 3개 — eval 렌더에서만 슈트를 CAD 로 보여준다(train 은 primitive):
- `hip_motor.stl` — ㄷ자 허리 프레임 + 힙 모터 구 → **Pelvis**
- `left_leg.stl` / `right_leg.stl` — 허벅지 스트럿 → **L_Hip / R_Hip**
- CAD 없는 파트(힙 링, 가방)는 eval 에서도 primitive. 좌표 정합: primitive 배치가 원본 HyperShell
  XML(2026-07-29)과 동일해 CAD 가 그대로 맞음.

### 생성·명세 출처

- XML 생성기: `tasks_for_smpl/script/make_exosuit_train.py hs`
- 명세(질량/기하/앵커/모터): `tasks_for_smpl/script/exosuit_spec.py` 의 `SUITS["hs"]`
- robot_config: `tasks_for_smpl/mimic_smpl/robot_configs/smpl_exosuitHS.py`

---

## 0. 용도별 어느 파일

| 용도 | 파일 |
|---|---|
| **학습**(S1/S2, Newton·IsaacLab 공통) | `smpl_humanoid_exosuitHS_for_train.xml` |
| **LBS 렌더 입력**(cmu38 영상, 이음새 없는 살) | `smpl_humanoid_exosuitHS_for_eval_lbskin.xml` + `render_exosuit_isaacsim.py eval` |

## 1. 파일 (정리 후 — 2개만)

| 파일 | mesh | 설명 |
|---|---|---|
| `…_for_train.xml` | 0 | **학습용.** primitive 만(인체 캡슐/박스 + 슈트 도형). 물리 재현의 기준. USD 변환 원본 |
| `…_for_eval_lbskin.xml` | 3 | LBS 렌더 입력. **슈트 CAD 3개(`../mesh/exosuit_hs`)만** — 살은 렌더러가 매 프레임 생성 |

> 격리됨(`_backup/unused_20260827`): `*_for_eval.xml`(rigid-STL eval, 폐기 — 런타임 override 0곳),
> `*_for_eval_skeleton.xml`(뼈 STL), `*_neutral*`(표기 중복), `suitV3_*`(구세대), `*.bak_*`.

## 2. ★ `_lbskin` 의 뜻 (오해 주의)

`_lbskin` 은 "살 메시가 들어 있다"는 뜻이 **아니다.** 그 파일엔 살이 없고 슈트 CAD 3개뿐 —
**렌더러(`render_exosuit_isaacsim.py`)가 매 프레임 LBS 로 살을 만들어 씌운다**(살 정점은 SMPL npz +
`THIGH/TORSO_SLIM=0.85`). 그래서 이음새가 없다. (예전 `_for_eval.xml` 은 살 STL 을 body 에 강체로
박은 판이라 이음새가 보였는데, 폐기되어 격리됐다.)

## 3. 어디서 / 어떻게 생성

- 원본 `../mjcf/smpl_humanoid.xml`(맨몸)에 exosuitHS 도형/CAD 를 얹어 만든 슈트 MJCF (`make_exosuit_train.py hs`).
- IsaacLab 학습용 USD 는 `…_for_train.xml` 을 `MjcfConverter` 로 변환 → `../usd_isaaclab_exosuitHS/`.
- (구 rigid-STL eval 조립 `make_exosuit_eval_asset.py` 는 폐기. 소스 `mjcf_skin/*`·`mesh/skin*` 도 격리됨.)

## 4. 어디에 쓰이나 (런타임 결선)

`robot_configs/smpl_exosuitHS.py` 가 `asset_file_name=mjcf_exosuitHS/…_for_train.xml` 로 등록.
실행은 CLI override `robot.asset.etri_prebuilt_usd=…/usd_isaaclab_exosuitHS/…_for_train.usda` 로 USD 직접 로드.
→ resolved_configs 의 MJCF 경로는 런타임에 안 읽힘(USD override 가 이김).

## 5. 릴리즈 지위

유지 — 슈트 학습(S1 weight adaptation / S2 ActionNet)·추론·렌더의 핵심.
