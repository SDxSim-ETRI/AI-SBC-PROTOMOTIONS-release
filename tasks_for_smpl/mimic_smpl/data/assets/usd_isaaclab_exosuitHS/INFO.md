# usd_isaaclab_exosuitHS/ — 슈트(exosuitHS) 로봇의 **IsaacLab 용 USD**

HyperShell 힙 외골격(exosuitHS)을 입은 SMPL 의 IsaacLab USD. 실제 **S1/S2 학습·추론이 로드하는 자산**.

---

## 1. 파일

| 파일 | 용도 |
|---|---|
| `smpl_humanoid_exosuitHS_for_train.usda` | **학습용·추론용**(S1/S2). primitive 물리 도형 기반. 모든 런타임 override 가 이걸 로드 |

> 예전 `smpl_humanoid_exosuitHS_for_eval.usda`(살 STL 24 강체판)는 런타임 override 0곳(폐기)이라
> 2026-08-27 `_backup/unused_20260827/` 로 격리했다. 추론시각화는 for_train USD + LBS(npz) 또는
> 무이음새(usda)로 한다.

## 2. 어디서 / 어떻게 생성

- **원본은 MJCF**: `../mjcf_exosuitHS/smpl_humanoid_exosuitHS_for_{train,eval}.xml`.
  (그 폴더는 rename 전 `mjcf_newton_exosuitHS` 였음.)
- 이 USD 는 그 MJCF 를 IsaacLab `MjcfConverter` 로 변환한 **빌드 산출물**.
- 슈트 정의 계보: SMPL 맨몸(`../mjcf/smpl_humanoid.xml`)에 exosuitHS 도형/CAD 를 얹어 MJCF 를 만들고
  (`make_exosuit_eval_asset.py` 등), 그걸 USD 로 변환. 슈트 CAD STL 은 `../mesh/exosuit_hs/`.
- **손편집 금지.** MJCF 를 고치고 재변환.

## 3. 어디에 쓰이나 (런타임 결선)

`robot_configs/smpl_exosuitHS.py` 가 `asset_file_name=mjcf_exosuitHS/..._for_train.xml` 로 등록하고,
실제 실행은 CLI override 로 이 USD 를 직접 지정한다:

```
--overrides robot.asset.etri_prebuilt_usd=.../usd_isaaclab_exosuitHS/smpl_humanoid_exosuitHS_for_train.usda
```

> ⚠️ 그래서 체크포인트 `resolved_configs` 의 `asset_file_name`(MJCF 경로)은 **런타임에 로드되지 않는다**
> (USD override 가 이김). 폴더 rename 이 체크포인트를 깨지 않는 이유가 이것이다.

## 4. 릴리즈 지위

유지 — 슈트 학습/추론의 핵심 자산. S1(weight adaptation)·S2(ActionNet) 모두 이 USD 로 돈다.
