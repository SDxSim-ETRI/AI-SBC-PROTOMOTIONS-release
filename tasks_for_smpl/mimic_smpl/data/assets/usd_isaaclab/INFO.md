# usd_isaaclab/ — 맨몸 SMPL 의 **IsaacLab 용 USD (빌드 산출물)**

`smpl_humanoid.usda` = `../mjcf/smpl_humanoid.xml` 을 IsaacLab 용으로 변환한 USD.

---

## 1. 무엇 / 어디서

- 맨몸 SMPL 인체(24 body / 69 DOF)의 **USD 표현**. Isaac Sim 은 물리 씬을 USD 로만 읽으므로 필요.
- **원본이 아니라 파생물**이다. 원본은 `../mjcf/smpl_humanoid.xml` 하나. 이 USD 는 거기서 변환됐다.

## 2. 어떻게 생성 (재빌드 방법)

MJCF→USD 변환은 IsaacLab 3 `MjcfConverter` 로 한다 (`protomotions/simulator/isaaclab/utils/mjcf_to_usd.py`).
직접 굽고 싶으면 IsaacLab 변환기를 호출하는 스크립트(예: `record_smpl_isaaclab.sh`, `release_assemble.sh`
안의 변환 단계)를 참고. **USDA 는 절대 손편집하지 말 것** — MJCF 를 고치고 재변환한다.

```
mjcf/smpl_humanoid.xml  ──MjcfConverter──▶  usd_isaaclab/smpl_humanoid.usda
   (원본, 손저작)              (빌드)              (파생, 커밋)
```

## 3. 어디에 쓰이나

- IsaacLab 학습/추론에서 SMPL 맨몸(S0 등) 로봇의 씬 자산으로 로드.
- 런타임에는 `robot.asset.etri_prebuilt_usd=<이 파일>` override 로 **바로 로드**해 시작 변환을 건너뛴다.
  (자동캐시 대신 미리 구운 것을 쓰는 이유는 `../mjcf/INFO.md` §4 참조.)

## 4. 릴리즈 지위

유지. 맨몸 SMPL 을 IsaacLab 에서 돌리는 데 필요. `smplx_humanoid.usda`(SMPL-X)는 우리가 안 쓰므로
`_backup/unused_20260827/` 로 격리했다.
