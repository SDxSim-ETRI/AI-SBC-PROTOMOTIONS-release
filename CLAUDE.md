# CLAUDE.md — 이 저장소에서 작업할 때의 규칙

이 저장소는 NVIDIA `NVlabs/ProtoMotions` 를 clone 한 것이다 (기준 `e3127726d`, 2026-07-29).
ETRI 작업물은 **`etri_` 접두어가 붙은 폴더**에만 둔다.

> upstream 에는 `CLAUDE.md` 가 없다 (NVIDIA 가 삭제함). 이 파일은 ETRI 가 추가한 것이다.

---

## 1. ⛔ git 관련 파일은 직접 수정하지 않는다

아래 파일은 **에이전트가 직접 편집하지 말 것.** 사용자가 직접 다루거나, 반드시 필요하면
무엇을 왜 바꿔야 하는지 먼저 보고하고 승인을 받는다.

```
.git/config          .git/ 하위 전부
.gitignore
.gitattributes       ← LFS 규칙. 잘못 건드리면 대용량 파일이 깨진다
.github/             워크플로
.gitmodules
```

**이유**

1. **머지 충돌의 주범.** 이전 fork 에서 `git merge origin/main` 을 시험했을 때
   실제 충돌 13개 중 3개가 `.gitignore`, `.github/workflows/deploy-docs.yml`, `CLAUDE.md`
   였다. upstream 이 이 파일들을 자주 고치므로 우리가 손대면 매 업데이트마다 충돌한다.
2. **`.gitattributes` 는 LFS 규칙이다.** `*.npz *.pkl *.json *.csv *.stl *.obj`,
   `data/**/*.yaml` 이 LFS 로 관리된다. 규칙을 바꾸면 기존 파일이 포인터/실체 불일치로 깨진다.
3. **`.git/config` 에 인증정보가 들어갈 수 있다.** 이전 fork 에서 remote URL 에 GitHub PAT
   가 평문으로 박혀 `git remote -v` 만 실행해도 노출되는 상태였다.
   토큰은 `.env` + credential helper 로 관리한다 (아래 §4).

**대신 이렇게 한다**

| 하려는 것 | 올바른 방법 |
|---|---|
| 산출물을 git 에서 제외 | `.gitignore` 편집 대신 **사용자에게 요청**. 또는 `etri_*` 폴더 안에 두고 그 폴더 하나만 제외 규칙 추가를 제안 |
| 인증 설정 | `.env` 에 `GITHUB_USER` / `GITHUB_PAT` 를 두고 credential helper 가 읽게 한다 |
| LFS 대상 추가 | `.gitattributes` 편집 전 반드시 보고. 기존 파일 영향 범위를 먼저 조사 |

---

## 2. 폴더 규칙 — `etri_` 접두어

**원래 이름 앞에 `etri_` 만 붙인다.**

```
tasks_for_smpl   →  etri_tasks_for_smpl
tasks            →  etri_tasks
tasks_for_v3     →  etri_tasks_for_v3
checkpoints      →  etri_checkpoints
scripts (우리 것) →  etri_scripts
docs (우리 것)    →  etri_docs
```

### ⭐ 새 파일을 만들 때의 규칙 (가장 중요)

**upstream 과 merge 할 때 충돌이 없도록** 아래 둘 중 하나를 택한다.

| 상황 | 규칙 | 예 |
|---|---|---|
| **별도 폴더를 만들 수 있다** | 폴더 이름에 `etri_` 접두어 | `etri_terrains/`, `etri_rewards/`, `etri_scripts/` |
| **원본 폴더 안에 만들어야 한다** | **파일명 앞에** `etri_` 접두어 | `protomotions/robot_configs/etri_registry.py` |

**폴더 분리가 항상 우선이다.** 원본 폴더 안에 두는 것은 프레임워크가 그 위치를 요구할 때만.

이렇게 하면:
- `git status` 에서 우리 파일이 이름만으로 즉시 구분된다 (히스토리 불필요)
- upstream 이 같은 이름의 파일을 추가할 확률이 사실상 0 → **merge 충돌이 나지 않는다**
- 어느 파일을 옮겨도 되는지 판단이 쉬워진다

이전 fork 에서 이 규칙이 없어 upstream 디렉터리 안에 우리 파일 271개가 이름 구분 없이
섞였고, NVIDIA 것과 구분하려면 git 히스토리를 뒤져야 했다.

### upstream 파일을 **수정**해야 할 때

새 파일이 아니라 기존 upstream 파일을 고쳐야 하는 경우(등록 지점, 호출 지점 등):

1. **본문은 `etri_` 파일로 빼고, upstream 파일에는 호출 한두 줄만 남긴다.**
   수정 행수가 적을수록 다음 merge 에서 충돌 가능성이 낮아진다.
2. 수정 지점에 **`# [ETRI patch] 이유`** 주석을 남긴다.

```python
# protomotions/robot_configs/factory.py  ← upstream 파일: 최소 수정
    # [ETRI patch] ETRI 로봇 등록은 etri_registry.py 에 분리
    from protomotions.robot_configs.etri_registry import etri_robot_config
    cfg = etri_robot_config(robot_name)
    if cfg is not None:
        return cfg
```

실제 사례: 이전 fork 는 `factory.py` 에 `elif` 25행을 직접 넣어 upstream 개편과 충돌했다.
위 방식이면 3행이라 충돌 확률이 훨씬 낮다.

### 예외 — 어쩔 수 없이 upstream 안에 둬야 하는 것

체크포인트의 `resolved_configs` 가 **경로/모듈 경로를 문자열로 기억**하기 때문에
아래는 옮기면 기존 체크포인트가 깨진다. 새로 만들 때부터 `etri_*` 에 두면 문제없다.

```yaml
# 체크포인트에 실제로 저장되는 값
_target_: protomotions.components.terrains.terrain_linear_stairs.LinearStairsTerrain
_target_: tasks_for_smpl.mimic_smpl_exosuitCR.envs.frozen_human_exo_env.FrozenHumanExoEnv
motion_file: tasks_for_smpl/mimic_smpl_exosuitCR/motions/walk_cmu_07_04_30s_aligned.motion
asset_file_name: mjcf/skeleton_torque.xml
```

→ **폴더 이름을 바꿀 때는 기존 체크포인트 로딩이 깨지는지 먼저 확인**하고,
필요하면 옛 경로에 alias 모듈을 남긴다.

---

## 3. upstream 코드를 고치기 전에

`protomotions/` 안의 upstream 파일을 직접 수정하면 **다음 업데이트에서 충돌한다.**
이전 fork 에서 27개 파일 `+1,032/−186` 을 수정했고, 그중 상당수가 다음 이유로 불필요했다:

| 상황 | 우선 검토할 대안 |
|---|---|
| 새 리워드/환경/로봇을 추가 | config 의 `_target_` 로 `etri_*` 모듈을 가리킨다. 함수는 `compute_func` 에 직접 넘길 수 있다 |
| 기존 클래스 동작 변경 | **서브클래스**를 `etri_*` 에 만들고 `_target_` 로 지정 |
| 하드코딩된 if/elif 에 분기 추가 | 레지스트리 훅을 추가하는 **upstream PR** 을 검토 |
| upstream 버그 | **upstream PR**. 반영되면 우리 수정이 사라진다 |
| 전역 동작을 바꿔야 함 | `etri_*` 모듈에서 런타임 monkey-patch |

**그래도 고쳐야 한다면**: 수정 지점에 `# [ETRI patch] 이유` 주석을 남겨 다음 머지 때
무엇이 우리 것인지 즉시 알 수 있게 한다.

---

## 4. 환경 (인터프리터가 용도별로 다르다)

| 인터프리터 | 용도 |
|---|---|
| `/home/user/venv_newton/bin/python` | Newton 학습·시뮬, 모션 변환·리타겟 |
| `/home/user/miniforge3/envs/env_isaaclab/bin/python` | IsaacLab 추론·렌더 |
| `/home/user/venv_zshot/bin/python` | **Kimodo 모션 생성 전용** (protomotions 없음) |

- **conda 계열(miniconda/mamba) 신규 설치 금지.** 위 환경만 사용한다.
- 비밀값은 `.env` (권한 600, `.gitignore` 대상). `HF_TOKEN`, `GITHUB_PAT` 등.
- Hugging Face `nvidia/Kimodo-*` 는 gated — 모델별 개별 승인이 필요하다.

---

## 5. GPU 사용 주의

- 단일 GPU(약 15.5 GiB)를 학습·추론·생성이 공유한다. **실행 전 `nvidia-smi` 로 확인**할 것.
- 학습이 돌고 있으면 추론/생성을 동시에 띄우지 않는다 — 학습 쪽이 OOM 으로 죽을 수 있다.
- Kimodo 텍스트 인코더(LLM2Vec Llama-3.1-8B, bf16 약 16 GB)는 **GPU 에 안 들어간다.**
  항상 CPU 로 올린다 (`--text-encoder-device cpu`).

---

## 6. 참고 문서

> ⚠️ [ETRI 2026-08-26] **조사·이관 문서는 이 저장소에서 나갔다.**
> 이 저장소는 **체크포인트와 그 실행에 필요한 파일만** 담는 릴리즈 저장소이므로,
> 문서·이관기록을 `~/PM_Tasks/docs/` 로 옮겼다(아래 목록은 그 새 위치 기준).
> DOFC 분석물은 `~/PM_Tasks/tasks_for_skeleton/docs/dofc/` 에 있다.

| 문서 (`~/PM_Tasks/docs/`) | 내용 |
|---|---|
| **`MIGRATION_STATUS.md`** | **이관 현황·다음 할 일·함정** ← 새 세션은 여기부터 |
| `OWNERSHIP.md` | NVIDIA 원본 vs ETRI 파일 분류 (20,616 파일). **git 히스토리 없이는 재생성 불가** |
| `CORE_PATCH_CLASSIFICATION.md` | upstream 수정 27개의 성격별 분류와 처리 방침(P1~P5) |
| `upstream_core_*.patch` | 그룹별 패치. 새 upstream 에 얹을 때 `git apply --3way` 사용 |
| `KIMODO-PROTOMOTIONS.md` | 텍스트 → 인체 물리 시뮬레이션 파이프라인 (명령어·실측 결과·함정) |

## 7. 이 저장소의 범위 — 체크포인트 + 실행 필수 파일만

> ⚠️ [ETRI 2026-08-26] 저장소 목적을 좁혔다. **이 저장소는 체크포인트와 그것을
> 실행하는 데 필요한 파일(upstream 코드 + ETRI 로봇/자산 정의)만** 담는다.
> 학습·변환·이관·분석 **도구와 문서·테스트는 여기 두지 않는다.**

같은 날 아래를 `~/PM_Tasks/` 로 옮겼다(전부 실행 필수가 아님을 확인 후):

| 옮긴 것 | 새 위치 |
|---|---|
| `etri_scripts/` (86개: 학습 runner·변환·이관·추론 래퍼·분석) | `~/PM_Tasks/scripts/` |
| `etri_docs/` (이관·조사 문서) | `~/PM_Tasks/docs/` |
| `docs/etri_dofc/` (DOFC 분석물) | `~/PM_Tasks/tasks_for_skeleton/docs/dofc/` |
| `etri_tests/` (케이블 물리 테스트) | `~/PM_Tasks/tasks_for_skeleton/tests/` |

**학습 산출물**(`output_*/`, epoch ckpt·로그), AMASS 원본, 녹화 영상, zarr 데이터는
애초에 여기 오지 않는다 — zarr 은 NAS 로 백업한다(`~/PM_Tasks/issue/zarr_to_nas_upload.md`).
가져오는 것은 소스·설정·패키징된 모션(`.motion`/`.pt`)·최종 가중치(`last.ckpt`, `score_based.ckpt`).

> 참고: `~/PM_Tasks` 는 **fork 가 아니라 ETRI 전용**이므로 그 안에서는 `etri_` 접두를 쓰지 않는다
> (`scripts/`, `docs/`, `tests/`). `etri_` 규칙(§2)은 이 upstream fork 안에서만 적용된다.

⚠️ **`resolved_configs.pt` 의 경로는 치환되지 않는다.** 릴리즈 전
`warm_start` 로 새 경로를 기록해야 한다 — `~/PM_Tasks/docs/MIGRATION_STATUS.md` §3.
