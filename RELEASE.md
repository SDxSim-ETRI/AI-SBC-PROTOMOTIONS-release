# ETRI AI-SBC — 착용형 보행보조 슈트 (exosuitHS) 릴리즈

SMPL 휴머노이드에 엉덩관절 모터 2개짜리 보조 슈트를 입히고 걷게 한 정책과, 그것을
링크 없이 바로 실행할 수 있는 최소 자산 묶음이다.

- **코드 리비전**: `8b172e9131326c7e47a97f47c4923530dc974407` (2026-08-26, upstream 병합)
- **조립일**: 2026-09-01
- **릴리즈 태그**: hs-flat

## 담긴 것

| 경로 | 내용 |
|---|---|
| `tasks_for_smpl/mimic_smpl/checkpoints/S1/last.ckpt` | **S1** — 슈트 무게 적응(모터 끔). 사람 걷기 정책 |
| `tasks_for_smpl/mimic_smpl/checkpoints/S2_flat/last.ckpt` | **S2** — 평지 모터 보조. S1을 동결하고 모터 출력만 학습 |
| `tasks_for_smpl/mimic_smpl/motions/test_motion_36_foot.pt` | 참조 모션 36종 패키징본 (AMASS/CMU) — 학습·평가용 |
| `tasks_for_smpl/mimic_smpl/motions/test-motion-36-foot/` | 같은 36종의 **개별 `.motion` 파일** — 한 동작씩 추론·녹화할 때 |
| `tasks_for_smpl/mimic_smpl/motions/test-motion-loop1200/` | **시각화용 1200스텝 루프 클립 3종** (02_01 느림 / 103_07 보통 / 39_03 빠름) |
| `tasks_for_smpl/mimic_smpl/data/assets/usd_isaaclab*/` | SMPL·슈트 USD 자산 |
| `tasks_for_smpl/mimic_smpl/robot_configs/` | 로봇 설정 모듈 (체크포인트 언피클이 import) |
| `tasks_for_smpl/mimic_smpl_exosuitHS/envs/` | **S2 추론에 필수인 env 모듈** — 체크포인트가 `FrozenHumanExoEnv` 를 이름으로 참조한다 |
| `run_release.sh` | ① 추론 실행기 (상대경로만 넘긴다) |
| `render_release.sh` | ② 영상 렌더 실행기 |
| `tasks_for_smpl/script/render_*.{py,env}` | 렌더 스크립트·룩 프로파일 |
| `tasks_for_smpl/mimic_smpl/data/assets/mesh/exosuit_hs/` | 슈트 CAD 메시 |

심볼릭 링크는 **0개**다. `git clone` 후 경로 수정 없이 바로 돌아간다.

## 실행 — 2단계다

영상까지 만들려면 **추론(①)과 렌더(②)를 따로** 돌린다. 나눈 이유는 렌더가 IsaacSim kit
파이썬과 SMPL 바디모델을 따로 요구하기 때문이다 — ①만으로도 물리 동작 확인은 된다.

### ① 추론 — 정책을 돌려 물리 롤아웃을 남긴다

```bash
source venv_il3/bin/activate          # 위에서 만든 환경
bash run_release.sh S1                # 슈트 입고 걷기 (모터 끔)
bash run_release.sh S2_flat           # 모터 보조 켬

# GPU 를 골라 쓰려면
CUDA_VISIBLE_DEVICES=3 bash run_release.sh S2_flat
```

> 두 스크립트 모두 `OMNI_KIT_ACCEPT_EULA=YES` 를 넘긴다. 이걸 빼면 IsaacSim 이
> 대화형 EULA 프롬프트에서 멈춘다(비대화형 실행이면 그대로 죽는다).

`cwd = 릴리즈 루트`에서 실행해야 한다 — 체크포인트에 박힌 상대경로가 그 기준으로 풀린다.

산출물은 `recordings/<STAGE>/<타임스탬프>/` 아래에 두 가지가 생긴다:

| 파일 | 내용 |
|---|---|
| `*.mp4` | IsaacLab **뷰포트 캡처** — 물리 동작 확인용(품질은 낮다) |
| `*.motion` | **물리 롤아웃** — ②의 입력이자, 관절 토크 분석의 원자료 |

환경변수로 조절한다:

```bash
MOTION=tasks_for_smpl/mimic_smpl/motions/test-motion-36-foot/walk_cmu_02_01.motion STEPS=86 OUT=recordings/my_test bash run_release.sh S2_flat
```

> ★ `run_release.sh` 는 `env.ref_respawn_offset=0.0` 을 넘긴다. **이 값을 빼면 안 된다** —
> 프레임워크 기본값 0.05(50mm)는 리셋 때 캐릭터를 공중에 띄워 착지 동작을 만든다.
> 이 릴리즈의 정책은 0.0 으로 학습했으므로 추론도 0.0 이어야 학습과 같은 조건이다.

### ② 렌더 — 롤아웃을 발표용 영상으로

```bash
source venv_il3/bin/activate
ISAACSIM=/path/to/IsaacSim-6.0.1/python.sh \
  bash render_release.sh recordings/S2_flat/*/*.motion
```

> `ISAACSIM` 은 **생략할 수 없다** — pip `isaacsim` 만으로는 렌더가 안 된다(위 ★★ 참조).

GPU 1장이면 충분하다. 여러 장인 머신에서는 `CUDA_VISIBLE_DEVICES` 로 고른다.

갈색 무광 고무 스킨 + 슈트 CAD + 측면 고정 카메라 + HUD 로 1920×1080 mp4 를 만든다.

#### 보조력 색상 — S1 과 S2 를 눈으로 구분한다

S1(모터 끔)과 S2(모터 켬)는 **걷는 모습만으로는 구분되지 않는다.** 그래서 S2 영상은
슈트 허벅지 스트럿을 **모터가 내는 힘 세기**로 칠한다.

| \|τ\| (모터 토크) | 색 |
|---|---|
| < 0.5 N·m | 회색 — 보조 없음 |
| 7.9 (정격의 1/3) | 노랑 |
| 15.8 (2/3) | 주황 |
| 23.7 (정격) | **빨강** |

정격을 3등분한 지점을 앵커로 **선형 보간**하므로 힘이 오르내리는 과정이 매끄럽게 이어진다.
좌우 스트럿을 각 힙 모터로 **독립 표시**하므로, 보행에서 좌우가 번갈아 일하는 것이 그대로 보인다.

- 데이터 출처: `run_release.sh S2_flat` 이 롤아웃 옆에 남기는 `exo_torque.pt`
  (롤아웃 `.motion` 에는 사람 관절 69개만 담기고 모터 2개는 빠지기 때문이다)
- `render_release.sh` 가 이 파일을 자동으로 찾는다. 다른 위치면 `EXO_TORQUE_LOG=<경로>`
- **S1 은 사이드카가 없어 자동으로 기본 회색**이 된다 — 이 대비가 곧 두 단계의 차이다
- 경계가 뚜렷한 이산 구간이 필요하면 `EXO_COLOR_MODE=band`
룩은 `tasks_for_smpl/script/render_look.env` 하나가 단일 진실 원천이고, 렌더가 끝나면
mp4 옆에 `RENDER_LOOK.txt` 가 생겨 **실제 적용된 값이 기록**된다.

카메라만 용도에 따라 고른다:

| 환경변수 | 쓸 때 |
|---|---|
| `CAM=side` (기본) | 월드 고정 측면. **선회가 섞인 동작에 안정적** |
| `CAM=front_rel` | 진행방향 정면. 직선 보행 데모용 — 선회가 많으면 화면이 돌아 어지럽다 |
| `CAM=turntable` | 360° 회전. 슈트 전방위 확인 |

**필요한 것은 하나뿐이다** — ★ **SMPL 바디모델** `SMPL_NEUTRAL.npz`.
라이선스 자산이라 이 릴리즈에 없다(아래 "받는 법"). 없으면 ②만 실패하고 ①은 정상 동작한다.

> ★★ **렌더에는 standalone IsaacSim 6.0.1 이 필요하다** (추론①은 venv 만으로 된다).
> `requirements-lock.txt` 의 pip `isaacsim` 패키지만으로는 **렌더가 안 된다** —
> 기본 experience(`isaacsim.exp.base`)가 요구하는 확장이 pip 배포본에 빠져 있어
> 앱이 뜨지 않는다(`No versions of isaacsim.anim.robot.schema …`, 이어서
> `replicator.agent.schema`, `util.debug_draw` … 계속 나온다).
>
> 대체 experience(`isaaclab.python.headless.rendering.kit`)를 지정하면 앱은 뜨지만
> **writer 가 프레임당 PNG 를 40~100장** 쓴다. 1222프레임이면 11만 장이 되고 마지막
> `imageio.mimsave` 가 전부 RAM 에 올려 283 GB 를 요구해 OOM 으로 죽는다
> (2026-09-01 실측). `rt_subframes` 축소와 `set_capture_on_play(False)` 로는 해결되지
> 않는다 — standalone 을 쓰는 것만이 확인된 해법이다(PNG 60/60 프레임 검증).
>
> ```bash
> ISAACSIM=/path/to/IsaacSim-6.0.1/python.sh bash render_release.sh <rollout>.motion
> ```
> IsaacSim 6.0.1 은 NVIDIA 에서 내려받는다(약 28 GB). 설치 위치는 어디든 되고
> `python.sh` 가 자기 위치 기준으로 동작하므로 폴더째 옮겨도 된다.

### SMPL 바디모델 받는 법

LBS 살 렌더에 필수다. Max Planck 연구소 라이선스라 **재배포가 금지**되어 각자 동의하고
받아야 한다.

1. https://smpl.is.tue.mpg.de 에서 계정 등록 후 라이선스 동의
2. **SMPL for Python users** 내려받기
3. `SMPL_NEUTRAL.npz` 를 아래 경로에 둔다

```
etri_라이선스동의_다운로드/smpl_models/smpl/SMPL_NEUTRAL.npz
```

경로 해석은 `tasks_for_smpl/script/etri_smpl_model_path.py` 가 담당하므로 이 자리에만 두면 된다.

### 모션 파일 고르기

| 용도 | 쓸 것 |
|---|---|
| 학습·평가(ablation) | `motions/test_motion_36_foot.pt` (36클립 한 덩어리) |
| 한 동작씩 추론·영상 | `motions/test-motion-36-foot/walk_cmu_<번호>.motion` |
| 시각화 데모 | `motions/test-motion-loop1200/walk_cmu_<번호>_loop1200.motion` |

루프 클립은 같은 걸음이 1200스텝 동안 이어지도록 이어붙인 것이라 데모 영상에 적합하다.
평가 수치는 반드시 `test_motion_36_foot` 쪽으로 낸다 — 학습에 쓴 바로 그 클립이다.

## 함께 넣지 않은 것 — upstream 사전학습 모델 받는 법

이 릴리즈는 **exosuitHS 평지 보행보조** 하나에 필요한 것만 담았다. upstream ProtoMotions 가
함께 배포하는 **다른 로봇·다른 과제**의 사전학습 모델 약 2.2GB 는 뺐다(clone 부담 때문).

| 뺀 것 | 크기 | 쓰임 |
|---|---|---|
| `data/pretrained_models/gpc_prior/` | 794MB | GPC 과제 |
| `data/pretrained_models/motion_tracker/soma-bones`, `soma_bones_fsq`, `soma_bones_fsq_amp_muon` | 803MB | soma 로봇 |
| `data/pretrained_models/motion_tracker/g1-bones-deploy` | 249MB | Unitree G1 |
| `data/pretrained_models/masked_mimic/smpl` | 219MB | masked-mimic 과제 |
| `data/pretrained_models/motion_tracker/smpl-terrains` | 127MB | 지형 보행 (다음 릴리즈 대상) |
| `examples/experiments/gpc/` | — | 위 모델이 없으면 실행 불가 |
| `protomotions/tests/test_{pretrained_model_cards,installation_docs,release_docs_and_legal}.py` | — | 위 모델을 검사하는 테스트 — 남기면 반드시 실패 |

> ✅ **우리 계보의 출발점인 `data/pretrained_models/motion_tracker/smpl` (135MB)은 들어 있다.**
> S1 재학습을 재현하려면 이것만 있으면 된다.

### 받는 법

upstream 저장소에 **git LFS** 로 들어 있다. 전체를 받으려면:

```bash
git clone https://github.com/NVlabs/ProtoMotions.git
cd ProtoMotions
git lfs install && git lfs pull
# 필요한 폴더만 이 릴리즈 트리로 복사
cp -r data/pretrained_models/motion_tracker/smpl-terrains \
      <이 릴리즈 루트>/data/pretrained_models/motion_tracker/
```

3GB 전부 받기 싫으면 **필요한 폴더만** 골라 받는다:

```bash
git clone --filter=blob:none --no-checkout https://github.com/NVlabs/ProtoMotions.git
cd ProtoMotions
git sparse-checkout set data/pretrained_models/motion_tracker/smpl-terrains
git checkout main && git lfs pull --include="data/pretrained_models/motion_tracker/smpl-terrains/**"
```

`git lfs` 가 없으면 먼저 설치해야 한다(`apt install git-lfs` 또는 https://git-lfs.com).
없이 clone 하면 실제 파일 대신 **포인터 텍스트**가 받아진다.

## 학습 계보 — S0 가 없는 이유

```
NVIDIA 사전학습(SMPL 모션 트래킹)  →  S1(슈트 무게 적응)  →  S2(모터 보조)
```

**출발점은 우리가 학습한 것이 아니다.** NVIDIA 가 ProtoMotions 저장소에 함께 배포한 SMPL
전신 모션 트래킹 사전학습 체크포인트에서 시작한다.

| | |
|---|---|
| 원본 경로 | `data/pretrained_models/motion_tracker/smpl/` (ProtoMotions 저장소 안) |
| upstream | https://github.com/NVlabs/ProtoMotions |
| 추가 커밋 | `fdd5ba9aa` — "ProtoMotions v3.1: Modular architecture and domain randomization (#172)", Chen Tessler, 2026-01-23 |
| 배포 방식 | git LFS (`.gitattributes`: `*.ckpt`, `*.pt`) |

따라서 별도의 S0(맨몸 적응) 단계가 없고, 릴리즈에도 넣지 않았다. 사전학습 가중치는 위
경로에서 `git lfs` 로 그대로 받을 수 있으므로 재현 경로는 끊기지 않는다.

## 체크포인트 선택 기준

`last.ckpt`·`score_based.ckpt` 를 그대로 쓰지 않았다 — `score_based` 는 `success_rate` 기반인데
이 지표는 epoch 1 부터 포화해 판별력이 없다.

| 단계 | 기준 | 선택된 저장본 |
|---|---|---|
| S1 | `success = 1.0` 이면서 `eval/gt_error/mean` 최저 | `epoch_200.ckpt` (gt_error 0.03107, 36/36 성공) |
| S2 | ablation 저감률(`tau_Hip_y`) 최대 (마스킹 적용) | `epoch_6400.ckpt` (저감 26.2%, 후보 9개 중 정점) |

S1 은 epoch 400 에서 접촉이 미세하게 나았으나(1.9%) 추종이 15% 악화되어 200 을 택했다.

## 결과 요약

모터를 끈 상태와 켠 상태를 같은 조건에서 비교해, **엉덩관절(`Hip_y`) 평균 |τ|** 이 얼마나
줄었는지 측정했다. 36동작 전부에 대해 잰 값이다.

| | 값 |
|---|---|
| 모터 끔 (사람이 낸 힘) | 48.93 N·m |
| 모터 켬 (사람이 낸 힘) | 36.14 N·m |
| **저감** | **26.2%** (절대 12.80 N·m) |
| 모터가 낸 힘 | 14.68 N·m (정격 23.7 의 62%) |
| 걷기 정확도 변화 | −1.3% (개선) |
| 팔 부담 변화 | −0.9% (개선) |

**36동작 전부에서 효과가 났고 부담이 늘어난 동작은 없었다.** 동작별 저감은 10.6~35.8%,
중앙값 25.8% 로 분포한다. 걷는 속도·양발지지·원래 부담 어느 것으로도 저감률이 설명되지
않았다(전부 p>0.05) — 보조 효과는 걸음의 종류를 가리지 않는다.

> 측정 프로토콜: 36동작 · 600스텝 · 512환경 · seed 0. **본보기 동작이 끝난 뒤의 구간은
> 측정에서 제외**한다(짧은 동작에서 그 구간이 최대 88% 를 차지해 수치를 왜곡했다).

## 실행 환경 (학습·검증에 쓴 것)

**아래는 이 릴리즈를 clone 해 실제로 환경을 만들고 검증한 기준 시스템이다**(2026-09-01).

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS (커널 7.0.0-28-generic) |
| GPU | NVIDIA RTX PRO 6000 Blackwell **Server** Edition · 97.9 GB × 4 |
| 드라이버 | 595.84 (CUDA 13.0) |
| Python | 3.12.3 |
| 패키지 관리 | **uv** 0.12.8 (pip 아님) |

> GPU 는 1 장이면 충분하다. 검증은 `CUDA_VISIBLE_DEVICES=3` 으로 **한 장만** 써서 했다.
> 개발은 RTX PRO 6000 Blackwell **Workstation** Edition × 1 (드라이버 580.178.04)에서 했고,
> 위 서버에서 재현·검증했다. 두 머신 모두 sm_120 이다.

### 핵심 패키지

| 패키지 | 버전 |
|---|---|
| torch | **2.11.0+cu130** (CUDA 13.0, cuDNN 9.19.0) |
| isaaclab 소스 커밋 | **`4ecd0b0`** (2026-07-16) |
| torchvision | 0.26.0 |
| numpy | 2.3.1 |
| scipy | 1.17.0 |
| lightning / pytorch-lightning | 2.6.5 |
| tensordict | 0.14.0 |
| **isaacsim** | **6.0.1.0** |
| **isaaclab** | **12.0.0** |
| isaaclab_physx | 2.9.1 |
| isaaclab_newton | 1.9.1 |
| warp-lang | 1.13.0 |
| mujoco | 3.12.0 |
| mujoco-warp | 3.8.0.3 |
| usd-exchange | 3.0.0 |
| omegaconf | 2.4.0.dev15 |
| tensorboard | 2.21.0 |

### 환경 만들기

```bash
# 1) uv 설치 (pip 대신 uv 를 쓴다 — 이 venv 가 uv 로 만들어졌다)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) venv + 패키지 206개
uv venv --python 3.12 venv_il3
VIRTUAL_ENV=$PWD/venv_il3 uv pip install --no-deps \
  --index-strategy unsafe-best-match --extra-index-url https://pypi.nvidia.com \
  -r requirements-lock.txt

# 3) isaaclab 5종 — PyPI 가 아니라 소스 editable 이다
git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout 4ecd0b0          # ← 이 커밋에서 검증됨
for m in isaaclab isaaclab_assets isaaclab_contrib isaaclab_newton isaaclab_physx; do
  VIRTUAL_ENV=<venv 경로> uv pip install --no-deps -e source/$m
done
```

> ★★ **`--no-deps` 를 빼면 설치가 안 된다.** `isaacsim-core==6.0.1.0` 이 `mujoco==3.8.0` 을
> 요구하는데 이 환경은 `mujoco==3.12.0` 을 쓴다. 의존성 해석을 켜면 이 모순에서 멈춘다.
> 전이 의존성은 `requirements-lock.txt` 에 이미 다 들어 있다.

> **`isaaclab*` 5종은 목록에 없다.** PyPI 패키지가 아니라 IsaacLab 저장소 소스의
> editable 설치이기 때문이다. 버전(12.0.0)만 맞추고 커밋이 다르면 미묘하게 다른
> 환경이 되므로 `4ecd0b0` 을 쓴다.

`requirements-lock.txt` 는 **206개**다. 개발 환경에는 383개가 있었으나 177개를 뺐다 —
ROS2 시스템 패키지(개발 머신이 시스템 ROS 를 흡수해 딸려온 것으로 PyPI 에 없다),
사내 전용 패키지, 그리고 위 `isaaclab*` 5종이다. 학습·추론·렌더에는 쓰이지 않는다.
이 206개는 별도 서버에서 **실제 설치와 import 검증을 통과**했다(2026-09-01).

> **주의**: `omegaconf` 가 안정판이 아니라 **dev 버전**(2.4.0.dev15)이다. config override 동작이
> 다르게 보이면 여기를 먼저 의심할 것.

> 렌더링은 이 venv 가 아니라 IsaacSim 자체 kit 파이썬(`IsaacSim-6.0.1/python.sh`)으로 돈다.
> 학습·추론은 venv, 렌더는 kit 파이썬인 이원 구조다 — **선택이 아니라 필수다.**
> 이유는 "① 추론 / ② 렌더" 절의 ★★ 항목에 있다.

## 알려진 사항

- **스폰 오프셋**: 프레임워크 기본 `env.ref_respawn_offset = 0.05`(50mm)는 리셋 시 캐릭터를
  공중에 띄워 착지 동작을 만든다. 이 릴리즈의 정책은 **0.0 으로 학습**했다. 추론 시에도
  `run_release.sh` 가 0.0 을 넘긴다. 직접 실행할 때는 이 값을 확인할 것.
- **시뮬레이터 선택**: IsaacLab 으로 학습한 체크포인트는 **IsaacLab 으로 추론**해야 한다.
  Newton 추론은 sim2sim 격차가 커서 별도 파인튜닝이 필요하다.
- 참조 모션의 발 접지는 프레임별로 보정된 판(`test-motion-36-foot`)이다. 원본 AMASS/CMU
  리타깃 결과는 발이 지면에서 평균 23mm 떠 있어 그대로 쓰면 학습이 나빠진다.
