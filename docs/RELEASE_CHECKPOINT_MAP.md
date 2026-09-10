# 릴리즈 체크포인트 지도 — v1.0 (hs2)

어느 학습 산출물이 어느 릴리즈 체크포인트가 됐는지, **왜 그 시드를 골랐는지** 남긴다.

## 선택 규칙

> **주지표(힙_y 저감률)의 중앙값 시드를 릴리즈한다.**
> 단, 다른 실행이 동결 인체로 물고 있는 체크포인트는 **사슬 일관성이 우선**한다.

**최고 시드를 고르지 않는다.** 논문은 평균 ± σ̂ 을 보고하므로, 최고를 릴리즈하면
받은 사람이 논문 수치를 재현하지 못한다. 중앙값이면 일치한다.

## 4종

| 릴리즈 | 원본 (144) | 파일 | 근거 |
|---|---|---|---|
| `checkpoints/S1/` | `mimic_smpl/output_IL_s1_flat_hs2_motions36/` | `epoch_200.ckpt` | **선택 불가** — S2_flat 3시드 전부가 이것을 동결 인체로 학습했다. 다른 시드를 넣으면 짝이 어긋난다 |
| `checkpoints/S2_flat/` | `mimic_smpl_exosuitHS/output_IL_s2_flat_hs2_seed2_motions36/` | `epoch_20200.ckpt` | **seed2 = 중앙값.** −25.35 % (seed0 −25.13 / seed1 −27.06, 평균 −25.85 · σ̂ 1.06). 추종도 셋 중 최良(gt_err 0.03130) |
| `checkpoints/S1_rough/` | `mimic_smpl/output_IL_s1_rough_hs2_E_motions3_env16k/` | `last.ckpt` | 커리큘럼 단일 계보의 종점 (`cmu_02_01 → A → B → C → D → E`) |
| `checkpoints/S2_rough/` | `mimic_smpl_exosuitHS/output_IL_s2_rough_hs2_thresholdonly_3motions_seed0/` | `epoch_4000.ckpt` | **seed0 = 중앙값.** −15.84 % (seed1 −15.54 / seed2 −16.38, 평균 −15.92 · σ̂ 0.43) |

각 폴더에 5개 파일을 함께 담는다 — `last.ckpt` · `resolved_configs_inference.pt` ·
`resolved_configs.pt` · `experiment_config.py` · `config.yaml`.

> `resolved_configs.pt` 는 **S2 가 동결 인체 actor 구조를 복원하는 데 필요**하다. 빠뜨리면
> S2 추론이 실패한다.

## 동결 인체 사슬

```
S1        ← 웜스타트 output_IL_pretrained
S2_flat   ← 동결 인체 = S1
S1_rough  ← 웜스타트 S1 → 커리큘럼 A~E
S2_rough  ← 동결 인체 = S1_rough
```

**`run_release.sh` 가 `env.frozen_human_ckpt` 를 이 트리 기준으로 덮어쓴다.** 체크포인트에
박힌 경로는 학습 머신의 절대경로(`/home/user/PM_Tasks/…`)라 받는 쪽에서 그대로 쓸 수 없다.

## 자산 세대

**전부 hs2** (`_for_train_v2`). hs1 자산(`_for_train.usda` / `.xml`)은 v1.0 에서 **제거**했다.
hs1 과 hs2 는 가방 부착 위치(Torso→Pelvis)와 대퇴 box 높이가 달라 **기준선 Nm 을 서로 쓸 수 없다.**
근거: `docs/NAMING.md` 자산 세대 정의.

## 체형 자산 (평가 전용, 학습 없음)

| 라벨 | betas[0] | 맨몸 | 무릎 여유 | 성격 |
|---|---|---|---|---|
| `short` | −1.0 | 54.04 kg | **+0.1 mm** | 사전등록 요인 수준 |
| `std`(= v2) | 0.0 | 63.31 kg | +20.2 mm | 〃 · **학습 체형** |
| `tall` | +1.0 | 74.00 kg | +40.3 mm | 〃 |
| `tall15` | +1.5 | 79.92 kg | +50.3 mm | 상한 탐색점 (접미 숫자 = betas × 10) |
| `tall20` | +2.0 | 86.14 kg | +60.4 mm | 〃 |

`BODY=<라벨>` 로 고른다. 체형별 레퍼런스 모션도 함께 담았다(`test-motion-loop1200-<라벨>/`) —
관절각은 네 체형이 동일하고 body 위치만 각 체형의 FK 로 다시 푼 것이다(프로토콜 P2).

## 재현 근거

| | |
|---|---|
| 시드 변동·통계 규칙 | `docs/PAPER_BASIS.md` |
| 평지 동작 분해 | `PAPER4_1.md` |
| 지형 × 체형 | `PAPER4_2.md` |
| 명명 규칙 | `docs/NAMING.md` |
