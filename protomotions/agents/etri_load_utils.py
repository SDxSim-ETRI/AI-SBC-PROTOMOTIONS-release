# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ETRI warm-start 관용 로딩 — 형태가 달라진 체크포인트에서 이어받기
================================================================

관측/행동 차원이 바뀐 뒤 기존 체크포인트로 warm-start 하면 upstream 의 엄격한
`load_state_dict` 가 `RuntimeError` 로 죽는다. 이 모듈은 **형태가 맞는 텐서만
골라 싣는** 폴백을 제공한다.

upstream 이 2026-07 에 로딩 경로를 `_load_model_state_dict` /
`_load_ppo_training_state` 훅으로 리팩터링해 두었으므로, upstream 파일에는
try/except 몇 줄만 남기고 본문은 여기에 둔다 (CLAUDE.md §2).

upstream 이 이미 해결한 것과의 구분:
  - upstream: optimizer state 가 **없을 때** 관용 (`require_optimizers or "..." in state_dict`)
  - ETRI:     optimizer state 가 **있지만 형태가 다를 때** 관용  ← 이 모듈
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

log = logging.getLogger(__name__)


def partial_load_model_state(model: torch.nn.Module, saved_state: Dict[str, Any]) -> None:
    """형태가 일치하는 파라미터만 싣는다 (warm-start 용 폴백).

    엄격한 `load_state_dict` 가 실패한 뒤에 호출한다. 현재 모델의 state_dict 를
    기준으로, 저장본에서 **키가 있고 shape 까지 같은** 텐서만 덮어쓴 뒤
    한 번에 로드한다. 나머지는 현재(랜덤 초기화) 값을 유지한다.

    Args:
        model: 로드 대상 모델.
        saved_state: 체크포인트의 model state_dict.
    """
    current_state = model.state_dict()
    loaded, skipped = [], []

    for k, v in saved_state.items():
        if k not in current_state:
            skipped.append(k)
            continue
        try:
            if isinstance(v, torch.Tensor) and v.shape == current_state[k].shape:
                current_state[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        except Exception:
            skipped.append(k)

    model.load_state_dict(current_state)
    log.info(
        f"[ETRI] 부분 로드 완료: {len(loaded)}개 텐서 적용, {len(skipped)}개 건너뜀 "
        f"(형태 불일치 또는 미존재)"
    )
    if skipped:
        log.debug(f"[ETRI] 건너뛴 키(최대 10개): {skipped[:10]}")


def optimizer_state_is_compatible(
    optimizer: torch.optim.Optimizer, saved: Dict[str, Any], key: str
) -> bool:
    """저장된 optimizer state 를 지금 optimizer 에 실을 수 있는지 검사.

    param_group 개수와 각 group 의 파라미터 개수가 같아야 한다. 모델 구조가
    바뀌면 이 값들이 달라지고, 그대로 로드하면 `load_state_dict` 가 죽거나
    (더 나쁘게) 엉뚱한 파라미터에 모멘텀이 실린다.

    Args:
        optimizer: 현재 optimizer.
        saved: 체크포인트의 optimizer state_dict.
        key: 로그용 이름 (예: "actor_optimizer").

    Returns:
        실어도 안전하면 True.
    """
    try:
        saved_groups = saved.get("param_groups", [])
        cur_groups = optimizer.state_dict().get("param_groups", [])
        if len(saved_groups) != len(cur_groups):
            log.warning(
                f"[ETRI] {key} 건너뜀: param_group 수 불일치 "
                f"(체크포인트 {len(saved_groups)} vs 현재 {len(cur_groups)})"
            )
            return False
        for i, (sg, cg) in enumerate(zip(saved_groups, cur_groups)):
            if len(sg.get("params", [])) != len(cg.get("params", [])):
                log.warning(
                    f"[ETRI] {key} 건너뜀: group {i} 파라미터 수 불일치 "
                    f"({len(sg.get('params', []))} vs {len(cg.get('params', []))})"
                )
                return False
        return True
    except Exception as e:
        log.warning(f"[ETRI] {key} 호환성 검사 실패({e}) — 건너뜀")
        return False


def load_optimizer_state_if_compatible(
    optimizer: torch.optim.Optimizer, state_dict: Dict[str, Any], key: str
) -> bool:
    """호환될 때만 optimizer state 를 싣는다.

    Returns:
        실제로 실었으면 True. False 면 optimizer 는 초기 상태로 남는다
        (호출부는 이 값으로 adaptive LR 복원 여부를 결정해야 한다 —
        optimizer 를 안 실었는데 LR 스케일만 복원하면 어긋난다).
    """
    saved = state_dict.get(key)
    if saved is None:
        log.info(f"[ETRI] {key} 없음 — 건너뜀")
        return False
    if not optimizer_state_is_compatible(optimizer, saved, key):
        return False
    optimizer.load_state_dict(saved)
    return True
