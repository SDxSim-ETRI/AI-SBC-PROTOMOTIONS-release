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
ETRI 로봇 설정 레지스트리 — upstream `factory.py` 와 분리
========================================================

`factory.py` 는 upstream 파일이다. 이전 fork 는 거기에 `elif` 25행을 직접 넣어
upstream 이 그 함수를 개편할 때마다 충돌했다 (2026-07-31 merge 시험에서 91행 충돌).

이 파일은 그 등록 내용을 **딕셔너리 하나**로 옮긴 것이다. `factory.py` 에는
3행짜리 훅만 남으므로 앞으로 충돌 가능성이 거의 없다.

CLAUDE.md §2 규칙:
  - 별도 폴더를 만들 수 있으면 폴더에 `etri_` 접두어
  - 원본 폴더 안에 만들어야 하면 **파일명 앞에** `etri_` 접두어  ← 이 파일이 그 경우
    (`robot_config()` 가 이 위치에서 로봇을 찾으므로 폴더 분리가 불가)

## 로봇 추가 방법

`ETRI_ROBOT_CONFIGS` 에 한 줄 추가하면 된다. `factory.py` 는 건드리지 않는다.

    "my_robot": ("etri_tasks_xxx.robot_configs.my_robot", "MyRobotConfig"),

import 는 **지연(lazy)** 이다 — 요청된 로봇만 import 하므로, 아직 이관하지
않은 태스크 폴더가 있어도 다른 로봇 사용에 영향이 없다.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from typing import Dict, Optional, Tuple

# ── [ETRI 2026-08-26] `etri_tasks_for_smpl` → `tasks_for_smpl` 모듈 별칭 ────────
#
# 왜 필요한가:
#   PM_Tasks 루트에 `etri_tasks_for_smpl -> tasks_for_smpl` 심볼릭 링크가 있었고,
#   그 이름이 **파이썬 모듈 경로로** 두 곳에 굳어 있었다 —
#     (1) 이 레지스트리, (2) 구 체크포인트의 `resolved_configs*.pt` 피클.
#   (1)은 위에서 실경로 이름으로 고쳤지만 (2)는 파일 안에 박혀 있어 못 고친다.
#   링크를 지우면 그 체크포인트들이 ModuleNotFoundError 로 언피클에 실패한다.
#
#   그래서 옛 이름을 **새 이름으로 넘겨주는 별칭**을 둔다. 링크가 있든 없든 동작하며,
#   앞으로 저장되는 체크포인트에는 `tasks_for_smpl.*` 만 들어간다.
#
# 영향받는 체크포인트 (2026-08-26 실측):
#   tasks_for_smpl/mimic_smpl/output_isaaclab_s1_exosuitHS_motions36/resolved_configs*.pt
# 2026-08-26 저장소 재편으로 옛 이름이 두 벌 생겼다. 구 체크포인트 피클이
# 이 이름들로 모듈을 요청하므로 새 이름으로 넘겨준다.
#   · etri_tasks_for_smpl.*  → tasks_for_smpl.*        (심볼릭 링크 제거)
#   · tasks.mimic_*          → tasks_for_skeleton.mimic_*  (tasks/ 한 단계 제거)
_ALIASES = {
    "etri_tasks_for_smpl": "tasks_for_smpl",
    "tasks": "tasks_for_skeleton",
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec):
        return importlib.import_module(self._target)

    def exec_module(self, module) -> None:  # 이미 로드된 모듈이므로 할 일 없음
        pass


class _EtriTasksAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for old, new in _ALIASES.items():
            if fullname == old or fullname.startswith(old + "."):
                return importlib.util.spec_from_loader(
                    fullname, _AliasLoader(new + fullname[len(old):])
                )
        return None


if not any(isinstance(f, _EtriTasksAliasFinder) for f in sys.meta_path):
    sys.meta_path.append(_EtriTasksAliasFinder())
# ──────────────────────────────────────────────────────────────────────────────

# robot_name → (모듈 경로, 클래스 이름)
#
# ⚠️ 2026-07-31 이관 상태 — **아래 모듈은 아직 이 저장소에 없다.**
#   학습은 old fork(`~/ProtoMotions`)에서 계속하고, **릴리즈 시점에만** 가져오는 방침이다.
#   (학습 과정 산출물까지 이관하면 20 GB, 릴리즈만 가져오면 1.4 GB)
#     - `tasks_for_smpl.*` : 릴리즈 대기 — `(PM_Tasks)/scripts/etri_import_task_release.sh`
#     - `tasks.*`               : 미이관. 가져온 뒤 아래 경로를 `etri_tasks.*` 로 갱신할 것.
#   지연 import 이므로 해당 로봇을 요청할 때만 ImportError 가 난다.
ETRI_ROBOT_CONFIGS: Dict[str, Tuple[str, str]] = {
    # --- SMPL 계열 (외골격) ---
    # SMPL + Cosmo Robotics 전신 외골격. 24 body / 69 DOF = 맨몸과 동일하고
    # 슈트는 질량 8.00kg 만 추가한다.
    "smpl_exosuitCR": (
        "tasks_for_smpl.mimic_smpl_exosuitCR.robot_configs.smpl_exosuitCR",
        "SmplExosuitCRRobotConfig",
    ),
    # [ETRI 2026-08-25] HS 등록 누락 복구.
    #   `smpl_exosuitHS.py` 는 tasks_for_smpl/mimic_smpl/robot_configs/ 에 존재하는데
    #   이 레지스트리에 CR 만 있어 `--robot-name smpl_exosuitHS` 학습이
    #   `ValueError: Invalid robot name` 으로 실패했다(저장소 통합 시 유실 추정).
    #   추론은 resolved_configs 피클로 로봇 설정을 복원하므로 드러나지 않았고,
    #   **학습에서만** 막혔다.
    # SMPL + HyperShell 힙 외골격. 24 body / 69 DOF = 맨몸과 동일, 슈트 질량 4.70 kg.
    "smpl_exosuitHS": (
        "tasks_for_smpl.mimic_smpl.robot_configs.smpl_exosuitHS",
        "SmplExosuitHSRobotConfig",
    ),
    # --- OpenSim 유래 골격 (skeleton_torque 계열) ---
    "skeleton_torque_paper1": (
        "protomotions.robot_configs.etri_skeleton_torque_paper1",
        "SkeletonTorquePaper1RobotConfig",
    ),
    "skeleton_torque_27dof": (
        "protomotions.robot_configs.etri_skeleton_torque_27dof",
        "SkeletonTorque27DofRobotConfig",
    ),
    "skeleton_torque_27dof_anatrange_smplpd": (
        "protomotions.robot_configs.etri_skeleton_torque_27dof_anatrange_smplpd",
        "SkeletonTorque27DofAnatRangeSmplPdRobotConfig",
    ),
    "skeleton_torque_27dof_anatrange_velonly": (
        "protomotions.robot_configs.etri_skeleton_torque_27dof_anatrange_velonly",
        "SkeletonTorque27DofAnatRangeVelOnlyRobotConfig",
    ),
    "skeleton_torque_27dof_smplgains": (
        "protomotions.robot_configs.etri_skeleton_torque_27dof_smplgains",
        "SkeletonTorque27DofSmplGainsRobotConfig",
    ),
    "skeleton_torque_27dof_smplrange_origpd": (
        "protomotions.robot_configs.etri_skeleton_torque_27dof_smplrange_origpd",
        "SkeletonTorque27DofSmplRangeOrigPdRobotConfig",
    ),
    "skeleton_torque": (
        "protomotions.robot_configs.etri_skeleton_torque",
        "SkeletonTorqueRobotConfig",
    ),
    "skeleton_torque_suit": (
        "protomotions.robot_configs.etri_skeleton_torque_suit",
        "SkeletonTorqueSuitRobotConfig",
    ),
    "skeleton_torque_suit_muscle": (
        "protomotions.robot_configs.etri_skeleton_torque_suit_muscle",
        "SkeletonTorqueSuitMuscleRobotConfig",
    ),
    "skeleton_torque_suit_passive_cable": (
        "protomotions.robot_configs.etri_skeleton_torque_suit_passive_cable",
        "SkeletonTorqueSuitPassiveCableRobotConfig",
    ),
    "skeleton_torque_suit_active_cable": (
        "protomotions.robot_configs.etri_skeleton_torque_suit_active_cable",
        "SkeletonTorqueSuitActiveCableRobotConfig",
    ),
    # 31 DOF 변형 (별칭 2개)
    "skeleton_torque_31dof": (
        "protomotions.robot_configs.etri_skeleton",
        "SkeletonRobotConfig",
    ),
    "skeleton": (
        "protomotions.robot_configs.etri_skeleton",
        "SkeletonRobotConfig",
    ),
    "skeleton_torque_suit_31dof": (
        "protomotions.robot_configs.etri_etrisuit",
        "EtriSuitRobotConfig",
    ),
    "etrisuit": (
        "protomotions.robot_configs.etri_etrisuit",
        "EtriSuitRobotConfig",
    ),
    "etrisuit_active_cable": (
        "protomotions.robot_configs.etri_etrisuit_active_cable",
        "EtriSuitActiveCableRobotConfig",
    ),
    # --- 태스크 폴더에 있는 것 (tasks/ 미이관 — 아래 경로는 이관 후 갱신 필요) ---
    "skeleton_torque_suit_passive_cable_v2": (
        "tasks_for_skeleton.mimic_suit_passive_cable_motions14_23dof_v2.robot_configs"
        ".skeleton_torque_suit_passive_cable_v2",
        "SkeletonTorqueSuitPassiveCableV2RobotConfig",
    ),
    "skeleton_torque_suit_passive_cable_v2_3": (
        "tasks_for_skeleton.mimic_suit_passive_cable_motions14_23dof_v2.robot_configs"
        ".skeleton_torque_suit_passive_cable_v2_3",
        "SkeletonTorqueSuitPassiveCableV23RobotConfig",
    ),
    "skeleton_torque_suit_active_cable_v2": (
        "tasks_for_skeleton.mimic_suit_active_cable_walk_23dof_v2.robot_configs"
        ".skeleton_torque_suit_active_cable_v2",
        "SkeletonTorqueSuitActiveCableV2RobotConfig",
    ),
    "skeleton_torque_suit_active_cable_v2_3": (
        "tasks_for_skeleton.mimic_suit_active_cable_walk_23dof_v2.robot_configs"
        ".skeleton_torque_suit_active_cable_v2_3",
        "SkeletonTorqueSuitActiveCableV2_3RobotConfig",
    ),
}


def etri_robot_config(robot_name: str):
    """ETRI 로봇이면 설정 인스턴스를, 아니면 None 을 반환한다.

    `factory.py` 의 upstream 분기보다 **먼저** 호출된다. None 을 반환하면
    upstream 이 자기 로봇(smpl, g1, ...)을 계속 처리한다.

    Args:
        robot_name: `--robot-name` 으로 넘어오는 이름.

    Returns:
        RobotConfig 인스턴스, 또는 ETRI 로봇이 아니면 None.

    Raises:
        ImportError: 등록은 돼 있으나 해당 모듈을 찾을 수 없을 때
            (예: `tasks/` 를 아직 이관하지 않은 상태에서 그 로봇을 요청).
            어느 경로가 없는지 알려 주기 위해 원본 예외를 감싸 다시 던진다.
    """
    entry = ETRI_ROBOT_CONFIGS.get(robot_name)
    if entry is None:
        return None

    module_path, class_name = entry
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"ETRI 로봇 '{robot_name}' 의 모듈을 찾을 수 없습니다: {module_path}\n"
            f"  원인: {e}\n"
            f"  해당 태스크 폴더가 아직 이관되지 않았을 수 있습니다 "
            f"(protomotions/robot_configs/etri_registry.py 의 경로를 확인하세요)."
        ) from e

    return getattr(module, class_name)()


def etri_robot_names() -> Tuple[str, ...]:
    """등록된 ETRI 로봇 이름 목록 (도움말·검증용)."""
    return tuple(sorted(ETRI_ROBOT_CONFIGS))
