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
#
"""
v2 locomujoco MJCF 가공 스크립트
=================================
mjcf_v2/ 파일에서 불필요 요소를 제거하여 두 버전 생성:

  27 DOF 버전 (기본, mjcf_v2/):
    - 신경근 경로 site 제거 (-P1, -P2, ... 패턴)
    - 발목 세부 DOF 제거: subtalar_angle_r/l, mtp_angle_r/l
    - 손목 세부 DOF 제거: wrist_flex_r/l, wrist_dev_r/l
    - 관련 equality constraint / actuator 제거

  31 DOF 버전 (mjcf_v2/31dof/):
    - 신경근 경로 site 제거만 적용 (발목/손목 DOF 유지)

Usage:
    python usd_convert/strip_v2_mjcf.py
"""

import os
import re
import xml.etree.ElementTree as ET

# ET 기본 namespace 처리
ET.register_namespace("", "")

# ── 제거 대상 정의 ────────────────────────────────────────────────

# 발목/손목 세부 DOF (27 DOF 버전에서 제거)
REMOVE_JOINTS_27 = {
    "subtalar_angle_r", "mtp_angle_r",
    "subtalar_angle_l", "mtp_angle_l",
    "wrist_flex_r",    "wrist_dev_r",
    "wrist_flex_l",    "wrist_dev_l",
}

# 대응 actuator 이름
REMOVE_ACTUATORS_27 = {
    "mot_subtalar_angle_r", "mot_mtp_angle_r",
    "mot_subtalar_angle_l", "mot_mtp_angle_l",
    "mot_wrist_flex_r",     "mot_wrist_dev_r",
    "mot_wrist_flex_l",     "mot_wrist_dev_l",
}

# 대응 equality constraint 이름 (joint1 == 제거 joint → equality 전체 제거)
REMOVE_EQUALITY_JOINTS = REMOVE_JOINTS_27

# 신경근 경로 site 패턴: 이름에 "-P" 가 포함된 것
MUSCLE_SITE_RE = re.compile(r".+-P\d+$")


# ── 헬퍼 함수 ─────────────────────────────────────────────────────

def remove_muscle_sites(root: ET.Element) -> int:
    """근육 경로 site(-P1, -P2, ...) 제거. 반환: 제거 수."""
    count = 0
    for parent in root.iter():
        to_remove = [
            child for child in parent
            if child.tag == "site" and MUSCLE_SITE_RE.match(child.get("name", ""))
        ]
        for child in to_remove:
            parent.remove(child)
            count += 1
    return count


def remove_dof_joints(root: ET.Element, joint_names: set) -> int:
    """지정 이름의 joint 요소를 body 내에서 제거. 반환: 제거 수."""
    count = 0
    for body in root.iter("body"):
        to_remove = [j for j in body.findall("joint") if j.get("name") in joint_names]
        for j in to_remove:
            body.remove(j)
            count += 1
    return count


def remove_equality_constraints(root: ET.Element, joint_names: set) -> int:
    """joint1 이 joint_names 에 속하는 equality constraint 제거. 반환: 제거 수."""
    count = 0
    equality = root.find("equality")
    if equality is None:
        return 0
    to_remove = [
        child for child in equality
        if child.get("joint1") in joint_names
    ]
    for child in to_remove:
        equality.remove(child)
        count += 1
    # equality 섹션이 비면 제거
    if len(list(equality)) == 0:
        root.remove(equality)
    return count


def remove_actuators(root: ET.Element, actuator_names: set) -> int:
    """지정 이름의 actuator 제거. 반환: 제거 수."""
    count = 0
    actuator_sec = root.find("actuator")
    if actuator_sec is None:
        return 0
    to_remove = [a for a in actuator_sec if a.get("name") in actuator_names]
    for a in to_remove:
        actuator_sec.remove(a)
        count += 1
    return count


def remove_sensor_refs(root: ET.Element, joint_names: set) -> int:
    """joint= 속성이 제거된 joint를 참조하는 sensor 제거. 반환: 제거 수."""
    count = 0
    sensor_sec = root.find("sensor")
    if sensor_sec is None:
        return 0
    to_remove = [s for s in sensor_sec if s.get("joint") in joint_names]
    for s in to_remove:
        sensor_sec.remove(s)
        count += 1
    return count


def process(input_path: str, output_path: str, remove_dof: bool) -> None:
    """
    input_path MJCF를 읽어 가공 후 output_path에 저장.
    remove_dof=True  → 27 DOF (신경근 + 발목/손목 제거)
    remove_dof=False → 31 DOF (신경근만 제거)
    """
    tree = ET.parse(input_path)
    root = tree.getroot()

    n_muscle = remove_muscle_sites(root)
    print(f"  신경근 site 제거: {n_muscle}개")

    if remove_dof:
        n_joints  = remove_dof_joints(root, REMOVE_JOINTS_27)
        n_eq      = remove_equality_constraints(root, REMOVE_EQUALITY_JOINTS)
        n_act     = remove_actuators(root, REMOVE_ACTUATORS_27)
        n_sensor  = remove_sensor_refs(root, REMOVE_JOINTS_27)
        print(f"  DOF 제거: joint {n_joints}개, equality {n_eq}개, actuator {n_act}개, sensor {n_sensor}개")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path, xml_declaration=True, encoding="unicode")
    size_kb = os.path.getsize(output_path) // 1024
    print(f"  → {output_path} ({size_kb} KB)")


# ── 메인 ──────────────────────────────────────────────────────────

ASSETS = "protomotions/data/assets"
STEMS = [
    "skeleton_cable_suit_v2_for_train",
    "skeleton_cable_suit_v2_for_eval",
]

for stem in STEMS:
    src = f"{ASSETS}/mjcf_v2/{stem}.xml"
    print(f"\n=== {stem} ===")

    # 31 DOF 버전 (신경근만 제거)
    print(" [31dof] 신경근만 제거:")
    process(src, f"{ASSETS}/mjcf_v2/31dof/{stem}.xml", remove_dof=False)

    # 27 DOF 버전 (신경근 + 발목/손목 제거) — 원본 파일 덮어쓰기
    print(" [27dof] 신경근 + 발목/손목 제거:")
    process(src, src, remove_dof=True)

print("\n완료!")
print(f"mjcf_v2/          : 27 DOF (신경근 제거 + 발목/손목 제거)")
print(f"mjcf_v2/31dof/    : 31 DOF (신경근 제거만)")
