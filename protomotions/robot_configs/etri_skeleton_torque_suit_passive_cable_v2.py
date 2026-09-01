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
# Backward-compatibility shim — class moved to tasks/mimic_suit_passive_cable_motions14_23dof_v2/robot_configs/
# Kept here so old resolved_configs.pt pickles (which record the original module path) can still be loaded.
from tasks.mimic_suit_passive_cable_motions14_23dof_v2.robot_configs.skeleton_torque_suit_passive_cable_v2 import (
    SkeletonTorqueSuitPassiveCableV2RobotConfig,
)

__all__ = ["SkeletonTorqueSuitPassiveCableV2RobotConfig"]
