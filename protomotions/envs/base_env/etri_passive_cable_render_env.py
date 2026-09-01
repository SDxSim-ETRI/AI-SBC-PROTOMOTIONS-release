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
"""PassiveCableRenderEnv: plain BaseEnv + live cable-capsule rendering.

Unlike the active-cable suit, the passive-cable suit's tendons are driven by
Newton's own native spatial-tendon physics -- there is no body-wrench force to
apply, so no custom step() logic is needed. But Newton's viewer never renders
<tendon> paths at all (see protomotions.envs.obs.cable.make_cable_render_hook
docstring), so without this the cables are invisible during --auto-record /
interactive play, exactly like the active-cable suit was before 2026-07-10.

This class exists purely to register that same render hook; everything else
is identical to BaseEnv. Safe to use for training too -- the hook is a no-op
whenever the simulator is headless (see NewtonSimulator.render()).
"""
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.obs.etri_cable import make_cable_render_hook


class PassiveCableRenderEnv(BaseEnv):
    """BaseEnv with the shared cable-capsule render hook registered."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.simulator, "_render_hook"):
            self.simulator._render_hook = make_cable_render_hook(self.simulator)
