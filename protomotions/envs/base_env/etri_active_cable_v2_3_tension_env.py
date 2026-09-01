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
"""ActiveCableV2TensionEnv: real cable-tension force control for the v2 (spool) suit.

Background
==========
The v2 suit's spool_hinge joints have no physical link to the cable tendon
(cable_exit is fixed to exo_main_body, a static body -- spool rotation never
changes tendon path length; see
tasks/mimic_suit_active_cable_walk_23dof_v2/robot_configs/skeleton_torque_suit_active_cable_v2.py).
PD-controlling spool angle (the old approach) has zero physical effect.

The original locomujoco design used a genuine tendon-force actuator
(`<motor tendon="cable1_tendon" ctrlrange="-140 0">`), but neither Newton nor
IsaacLab support tendon actuators (see
tasks/.../scripts/convert_mujoco_to_newton.py, step 5). Both engines DO support
applying an arbitrary world-frame wrench directly to a body's center of mass
(Newton: state.body_f; IsaacLab: set_external_force_and_torque), so this env
reproduces the tendon-actuator's physics manually via
protomotions.envs.obs.cable.compute_cable_body_wrench(), computed from the same
exit/anchor/insertion waypoint geometry already used for the cable_obs.

DOF mapping (skeleton_torque_suit_active_cable_v2_3, 27 DOFs)
===============================================================
  0-22: skeleton DOFs -- normal per-DOF PD, unchanged.
  23-26: cable[1-4]_tension (renamed from spool[1-4]_hinge in the v2-1 XML) --
         not PD-driven. The raw policy output for these 4 dims is read as a
         cable tension command (tanh -> [0, 140] N pull, matching the original
         motor's rated pull) and applied as a body wrench. The underlying
         joints themselves are left passive (zero PD target) since they carry
         no force -- matching the *original* locomujoco design, where spool
         bodies are explicitly documented as "visual/encoder references only".
"""
import torch
from torch import Tensor

from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.obs.etri_cable import compute_cable_body_wrench, make_cable_render_hook

_TENSION_DOF_START = 23
_NUM_CABLES = 4
_MAX_TENSION_N = 140.0  # matches original <motor ctrlrange="-140 0"> rated pull


class ActiveCableV2TensionEnv(BaseEnv):
    """BaseEnv with real cable-tension force control injected at each step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Newton-only: draw cable capsules every rendered frame (record_newton.sh
        # --auto-record included), not just in the offline render_cable_video.py
        # re-render. Newton's simulator already exposes a no-arg `_render_hook`
        # slot (called between viewer.log_state() and end_frame(), see e.g.
        # _setup_scene_box_render_hook) -- reuse it rather than adding a new
        # mechanism. No-op under IsaacLab (no such hook on that simulator).
        if hasattr(self.simulator, "_render_hook"):
            self.simulator._render_hook = make_cable_render_hook(self.simulator)

    def step(self, action: Tensor):
        self.extras = {}
        self._current_context = None
        self._current_noisy_obs = None
        self._current_raw_action[:] = action

        # Process PPO action (this also rebuilds self.context)
        action_dict = self._process_action(action, self.context)
        processed_action = action_dict["processed_action"].clone()

        # Cable tension: raw policy output -> [0, _MAX_TENSION_N] N, pull-only.
        # zero-action -> zero-tension: tanh 음수 영역을 0으로 잘라 학습 초기에
        # 케이블이 당기지 않는 상태에서 출발 ((tanh+1)/2 매핑은 초기 ~70N 편향
        # 으로 mimic 초기 학습을 불안정하게 함). actor_logstd 수준의 탐색이면
        # 절반가량 양수 영역이라 gradient 신호는 유지됨.
        raw_tension = action[:, _TENSION_DOF_START:_TENSION_DOF_START + _NUM_CABLES]
        tension = torch.tanh(raw_tension).clamp(min=0.0) * _MAX_TENSION_N

        body_pos = self.context.current.rigid_body_pos
        body_rot = self.context.current.rigid_body_rot
        wrench = compute_cable_body_wrench(
            body_pos, body_rot, self.robot_config.kinematic_info.num_bodies, tension
        )
        self.simulator.set_external_body_wrench(wrench)

        # Spool DOFs carry no force now -- hold at zero target (passive/cosmetic).
        processed_action[:, _TENSION_DOF_START:_TENSION_DOF_START + _NUM_CABLES] = 0.0

        self._current_processed_action[:] = processed_action
        self.simulator.step(processed_action, markers_callback=self.get_markers_state)
        self.post_physics_step()

        # Phase 1 판정 지표 로깅 (TensorBoard: env/<key>_mean)
        #  - cable_tension_N: 케이블별 명령 장력 (케이블이 실제로 쓰이는지)
        #  - skeleton_torque_abs: skeleton DOF PD 토크 절대평균 (effort proxy,
        #    passive 대비 감소 여부 판정용). BUILT_IN_PD는 토크가 솔버 내부에서
        #    계산되어 get_dof_forces()가 0을 반환하므로 명시적 PD 공식으로 추정:
        #    tau = kp*(target - q) - kd*qd, effort_limit 클램프.
        for i in range(_NUM_CABLES):
            self.extras[f"cable_tension_{i + 1}"] = tension[:, i]
        self.extras["skeleton_torque_abs"] = self._estimate_pd_torque_abs(
            processed_action
        )

        # BaseEnv.step() also consumes an interactive viewer reset request here,
        # but the API for that differs between local/remote protomotions
        # versions (consume_reset_request() vs simulator.user_requested_reset)
        # and it's a no-op for headless training/auto-record anyway (no
        # keyboard input to consume), so it's intentionally omitted for
        # cross-server portability.
        obs = self.get_obs()
        return obs, self.rew_buf, self.reset_buf, self.terminate_buf, self.extras

    def _estimate_pd_torque_abs(self, pd_targets: Tensor) -> Tensor:
        """Skeleton DOF(0-22)의 실현 PD 토크 절대평균 추정 [num_envs]."""
        if not hasattr(self, "_pd_kp"):
            info = self.robot_config.control.control_info
            dof_names = self.robot_config.kinematic_info.dof_names
            device = pd_targets.device
            self._pd_kp = torch.tensor(
                [info[n].stiffness or 0.0 for n in dof_names], device=device
            )
            self._pd_kd = torch.tensor(
                [info[n].damping or 0.0 for n in dof_names], device=device
            )
            self._pd_effort = torch.tensor(
                [info[n].effort_limit or torch.inf for n in dof_names], device=device
            )
        dof_state = self.simulator.get_dof_state()
        tau = self._pd_kp * (pd_targets - dof_state.dof_pos) - self._pd_kd * dof_state.dof_vel
        tau = tau.clamp(-self._pd_effort, self._pd_effort)
        return tau[:, :_TENSION_DOF_START].abs().mean(dim=-1)
