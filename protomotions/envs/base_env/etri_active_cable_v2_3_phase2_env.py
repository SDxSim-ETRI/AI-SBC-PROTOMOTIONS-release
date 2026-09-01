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
"""ActiveCableV2Phase2Env: effort-optimization on top of ActiveCableV2TensionEnv.

v1 of this env (2026-07-09) used a pure-penalty reward:

    reward -= lambda_torque*torque_cost + lambda_tension*tension_cost
             + lambda_smoothness*smoothness_cost + lambda_symmetry*symmetry_cost

Trained to convergence (~epoch 1100, 2026-07-10) it collapsed to near-zero
tension on all 4 cables (0.001N) -- every term in that reward decreases
together when the suit does nothing, so "don't use the cable" is the trivial
global optimum. Tracking stayed perfect (success 100%, gt_error ~0.10m) and
skeleton torque did drop (32->10.5), but only because *removing* the
asymmetric, uncoordinated Phase-1 pulling stopped fighting the walk -- not
because the policy learned to use the cable well. See INFO.md "Phase 2 종료".

v2 (this version, 2026-07-10) is a rewrite based on:
  - reading a sister project (LocoMujoco_with_mjwarp/walking_assistance) that
    hit the same "assist usage collapses to ~0" failure and fixed it by
    replacing usage *penalties* with a positive *assist power* reward
    (assist:anti-assist:effort-penalty ratio was roughly 1.0 : 0.4 : 0.08 --
    the usage incentive dominates), demoting L/R symmetry from a reward term
    to a monitoring-only metric, and zeroing/removing the raw tension penalty
  - a second opinion on that research emphasizing credit assignment: reward
    terms tied directly to the current action (assist power is computed from
    this step's tension) train far better with PPO than terms tied to a
    delayed, confounded outcome (human_torque_cost alone can drop for reasons
    unrelated to the cable, so it's a weak training signal on its own), and
    that a tension penalty should gate on a safety threshold rather than
    penalizing raw magnitude unconditionally

New reward:

    reward = (Phase 1 reward, unchanged)
           + lambda_assist * assist_power        # NEW, positive, dominant
           - lambda_anti   * anti_assist_power    # NEW, penalizes fighting the motion
           - lambda_torque * human_torque_cost    # kept, small: secondary/weak signal
           - lambda_smooth * cable_smoothness_cost
           - lambda_safety * tension_overlimit_cost  # NEW: 0 below threshold, quadratic above

symmetry_cost is still computed and written to self.extras (so it still shows
up in TensorBoard as env/cable_symmetry_cost_mean) but is no longer added to
self.rew_buf -- monitoring only, per the research above.

Physical meaning of assist/anti-assist power: a cable is doing positive
(assistive) mechanical work on the body exactly when it is under tension AND
its path length is shortening (a winch pulling in slack cable under load).
compute_cable_velocities() already gives d(path_length)/dt per cable (reused
from the cable_obs machinery); relu(-dlen/dt) isolates the shortening
(assistive) direction and relu(+dlen/dt) isolates the lengthening (the cable
resisting/fighting the motion while still under tension) direction. This is
the same distinction the sister project's exo_assist_power_sum /
exo_anti_assist_sum makes.
"""
import torch
from torch import Tensor

from protomotions.envs.base_env.etri_active_cable_v2_3_tension_env import ActiveCableV2TensionEnv
from protomotions.envs.obs.etri_cable import compute_cable_velocities
from protomotions.utils.rotations import quat_angle_diff_norm

# Initial weights (v2, first smoke test) were assist:anti:torque = 1.0:0.4:0.0005
# raw lambdas, following the sister-project *ratio* (1.0 : 0.4 : 0.08) without
# renormalizing for the fact that our reward isn't bounded/exp(-cost) like theirs.
# A 39-epoch smoke test (2026-07-10) showed why that matters: at the reached
# operating point (assist_power~13.7, torque_cost~1580), the SCALED contribution
# was lambda_assist*assist_power ~= 13.7 vs lambda_torque*torque_cost ~= 0.79 --
# a ~17:1 ratio far more lopsided than intended. Result: cable usage climbed
# nicely (active_frac 0.38->0.74, no collapse) but human_torque_cost also rose
# (1132->1580) instead of falling -- the policy was chasing raw assist_power
# (cranking tension against +/- 0 net benefit) rather than torque reduction.
#
# v2.1 (this revision) rebalances so the two scaled contributions are close to
# parity at that same operating point (0.1*13.7=1.37 vs 0.001*1580=1.58) instead
# of assist dominating 17:1 -- assist still meaningfully positive (won't collapse
# to the v1 zero-tension optimum) but no longer free to inflate tension with only
# a token torque cost pushing back.
#
# A 181-epoch smoke test of v2.1 (2026-07-10) confirmed the fix: human_torque_cost
# now falls (930->412) instead of rising. But it surfaced a *different* regression:
# cable1/cable4 tension rose (8.4->16.6N / 2.8->9.1N) while cable2/cable3 collapsed
# toward 0 (0.67->0.14N / 1.0->0.23N) -- symmetry_cost (monitoring-only) got 4x
# worse (140->590). Averaging assist_power across the 4 cables before rewarding
# lets 1-2 cables carry the whole assist term while the others free-ride at zero;
# nothing in a *mean* pushes usage toward all 4.
#
# v2.2 (this revision) rewards assist_power *per cable* through a concave (sqrt)
# transform before summing, instead of averaging the raw linear power: sum(sqrt(p_i))
# is maximized, for a fixed total power budget, when that budget is spread evenly
# across cables (concavity == diminishing marginal return per cable) -- so getting
# the same total assist from 4 cables at ~2W each is worth more reward than 1 cable
# at ~8W and 3 at 0W. This directly targets the free-riding failure above without
# reintroducing symmetry_cost as a penalty (which the sister-project research
# advised against -- see v2 notes below).
#
# A 254-epoch smoke test of v2.2 (2026-07-10) confirmed both earlier fixes held:
# all 4 cable tensions rose together (no free-riding) and human_torque_cost kept
# falling (928->475). But visual review of the recording (user-reported, then
# confirmed numerically) found a *new* failure the aggregate tracking metrics
# never caught: pelvis yaw oscillates ~80 deg peak-to-peak (~38 deg detrended)
# during rollout vs ~7.7 deg in the reference "walk" mocap clip itself -- roughly
# 5x the natural range. eval/gt_error and env/termination/tracking_error_mean
# both stayed at their usual low values throughout, because the existing gr_rew
# tracking term (mimic_tracking_rewards_factory, gr_weight=0.3) averages
# quaternion error over all 29 bodies -- a single body (pelvis) being very wrong
# is diluted to 1/29th of that term and barely moves the average. This reads as
# reward hacking: twisting the pelvis changes cable path lengths (dlen/dt) without
# actually walking better, so it's a cheap way to pump assist_reward.
#
# v2.3 (this revision) adds a pelvis-orientation penalty computed directly against
# the reference motion's pelvis quaternion at the current timestep (via
# self.context.mimic.ref_state, the same reference the mimic reward already reads),
# so it isn't diluted across other bodies the way gr_rew is. Uses the raw squared
# angle (quat_angle_diff_norm), not an exp(-cost) saturating form, so the gradient
# stays informative even at the large errors seen above instead of flattening out.
#
# A 203-epoch smoke test of v2.3 (2026-07-10) confirmed the pelvis fix:
# cable_pelvis_ori_cost fell 13x (0.236->0.018) and a recording looked visually
# natural (previously-obvious twisting gone), while human_torque_cost kept falling
# (937->560). But with the twist "cheat" closed off, cable1/cable4 tension kept
# climbing (8.4->20.4N / 2.9->9.7N) while cable2/cable3 stayed flat and low
# (0.67->1.1N / 1.0->1.2N) -- the same diagonal-pair imbalance from the original
# Phase-1 problem this whole redesign started from, just without the twist this
# time. sum(sqrt(p_i)) (v2.2) gives diminishing returns per cable but never
# actively rewards the *weakest* cable specifically -- if cable1/4 are genuinely
# more effective at this gait phase, concavity alone doesn't force parity.
#
# v2.4 (this revision) adds an explicit max-min fairness bonus on top of the v2.2
# sum(sqrt(p_i)) term: lambda_min_assist * sqrt(min_i(p_i)) rewards specifically
# whatever cable is currently doing the *least* work, independent of how much the
# other 3 are doing. A policy that lets cable2/3 sit near 0 gets ~0 from this term
# no matter how much cable1/4 contribute, giving a direct, undiluted incentive to
# raise the floor rather than relying on concavity of the sum to do it indirectly.
#
# A 219-epoch smoke test of v2.4 (2026-07-10) showed only a partial fix: cable2
# rose 3x (0.67->2.23N, still small) and cable3 barely moved (1.0->1.32N), while
# cable1/cable4 kept climbing (21.2N/10.8N). env/cable_min_assist_power_mean stayed
# at 0.0000 for the *entire* run. Root cause: min_i(p_i) is taken at a single
# instant. In a normal gait, every cable is briefly near-idle at some point in the
# cycle (e.g. during its leg's swing phase) -- that's expected, not a fairness
# problem. A per-timestep min can't distinguish "this cable is idle right now,
# like all 4 are in turn" from "this cable is idle *every* time" (chronic
# under-use), so it stayed ~0 and provided almost no gradient either way.
#
# v2.5 (this revision) replaces the instantaneous min with an EMA of each cable's
# assist_power_per_cable (~2.5s / ~2 gait-cycle window at 20fps), then takes
# lambda_min_assist * sqrt(min_i(ema_i)) -- this rewards whichever cable has the
# lowest *time-averaged* usage, which only stays low if that specific cable is
# chronically neglected across cycles, not merely mid-swing this instant.
#
# A ~300-epoch smoke test of v2.5 (2026-07-10) + a 400-step offline rollout
# analysis (2026-07-13, scripts/analyze_cable_balance.py) showed the diagonal
# imbalance fully intact: cable1/cable4 mean tension 20.9N/9.3N vs cable2/cable3
# ~0N (active <5% of steps), EMA assist power 1.9W/1.3W vs ~0.006W. Root cause
# of the min-assist term's impotence: it is a *bonus* -- when a cable is fully
# dead the term contributes ~sqrt(1e-6)~=0 and, with exploration noise this low
# (actor_logstd=-2.9, sigma~=0.055 pre-tanh), the policy never stumbles onto the
# reward, so there is no learnable signal at the exact operating point we need
# to escape. A colleague's muscle-model project hit the same failure mode and
# escaped it with per-cable *penalties* on inactivity rather than bonuses.
#
# v2.6 (this revision) adds exactly that: a per-cable hinge penalty on the same
# EMA the min-assist term already maintains,
#     dead_penalty = lambda_dead * sum_i max(0, floor - ema_i)
# Unlike sqrt(min(ema)): (a) it fires for EVERY chronically idle cable at once,
# not just the argmin; (b) while a cable stays dead it exerts a constant
# (linear-hinge) pressure that cannot fade to zero the way a bonus does; (c) the
# floor gives an explicit usage target. Scale: measured active-cable EMA is
# 1.3-1.9W, so floor=0.3W (~20% of active level) with lambda_dead=0.3 makes a
# fully dead cable cost 0.09/step -- same order as the whole assist bonus
# (~0.25/step at the current operating point), strong enough to matter through
# advantage normalization but well below the ~1.3/step tracking term.
# min-assist (v2.5) is kept unchanged alongside it.
#
# A 200-epoch smoke test of v2.6 (2026-07-13) showed the dead penalty works --
# all 4 cables came alive (assist EMA above the floor by epoch ~150) -- but
# uncovered a runaway the diagonal-pair regime had been masking: once a cable is
# "on", the UNBOUNDED assist term sum(sqrt(p_i)) pays for pumping it harder
# forever. The policy cranked all 4 tensions to ~27N with violent oscillation
# (smoothness_cost 40->168), assist power 8W+/cable, and sacrificed tracking to
# do it: episode length collapsed 132->10 steps and the epoch-200 full eval
# diverged completely (gt_error 37m, first-ever motion failure -> crashed an
# unexercised failure-handling path in motion sampling). v2.5 never triggered
# this only because 2 dead cables kept total pumping income below critical.
#
# v2.6.1 (this revision) caps the per-cable assist power credited to the reward:
#     assist_reward = sum_i sqrt(min(p_i, p_cap)),  p_cap = 2.0 W
# i.e. roughly the healthy active-cable level measured on v2.5 (1.3-1.9W). Below
# the cap behavior is unchanged; at the cap the pumping gradient is exactly zero,
# so "turn everything to 27N and shake" earns nothing over normal use. Together
# with the dead-penalty floor this states the actual objective: keep every
# cable's time-averaged assist power in the [0.3, 2.0] W band while tracking.
#
# A 100-epoch smoke test of v2.6.1 (2026-07-13) confirmed the cap kills the
# pumping regime (tensions stayed <11N, smoothness_cost ~51 vs 168 in the
# runaway) but tracking still collapsed the same way (episode length 712 at
# epoch 25 -> 9.5 at epoch 100, terminate 10.5%/step vs 0% in every pre-v2.6
# run): lambda_dead=0.3 pushes the policy toward cable2/3 activation harder
# than PPO (actor lr 2e-5) can re-stabilize the gait it perturbs. The penalty
# was sized for "noticeable vs assist bonus" but that turned out to be
# "dominant vs the tracking margin".
#
# v2.6.2 (this revision): lambda_dead 0.3 -> 0.1. A fully dead cable now costs
# 0.03/step (vs 0.09) -- still a persistent, non-vanishing pressure (unlike the
# v2.5 bonus whose gradient dies at 0), but small enough that breaking the gait
# is never worth it. Expect slower cable2/3 activation; judge by trend, not
# level, at smoke-test scale.
#
# A 100-epoch smoke test of v2.6.2 (2026-07-13) collapsed identically, just
# slower (episode length 720 at epoch 25 -> 232 at 50 -> 10 at 100). So the
# instability is not about magnitude: the penalty keeps pulling toward cable
# activation even WHILE the gait is degrading, fighting the recovery gradient
# exactly when tracking needs it most -- lowering lambda only delays the spiral.
#
# v2.6.3 (this revision) gates the dead penalty on tracking health instead of
# shrinking it further: it applies only in states whose mean body-position error
# is below _DEAD_PENALTY_GT_ERR_GATE (0.25m, half the 0.5m termination
# threshold). Lexicographic intent: tracking first, cable-usage fairness only
# where tracking is already sound. When the gait wobbles the pressure vanishes
# and the pure tracking gradient takes over; once recovered, the pressure
# resumes. lambda_dead stays 0.1 (v2.6.2) since the gate, not the size, was the
# missing piece.
_DEFAULT_LAMBDA_ASSIST = 0.1
_DEFAULT_LAMBDA_ANTI = 0.04
_DEFAULT_LAMBDA_TORQUE = 0.001  # back to v1's value -- needs real weight to matter now
_DEFAULT_LAMBDA_SMOOTHNESS = 0.001  # unchanged from v1
_DEFAULT_LAMBDA_SAFETY = 0.01
_DEFAULT_LAMBDA_PELVIS_ORI = 5.0  # new in v2.3 -- see module docstring
_DEFAULT_LAMBDA_MIN_ASSIST = 0.1  # v2.4 -- see module docstring
_ASSIST_EMA_ALPHA = 0.02  # v2.5 -- ~50-step (~2.5s @ 20fps) smoothing window
_DEFAULT_LAMBDA_DEAD = 0.1  # v2.6.2 (was 0.3 in v2.6) -- see module docstring
_DEAD_ASSIST_FLOOR_W = 0.3  # v2.6 -- ~20% of the measured active-cable EMA (1.3-1.9W)
_ASSIST_POWER_CAP_W = 2.0  # v2.6.1 -- healthy active-cable level; kills the pumping gradient
_DEAD_PENALTY_GT_ERR_GATE = 0.25  # v2.6.3 -- apply dead penalty only below this tracking error [m]
_SAFE_TENSION_N = 100.0  # headroom below the 140N pull cap (_MAX_TENSION_N in the parent env)


class ActiveCableV2Phase2Env(ActiveCableV2TensionEnv):
    """ActiveCableV2TensionEnv + positive assist-power reward (v2, see module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_assist = _DEFAULT_LAMBDA_ASSIST
        self.lambda_anti = _DEFAULT_LAMBDA_ANTI
        self.lambda_torque = _DEFAULT_LAMBDA_TORQUE
        self.lambda_smoothness = _DEFAULT_LAMBDA_SMOOTHNESS
        self.lambda_safety = _DEFAULT_LAMBDA_SAFETY
        self.lambda_pelvis_ori = _DEFAULT_LAMBDA_PELVIS_ORI
        self.lambda_min_assist = _DEFAULT_LAMBDA_MIN_ASSIST
        self.lambda_dead = _DEFAULT_LAMBDA_DEAD
        self.dead_assist_floor_w = _DEAD_ASSIST_FLOOR_W
        self.assist_power_cap_w = _ASSIST_POWER_CAP_W
        self.dead_penalty_gt_err_gate = _DEAD_PENALTY_GT_ERR_GATE
        self.assist_ema_alpha = _ASSIST_EMA_ALPHA
        self.safe_tension_n = _SAFE_TENSION_N
        self._prev_tension = None
        self._assist_ema = None

    def step(self, action: Tensor):
        obs, rewards, dones, terminated, extras = super().step(action)

        tension = torch.stack(
            [extras[f"cable_tension_{i + 1}"] for i in range(4)], dim=-1
        )  # [E, 4], N

        if self._prev_tension is None:
            self._prev_tension = tension.clone()

        body_pos = self.context.current.rigid_body_pos
        body_rot = self.context.current.rigid_body_rot
        body_vel = self.context.current.rigid_body_vel
        dlen_dt = compute_cable_velocities(body_pos, body_rot, body_vel)  # [E, 4], m/s

        # Positive work (assist): cable under tension while shortening (dlen/dt < 0).
        assist_power_per_cable = tension * (-dlen_dt).clamp(min=0.0)  # [E, 4]
        # Negative work (fighting): cable under tension while lengthening (dlen/dt > 0).
        anti_assist_power = (tension * dlen_dt.clamp(min=0.0)).mean(dim=-1)  # [E]

        # Reward assist through a concave (sqrt) transform per cable, summed, rather
        # than averaging the raw linear power: sum(sqrt(p_i)) is maximized, for a
        # fixed total power budget, when that budget is spread evenly across the 4
        # cables -- a linear mean has no such preference and let 1-2 cables carry
        # the whole assist term while the others free-rode at ~0 (see module
        # docstring, v2.2 note). assist_power (linear mean) is kept in extras for
        # monitoring/comparison; assist_reward is what actually enters rew_buf.
        assist_power = assist_power_per_cable.mean(dim=-1)  # [E], monitoring only
        # v2.6.1: per-cable cap -- above p_cap the pumping gradient is zero, so
        # cranking tension past the healthy band earns nothing (module docstring).
        assist_reward = torch.sqrt(
            assist_power_per_cable.clamp(max=self.assist_power_cap_w) + 1e-6
        ).sum(dim=-1)  # [E]

        # Max-min fairness bonus over a time-averaged (EMA) per-cable usage, not
        # the instantaneous value -- see module docstring, v2.5 note. Rewards
        # whichever cable has the lowest usage *across recent cycles*, not
        # whichever happens to be mid-swing this exact step.
        if self._assist_ema is None:
            self._assist_ema = assist_power_per_cable.clone()
        else:
            self._assist_ema = (
                self.assist_ema_alpha * assist_power_per_cable
                + (1.0 - self.assist_ema_alpha) * self._assist_ema
            )
        min_assist_power = self._assist_ema.min(dim=-1).values  # [E]
        min_assist_reward = torch.sqrt(
            min_assist_power.clamp(max=self.assist_power_cap_w) + 1e-6
        )  # [E]

        # v2.6: per-cable hinge penalty on chronic (EMA) inactivity -- unlike the
        # min-assist bonus above, this keeps a constant pressure on EVERY cable
        # sitting below the usage floor. See module docstring, v2.6 note.
        # v2.6.3: gated on tracking health -- pressure only where the gait is
        # sound, so it can never fight the recovery gradient (docstring).
        ref_body_pos = self.context.mimic.ref_state.rigid_body_pos  # [E, B, 3]
        gt_err = (
            (self.context.current.rigid_body_pos - ref_body_pos)
            .norm(dim=-1)
            .mean(dim=-1)
        )  # [E], m
        tracking_healthy = (gt_err < self.dead_penalty_gt_err_gate).float()  # [E]
        dead_penalty = (
            (self.dead_assist_floor_w - self._assist_ema).clamp(min=0.0).sum(dim=-1)
            * tracking_healthy
        )  # [E]

        torque_cost = extras["skeleton_torque_abs"].pow(2)  # [E]
        smoothness_cost = (tension - self._prev_tension).pow(2).mean(dim=-1)  # [E]
        safety_cost = (tension - self.safe_tension_n).clamp(min=0.0).pow(2).mean(dim=-1)  # [E]
        # Monitoring-only (not added to reward) -- see module docstring.
        symmetry_cost = (
            (tension[:, 0] - tension[:, 1]).pow(2)  # cable1(L-post) vs cable2(R-post)
            + (tension[:, 2] - tension[:, 3]).pow(2)  # cable3(L-ant) vs cable4(R-ant)
        )  # [E]

        # Pelvis-orientation tracking, computed directly against the reference
        # motion (not diluted across all 29 bodies like the existing gr_rew term)
        # -- see module docstring, v2.3 note.
        ref_pelvis_rot = self.context.mimic.ref_state.rigid_body_rot[:, 0, :]  # [E, 4] xyzw
        pelvis_rot = body_rot[:, 0, :]  # [E, 4] xyzw
        pelvis_ori_cost = quat_angle_diff_norm(pelvis_rot, ref_pelvis_rot, w_last=True)  # [E], rad^2

        self._prev_tension = tension.clone()
        # Reset the EMA for envs whose episode just ended so the next episode's
        # window doesn't start blended with the previous one's usage.
        done_mask = dones.bool() if dones.dtype != torch.bool else dones
        if done_mask.any():
            self._assist_ema[done_mask] = assist_power_per_cable[done_mask]

        extras["cable_assist_power"] = assist_power
        extras["cable_assist_reward"] = assist_reward
        extras["cable_min_assist_power"] = min_assist_power
        extras["cable_min_assist_reward"] = min_assist_reward
        extras["cable_dead_penalty"] = dead_penalty
        extras["cable_dead_gate_open"] = tracking_healthy
        for i in range(4):
            extras[f"cable_assist_power_{i + 1}"] = assist_power_per_cable[:, i]
        extras["cable_anti_assist_power"] = anti_assist_power
        extras["human_torque_cost"] = torque_cost
        extras["cable_smoothness_cost"] = smoothness_cost
        extras["cable_safety_cost"] = safety_cost
        extras["cable_symmetry_cost"] = symmetry_cost
        extras["cable_pelvis_ori_cost"] = pelvis_ori_cost
        # Utilization metrics recommended alongside the reward change (see INFO.md):
        # mean tension and fraction of steps with any cable meaningfully engaged.
        extras["cable_tension_mean_all"] = tension.mean(dim=-1)
        extras["cable_active_frac"] = (tension > 1.0).float().mean(dim=-1)

        self.rew_buf += (
            self.lambda_assist * assist_reward
            + self.lambda_min_assist * min_assist_reward
        )
        self.rew_buf -= (
            self.lambda_anti * anti_assist_power
            + self.lambda_torque * torque_cost
            + self.lambda_smoothness * smoothness_cost
            + self.lambda_safety * safety_cost
            + self.lambda_pelvis_ori * pelvis_ori_cost
            + self.lambda_dead * dead_penalty
        )

        return obs, self.rew_buf, dones, terminated, extras
