# SPDX-FileCopyrightText: Copyright (c) 2026 ETRI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# [ETRI patch] ETRI 가 추가한 컴포넌트 팩토리. upstream `component_factories.py` 를
# 직접 늘리는 대신 여기로 분리해 다음 upstream 병합에서 충돌하지 않게 한다
# (CLAUDE.md §3). 두 팩토리가 쓰는 compute 함수는 이미 upstream 쪽에 있다.
#
# 구 fork(`~/ProtoMotions_a5000`)에서 이관. 2026-08-19.
"""ETRI 추가 컴포넌트 팩토리."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent


def dof_pos_rew_factory(
    weight: float = 0.3,
    coefficient: float = -25.0,
    dof_names: list = None,
    dof_weight_map: dict = None,
) -> MdpComponent:
    """Factory for per-DOF joint angle tracking reward.

    Complements gt_rew/gr_rew (body position/rotation, averaged/diluted across
    all bodies) with a direct joint-angle term. See compute_dof_pos_rew
    docstring for the motivating failure mode (short-lever joints like elbow
    can visibly diverge from the reference while body-position tracking still
    reports low error, and can get stuck there under uniform per-DOF weight
    since 1 lagging DOF only gets 1/N of a uniform mean's gradient).

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.
        dof_names: robot_config.kinematic_info.dof_names, in the same order as
            current_dof_pos/ref_dof_pos. Required only if dof_weight_map is set.
        dof_weight_map: Optional {name_prefix: multiplier} to upweight specific
            lagging DOFs, e.g. {"elbow_flex": 4.0}. Unmatched DOFs default to
            1.0; internally renormalized to sum to num_dofs (see
            compute_dof_pos_rew). None = uniform weighting.

    Returns:
        MdpComponent configured for joint-angle tracking.
    """
    import torch as _torch

    from protomotions.envs.rewards import compute_dof_pos_rew

    static_params = {"weight": weight, "coefficient": coefficient}
    if dof_weight_map:
        if not dof_names:
            raise ValueError("dof_pos_rew_factory: dof_weight_map requires dof_names")
        w = [
            next((float(v) for prefix, v in dof_weight_map.items() if name.startswith(prefix)), 1.0)
            for name in dof_names
        ]
        static_params["dof_weights"] = _torch.tensor(w, dtype=_torch.float32)

    return MdpComponent(
        compute_func=compute_dof_pos_rew,
        dynamic_vars={
            "current_dof_pos": EnvContext.current.dof_pos,
            "ref_dof_pos": EnvContext.mimic.ref_state.dof_pos,
        },
        static_params=static_params,
    )


def corrupted_xy_offset_factory(
    log_noise_std: float = 0.12,
    soft_threshold: float = 0.15,
) -> MdpComponent:
    """Factory for odometer-corrupted XY offset observation.

    Produces a heading-local 2D vector from the robot's current position to
    the reference anchor position, with per-episode affine corruption (scale +
    yaw bias, sampled at reset from EnvConfig.odom_scale_range /
    odom_yaw_range_deg) and per-step proportional log-space noise.

    Applied identically in simulation and on the real G1 by passing the real
    odometer reading through the same corruption parameters — eliminating the
    sim-to-real gap on this observation channel.

    See ``build_corrupted_xy_offset`` in target_poses.py for full design rationale,
    and ``data/scripts/visualize_odometer_corruption.py`` for interactive tuning.

    Args:
        log_noise_std: Std of per-step noise in log(1+mag) space (default 0.12).
        soft_threshold: Noise ramp characteristic length in metres (default 0.15).

    Returns:
        MdpComponent producing corrupted XY offset [envs, 2].
    """
    from protomotions.envs.obs import build_corrupted_xy_offset

    return MdpComponent(
        compute_func=build_corrupted_xy_offset,
        dynamic_vars={
            "current_state_anchor_pos": EnvContext.current.anchor_pos,
            "current_state_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
            "odom_scale": EnvContext.odom_scale,
            "odom_yaw_cos_sin": EnvContext.odom_yaw_cos_sin,
        },
        static_params={
            "w_last": True,
            "log_noise_std": log_noise_std,
            "soft_threshold": soft_threshold,
        },
    )
