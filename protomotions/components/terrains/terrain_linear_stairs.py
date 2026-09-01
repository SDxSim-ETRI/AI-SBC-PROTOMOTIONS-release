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
"""One-directional staircase terrain for stair-climbing training.

Each subterrain layout (side view, x direction):

  Stairs rise in the -x direction (character walks from approach toward low x).
  This matches skeleton_torque_stairs_koo.pt motion which climbs in -x direction.

  z=N*h  |___/
  ...       /___/
  z=h           /___/
  z=0               |_______________|
                    [stairs N steps ][approach (flat)]
                    [start_x      ][end_x-ap ][end_x]

  - approach section (HIGH-x end): flat at z=0, character spawns here
  - staircase (LOW-x end): N steps, height rises as x decreases
  - character faces -x → first step is right in front

Config fields used (reused from TerrainConfig):
  pyramid_stairs_platform_size  → approach flat length [m]
  pyramid_stairs_step_height    → step riser height [m]
  pyramid_stairs_step_width     → step tread depth [m]
  map_length = N × step_width + approach  (total x length)
  map_width  = corridor width
"""

import torch
from protomotions.components.terrains.terrain import Terrain


class LinearStairsTerrain(Terrain):
    """Terrain consisting of a flat approach section followed by a rising staircase."""

    def is_flat(self) -> bool:
        return False

    def find_terrain_height_for_max_below_body(self, respawned_rigid_body_pos):
        """Shift character down by one step height to correct for motion coordinate offset.

        stairs_koo motion ground level is at z=+step_height (0.17m), not z=0:
          calcn_r at t=0 → z=0.2006m = step1(0.17m) + heel_clearance(0.03m)
        Returning -step_height aligns motion ground with Newton's z=0:
          calcn_r world z = 0.2006 - 0.17 + 0.05(ref_respawn) = 0.08m ≈ on approach ground

        Applied at spawn AND every step (kinematic_replay/mimic call this each frame).
        Returning a constant avoids double-counting stair heights from the heightfield.
        """
        offset = -self.config.pyramid_stairs_step_height  # -0.17m
        return torch.full(
            (respawned_rigid_body_pos.shape[0],),
            offset,
            device=respawned_rigid_body_pos.device,
        )

    def generate_subterrains(self):
        cfg = self.config
        step_h_px = int(abs(cfg.pyramid_stairs_step_height) / cfg.vertical_scale)
        step_w_px = max(1, int(cfg.pyramid_stairs_step_width / cfg.horizontal_scale))
        approach_px = int(cfg.pyramid_stairs_platform_size / cfg.horizontal_scale)

        for subterrain_idx in range(self.env_cols):
            for level_idx in range(self.env_rows):
                start_x = self.border + level_idx * self.length_per_env_pixels
                end_x = self.border + (level_idx + 1) * self.length_per_env_pixels
                start_y = self.border + subterrain_idx * self.width_per_env_pixels
                end_y = self.border + (subterrain_idx + 1) * self.width_per_env_pixels

                # Approach section: HIGH-x end (flat, z=0) → approach_px wide
                # Stair section:   LOW-x end, steps rise as x decreases
                stair_end_x = end_x - approach_px   # boundary between stairs and approach
                available_px = stair_end_x - start_x
                num_steps = max(0, available_px // step_w_px)

                # Step i=0 is the HIGHEST step (farthest from approach)
                # Step i=num_steps-1 is the LOWEST step (adjacent to approach, z=step_h_px)
                for i in range(num_steps):
                    height = (num_steps - i) * step_h_px  # decreases as i increases
                    x0 = start_x + i * step_w_px
                    x1 = min(start_x + (i + 1) * step_w_px, stair_end_x)
                    self.height_field_raw[x0:x1, start_y:end_y] = height

    def sample_valid_locations(self, num_envs, sample_flat=False):
        """Spawn 2.6 cm into the flat approach, matching the stairs_koo motion start.

        stairs_koo motion starts with the root 26 mm in front of step 1
        (approach side), so spawning at stair_boundary_x + 0.026 m aligns the
        motion coordinate system with the terrain:
          - step 1 in world  =  stair_boundary_x - [0, step_width]
          - root at spawn    =  stair_boundary_x + 0.026 m  (flat, z=0)
          - first foot hits step 1 within ~3 frames (walking at ~1 m/s)

        No x jitter — x offset must stay precise so foot contacts align with steps.
        Small y jitter for environment diversity.
        """
        cfg = self.config
        centers = []
        for level_idx in range(self.env_rows):
            for subterrain_idx in range(self.env_cols):
                stair_boundary_x = (self.border_size
                                    + (level_idx + 1) * self.env_length
                                    - cfg.pyramid_stairs_platform_size)
                x_c = stair_boundary_x + 0.026  # stairs_koo 모션 시작 정렬 (step1 직전 2.6cm)
                y_c = (self.border_size
                       + subterrain_idx * self.env_width
                       + self.env_width / 2)
                centers.append([x_c, y_c])

        centers_t = torch.tensor(centers, device=self.device, dtype=torch.float32)
        idx = torch.arange(num_envs) % len(centers_t)
        pos = centers_t[idx]

        # y 지터만 적용 (x 지터 없음 — 모션-지형 정렬 유지)
        jitter_y = (torch.rand(num_envs, device=self.device) - 0.5) * 1.0
        jitter = torch.zeros(num_envs, 2, device=self.device)
        jitter[:, 1] = jitter_y
        return pos + jitter
