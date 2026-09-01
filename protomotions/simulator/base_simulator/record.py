# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording and rendering logic extracted from the Simulator base class.

This module contains the RecordingMixin class which provides video recording,
frame capture, marker management, and motion/object serialization functionality.
The mixin is designed to be used with the Simulator class, accessing simulator
state and methods via self.
"""

from collections import deque
from datetime import datetime
import logging
import os
import glob
from typing import Dict, Optional

import torch

from protomotions.utils import rotations
from protomotions.simulator.base_simulator.config import MarkerState
from protomotions.simulator.base_simulator.utils import build_motion_data

log = logging.getLogger(__name__)


class RecordingMixin:
    """Mixin providing recording and rendering capabilities for simulators.

    This mixin expects the following attributes/methods to be provided by the
    host class (Simulator):
        - self.headless, self.config, self.scene_lib, self.num_envs
        - self._num_dof, self._proj_config
        - self._original_marker_configs
        - self.get_robot_state(), self.get_object_root_state()
        - self._get_projectile_positions_rotations()
        - self._write_viewport_to_file(file_name)
        - self._update_simulator_markers(markers_state)
    """

    # -------------------------
    # Initialization
    # -------------------------

    def _init_recording_state(self) -> None:
        """Initialize all recording-related attributes."""
        self._camera_target: Dict[str, int] = {"env": 0, "element": 0}
        self._show_markers: bool = False

        self._user_is_recording, self._user_recording_state_change = False, False
        self._user_recording_video_queue_size = 100000
        self._delete_user_viewer_recordings = False
        os.makedirs("output/renderings", exist_ok=True)
        self._user_recording_video_path = os.path.join(
            "output/renderings", f"{self.config.experiment_name}-%s"
        )
        # True when headless rendering is enabled (headless=True + enable_cameras=True)
        # so that recording runs even though self.headless is True.
        self._rendering_enabled: bool = False

        # Last markers state for recording (set each step)
        self._last_markers_state: Optional[Dict[str, MarkerState]] = None

    # -------------------------
    # Recording state control
    # -------------------------

    def _toggle_video_record(self):
        self._user_is_recording = not self._user_is_recording
        self._user_recording_state_change = True

    def _cancel_video_record(self):
        self._user_is_recording = False
        self._user_recording_state_change = False
        self._delete_user_viewer_recordings = True

    # -------------------------
    # Camera target
    # -------------------------

    def _toggle_camera_target(self) -> None:
        """
        Toggle the camera target between different environments and objects.

        The target cycles through all objects in the scene, with 0 referring to the environment.
        """
        if self.scene_lib.num_objects_per_scene > 0:
            self._camera_target["element"] = (self._camera_target["element"] + 1) % (
                self.scene_lib.num_objects_per_scene + 1
            )
            print("Updated camera target to element", self._camera_target["element"])

        if self._camera_target["element"] == 0:
            self._camera_target["env"] = (
                self._camera_target["env"] + 1
            ) % self.num_envs
            print("Updated camera target to env", self._camera_target["env"])

    # -------------------------
    # Marker management
    # -------------------------

    def _toggle_markers(self):
        self._show_markers = not self._show_markers
        print(f"Markers are now {'visible' if self._show_markers else 'hidden'}")

    def _update_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """
        Update visualization markers for the simulator.

        Converts marker orientations if necessary and delegates to the simulator-specific update.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary containing marker states.
        """

        if not markers_state or len(markers_state) == 0:
            return

        if not self.config.w_last:
            for key in markers_state.keys():
                markers_state[key].orientation = rotations.xyzw_to_wxyz(
                    markers_state[key].orientation
                )
        if not self._show_markers:
            for key in markers_state.keys():
                # Throw it out of view
                markers_state[key].translation = (
                    torch.zeros_like(markers_state[key].translation) - 1000000
                )
        self._update_simulator_markers(markers_state)

    def _build_markers_save_data(self) -> dict:
        """Build markers data dictionary for saving to .markers.pt file."""
        markers_data = {"fps": round(1.0 / self.dt), "markers": {}}
        for name, frame_list in self._recorded_markers.items():
            translations = torch.stack([f[0] for f in frame_list], dim=0)
            orientations = torch.stack([f[1] for f in frame_list], dim=0)
            # Get marker config metadata from the original (pre-simulator)
            # configs, since simulator-specific init may wrap/replace them
            marker_config = self._original_marker_configs.get(name)
            marker_type = "sphere"
            color = (1.0, 0.0, 0.0)
            sizes = []
            if marker_config is not None:
                marker_type = marker_config.type
                color = marker_config.color
                sizes = [m.size for m in marker_config.markers]

            markers_data["markers"][name] = {
                "type": marker_type,
                "color": color,
                "sizes": sizes,
                "translation": translations,
                "orientation": orientations,
            }
        return markers_data

    # -------------------------
    # Object serialization
    # -------------------------

    def _build_terrain_save_data(self) -> Optional[dict]:
        """Build terrain data dictionary for saving to .terrain.pt file.

        Returns None if terrain is flat or not available.
        """
        terrain = getattr(self, "terrain", None)
        if terrain is None:
            return None

        # Skip saving for flat terrains — Blender's default ground plane suffices
        if terrain.is_flat():
            return None

        return {
            "height_field_raw": terrain.height_field_raw,
            "horizontal_scale": terrain.horizontal_scale,
            "vertical_scale": terrain.vertical_scale,
        }

    def _build_objects_save_data(self) -> dict:
        """Build objects data dictionary for saving to .objects.pt file."""
        objects_list = []

        # Scene objects
        if self._recorded_objects:
            from protomotions.components.scene_lib import (
                BoxSceneObject,
                SphereSceneObject,
                CylinderSceneObject,
                MeshSceneObject,
            )

            translations = torch.stack([f[0] for f in self._recorded_objects], dim=0)
            rotations = torch.stack([f[1] for f in self._recorded_objects], dim=0)

            eid = self._recording_env_id
            scene_idx = self.scene_lib._scene_to_original_scene_id[eid].item()
            scene = self.scene_lib._original_scenes[scene_idx]

            for obj_idx, obj in enumerate(scene.objects):
                obj_info = {
                    "name": f"object_{obj_idx}",
                    "translation": translations[:, obj_idx, :],
                    "rotation": rotations[:, obj_idx, :],
                }
                if isinstance(obj, BoxSceneObject):
                    obj_info["shape"] = "box"
                    obj_info["size"] = [obj.width, obj.depth, obj.height]
                elif isinstance(obj, SphereSceneObject):
                    obj_info["shape"] = "sphere"
                    obj_info["size"] = [obj.radius]
                elif isinstance(obj, CylinderSceneObject):
                    obj_info["shape"] = "cylinder"
                    obj_info["size"] = [obj.radius, obj.height]
                elif isinstance(obj, MeshSceneObject):
                    obj_info["shape"] = "mesh"
                    obj_info["size"] = []
                    obj_info["mesh_path"] = obj.object_path
                    obj_info["scale"] = list(obj.scale)
                else:
                    obj_info["shape"] = "box"
                    dims = obj.object_dims
                    if dims is not None:
                        obj_info["size"] = [
                            dims[1] - dims[0],
                            dims[3] - dims[2],
                            dims[5] - dims[4],
                        ]
                    else:
                        obj_info["size"] = [0.1, 0.1, 0.1]

                objects_list.append(obj_info)

        # Projectiles
        if self._recorded_projectiles:
            proj_pos = torch.stack(
                [f[0] for f in self._recorded_projectiles], dim=0
            )  # [num_frames, num_proj, 3]
            proj_rot = torch.stack(
                [f[1] for f in self._recorded_projectiles], dim=0
            )  # [num_frames, num_proj, 4]

            half_sizes = self._proj_config.get_sizes()
            hide_z = self._proj_config.hide_z

            for p in range(proj_pos.shape[1]):
                # Only include projectiles that were visible at some point
                if (proj_pos[:, p, 2] > hide_z + 0.5).any():
                    hs = half_sizes[p]
                    full_size = hs * 2
                    objects_list.append(
                        {
                            "name": f"projectile_{p}",
                            "shape": "box",
                            "size": [full_size, full_size, full_size],
                            "translation": proj_pos[:, p, :],
                            "rotation": proj_rot[:, p, :],
                        }
                    )

        return {"fps": round(1.0 / self.dt), "objects": objects_list}

    # -------------------------
    # Main render loop
    # -------------------------

    def render(self):
        """
        Render the current simulation state and handle video recording if enabled.

        This method manages:
        1. Video recording state transitions and initialization
        2. Frame capture and saving during recording
        3. Video compilation when recording ends
        4. Cleanup of temporary image files
        """
        if not self.headless or self._rendering_enabled:
            # Handle recording state transitions
            if self._user_recording_state_change:
                if self._user_is_recording:
                    # Initialize new recording
                    self._user_recording_video_queue = deque(
                        maxlen=self._user_recording_video_queue_size
                    )
                    curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    self._curr_user_recording_name = (
                        self._user_recording_video_path % curr_date_time
                    )
                    self._user_recording_frame = 0

                    self._recorded_motion = {
                        "gts": [],  # rigid_body_pos (global translations)
                        "grs": [],  # rigid_body_rot (global rotations)
                        "gvs": [],  # rigid_body_vel (global velocities)
                        "gavs": [],  # rigid_body_ang_vel (global angular velocities)
                        "dps": [],  # dof_pos
                        "dvs": [],  # dof_vel
                        "dts": [],  # dof_forces (applied joint torques)
                        "contacts": [],  # rigid_body_contacts
                    }
                    self._recorded_markers = {}
                    self._recorded_objects = []
                    self._recorded_projectiles = []
                    self._recording_env_id = self._camera_target["env"]

                    os.makedirs(os.path.join(self._curr_user_recording_name, "_frames"), exist_ok=True)
                    print(
                        f"Started recording to folder {self._curr_user_recording_name}"
                    )
                else:
                    # Finalize recording and create video
                    import subprocess

                    # PNG frames are in _frames/ subfolder; output files go in the main folder
                    image_dir = os.path.join(self._curr_user_recording_name, "_frames")
                    _stem = os.path.basename(self._curr_user_recording_name)
                    out_mp4 = os.path.join(self._curr_user_recording_name, f"{_stem}.mp4")

                    # Use ffmpeg directly — robust to async-captured frames with
                    # minor size variations (unlike moviepy's ImageSequenceClip).
                    # framerate must match the actual policy step rate (1/self.dt)
                    # so every task's recording plays back at 1x speed -- a fixed
                    # 30 here doesn't match e.g. this suit's 20fps policy rate and
                    # played back 1.5x too fast.
                    _record_fps = round(1.0 / self.dt)
                    # [ETRI patch 2026-08-25] mp4 인코딩 실패가 .motion/.terrain.pt
                    # 저장을 막지 않도록 한다.
                    #   헤드리스(--headless) 실행에서는 뷰어가 없어 _frames/ 에 PNG 가
                    #   하나도 없다. 기존 코드는 check=True 로 ffmpeg 을 호출해
                    #   CalledProcessError(exit 254) 가 여기서 터졌고, 아래의 모션·지형
                    #   저장 블록에 **도달하지 못했다** — 즉 헤드리스에서는 물리 롤아웃
                    #   .motion 을 얻는 것이 원리적으로 불가능했다.
                    #   오프라인 렌더 워크플로우(무이음새 UsdSkel + 지형)는 그 .motion 이
                    #   입력이므로, 인코딩은 best-effort 로 내리고 저장을 보장한다.
                    _frames = sorted(glob.glob(os.path.join(image_dir, "*.png")))
                    if not _frames:
                        print(
                            "[record] _frames/ 에 PNG 가 없다 (헤드리스로 보임) — "
                            "mp4 인코딩을 건너뛰고 .motion/.terrain.pt 저장을 계속한다."
                        )
                    else:
                        try:
                            subprocess.run(
                                [
                                    "ffmpeg", "-y",
                                    "-framerate", str(_record_fps),
                                    "-i", os.path.join(image_dir, "%04d.png"),
                                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                    "-c:v", "libx264",
                                    "-profile:v", "main",
                                    "-level", "4.0",
                                    "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart",
                                    "-crf", "23",
                                    "-x264-params", "keyint=60:min-keyint=30",
                                    "-threads", "32",
                                    out_mp4,
                                ],
                                check=True,
                                capture_output=True,
                            )
                            self._delete_user_viewer_recordings = True
                            print(f"Video saved to {out_mp4}")
                        except (subprocess.CalledProcessError, FileNotFoundError) as e:
                            _err = getattr(e, "stderr", b"") or b""
                            print(
                                "[record] ffmpeg 인코딩 실패 — .motion/.terrain.pt 저장은 계속한다.\n"
                                f"         {type(e).__name__}: {_err.decode('utf-8', 'replace')[-400:]}"
                            )

                    # Save the recorded motion as a .motion file
                    motion_data = build_motion_data(
                        self._recorded_motion,
                        fps=_record_fps,  # actual policy step rate, not a fixed 30
                        num_dof=self._num_dof,
                    )
                    motion_file_path = os.path.join(self._curr_user_recording_name, f"{_stem}.motion")
                    torch.save(motion_data, motion_file_path)
                    print(f"Motion saved to {motion_file_path}")
                    self._recorded_motion = None

                    # Save markers and objects files
                    try:
                        if self._recorded_markers:
                            markers_data = self._build_markers_save_data()
                            markers_path = os.path.join(self._curr_user_recording_name, f"{_stem}.markers.pt")
                            torch.save(markers_data, markers_path)
                            print(f"Markers saved to {markers_path}")

                        if self._recorded_objects or self._recorded_projectiles:
                            objects_data = self._build_objects_save_data()
                            objects_path = os.path.join(self._curr_user_recording_name, f"{_stem}.objects.pt")
                            torch.save(objects_data, objects_path)
                            print(f"Objects saved to {objects_path}")
                        terrain_data = self._build_terrain_save_data()
                        if terrain_data is not None:
                            terrain_path = (
                                f"{self._curr_user_recording_name}.terrain.pt"
                            )
                            torch.save(terrain_data, terrain_path)
                            print(f"Terrain saved to {terrain_path}")

                    except Exception as e:
                        print(f"Warning: failed to save markers/objects/terrain: {e}")
                    self._recorded_markers = None
                    self._recorded_objects = None
                    self._recorded_projectiles = None

                self._user_recording_state_change = False

            # Capture frame if recording
            if self._user_is_recording:
                file_name = os.path.join(
                    self._curr_user_recording_name,
                    "_frames",
                    "%04d.png" % self._user_recording_frame,
                )
                self._write_viewport_to_file(file_name)
                self._user_recording_frame += 1

                eid = self._recording_env_id

                # Record motion (single env only)
                robot_state = self.get_robot_state()
                self._recorded_motion["gts"].append(
                    robot_state.rigid_body_pos[eid].cpu().clone()
                )
                self._recorded_motion["grs"].append(
                    robot_state.rigid_body_rot[eid].cpu().clone()
                )
                if robot_state.rigid_body_vel is not None:
                    self._recorded_motion["gvs"].append(
                        robot_state.rigid_body_vel[eid].cpu().clone()
                    )
                if robot_state.rigid_body_ang_vel is not None:
                    self._recorded_motion["gavs"].append(
                        robot_state.rigid_body_ang_vel[eid].cpu().clone()
                    )
                if robot_state.dof_pos is not None:
                    self._recorded_motion["dps"].append(
                        robot_state.dof_pos[eid].cpu().clone()
                    )
                if robot_state.dof_vel is not None:
                    self._recorded_motion["dvs"].append(
                        robot_state.dof_vel[eid].cpu().clone()
                    )
                if robot_state.dof_forces is not None:
                    self._recorded_motion["dts"].append(
                        robot_state.dof_forces[eid].cpu().clone()
                    )
                if robot_state.rigid_body_contacts is not None:
                    self._recorded_motion["contacts"].append(
                        robot_state.rigid_body_contacts[eid].cpu().clone()
                    )

                # Record markers (single env only, skip terrain markers)
                if self._last_markers_state:
                    for name, ms in self._last_markers_state.items():
                        if name == "terrain_markers":
                            continue
                        if name not in self._recorded_markers:
                            self._recorded_markers[name] = []
                        self._recorded_markers[name].append(
                            (
                                ms.translation[eid].cpu().clone(),
                                ms.orientation[eid].cpu().clone(),
                            )
                        )

                # Record objects (single env only)
                if (
                    self.scene_lib is not None
                    and self.scene_lib.num_objects_per_scene > 0
                ):
                    obj_state = self.get_object_root_state()
                    if obj_state is not None and obj_state.root_pos is not None:
                        self._recorded_objects.append(
                            (
                                obj_state.root_pos[eid].cpu().clone(),
                                obj_state.root_rot[eid].cpu().clone(),
                            )
                        )

                # Record projectiles (single env only)
                if (
                    self._proj_config is not None
                    and self._proj_config.num_projectiles > 0
                ):
                    pos, rot = self._get_projectile_positions_rotations()
                    self._recorded_projectiles.append(
                        (
                            pos[eid].cpu().clone(),
                            rot[eid].cpu().clone(),
                        )
                    )

            # Clean up temporary PNG frames (_frames/ subfolder only)
            if self._delete_user_viewer_recordings:
                frames_dir = os.path.join(self._curr_user_recording_name, "_frames")
                if os.path.exists(frames_dir):
                    for image in os.listdir(frames_dir):
                        if image.endswith(".png"):
                            os.remove(os.path.join(frames_dir, image))
                    os.rmdir(frames_dir)
                self._delete_user_viewer_recordings = False
                self._recorded_motion = None
