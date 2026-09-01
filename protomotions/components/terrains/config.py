# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for terrain generation and simulation properties."""

from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum


class CombineMode(Enum):
    """Physics material combine mode for friction/restitution."""
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MULTIPLY = "multiply"

    @classmethod
    def from_str(cls, value: str) -> "CombineMode":
        """Create enum from string, case-insensitive."""
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. "
                f"Valid values are: {[e.value for e in cls]}"
            )
        return cls(value)


@dataclass
class TerrainSimConfig:
    """Configuration for terrain simulation properties (friction, restitution, height offset).

    These properties affect the physical behavior of the terrain in simulation.
    Separate from TerrainConfig which defines terrain geometry.
    """

    static_friction: float = field(
        default=1.0,
        metadata={"help": "Static friction coefficient.", "min": 0.0}
    )
    dynamic_friction: float = field(
        default=1.0,
        metadata={"help": "Dynamic friction coefficient.", "min": 0.0}
    )
    restitution: float = field(
        default=0.0,
        metadata={"help": "Restitution (bounciness) coefficient.", "min": 0.0, "max": 1.0}
    )
    height_offset: float = field(
        default=0.0,
        metadata={"help": "Height offset for terrain (negative = lower)."}
    )
    combine_mode: CombineMode = field(
        default=CombineMode.AVERAGE,
        metadata={"help": "How to combine friction values between objects."}
    )


@dataclass
class TerrainConfig:
    """Configuration for terrain generation.
    
    Defines terrain geometry, procedural generation parameters, and simulation properties.
    """

    _target_: str = "protomotions.components.terrains.terrain.Terrain"
    map_length: float = field(
        default=20.0,
        metadata={"help": "Length of terrain map in meters.", "min": 1.0}
    )
    map_width: float = field(
        default=20.0,
        metadata={"help": "Width of terrain map in meters.", "min": 1.0}
    )
    border_size: float = field(
        default=40.0,
        metadata={"help": "Border size to ensure space from edges.", "min": 0.0}
    )
    num_levels: int = field(
        default=10,
        metadata={"help": "Number of difficulty levels for curriculum.", "min": 1}
    )
    num_terrains: int = field(
        default=10,
        metadata={"help": "Number of terrain variations to generate.", "min": 1}
    )

    terrain_proportions: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        metadata={"help": "Proportions: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping, poles, flat]"}
    )
    terrain_sequence: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Per-column terrain type list (overrides terrain_proportions). "
                          "Indices: 0=smooth_slope, 1=rough_slope, 2=stairs_up, 3=stairs_down, "
                          "4=discrete, 5=stepping_stones, 6=poles, 7=flat. "
                          "Example: [7,0,4,0,4,0,4,0,4,7] → flat|slope|disc alternating|flat"}
    )
    pyramid_stairs_step_height: Optional[float] = field(
        default=None,
        metadata={"help": "Fixed step height [m] for pyramid stairs. None = curriculum (0.05+range*difficulty)."}
    )
    pyramid_stairs_step_height_max: Optional[float] = field(
        default=None,
        metadata={"help": "Max step height [m] for pyramid stairs curriculum. None = 0.225m (0.05+0.175). "
                          "Set e.g. 0.17 for 5~17cm range."}
    )
    pyramid_stairs_step_width: float = field(
        default=0.31,
        metadata={"help": "Step tread depth [m] for pyramid stairs.", "min": 0.05}
    )
    pyramid_stairs_platform_size: float = field(
        default=3.0,
        metadata={"help": "Flat center platform diameter [m] for pyramid stairs.", "min": 0.5}
    )
    discrete_obstacles_min_height: float = field(
        default=0.025,
        metadata={"help": "Discrete obstacles minimum height at difficulty 0 [meters]."}
    )
    discrete_obstacles_max_height: float = field(
        default=0.175,
        metadata={"help": "Discrete obstacles maximum height at difficulty 1 [meters]."}
    )
    discrete_obstacles_bevel_size: float = field(
        default=0.0,
        metadata={"help": "Bevel (chamfer) width [meters] applied to obstacle edges. 0 = sharp right-angle corners."}
    )
    rough_terrain_amplitude: float = field(
        default=0.10,
        metadata={"help": "Peak amplitude [meters] of random noise added to rough_slope terrain. ±amplitude."}
    )
    rough_terrain_period: float = field(
        default=0.20,
        metadata={"help": "Spatial period [meters] of rough_slope noise (downsampled_scale). Larger = longer waves."}
    )
    slope_threshold: float = field(
        default=0.9,
        metadata={"help": "Maximum slope angle threshold.", "min": 0.0, "max": 1.0}
    )
    num_samples_per_axis: int = field(
        default=16,
        metadata={"help": "Samples per axis for height observation.", "min": 1}
    )
    sample_width: float = field(
        default=1.0,
        metadata={"help": "Width between sample points in meters.", "min": 0.01}
    )
    terrain_obs_num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Total terrain observation samples. Auto-computed if None."}
    )

    horizontal_scale: float = field(
        default=0.1,
        metadata={"help": "Horizontal resolution scale.", "min": 0.001}
    )
    vertical_scale: float = field(
        default=0.005,
        metadata={"help": "Vertical resolution scale.", "min": 0.001}
    )
    slope_scale: float = field(
        default=0.4,
        metadata={"help": "Slope multiplier for rough/smooth slope terrains. slope = difficulty * slope_scale.", "min": 0.0}
    )

    spacing_between_scenes: float = field(
        default=10.0,
        metadata={"help": "Distance between scenes in grid layout.", "min": 0.0}
    )

    minimal_humanoid_spacing: float = field(
        default=1.0,
        metadata={"help": "Minimum spacing between humanoids in non-scene regions.", "min": 0.0}
    )

    terrain_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save/load terrain file."}
    )
    load_terrain: bool = field(
        default=False,
        metadata={"help": "Load terrain from file instead of generating."}
    )
    save_terrain: bool = field(
        default=False,
        metadata={"help": "Save generated terrain to file."}
    )

    sim_config: TerrainSimConfig = field(
        default_factory=TerrainSimConfig,
        metadata={"help": "Simulation properties (friction, restitution)."}
    )

    def __post_init__(self):
        if self.terrain_obs_num_samples is None:
            self.terrain_obs_num_samples = self.num_samples_per_axis**2


@dataclass
class LinearStairsTerrainConfig(TerrainConfig):
    """한 방향 직선 계단 지형 설정.

    각 서브터레인 레이아웃 (측면도, x축 기준):
      HIGH-x end: 평지 접근 구간 (pyramid_stairs_platform_size m) → 캐릭터 스폰
      LOW-x  end: 계단 상승 구간 → 캐릭터가 -x 방향으로 걸어 올라감

    stairs.usda 치수와 일치:
      step_height=0.17m, step_width=0.30m
      approach=2m, num_steps=13 → map_length≈6m
    """

    _target_: str = "protomotions.components.terrains.terrain_linear_stairs.LinearStairsTerrain"

    map_length: float = field(
        default=6.0,
        metadata={"help": "각 타일의 X 길이 [m]. approach + stairs 합계."}
    )
    map_width: float = field(
        default=4.0,
        metadata={"help": "각 타일의 Y 폭 [m]."}
    )
    border_size: float = field(
        default=10.0,
        metadata={"help": "지형 가장자리 여백 [m]."}
    )
    num_levels: int = field(
        default=16,
        metadata={"help": "X 방향 타일 수 (curriculum 레벨 수)."}
    )
    num_terrains: int = field(
        default=16,
        metadata={"help": "Y 방향 타일 수 (terrain 변형 수)."}
    )
    pyramid_stairs_step_height: Optional[float] = field(
        default=0.17,
        metadata={"help": "계단 riser 높이 [m]. stairs.usda 치수 기준."}
    )
    pyramid_stairs_step_width: float = field(
        default=0.30,
        metadata={"help": "계단 tread 깊이 [m]. stairs.usda 치수 기준."}
    )
    pyramid_stairs_platform_size: float = field(
        default=2.0,
        metadata={"help": "스폰 평지(접근 구간) 길이 [m]."}
    )
    minimal_humanoid_spacing: float = field(
        default=0.0,
        metadata={"help": "환경 간 최소 간격 [m]."}
    )


@dataclass
class ComplexTerrainConfig(TerrainConfig):
    """Configuration for complex procedural terrain."""
    
    num_terrains: int = field(
        default=7,
        metadata={"help": "Number of terrain variations.", "min": 1}
    )
    num_levels: int = field(
        default=7,
        metadata={"help": "Number of difficulty levels.", "min": 1}
    )
    terrain_proportions: List[float] = field(
        default_factory=lambda: [0.2, 0.1, 0.1, 0.1, 0.05, 0.0, 0.0, 0.45],
        metadata={"help": "Proportions for different terrain types."}
    )
    minimal_humanoid_spacing: float = field(
        default=0.0,
        metadata={"help": "Minimum spacing between humanoids.", "min": 0.0}
    )
