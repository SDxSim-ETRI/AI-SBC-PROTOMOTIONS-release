# SPDX-FileCopyrightText: Copyright (c) 2026 ETRI
# SPDX-License-Identifier: Apache-2.0
"""HS-ROUGH v2 지형 — 난이도 하한을 세운 Terrain 파생 클래스.

**protomotions 는 손대지 않는다**(저장소 규칙: 우리 코드는 mimic_smpl 아래에,
protomotions 는 import 로만). `TerrainConfig._target_` 이 클래스 경로 문자열이고
`protomotions/utils/component_builder.py:32` 가 `get_class(_target_)` 로 이걸
그대로 인스턴스화하므로, 여기서 파생 클래스를 만들어 `_target_` 을 가리키면 된다.

무엇을 왜 바꾸나
---------------
upstream 은 난이도를 `difficulty = level_idx / num_levels` 로 주고
(`terrain.py:427`) 각 지형이 그것을 선형으로 소비한다:

    slope       = difficulty * slope_scale                     # terrain.py:353
    discrete_h  = min_h + difficulty * (max_h - min_h)         # terrain.py:356

따라서 **레벨 0 은 언제나 난이도 0**(경사 0° = 평지)이고, 마지막 레벨도
`9/10 = 0.9` 라 설정한 상한에 도달하지 못한다. 우리 정책은 이 지형에서
success_rate 1.000 / 1220스텝 생존으로 포화했으므로 (2026-09-02, epoch 4767)
**난이도를 중간부터 시작**시켜 하한을 올린다.

`slope_scale` 은 smooth_slope 와 rough_slope 가 **공유**한다. 러프는 난이도를
유지해야 하므로(사용자 지시) 설정값 하나로는 둘을 분리할 수 없다 — 그래서
smooth_slope 만 여기서 가로챈다. 러프는 super() 로 넘겨 upstream 식 그대로 둔다.

`slope` 단위 주의
-----------------
`pyramid_sloped_subterrain` 은 `max_height = slope * (h_scale/v_scale) * (width/2)`
로 쓰는데, 미터로 풀면 `rise = slope * (map_width/2)` 이고 run 도 `map_width/2`
이다. 즉 **`slope` 는 라디안이 아니라 tan(θ)** 다. 기존 `slope_scale=0.227` 은
`atan(0.227)=12.8°` 로 라디안 해석(13.0°)과 우연히 비슷해 구분이 안 됐다.
여기서는 혼동을 없애려고 **도(degree) 로 받아 tan 으로 변환**한다.

지형 타입 코드
-------------
0=smooth_slope 1=rough_slope 2=stairs_up 3=stairs_down
4=discrete 5=stepping_stones 6=poles 7=flat
100=smooth_slope(골짜기)  ← 우리 확장. upstream 의 proportions 경로가 첫 경사 열에
     음의 경사를 주던 것을 `terrain_sequence` 에서도 재현하기 위한 코드다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.terrains.terrain import Terrain
from protomotions.components.terrains.subterrain_generator import (
    pyramid_sloped_subterrain,
)

SMOOTH_SLOPE = 0
ROUGH_SLOPE = 1
SMOOTH_SLOPE_VALLEY = 100


@dataclass
class EtriRoughTerrainConfig(TerrainConfig):
    """HS-ROUGH v2. 상한은 마지막 레벨에서 **정확히** 도달한다."""

    _target_: str = "tasks_for_smpl.mimic_smpl.terrains.etri_terrain.EtriRoughTerrain"

    smooth_slope_min_deg: float = field(
        default=10.0,
        metadata={"help": "smooth_slope 최저 레벨의 경사[도]. 난이도 하한."},
    )
    smooth_slope_max_deg: float = field(
        default=20.0,
        metadata={"help": "smooth_slope 최고 레벨의 경사[도]."},
    )


class EtriRoughTerrain(Terrain):
    """smooth_slope 만 재정의하고 나머지는 난이도만 정규화해 upstream 에 위임."""

    def _generate_subterrain(self, subterrain, terrain_type, difficulty):
        # 러프 경사는 손대지 않는다 — 난이도 정규화조차 하지 않는다(현행 유지).
        if terrain_type == ROUGH_SLOPE:
            return super()._generate_subterrain(subterrain, terrain_type, difficulty)

        # difficulty 는 0 … (n-1)/n 이라 상한에 못 닿는다. 마지막 레벨이 정확히
        # 1.0 이 되도록 정규화한다.
        n = max(int(self.config.num_levels), 2)
        t = min(difficulty * n / (n - 1), 1.0)

        if terrain_type in (SMOOTH_SLOPE, SMOOTH_SLOPE_VALLEY):
            lo = self.config.smooth_slope_min_deg
            hi = self.config.smooth_slope_max_deg
            deg = lo + t * (hi - lo)
            slope = math.tan(math.radians(deg))
            if terrain_type == SMOOTH_SLOPE_VALLEY:
                slope = -slope
            return pyramid_sloped_subterrain(
                subterrain, slope=slope, platform_size=3.0
            )

        # discrete(4) 는 discrete_obstacles_{min,max}_height 를 그대로 쓰되
        # 정규화된 t 로 상한에 닿게 한다. flat(7) 등 나머지는 영향 없음.
        return super()._generate_subterrain(subterrain, terrain_type, t)


# =============================================================================
# 단계 C (2026-09-02) — smooth_slope 상한만 실제로 채운다
# =============================================================================
@dataclass
class EtriSlopeCapTerrainConfig(TerrainConfig):
    """단계 C. **smooth_slope 만** 재정의하고 나머지는 raw difficulty 로 위임한다.

    왜 별도 클래스인가
    -----------------
    위의 EtriRoughTerrain 은 discrete 에도 정규화된 t 를 넘겨(상한 20cm 에 도달)
    난이도를 두 곳에서 바꾼다. 단계 C 는 **경사 하나만** 바꾸는 단계이므로,
    러프·discrete·평지는 upstream 과 완전히 같게 두어야 한다.

    왜 필요한가
    ----------
    upstream 커리큘럼은 difficulty = level_idx / num_levels 이므로 최고 레벨이
    9/10 = 0.9 다. v1 은 slope_scale=0.227(= tan 12.8°)을 썼지만 실제 도달값은
    0.9 × 0.227 = 0.204 → **11.6°** 로, 설정한 13° 에 닿지 못했다.
    사용자 지시(2026-09-02): "예전 시험에서 13도 통과했으니 13도로".
    ondi 의 v27(mixed_slope13_discrete15)이 그 근거다.

    여기서는 최고 레벨이 정확히 smooth_slope_max_deg 가 되도록 정규화한다.
    러프는 slope_scale=0.227 을 raw difficulty 로 쓰던 그대로 둔다(지시: 러프 유지).

    ※ 건축법 참고(2026-09-02 조사): 국내 보행 경사로 상한은 1:8 = 7.1°
      (건축물의 피난·방화구조 규칙 제15조⑤), 주차장 경사로 17% = 9.6°,
      임도 종단경사 18% = 10.2°. 13° 는 이들보다 급하고 계단(26~35°)보다 완만하다.
    """

    _target_: str = (
        "tasks_for_smpl.mimic_smpl.terrains.etri_terrain.EtriSlopeCapTerrain"
    )
    smooth_slope_max_deg: float = field(
        default=13.0,
        metadata={"help": "최고 레벨에서 **실제로 도달할** smooth_slope 경사[도]."},
    )
    discrete_reach_max: bool = field(
        default=False,
        metadata={
            "help": "True 면 discrete 도 최고 레벨에서 상한에 정확히 도달한다. "
                    "upstream 은 difficulty 최고 0.9 라 20cm 설정이 18.1cm 로 끝난다. "
                    "단계 C 는 False(= B 와 동일), 단계 D 는 True."
        },
    )


class EtriSlopeCapTerrain(Terrain):
    """smooth_slope(및 골짜기)만 가로채고, 나머지는 upstream 그대로."""

    #: upstream 의 지형 타입 코드 — discrete
    DISCRETE = 4

    def _norm(self, difficulty):
        """difficulty(최고 (n-1)/n)를 최고 레벨에서 정확히 1.0 이 되도록 정규화한다."""
        n = max(int(self.config.num_levels), 2)
        return min(difficulty * n / (n - 1), 1.0)

    def _generate_subterrain(self, subterrain, terrain_type, difficulty):
        if terrain_type not in (SMOOTH_SLOPE, SMOOTH_SLOPE_VALLEY):
            # [단계 D] discrete 도 상한에 도달시킨다. rough(1)·flat(7) 은 손대지 않는다
            #   — rough 는 지시에 따라 유지, flat 은 난이도 무관.
            if terrain_type == self.DISCRETE and getattr(
                self.config, "discrete_reach_max", False
            ):
                return super()._generate_subterrain(
                    subterrain, terrain_type, self._norm(difficulty)
                )
            # 그 외는 raw difficulty 로 위임 = 단계 B·C 와 동일
            return super()._generate_subterrain(subterrain, terrain_type, difficulty)

        t = self._norm(difficulty)   # 최고 레벨 → 정확히 1.0
        slope = math.tan(math.radians(self.config.smooth_slope_max_deg)) * t
        if terrain_type == SMOOTH_SLOPE_VALLEY:
            slope = -slope   # col 0 = 골짜기 (upstream proportions 경로의 choice<0.05 재현)
        return pyramid_sloped_subterrain(subterrain, slope=slope, platform_size=3.0)
