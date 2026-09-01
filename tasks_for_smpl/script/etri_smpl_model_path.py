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
"""SMPL 바디모델 `.npz` 경로 해석 — 저장소 레이아웃이 두 종류라 한 곳에서 처리한다.

왜 필요한가
-----------
`SMPL_NEUTRAL.npz` 는 LBS 살 렌더·스킨 생성에 필수인데 저장소마다 위치가 다르다.

| 저장소 | 위치 | 이유 |
|---|---|---|
| `~/ProtoMotions` (학습용 fork) | `data/smpl_models/smpl/` | 원래 두던 자리 |
| `~/ProtoMotions_etri` (릴리즈) | `etri_라이선스동의_다운로드/smpl_models/smpl/` | ETRI 자산은 `etri_` 접두어 폴더(CLAUDE.md §2). **폴더 이름이 곧 경고**다 — 이 안의 것은 개별 라이선스 동의 후 직접 받아야 하는 자산이다 |

두 경로를 모두 찾아보므로 **스크립트를 저장소 간에 그대로 옮겨도 동작한다.**
`SMPL_MODEL_DIR` 환경변수로 임의 위치를 지정할 수도 있다.

라이선스: SMPL 바디모델은 Max Planck Institute 배포물이고 **개별 동의**가 필요하다.
자세한 내용과 획득 방법은 릴리즈 저장소의 `etri_라이선스동의_다운로드/README.md` 참고.

사용법
    from etri_smpl_model_path import smpl_npz
    NPZ = smpl_npz()                 # SMPL_NEUTRAL.npz (없으면 안내와 함께 예외)
    NPZ = smpl_npz("SMPL_MALE.npz")
"""

import os
from pathlib import Path

# 이 파일은 <repo>/tasks_for_smpl/script/ 또는 <repo>/tasks_for_smpl/script/ 에 있다
_REPO = Path(__file__).resolve().parents[2]

# 먼저 매칭된 것을 쓴다. 환경변수 > 릴리즈 레이아웃 > 학습 fork 레이아웃
_LICENSE_DIR = "etri_라이선스동의_다운로드"
_CANDIDATE_DIRS = [
    Path(os.environ["SMPL_MODEL_DIR"]) if os.environ.get("SMPL_MODEL_DIR") else None,
    _REPO / _LICENSE_DIR / "smpl_models/smpl",
    # [ETRI 2026-08-26] 라이선스 자산은 **저장소 트리 밖**의 고정 위치에 둔다.
    #   `~/OUR_HUMAN_DATA/` = 인체 모델(SMPL/SMPL-H/MANO/SMPL-X) 전용 저장소.
    #   저장소를 지우거나 다시 클론해도 살아남고, 여러 저장소가 공유한다.
    #   (모션 데이터는 `~/OUR_MOTION_DATA/` — 이름이 비슷하니 주의)
    Path.home() / "OUR_HUMAN_DATA/smpl_models/smpl",
    _REPO / "tasks_for_smpl/data/smpl_models/smpl",   # 구 레이아웃 (호환)
    _REPO / "data/smpl_models/smpl",                  # 더 구 레이아웃 (호환)
]

_HELP = """
SMPL 바디모델을 찾을 수 없습니다: {name}

찾아본 위치:
{tried}

SMPL 은 Max Planck Institute 배포물로 **개별 라이선스 동의**가 필요합니다.
  1. https://smpl.is.tue.mpg.de 에서 계정을 만들고 라이선스에 동의
  2. "SMPL for Python users" 를 받아 압축을 풀고
  3. `{dst}` 에 `SMPL_NEUTRAL.npz` 를 둡니다
     (`.pkl` 로 받았다면 npz 로 변환해야 합니다 — 같은 폴더의 README.md 참고)

또는 이미 가진 경로를 알려주세요:
  SMPL_MODEL_DIR=/path/to/smpl_models/smpl python ...
"""


def smpl_model_dir() -> Path:
    """SMPL npz 가 들어 있는 폴더. 없으면 릴리즈 레이아웃 경로를 반환한다."""
    for d in _CANDIDATE_DIRS:
        if d is not None and (d / "SMPL_NEUTRAL.npz").exists():
            return d
    return _REPO / _LICENSE_DIR / "smpl_models/smpl"


def smpl_npz(name: str = "SMPL_NEUTRAL.npz") -> Path:
    """SMPL `.npz` 경로. 없으면 획득 방법을 담은 예외를 던진다."""
    tried = []
    for d in _CANDIDATE_DIRS:
        if d is None:
            continue
        p = d / name
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(_HELP.format(
        name=name, tried="\n".join(f"  - {t}" for t in tried),
        dst=_REPO / _LICENSE_DIR / "smpl_models/smpl"))
