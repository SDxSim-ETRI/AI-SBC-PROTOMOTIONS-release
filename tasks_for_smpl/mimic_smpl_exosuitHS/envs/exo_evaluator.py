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
"""ActionNet 용 MimicEvaluator — eval action 버퍼를 모터 차원으로.

`MimicEvaluator._create_metrics()` 는 actions 버퍼를 로봇 DOF 수(69)로 할당하는데
Stage 2 의 학습 정책은 **모터 토크 차원만** 출력하므로 첫 전체 eval 에서
"value tensor of shape [N] cannot be broadcast to [1, 69]" 로 죽는다.
원격이 같은 지점에서 epoch 200 에 크래시했고(2026-07-15) 버퍼 크기만 바꾼
서브클래스로 해결했다 — 같은 수정.

★ 모터 수를 상수로 박지 않는다. 힙 전용(2개)·힙+무릎(4개)처럼 구성이 바뀌면
  상수는 조용히 어긋나고 **첫 전체 eval(epoch 200)에서야** 터진다. env 가
  `exo_motors` 로부터 계산해 둔 `num_motors` 를 그대로 쓴다.

`eval/action_delta_*` 는 이제 PD 목표각이 아니라 **모터 raw action 변화율**을
뜻한다(rad 라벨은 명목상). 모터 진동 모니터링 용도로는 그대로 유효하다.
"""

from typing import Dict

from torch import Tensor

from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator


class ExoActionNetMimicEvaluator(MimicEvaluator):
    """actions 트래젝토리 버퍼를 ActionNet 출력 차원으로 할당하는 evaluator."""

    def _create_metrics(
        self,
        num_motions: int,
        motion_num_frames: Tensor,
        max_eval_steps: int,
    ) -> Dict[str, MotionMetrics]:
        metrics = super()._create_metrics(num_motions, motion_num_frames, max_eval_steps)
        num_motor_actions = getattr(self.env, "num_motors", None)
        if not num_motor_actions:
            raise RuntimeError(
                "env 에 num_motors 가 없습니다 — FrozenHumanExoEnv 가 아닌 환경에 "
                "ExoActionNetMimicEvaluator 를 붙였는지 확인하세요."
            )
        metrics["actions"] = MotionMetrics(
            num_motions,
            motion_num_frames,
            max_eval_steps,
            num_motor_actions,
            device=self.device,
        )
        return metrics
