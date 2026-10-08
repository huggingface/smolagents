#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F

from .tools import Tool


class TorchAgenticPlannerTool(Tool):
    name = "torchagentic_planner"
    description = """
    Runs a differentiable planner (value iteration or MCTS) on randomly initialized reward and transition models and returns a plan summary.
    The planner uses torchagentic primitives to simulate planning over abstract states and actions.
    Input the planning task description (used as context), and the tool returns the top-ranked states or actions from the planner.
    """
    inputs = {
        "task": {
            "type": "string",
            "description": "A description of the planning task or goal.",
        },
        "num_states": {
            "type": "integer",
            "description": "Number of abstract states for the planner.",
            "nullable": True,
        },
        "num_actions": {
            "type": "integer",
            "description": "Number of abstract actions for the planner.",
            "nullable": True,
        },
        "planner_type": {
            "type": "string",
            "description": "Type of planner: 'vi' for value iteration, 'mcts' for Monte Carlo tree search.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        num_states: int = 64,
        num_actions: int = 8,
        planner_type: str = "vi",
        gamma: float = 0.99,
    ):
        super().__init__()
        try:
            from torchagentic.nn.planner import MCTSPlanner, ValueIteration
        except ImportError as e:
            raise ImportError(
                "TorchAgenticPlannerTool requires torchagentic. "
                "Install it with: pip install smolagents[torchagentic]"
            ) from e

        self._num_states = num_states
        self._num_actions = num_actions
        self._gamma = gamma
        self._planner_type = planner_type

        if planner_type == "vi":
            self._planner = ValueIteration(
                num_states=num_states,
                num_actions=num_actions,
                gamma=gamma,
                num_iters=20,
            )
        elif planner_type == "mcts":
            self._planner = MCTSPlanner(
                num_simulations=20,
                c_puct=1.25,
                gamma=gamma,
            )
        else:
            raise ValueError(f"Unknown planner_type: {planner_type}")

    def forward(
        self,
        task: str,
        num_states: int | None = None,
        num_actions: int | None = None,
        planner_type: str | None = None,
    ) -> str:
        S = num_states or self._num_states
        A = num_actions or self._num_actions
        pt = planner_type or self._planner_type

        with torch.no_grad():
            reward = torch.randn(1, S, A)
            kernel = torch.randn(1, S, A, S)
            kernel = F.softmax(kernel.reshape(1, S * A, S), dim=-1).reshape(1, S, A, S)

            if pt == "vi":
                from torchagentic.nn.planner import ValueIteration

                planner = ValueIteration(
                    num_states=S,
                    num_actions=A,
                    gamma=self._gamma,
                    num_iters=20,
                )
                values, q_values = planner(reward, kernel)
                best_q = q_values.max(dim=-1).values.squeeze(0)
                top_states = best_q.topk(min(3, S)).indices.tolist()
                plan_summary = f"top states: {top_states}"
            else:
                from torchagentic.nn.planner import MCTSPlanner

                prior = torch.randn(1, A)
                value = torch.randn(1)
                planner = MCTSPlanner(
                    num_simulations=20,
                    c_puct=1.25,
                    gamma=self._gamma,
                )
                probs, _ = planner(prior, value)
                top_actions = probs.topk(min(3, A), dim=-1).indices.squeeze(0).tolist()
                plan_summary = f"top actions: {top_actions}"

        return (
            f"[TorchAgentic | {pt.upper()} | {plan_summary} | "
            f"states: {S} | actions: {A} | task: {task[:80]}]"
        )


__all__ = ["TorchAgenticPlannerTool"]
