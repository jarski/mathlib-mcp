"""Models, conditions and prompts of the evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CONDITIONS = ("none", "docs", "mcp", "skill")
CONDITION_LABELS = {
    "none": "no help",
    "docs": "llms-full.txt in the system prompt",
    "mcp": "mathlib-mcp tools",
    "skill": "the Agent Skill: SKILL.md in the system prompt, its files and checker as tools",
}

MAX_AGENT_TURNS = 10  # model calls per attempt in the tool conditions (mcp, skill)


@dataclass(frozen=True)
class ModelConfig:
    key: str
    id: str  # requested model id
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_multiplier: float = 1.25  # 5-minute TTL writes
    max_tokens: int = 16000
    params: dict[str, Any] = field(default_factory=dict)  # extra request parameters

    def cost(self, usage: dict[str, int]) -> float:
        return (usage["input_tokens"] * self.input_per_mtok
                + usage["output_tokens"] * self.output_per_mtok
                + usage["cache_creation_input_tokens"] * self.input_per_mtok
                * self.cache_write_multiplier
                + usage["cache_read_input_tokens"] * self.cache_read_per_mtok) / 1e6


MODELS = {
    # Haiku 4.5 runs without extended thinking (its default).
    "haiku": ModelConfig("haiku", "claude-haiku-4-5", 1.00, 5.00, 0.10, max_tokens=8192),
    # Sonnet 5.5 runs adaptive thinking (its default) at medium effort, the recommended
    # starting point for agentic coding and multistep tool use.
    "sonnet": ModelConfig("sonnet", "claude-sonnet-5-5", 2.00, 10.00, 0.20,
                          params={"output_config": {"effort": "medium"}}),
}

SYSTEM_PROMPT = """\
You are an expert in GPU computing with Python. You write code that uses CuPy 14.2 \
(`cupy`, `cupyx`) and nvmath-python 1.0 (`nvmath`) on an NVIDIA GPU with CUDA 12.

Rules for every answer:
- Define the requested function with exactly the requested name and parameters.
- Inputs are already on the GPU: CuPy arrays, or `cupyx.scipy.sparse` matrices where the \
task says so. Return CuPy arrays (or `cupyx.scipy.sparse` matrices) that stay on the GPU; \
plain Python scalars are fine where the task asks for a scalar or a tuple containing one.
- Do the computation on the GPU. Do not move arrays to NumPy.
- Put the imports you need at the top of the code. No tests, no example usage and no \
`if __name__ == "__main__":` block.
- End your answer with a single ```python code block that contains the complete \
solution."""

DOCS_HEADER = "# Reference documentation for the libraries above\n\n"

FINAL_ANSWER_NUDGE = (
    "You have used all available tool calls. Reply now with your final solution in a "
    "single ```python code block."
)
