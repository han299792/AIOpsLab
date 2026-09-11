"""Anthropic Claude agent for AIOpsLab.

The repo ships clients for OpenAI, Azure, DeepSeek, Qwen, vLLM, OpenRouter and
Groq, but none for Anthropic directly -- Claude was reachable only through
OpenRouter's OpenAI-compatible endpoint.

Three differences from the existing clients, all deliberate:

* **Usage is captured, not discarded.** Every other client drops
  ``response.usage``, and the evaluator's ``in_tokens``/``out_tokens`` count the
  flattened trace once, which badly undercounts a multi-turn agent that resends
  its history each step. ``self.usage`` here accumulates what the API actually
  reported, so cost can be computed rather than guessed.
* **No response cache by default.** The shared ``Cache`` keys on the message
  list alone, so repeated runs of the same problem are served from disk and stop
  being independent -- which silently destroys the variance any repeated
  experiment is trying to measure. Pass ``use_cache=True`` to opt in.
* **Deterministic by default** (``effort="low"``, no sampling knobs), so that
  variance across repeats comes from the environment rather than from decoding.

Usage::

    agent = ClaudeAgent()
    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name="claude")
    desc, instructs, apis = orchestrator.init_problem(pid)
    agent.init_context(desc, instructs, apis)
    asyncio.run(orchestrator.start_problem(max_steps=30))
"""

import asyncio
import os

from dotenv import load_dotenv

from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.utils.templates import DOCS_SHELL_ONLY

load_dotenv()

#: Kept at module scope to match the other clients (GPT_MODEL, QWEN_MODEL, ...).
CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")


class ClaudeClient:
    """Thin wrapper over the Anthropic Messages API."""

    def __init__(
        self,
        model: str = CLAUDE_MODEL,
        max_tokens: int = 4096,
        effort: str = "low",
        use_cache: bool = False,
    ):
        # Imported here so that merely importing this module does not require
        # the SDK to be installed -- the agent registry imports every client.
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set to use the Claude client."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

        if use_cache:
            from clients.utils.llm import Cache

            self.cache = Cache(namespace=f"{model}|effort={effort}")
        else:
            self.cache = None

        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "calls": 0,
        }

    def _record(self, usage) -> None:
        self.usage["calls"] += 1
        self.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        self.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        self.usage["cache_read_tokens"] += (
            getattr(usage, "cache_read_input_tokens", 0) or 0
        )
        self.usage["cache_write_tokens"] += (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )

    def run(self, system: str, messages: list[dict[str, str]]) -> list[str]:
        """Return a single-element list, matching the other clients' shape."""
        key = None
        if self.cache is not None:
            key = [{"role": "system", "content": system}] + messages
            hit = self.cache.get_from_cache(key)
            if hit is not None:
                return hit

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self.effort},
        )
        self._record(response.usage)

        # A policy decline returns HTTP 200 with stop_reason "refusal" and no
        # usable content, so check it before reading blocks.
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            raise RuntimeError(
                f"model refused the request: "
                f"{getattr(detail, 'category', 'unknown')} "
                f"{getattr(detail, 'explanation', '')}"
            )

        text = "".join(b.text for b in response.content if b.type == "text")
        out = [text]

        if self.cache is not None:
            self.cache.add_to_cache(key, out)
            self.cache.save_cache()
        return out


class ClaudeAgent:
    def __init__(self, model: str = CLAUDE_MODEL, use_cache: bool = False):
        self.history = []
        self.llm = ClaudeClient(model=model, use_cache=use_cache)

    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        """Initialize the context for the agent."""
        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        stringify_apis = lambda apis: "\n\n".join(  # noqa: E731
            [f"{k}\n{v}" for k, v in apis.items()]
        )

        # Anthropic takes the system prompt as its own parameter rather than a
        # message, so it is held separately instead of pushed into history.
        self.system_message = DOCS_SHELL_ONLY.format(
            prob_desc=problem_desc,
            shell_api=stringify_apis(self.shell_api),
            submit_api=stringify_apis(self.submit_api),
        )
        self.task_message = instructions
        self.history.append({"role": "user", "content": self.task_message})

    async def get_action(self, input) -> str:
        """Wrapper to interface the agent with AIOpsLab."""
        self.history.append({"role": "user", "content": input})
        response = await asyncio.to_thread(
            self.llm.run, self.system_message, self.history
        )
        print(f"===== Agent ({self.llm.model}) ====\n{response[0]}")
        self.history.append({"role": "assistant", "content": response[0]})
        return response[0]

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}


if __name__ == "__main__":
    problems = ProblemRegistry().PROBLEM_REGISTRY
    for pid in problems:
        agent = ClaudeAgent()

        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="claude")

        problem_desc, instructs, apis = orchestrator.init_problem(pid)
        agent.init_context(problem_desc, instructs, apis)
        asyncio.run(orchestrator.start_problem(max_steps=30))
        print(f"usage: {agent.llm.usage}")
