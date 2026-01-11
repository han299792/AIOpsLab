"""Naive GPT4 client (with shell access) for AIOpsLab.

Achiam, Josh, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida et al. 
"Gpt-4 technical report." arXiv preprint arXiv:2303.08774 (2023).

Code: https://openai.com/index/gpt-4-research/
Paper: https://arxiv.org/abs/2303.08774
"""
import os
import asyncio
import tiktoken
import wandb
from openai import APITimeoutError
from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.utils.llm import GPTClient
from dotenv import load_dotenv

from clients.utils.templates import DOCS_SHELL_ONLY

# Load environment variables from the .env file
load_dotenv()

def count_message_tokens(message, enc):
    # Each message format adds ~4 tokens of overhead
    tokens = 4  # <|start|>role/name + content + <|end|>
    tokens += len(enc.encode(message.get("content", "")))
    return tokens

def trim_history_to_token_limit(history, max_tokens=120000, model="gpt-5.1-codex"):
    try:
        # 1. 먼저 자동으로 시도
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # 2. 실패하면 최신 인코딩(o200k_base)을 강제로 사용
        # GPT-5 계열은 보통 GPT-4o와 같은 o200k_base를 사용합니다.
        print(f"Warning: {model} not found in tiktoken. Using o200k_base.")
        enc = tiktoken.get_encoding("o200k_base")

    trimmed = []
    total_tokens = 0

    # Always include the last message
    last_msg = history[-1]
    last_msg_tokens = count_message_tokens(last_msg, enc)

    if last_msg_tokens > max_tokens:
        # If even the last message is too big, truncate its content
        truncated_content = enc.decode(enc.encode(last_msg["content"])[:max_tokens - 4])
        return [{"role": last_msg["role"], "content": truncated_content}]
    
    trimmed.insert(0, last_msg)
    total_tokens += last_msg_tokens

    # Add earlier messages in reverse until limit is reached
    for message in reversed(history[:-1]):
        message_tokens = count_message_tokens(message, enc)
        if total_tokens + message_tokens > max_tokens:
            break
        trimmed.insert(0, message)
        total_tokens += message_tokens

    return trimmed

class GPTAgent:
    def __init__(self):
        self.history = []
        self.llm = GPTClient()
    
    def test(self):
        return self.llm.run([{"role": "system", "content": "hello"}])

    def init_context(self, problem_desc: str, instructions: str, apis: str):
        """Initialize the context for the agent."""

        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        stringify_apis = lambda apis: "\n\n".join(
            [f"{k}\n{v}" for k, v in apis.items()]
        )

        self.system_message = DOCS_SHELL_ONLY.format(
            prob_desc=problem_desc,
            shell_api=stringify_apis(self.shell_api),
            submit_api=stringify_apis(self.submit_api),
        )

        self.task_message = instructions

        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": self.task_message})

    async def get_action(self, input) -> str:
        """Wrapper to interface the agent with OpsBench.

        Args:
            input (str): The input from the orchestrator/environment.

        Returns:
            str: The response from the agent.
        """
        import time
        
        self.history.append({"role": "user", "content": input})
        trimmed_history = trim_history_to_token_limit(self.history)
        
        # Retry logic for API timeouts
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.run(trimmed_history)
                print(f"===== Agent (GPT-4o-mini) ====\n{response}")
                self.history.append({"role": "assistant", "content": response[0]})
                return response[0]
            except APITimeoutError as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"API timeout in get_action (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API timeout after {max_retries} attempts in get_action. Raising exception.")
                    raise
            except Exception as e:
                # For other errors, don't retry
                print(f"Error in get_action: {repr(e)}")
                raise

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}


if __name__ == "__main__":
    # Load use_wandb from environment variable with a default of False
    use_wandb = os.getenv("USE_WANDB", "false").lower() == "true"
    
    if use_wandb:
        # Initialize wandb running
        wandb.init(project="AIOpsLab", entity="AIOpsLab")

    problems = ProblemRegistry().PROBLEM_REGISTRY
    for pid in problems:
        agent = GPTAgent()

        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="gpt-w-shell")

        problem_desc, instructs, apis = orchestrator.init_problem(pid)
        agent.init_context(problem_desc, instructs, apis)
        asyncio.run(orchestrator.start_problem(max_steps=30))

    if use_wandb:
        # Finish the wandb run
        wandb.finish()
