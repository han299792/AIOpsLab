# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A common abstraction for a cached LLM inference setup. Currently supports OpenAI's gpt-4-turbo and other models."""


import os
import json
import yaml
from groq import Groq
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

from groq import Groq
from openai import OpenAI, AzureOpenAI
from openai import APITimeoutError, RateLimitError
from azure.identity import get_bearer_token_provider, AzureCliCredential, ManagedIdentityCredential

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
"""An common abstraction for a cached LLM inference setup. Currently supports OpenAI's gpt-4-turbo and other models."""


CACHE_DIR = Path("./cache_dir")
CACHE_PATH = CACHE_DIR / "cache.json"
GPT_MODEL = "gpt-4o"


@dataclass
class AzureConfig:
    azure_endpoint: str
    api_version: str


class Cache:
    """A simple cache implementation to store the results of the LLM inference."""

    def __init__(self) -> None:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                self.cache_dict = json.load(f)
        else:
            os.makedirs(CACHE_DIR, exist_ok=True)
            self.cache_dict = {}

    @staticmethod
    def process_payload(payload):
        if isinstance(payload, (list, dict)):
            return json.dumps(payload)
        return payload

    def get_from_cache(self, payload):
        payload_cache = self.process_payload(payload)
        if payload_cache in self.cache_dict:
            return self.cache_dict[payload_cache]
        return None

    def add_to_cache(self, payload, output):
        payload_cache = self.process_payload(payload)
        self.cache_dict[payload_cache] = output

    def save_cache(self):
        with open(CACHE_PATH, "w") as f:
            json.dump(self.cache_dict, f, indent=4)


class GPTClient:
    """Abstraction for OpenAI's GPT series model."""

    def __init__(self, auth_type: str = "key", api_key: Optional[str] = None, azure_config_file: Optional[str] = None, use_cache: bool = True):
        self.cache = Cache()
        self.client = self._setup_client(auth_type, api_key, azure_config_file)

    def _load_azure_config(self, yaml_file_path: str) -> AzureConfig:
        with open(yaml_file_path, "r") as file:
            azure_config_data = yaml.safe_load(file)
            return AzureConfig(
                azure_endpoint=azure_config_data.get("azure_endpoint"),
                api_version=azure_config_data.get("api_version"),
            )

    def _setup_client(self, auth_type: str, api_key: Optional[str], azure_config_file: Optional[str]):
        azure_identity_opts = ["cli", "managed_identity"]
        if auth_type == "key":
            # TODO: support Azure OpenAI client.
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")
            return OpenAI(api_key=api_key)
        elif auth_type in azure_identity_opts:
            if not azure_config_file:
                raise ValueError("Azure configuration file must be provided for access via managed identity.\n Check AIOpsLab/clients/configs/example_azure_config.yml for an example.")
            azure_config = self._load_azure_config(azure_config_file)
            if auth_type == "cli":
                credential = AzureCliCredential()
            elif auth_type == "managed_identity":
                client_id = os.getenv("AZURE_CLIENT_ID")
                if client_id is None:
                    raise ValueError("Managed identity selected but AZURE_CLIENT_ID is not set.")
                credential = ManagedIdentityCredential(client_id=client_id)
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            return AzureOpenAI(
                api_version=azure_config.api_version,
                azure_endpoint=azure_config.azure_endpoint,
                azure_ad_token_provider=token_provider
            )
        else:
            raise ValueError("auth_type must be one of 'key', 'cli', or 'managed_identity'")

    def inference(self, payload: list[dict[str, str]], max_retries: int = 3) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        import time
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Increase timeout for longer requests (60s -> 120s)
                # Also add exponential backoff for retries
                timeout_seconds = 120 if attempt == 0 else 120 + (attempt * 30)
                
                response = self.client.chat.completions.create(
                    messages=payload,  # type: ignore
                    model=GPT_MODEL,
                    max_tokens=1024,
                    temperature=0.5,
                    top_p=0.95,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=1,
                    timeout=timeout_seconds,
                    stop=[],
                )
                return [c.message.content for c in response.choices]  # type: ignore
                
            except APITimeoutError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                    print(f"API timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API timeout after {max_retries} attempts. Giving up.")
                    raise
            except RateLimitError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # For rate limit, wait longer (exponential backoff with longer base)
                    wait_time = 10 + (attempt * 5)  # 10s, 15s, 20s
                    print(f"Rate limit error (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Rate limit error after {max_retries} attempts. Giving up.")
                    raise
            except Exception as e:
                # Check if it's a rate limit error (fallback for other rate limit formats)
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str or "rate_limit" in error_str:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = 10 + (attempt * 5)  # 10s, 15s, 20s
                        print(f"Rate limit error (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit error after {max_retries} attempts. Giving up.")
                        raise
                # For non-timeout, non-rate-limit errors, don't retry
                print(f"Exception: {repr(e)}")
                raise e
        
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in inference")

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response


class DeepSeekClient:
    """Abstraction for DeepSeek model."""

    def __init__(self):
        self.cache = Cache()

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                        base_url="https://api.deepseek.com")
        try:
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model="deepseek-reasoner",
                max_tokens=1024,
                stop=[],
            )

        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        return [c.message.content for c in response.choices]  # type: ignore

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response


class QwenClient:
    """Abstraction for Qwen's model. Some Qwen models only support streaming output."""

    def __init__(self):
        self.cache = Cache()

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        try:
            # TODO: Add constraints for the input context length
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model="qwq-32b",
                max_tokens=1024,
                n=1,
                timeout=60,
                stop=[],
                stream=True
            )
        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        reasoning_content = ""
        answer_content = ""
        is_answering = False

        for chunk in response:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    reasoning_content += delta.reasoning_content
                else:
                    if delta.content != "" and is_answering is False:
                        is_answering = True
                    answer_content += delta.content

        return [answer_content]

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response


class vLLMClient:
    """Abstraction for local LLM models."""

    def __init__(self,
                 model="Qwen/Qwen2.5-Coder-3B-Instruct",
                 repetition_penalty=1.0,
                 temperature=1.0,
                 top_p=0.95,
                 max_tokens=1024):
        self.cache = Cache()
        self.model = model
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        try:
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1,
                timeout=60,
                stop=[],
            )
        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        return [c.message.content for c in response.choices]  # type: ignore

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response


class OpenRouterClient:
    """Abstraction for OpenRouter API with support for multiple models."""

    def __init__(self, model="anthropic/claude-3.5-sonnet"):
        self.cache = Cache()
        self.model = model

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        try:
            response = client.chat.completions.create(
                messages=payload,  # type: ignore
                model=self.model,
                max_tokens=1024,
                temperature=0.5,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1,
                timeout=60,
                stop=[],
            )
        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        return [c.message.content for c in response.choices]  # type: ignore

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response


class LLaMAClient:
    """Abstraction for Meta's LLaMA-3 model."""

    def __init__(self):
        self.cache = Cache()

    def inference(self, payload: list[dict[str, str]]) -> list[str]:
        if self.cache is not None:
            cache_result = self.cache.get_from_cache(payload)
            if cache_result is not None:
                return cache_result

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        try:
            response = client.chat.completions.create(
                messages=payload,
                model="llama-3.1-8b-instant",
                max_tokens=1024,
                temperature=0.5,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1,
                timeout=60,
                stop=[],
            )
        except Exception as e:
            print(f"Exception: {repr(e)}")
            raise e

        return [c.message.content for c in response.choices]  # type: ignore

    def run(self, payload: list[dict[str, str]]) -> list[str]:
        response = self.inference(payload)
        if self.cache is not None:
            self.cache.add_to_cache(payload, response)
            self.cache.save_cache()
        return response
