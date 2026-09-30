"""Model inference callbacks used by evaluation runs.

Both callbacks are factories: pass the model name and they return a callback the
evaluator can run over resume items. ``ollama_inference`` targets a local Ollama
server; ``openai_inference`` targets any OpenAI-compatible chat-completions
endpoint (OpenRouter by default, plus Hugging Face router, DeepInfra, Novita,
llama.cpp, vLLM, and dedicated inference endpoints).
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.schema.resume_output import ResumeOutput

from .evaluator import (
    EvaluationItem,
    ModelCallback,
    ModelCallbackOutput,
    render_json_prompt,
    strict_json_schema,
)


OLLAMA_URL = "http://localhost:11434/api/generate"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 300


def ollama_inference(
    model_name: str = "llama3.2",
    *,
    url: str = OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> ModelCallback:
    """Return a callback that runs one resume through a local Ollama server."""

    def callback(item: EvaluationItem) -> ModelCallbackOutput:
        prompt = render_json_prompt(item.resume_text)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": ResumeOutput.model_json_schema(),
            "options": {"temperature": 0},
            "keep_alive": "5m",
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {url}. Start Ollama and confirm {model_name} is available."
            ) from error

        if result.get("error"):
            raise RuntimeError(f"Ollama error: {result['error']}")
        if "response" not in result:
            raise RuntimeError("Ollama response is missing its generated text.")
        return ModelCallbackOutput(
            raw_output=result["response"],
            prompt=prompt,
            model_name=model_name,
            metadata={
                "total_duration_ns": result.get("total_duration"),
                "prompt_eval_count": result.get("prompt_eval_count"),
                "eval_count": result.get("eval_count"),
            },
        )

    return callback


def openai_inference(
    model_name: str,
    *,
    base_url: str = OPENROUTER_URL,
    api_key_env: str = "OPENROUTER_API_KEY",
    timeout: float = DEFAULT_TIMEOUT,
    structured_output: bool = True,
) -> ModelCallback:
    """Return a callback for any OpenAI-compatible chat-completions endpoint.

    Defaults target OpenRouter with ``OPENROUTER_API_KEY``. For the Hugging Face
    router, pass ``base_url="https://router.huggingface.co/v1/chat/completions"``
    and ``api_key_env="HF_TOKEN"``; other providers follow the same pattern.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is missing from the environment or project .env.")

    def callback(item: EvaluationItem) -> ModelCallbackOutput:
        prompt = render_json_prompt(item.resume_text)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if structured_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "resume_output",
                    "strict": True,
                    "schema": strict_json_schema(ResumeOutput.model_json_schema()),
                },
            }
        request = Request(
            base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{base_url} returned HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"Cannot reach {base_url}: {error}") from error

        if result.get("error"):
            raise RuntimeError(f"Inference error: {result['error']}")
        choices = result.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if not content:
            raise RuntimeError("Response is missing its generated text.")
        return ModelCallbackOutput(
            raw_output=content,
            prompt=prompt,
            model_name=result.get("model", model_name),
            metadata=result.get("usage", {}),
        )

    return callback


__all__ = [
    "OPENROUTER_URL",
    "OLLAMA_URL",
    "ollama_inference",
    "openai_inference",
]
