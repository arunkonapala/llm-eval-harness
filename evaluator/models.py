"""Model providers: the candidate model being evaluated and the judge model
DeepEval uses to score it.

Two providers are supported, selected in config.ini:
  - openai:  any OpenAI-compatible endpoint (Groq, Gemini compat, Ollama, ...)
  - bedrock: AWS Bedrock via the Converse API (boto3 credentials/region apply)
"""

import configparser
import os

from deepeval.models import DeepEvalBaseLLM


def _load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    read = config.read(["config.ini", "config.ini.example"])
    if not read:
        raise FileNotFoundError("No config.ini found — copy config.ini.example to config.ini")
    return config


class OpenAICompatibleLLM(DeepEvalBaseLLM):
    """DeepEval judge wrapper for any OpenAI-compatible chat endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str):
        import openai

        self.model_name = model
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


class BedrockLLM(DeepEvalBaseLLM):
    """DeepEval judge wrapper for AWS Bedrock (Converse API)."""

    def __init__(self, model: str, region: str):
        import boto3

        self.model_name = model
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.converse(
            modelId=self.model_name,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
        )
        return response["output"]["message"]["content"][0]["text"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


def _build(section: configparser.SectionProxy) -> DeepEvalBaseLLM:
    provider = section.get("provider", "openai").lower()
    model = section.get("model")
    if provider == "bedrock":
        return BedrockLLM(model=model, region=section.get("region", "us-east-1"))
    api_key = os.getenv(section.get("api_key_env", "LLM_API_KEY"), "")
    return OpenAICompatibleLLM(
        model=model,
        base_url=section.get("base_url", "https://api.groq.com/openai/v1"),
        api_key=api_key,
    )


def get_judge() -> DeepEvalBaseLLM:
    """The model that scores responses (DeepEval's evaluation model)."""
    return _build(_load_config()["judge"])


def get_candidates() -> list[DeepEvalBaseLLM]:
    """The model(s) under evaluation. [candidate] may list several sections
    via `sections = candidate_a, candidate_b` for side-by-side comparison,
    or define a single model inline."""
    config = _load_config()
    section = config["candidate"]
    if "sections" in section:
        return [_build(config[name.strip()]) for name in section["sections"].split(",")]
    return [_build(section)]
