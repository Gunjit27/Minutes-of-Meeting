"""DeepEval judge models, picked with --judge provider:model.

    groq:llama-3.3-70b-versatile   (default, needs GROQ_API_KEY)
    gemini:gemini-2.5-flash        (needs GOOGLE_API_KEY)
    ollama:llama3.1:8b             (local, free, but a weak and lenient judge)

DeepEval has no built-in Groq model, so GroqJudge wraps ChatGroq. The judge
should be a bigger model than the one answering: an 8B model grading 8B
answers is too lenient to be worth reporting.
"""
import os

from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

DEFAULT_JUDGE = os.getenv("JUDGE_MODEL", "groq:llama-3.3-70b-versatile")


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model: str):
        self.model_name = model
        super().__init__(model)

    def load_model(self):
        # Free-tier Groq rate limits are tight; retry with backoff instead of failing the run.
        return ChatGroq(model=self.model_name, temperature=0, max_retries=8)

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.model.ainvoke(prompt)).content

    def get_model_name(self) -> str:
        return f"groq:{self.model_name}"


def get_judge(spec: str = DEFAULT_JUDGE):
    provider, _, model = spec.partition(":")
    if provider == "groq":
        return GroqJudge(model)
    if provider == "gemini":
        from deepeval.models import GeminiModel
        return GeminiModel(model=model, api_key=os.getenv("GOOGLE_API_KEY"))
    if provider == "ollama":
        from deepeval.models import OllamaModel
        return OllamaModel(model=model)
    raise ValueError(f"Unknown judge {spec!r}; use groq:, gemini: or ollama:")
