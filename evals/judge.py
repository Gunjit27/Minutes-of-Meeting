"""DeepEval judge backed by Groq. DeepEval has no built-in Groq model, so this wraps ChatGroq.

The judge is deliberately a bigger model than the one answering questions:
an 8B model grading its own answers is too lenient to be worth reporting.
"""
import os

from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

DEFAULT_JUDGE = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = DEFAULT_JUDGE):
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
        return f"groq/{self.model_name}"
