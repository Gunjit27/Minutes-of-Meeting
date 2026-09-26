# Single container for the live demo (Hugging Face Spaces or any Docker host):
# FastAPI on :8000 inside the container, Streamlit on :7860 exposed.
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/app/.venv/bin:$PATH \
    LLM_PROVIDER=groq EMBED_PROVIDER=hf RETRIEVAL_MODE=hybrid_rerank
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user . .

EXPOSE 7860
CMD uvicorn main:app --host 127.0.0.1 --port 8000 & \
    streamlit run app.py --server.port 7860 --server.address 0.0.0.0
