# 🧠 Debrief: Meeting Intelligence System

An end-to-end AI-powered system that converts meeting audio into actionable insights — including transcripts, Q&A, structured minutes (MoM), and presentation-ready slides.

---

## 🚀 Overview

This project is designed to automate the entire meeting workflow:

* Convert raw audio → structured knowledge
* Enable querying meeting content using LLMs
* Generate summaries and action items
* Export outputs into usable formats (MoM, PPT)

Unlike basic transcription tools, this system builds a **retrieval-based intelligence layer on top of meeting data**.

---

## ✨ Features

* 🎧 **Audio Transcription**

  * Uses Whisper to convert speech → text

* 🧠 **RAG-based Q&A**

  * Ask questions about meeting content
  * Hybrid retrieval: BM25 keyword search + vector search, merged with reciprocal rank fusion
  * Cross-encoder reranking of the merged candidates
  * Evaluated with DeepEval (see [Evaluation](#-evaluation))

* 📝 **Minutes of Meeting (MoM)**

  * Automatically extracts:

    * Key discussion points
    * Decisions
    * Action items

* 📊 **PPT Generation**

  * Converts insights into presentation-ready slides

* 🔊 **Text-to-Speech (Optional)**

  * Converts answers into audio responses

---

## 🏗️ Architecture

```text
Audio Input
   ↓
Whisper Transcription
   ↓
Text Processing
   ↓
Chunking + Embeddings
   ↓
Hybrid retrieval: BM25 + ChromaDB vectors (RRF)
   ↓
Cross-encoder reranker (top 3)
   ↓
LLM (Groq or local Ollama)
   ↓
Outputs:
  - Q&A
  - MoM
  - PPT
```

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI
* **Speech-to-Text:** Whisper
* **LLM:** gpt-oss-20b on Groq, or Llama 3.1 8B on local Ollama (`LLM_PROVIDER`)
* **Embeddings:** bge-small (sentence-transformers) or nomic-embed-text on Ollama (`EMBED_PROVIDER`)
* **Retrieval:** ChromaDB + BM25 (rank-bm25), ms-marco-MiniLM cross-encoder reranker
* **Evaluation:** DeepEval (LLM-as-judge), plus judge-free retrieval metrics against QMSum's human-marked evidence
* **TTS:** Edge-TTS
* **Dependency Management:** uv (pyproject.toml)

---

## 📂 Project Structure

```text
.
├── app.py                # Streamlit frontend
├── main.py              # FastAPI backend
├── state.py             # Shared state

├── logic/
│   ├── api/             # API endpoints
│   ├── transcription/   # Audio → text
│   ├── rag/             # Retrieval + Q&A
│   ├── mom/             # MoM generation
│   └── create_ppt/      # PPT generation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Gunjit27/Minutes-of-Meeting.git
cd Minutes-of-Meeting
```

---

### 2. Install dependencies (Recommended: uv)

```bash
pip install uv
uv sync
```

---

### 3. Configure

```bash
cp .env.example .env   # then set GROQ_API_KEY (free key: https://console.groq.com/keys)
```

To run fully local instead, set `LLM_PROVIDER=ollama` and `EMBED_PROVIDER=ollama`, and pull `llama3.1:8b` and `nomic-embed-text` in Ollama.

---

### 4. Run Backend

```bash
uvicorn main:app --reload
```

---

### 5. Run Frontend

```bash
streamlit run app.py
```

---

## 🧪 How It Works

1. Upload meeting audio
2. Transcribe audio using Whisper
3. Store embeddings in vector database
4. Query using RAG pipeline
5. Generate:

   * Answers
   * Meeting summary (MoM)
   * PPT slides

---

## 📏 Evaluation

The RAG pipeline is evaluated on **29 questions over 10 real meetings** from [QMSum](https://github.com/Yale-LILY/QMSum) (AMI product-design meetings, 10k to 40k characters each). QMSum's questions and answers are written by people, and each answer is tied to the transcript turns that support it, so retrieval can be scored without an LLM. `evals/build_qmsum.py` builds `evals/data/` from the QMSum repo.

Each question is run through all three retrieval modes, answered with the app's own prompt and LLM, and scored:

* **Retrieval, no LLM:** hit@3 (a top-3 chunk overlaps the evidence), MRR, and evidence recall (the share of the evidence the top 3 chunks cover)
* **DeepEval, with an LLM judge (gpt-oss-120b on Groq by default):** faithfulness, answer relevancy, contextual relevancy. Add `--metrics all` for contextual precision and recall.

```bash
uv run python -m evals.run --no-judge   # retrieval metrics only, no API calls
uv run python -m evals.run              # plus DeepEval, for the vector and hybrid_rerank modes
```

Results go to `evals/results/`: per-question rows with judge reasons in `<mode>.jsonl`, and the table below in `summary.md`. If the run hits a rate limit, it resumes where it stopped. `--judge gemini:gemini-2.5-flash` or `--judge ollama:llama3.1:8b` switches the judge.

### Results

Retrieval over 29 QMSum questions (10 meetings, 800-character chunks, top 3):

| Retrieval | hit@3 | MRR | Evidence recall | p50 retrieval |
|---|---|---|---|---|
| Vector only (MMR, original) | 0.79 | 0.68 | 0.33 | 23 ms |
| Hybrid (BM25 + vector, RRF) | 0.79 | 0.69 | 0.40 | 19 ms |
| **Hybrid + cross-encoder rerank** | **0.83** | **0.78** | **0.46** | 317 ms |

Reranking puts the right evidence first more often (MRR +15%), and the top 3 chunks cover 39% more of the evidence. The cost is about 300 ms more per query on a laptop CPU. Full per-question output is in `evals/results/`.

Answer quality with DeepEval. Answers come from gpt-oss-20b on Groq, and the judge is Llama 3.1 8B on local Ollama:

| Retrieval | Faithfulness | Answer relevancy | Contextual relevancy |
|---|---|---|---|
| Vector only (original) | 0.58 | 0.64 | 0.33 |
| Hybrid + cross-encoder rerank | 0.60 | 0.64 | 0.35 |

**What this shows.** Reranking clearly improves retrieval, but the answer scores barely move. A 0.02 change on 29 questions graded by an 8B judge is within noise. Two likely reasons:

* Most QMSum questions ask for a summary of a whole discussion ("What did the group discuss about X?"). Three 800-character chunks of spoken dialogue often don't hold the full evidence, since evidence recall is only 0.46 even after reranking. The answer step is now the bottleneck, not ranking.
* Contextual relevancy is low for both modes. Meeting speech is full of filler, so most statements in a retrieved chunk aren't about the question.

**Next experiments:** retrieve 5 to 6 chunks for summary-style questions, try smaller chunks with neighbour expansion, and re-judge with a stronger model. The eval makes each of these a one-command comparison.

---

## 💡 Use Cases

* Team meetings
* Client discussions
* Interview analysis
* Lecture summarization
* Knowledge extraction from recordings

---

