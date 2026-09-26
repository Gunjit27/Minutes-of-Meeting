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
* **LLM:** Llama 3.1 8B, on Groq or local Ollama (`LLM_PROVIDER`)
* **Embeddings:** bge-small (sentence-transformers) or nomic-embed-text on Ollama (`EMBED_PROVIDER`)
* **Retrieval:** ChromaDB + BM25 (rank-bm25), ms-marco-MiniLM cross-encoder reranker
* **Evaluation:** DeepEval with a Llama 3.3 70B judge on Groq
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

The RAG pipeline is evaluated on a fixed transcript of a sample earnings call (`evals/data/`), with a hand-checked question set in `evals/data/golden.json`. Every question is run through each retrieval mode, answered with the app's own prompt and LLM, and scored:

* **Retrieval, no LLM:** hit@3 and MRR. Each question carries a verbatim evidence quote, and a hit means a top-3 chunk contains it.
* **DeepEval, with a Llama 3.3 70B judge:** faithfulness, answer relevancy, contextual precision, contextual recall, contextual relevancy.

```bash
uv run python -m evals.transcribe EarningsCall.wav   # once, writes evals/data/
uv run python -m evals.run                           # all modes; --limit 5 for a quick run
```

Results are written to `evals/results/`: per-question rows with judge reasons in `<mode>.jsonl`, and the table below in `summary.md`. The run resumes where it stopped if Groq's free-tier limit cuts it off.

### Results

_Pending the first full run._

---

## 💡 Use Cases

* Team meetings
* Client discussions
* Interview analysis
* Lecture summarization
* Knowledge extraction from recordings

---

