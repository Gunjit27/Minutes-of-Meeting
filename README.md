# 🧠 Meeting Intelligence System

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
  * Context-aware answers using vector search

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
Vector Store (ChromaDB)
   ↓
LLM (Ollama)
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
* **LLM:** Ollama (LLaMA 3)
* **Vector Store:** ChromaDB
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

### 3. Run Backend

```bash
uvicorn main:app --reload
```

---

### 4. Run Frontend

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

## 💡 Use Cases

* Team meetings
* Client discussions
* Interview analysis
* Lecture summarization
* Knowledge extraction from recordings

---

