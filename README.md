# 🧠 Meeting Intelligence System

An end-to-end AI-powered system that transforms meeting audio into actionable insights — including transcripts, Q&A, summaries, and presentation-ready outputs.

---

## 🚀 Features

* 🎧 **Audio Transcription**

  * Converts meeting recordings into text using Whisper

* ❓ **Ask Questions (RAG-based)**

  * Query your meeting content using LLM-powered retrieval

* 📝 **Minutes of Meeting (MoM)**

  * Automatically generates structured meeting summaries

* 📊 **PPT Generation**

  * Converts meeting insights into presentation slides

* 🔊 **Text-to-Speech (optional)**

  * Listen to generated answers

---

## 🏗️ Architecture

```
Audio Input
   ↓
Whisper Transcription
   ↓
Text Cleaning
   ↓
Chunking + Embeddings
   ↓
Vector Store (RAG)
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
* **LLM:** Ollama (LLaMA 3)
* **Speech-to-Text:** Whisper
* **Vector DB:** ChromaDB
* **TTS:** Edge-TTS

---

## 📂 Project Structure

```
.
├── app.py                # Streamlit frontend
├── main.py              # FastAPI entrypoint
├── state.py             # Shared state

├── logic/
│   ├── api/             # API routes
│   ├── transcription/   # Audio → text
│   ├── rag/             # Retrieval + Q&A
│   ├── mom/             # MoM generation
│   └── create_ppt/      # PPT generation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/Gunjit27/Minutes-of-Meeting.git
cd Minutes-of-Meeting
```


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run backend

```bash
uvicorn main:app --reload
```

### 4. Run frontend

```bash
streamlit run app.py
```

---

## 🧪 Example Workflow

1. Upload meeting audio
2. Generate transcript
3. Ask questions about the meeting
4. Generate MoM
5. Export PPT

---

## 💡 Use Cases

* Team meetings
* Client calls
* Interview analysis
* Lecture summarization

---
