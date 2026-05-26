# Conversational Research Paper Assistant using RAG

## 📌 Overview

The Conversational Research Paper Assistant is an AI-powered system that enables users to upload research papers and interact with them through natural language conversations.

The project uses:

- Transformer Embeddings
- Retrieval-Augmented Generation (RAG)
- Semantic Vector Search
- Conversational AI

to retrieve relevant information from research papers and generate context-aware responses grounded in uploaded documents.

Unlike traditional keyword-based search systems, this assistant understands the semantic meaning of queries and supports multi-turn conversational interactions.

---

## 🚀 Key Features

- 📄 Upload and process research papers (PDFs)
- 💬 Conversational question answering
- 🧠 Semantic search using transformer embeddings
- ⚡ Fast vector retrieval using FAISS
- 📚 Retrieval-Augmented Generation (RAG)
- 🔄 Multi-turn conversational memory
- 🧾 Citation-aware responses
- 📊 Paper summarization
- 📉 Reduced hallucination through retrieval grounding

---

## 🏗️ System Architecture

```text
User Query
     ↓
PDF Processing → Text Chunking
     ↓
Embedding Generation
     ↓
FAISS Vector Database
     ↓
Semantic Retrieval
     ↓
Reranking
     ↓
LLM Response Generation
     ↓
Conversational Response

---


## 🔄 Workflow

- Upload a research paper
- Extract and preprocess document text
- Split content into semantic chunks
- Generate vector embeddings
- Store embeddings in FAISS
- Retrieve relevant chunks based on user query
- Generate grounded responses using RAG
- Maintain conversational context for follow-up queries

---

## 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) combines:

- Semantic retrieval  
- Vector databases  
- Large language models  

to generate accurate and context-aware responses using external knowledge sources.

### RAG Pipeline

Query → Embedding → Retrieval → Relevant Context → LLM Response

---

## 🧩 Tech Stack

- **Frontend:** Streamlit  
- **Backend:** FastAPI  
- **PDF Processing:** PyMuPDF, pdfplumber  
- **Embeddings:** Sentence Transformers (BGE / E5 / SPECTER2)  
- **Vector DB:** FAISS  
- **Reranker:** FlagEmbedding  
- **LLM:** Groq (Llama)  
- **Database:** SQLite + SQLAlchemy  

---

## ⚙️ Installation & Setup

### Clone Repository
```bash
git clone https://github.com/shruti-1809/research-paper-rag-assistant.git
cd research-paper-rag-assistant

Create Virtual Environment
-Windows
python -m venv venv
venv\Scripts\activate

-Mac/Linux
python -m venv venv
source venv/bin/activate

### Install Dependencies
pip install -r requirements.txt

### Run the Application
-Backend
python main.py
-Frontend
python -m streamlit run app.py

---


## 📈 Future Enhancements
-Hybrid Retrieval (BM25 + Vector Search)
-Adaptive Chunking
-Multi-Hop Reasoning
-Research Recommendation System
-AI-generated Literature Reviews
-Cloud Deployment

---

## 🎯 Applications
-Academic Research Assistant
-Enterprise Knowledge Assistant
-Medical Literature Assistant
-Legal Document Analysis
-AI Research Copilot

---

## 📌 Final Outcome

This project aims to build a research-focused AI assistant similar to:

Perplexity AI
Semantic Scholar AI
OpenAI Deep Research

by combining:

NLP
Information Retrieval
Conversational AI
Vector Databases
Retrieval-Augmented Generation (RAG)

into a unified intelligent research platform.

---