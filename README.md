# 🧠 Risk-Aware Financial RAG Agent (Fully Offline)

A routing-aware financial AI system that combines deterministic quantitative computation with retrieval-augmented generation (RAG), built fully offline using a local LLM.

## 🚀 Overview

This project implements a hybrid financial agent that:

Uses deterministic numeric tools for financial calculations

Uses vector retrieval (FAISS) for contextual QA

Routes queries using LangGraph

Implements hallucination guard

Outputs confidence scores

Includes evaluation framework

Runs fully offline (no OpenAI API required)


## 🏗 System Architecture

```md
```mermaid
flowchart TB
  Q[User Query] --> R[Parse & Route (LangGraph)]

  R --> N[Numeric Tool<br/>(deterministic)]
  R --> G[RAG Retrieval<br/>(FAISS)]

  N --> H[Hallucination Guard]
  G --> H

  H --> C[Confidence Scoring]
  C --> O[Structured Output]





## ✨ Key Features
### 1️⃣ Deterministic Financial Tools

Supports:

- Total return

- Annualized volatility

Numeric computations are performed directly with pandas — never delegated to the LLM.

---

### 2️⃣ RAG with Local LLM

- FAISS vector store

- HuggingFace embeddings (all-MiniLM-L6-v2)

- Local LLM via Ollama (gemma3:4b)

- Fully offline
---

### 3️⃣ Routing-Aware Agent (LangGraph)

The system dynamically routes queries:

- Numeric intent → deterministic tool

- Analytical / summary → RAG retrieval

---

### 4️⃣ Hallucination Guard

Heuristic checks for:

- Ticker inconsistencies

- Context insufficiency

- Unsupported claims

---

### 5️⃣ Confidence Scoring

| Route | Confidence |
|-------|------------|
|-------|------------|
| Numeric | 1.0 |
| RAG (grounded) | 0.9 |
| RAG (hallucination risk) | 0.4 |
| "I don't know" | 0.2 |

---

### 6️⃣ Evaluation Framework

Measures:

- Routing accuracy  
- Numeric correctness  
- Hallucination rate  
- Confidence calibration  

---

## 📁 Project Structure

risk-aware-financial-rag/
│
├── data/
│   └── prices.csv
│
├── vector_store/
│   ├── index.faiss
│   └── index.pkl
│
├── src/
│   ├── ingestion.py
│   ├── index_build.py
│   ├── workflow.py
│   └── evaluation.py
│
├── requirements.txt
└── README.md

---

## 🛠 Installation
### 1️⃣ Clone repo

git clone https://github.com/your-username/risk-aware-financial-rag.git
cd risk-aware-financial-rag

### 2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate

### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Install Ollama (Mac)

Download from:

https://ollama.com

Then:

ollama pull gemma3:4b

### ▶️ Run the Project
Step 1 – Download Market Data
python src/ingestion.py

Step 2 – Build Vector Store
python src/index_build.py

Step 3 – Run Agent
python src/workflow.py


Example output:

Q: What is the total return of AAPL?
Answer: AAPL total return over dataset window: 101.65%
Route: numeric
Confidence: 1.0

Step 4 – Run Evaluation
python src/evaluation.py


Example summary:

Routing Accuracy: 1.0
Numeric Accuracy: 1.0
Hallucination Risk Rate: 0.0

## 🧠 Design Philosophy

This project intentionally separates:

Deterministic	Generative
Numeric calculations	Contextual reasoning
Ground truth verifiable	Retrieval grounded
High confidence	Guarded confidence

This hybrid design improves reliability and reduces hallucination risk.


## 📚 Learning Outcomes

By completing this project, you will understand:

Retrieval-Augmented Generation

Agent routing with LangGraph

Deterministic vs generative AI separation

Hallucination mitigation strategies

Offline AI deployment

Evaluation of LLM systems

## 🧑‍💻 Tech Stack

Python

Pandas

FAISS

HuggingFace Embeddings

LangChain

LangGraph

Ollama

gemma3:4b

## 📜 License

MIT License

## 🎯 Final Note
This project demonstrates how to build a production-style, routing-aware AI system combining symbolic computation and neural retrieval — fully offline.

