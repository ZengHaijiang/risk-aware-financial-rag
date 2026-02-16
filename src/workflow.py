from __future__ import annotations

import re
from typing import TypedDict, Literal, Optional
import pandas as pd

from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END


# -------------------------
# Config
# -------------------------
DATA_PATH = "data/prices.csv"
VECTOR_DIR = "vector_store"
VALID_TICKERS = {"AAPL", "MSFT", "TSLA"}
DEFAULT_TICKER = "AAPL"


# -------------------------
# Load Data
# -------------------------
prices = pd.read_csv(DATA_PATH)
prices["date"] = pd.to_datetime(prices["date"])
prices["close"] = pd.to_numeric(prices["close"], errors="coerce")


# -------------------------
# Load LLM & Retriever
# -------------------------
llm = ChatOllama(model="gemma3:4b", temperature=0)

emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vs = FAISS.load_local(VECTOR_DIR, emb, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 6})


# -------------------------
# Utility Functions
# -------------------------
def extract_ticker(text: str) -> str:
    candidates = re.findall(r"\b[A-Z]{2,5}\b", text.upper())
    for c in candidates:
        if c in VALID_TICKERS:
            return c
    return DEFAULT_TICKER


def has_numeric_intent(text: str) -> bool:
    t = text.lower()
    keywords = [
        "return",
        "volatility",
        "std",
        "cagr",
        "drawdown",
        "average",
    ]
    return any(k in t for k in keywords)


def calc_total_return(df: pd.DataFrame, ticker: str) -> float:
    sub = df[df["ticker"] == ticker].sort_values("date")
    return float(sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0)


def calc_volatility(df: pd.DataFrame, ticker: str) -> float:
    sub = df[df["ticker"] == ticker].sort_values("date")
    rets = sub["close"].pct_change().dropna()
    return float(rets.std() * (252 ** 0.5))


# -------------------------
# LangGraph State
# -------------------------
class State(TypedDict, total=False):
    user_query: str
    route: Literal["numeric", "rag"]
    ticker: str
    answer: str
    hallucination_risk: bool
    confidence: float


# -------------------------
# Nodes
# -------------------------
def node_parse_and_route(state: State) -> State:
    q = state["user_query"]

    state["ticker"] = extract_ticker(q)
    state["route"] = "numeric" if has_numeric_intent(q) else "rag"

    return state


def node_numeric(state: State) -> State:
    ticker = state["ticker"]
    q = state["user_query"].lower()

    try:
        if "vol" in q:
            v = calc_volatility(prices, ticker)
            state["answer"] = f"{ticker} annualized volatility ≈ {v:.2%}."
        else:
            r = calc_total_return(prices, ticker)
            state["answer"] = f"{ticker} total return over dataset window: {r:.2%}."
    except Exception as e:
        state["answer"] = f"Numeric calculation failed: {e}"

    state["hallucination_risk"] = False
    return state


def node_rag(state: State) -> State:
    q = state["user_query"]

    docs = retriever.invoke(q)
    context = "\n\n".join([d.page_content.strip() for d in docs])

    prompt = f"""
You are a financial assistant.
Use ONLY the context to answer.
If context is insufficient, say "I don't know".

Context:
{context}

Question:
{q}

Answer concisely.
""".strip()

    resp = llm.invoke(prompt)
    answer = getattr(resp, "content", str(resp))

    # -------- Hallucination Guard --------
    hallucination_risk = False

    for t in VALID_TICKERS:
        if t in answer and t not in context:
            hallucination_risk = True

    if len(context) < 50:
        hallucination_risk = True

    state["answer"] = answer
    state["hallucination_risk"] = hallucination_risk

    return state


def node_confidence(state: State) -> State:
    if state["route"] == "numeric":
        state["confidence"] = 1.0
    else:
        if state.get("hallucination_risk", False):
            state["confidence"] = 0.4
        elif "I don't know" in state["answer"]:
            state["confidence"] = 0.2
        else:
            state["confidence"] = 0.9

    return state


def route(state: State) -> str:
    return state["route"]


# -------------------------
# Build Graph
# -------------------------
graph = StateGraph(State)

graph.add_node("parse", node_parse_and_route)
graph.add_node("numeric", node_numeric)
graph.add_node("rag", node_rag)
graph.add_node("confidence", node_confidence)

graph.set_entry_point("parse")

graph.add_conditional_edges(
    "parse",
    route,
    {"numeric": "numeric", "rag": "rag"},
)

graph.add_edge("numeric", "confidence")
graph.add_edge("rag", "confidence")
graph.add_edge("confidence", END)

app = graph.compile()


# -------------------------
# Test Run
# -------------------------
if __name__ == "__main__":
    tests = [
        "What is the total return of AAPL?",
        "Compute TSLA volatility",
        "Summarize the recent trend of MSFT",
        "What happened on 2024-01-05 for AAPL?",
    ]

    for t in tests:
        out = app.invoke({"user_query": t})
        print("\n==============================")
        print("Q:", t)
        print("Answer:", out["answer"])
        print("Route:", out["route"])
        print("Confidence:", out["confidence"])
        print("Hallucination Risk:", out.get("hallucination_risk"))
