import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

print("Loading price data...")

df = pd.read_csv("data/prices.csv")

docs = []

for _, row in df.iterrows():
    content = f"""
    ticker: {row['ticker']}
    date: {row['date']}
    open: {row['open']}
    high: {row['high']}
    low: {row['low']}
    close: {row['close']}
    volume: {row['volume']}
    """
    docs.append(Document(page_content=content))

print(f"Total documents: {len(docs)}")

print("Loading embedding model...")
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Building FAISS index...")
vs = FAISS.from_documents(docs, emb)

os.makedirs("vector_store", exist_ok=True)
vs.save_local("vector_store")

print("FAISS index saved to vector_store/")
