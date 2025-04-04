# pip install sentence-transformers faiss-cpu langchain python-docx pymupdf pandas torch

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import pandas as pd
import fitz
import torch
import requests

# === Paths for persistence ===
INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"
PROCESSED_FILES_PATH = "processed_files.pkl"

# === Load embedding model ===
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
if torch.cuda.is_available():
    embedding_model = embedding_model.to("cuda")
    print("[INFO] Using GPU for embeddings.")
else:
    print("[INFO] Using CPU for embeddings.")
embedding_dim = embedding_model.get_sentence_embedding_dimension()

# === Helper Functions ===
def read_pdf_with_pymupdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

def process_file(file_path):
    """Process a single file and return its text content."""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    elif file_path.endswith(".pdf"):
        return read_pdf_with_pymupdf(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        return df.to_string(index=False)
    return ""

def encode_passages(passages, file_name, batch_size=32):
    """Encode text passages in batches."""
    embeddings = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        print(f"[LOG] Embedding batch {i // batch_size + 1}/{(len(passages) + batch_size - 1) // batch_size}")
        print(f"Filename: {file_name}, Batch size: {len(batch)}")
        batch_embeddings = embedding_model.encode([f"passage: {p}" for p in batch], convert_to_tensor=torch.cuda.is_available())
        embeddings.extend(batch_embeddings.cpu().numpy() if torch.cuda.is_available() else batch_embeddings)
    return embeddings

# === Load or Build Index ===
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    print("[INFO] Loading FAISS index and chunks from disk...")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunk_id_to_text = pickle.load(f)
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, "rb") as f:
            processed_files = pickle.load(f)
    else:
        processed_files = set()
else:
    print("[INFO] Initializing new FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)
    chunk_id_to_text = {}
    processed_files = set()

# === Process New Documents ===
folder_path = "./documents"
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if file_name not in processed_files and os.path.isfile(file_path) and file_name.endswith((".txt", ".docx", ".pdf", ".xlsx")):
        print(f"[INFO] Processing new file: {file_name}")
        document_text = process_file(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(document_text)
        chunk_embeddings = encode_passages(chunks, file_name)
        index.add(np.array(chunk_embeddings))
        chunk_id_to_text.update({len(chunk_id_to_text) + i: chunk for i, chunk in enumerate(chunks)})
        processed_files.add(file_name)

# === Save Updated Index and Metadata ===
print("[INFO] Saving updated FAISS index and metadata...")
faiss.write_index(index, INDEX_PATH)
with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunk_id_to_text, f)
with open(PROCESSED_FILES_PATH, "wb") as f:
    pickle.dump(processed_files, f)
print("[INFO] FAISS index and metadata saved successfully.")

# === Retrieval function ===
def retrieve_chunks(query, top_k=5):
    query_vector = embedding_model.encode([f"query: {query}"])
    D, I = index.search(np.array(query_vector), top_k)
    return [chunk_id_to_text[i] for i in I[0]]

# === Prompt builder ===
def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""
    return prompt

# === Query LLaMA via Ollama ===
def query_llama_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2-uncensored",  # Replace if your model name is different
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()['response']

# === Ask user ===
while True:
    user_query = input("\nAsk your question (or type 'exit' to quit): ")

    if user_query.lower() == "exit":
        break
    
    retrieved = retrieve_chunks(user_query)

    print("\nRetrieved chunks:")
    for i, chunk in enumerate(retrieved):
        print(f"{i + 1}: {chunk}")
    
    final_prompt = build_prompt(user_query, retrieved)
    response = query_llama_ollama(final_prompt)
    
    print("\nAnswer:\n", response)
