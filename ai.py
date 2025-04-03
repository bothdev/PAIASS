# pip install sentence-transformers faiss-cpu langchain python-docx pymupdf pandas

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import requests
import os
import fitz
from docx import Document
import pickle
import pandas as pd
import torch

# === Paths for persistence ===
INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"

# === Load embedding model ===
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
if torch.cuda.is_available():
    embedding_model = embedding_model.to("cuda")  # Move model to GPU if available
    print("[INFO] Using GPU for embeddings.")
else:
    print("[INFO] Using CPU for embeddings.")
embedding_dim = embedding_model.get_sentence_embedding_dimension()

# === PDF Reader using PyMuPDF ===
def read_pdf_with_pymupdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

# === Load documents from folder ===
def load_documents_from_folder(folder_path):
    all_text = ""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n"
        elif file_name.endswith(".docx"):
            doc = Document(file_path)
            for para in doc.paragraphs:
                all_text += para.text + "\n"
        elif file_name.endswith(".pdf"):
            all_text += read_pdf_with_pymupdf(file_path) + "\n"
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(file_path)
            all_text += df.to_string(index=False) + "\n"
    return all_text

# === Load or build index ===
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
    """Encode text passages in batches and log embedding details."""
    embeddings = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        print(f"[LOG] Embedding batch {i // batch_size + 1}/{(len(passages) + batch_size - 1) // batch_size}")
        print(f"Filename: {file_name}, Batch size: {len(batch)}")
        batch_embeddings = embedding_model.encode([f"passage: {p}" for p in batch], convert_to_tensor=torch.cuda.is_available())
        embeddings.extend(batch_embeddings.cpu().numpy() if torch.cuda.is_available() else batch_embeddings)
    return embeddings

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    print("[INFO] Loading FAISS index and chunks from disk...")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunk_id_to_text = pickle.load(f)
else:
    print("[INFO] Building FAISS index from documents...")
    folder_path = "./documents"
    index = faiss.IndexFlatL2(embedding_dim)
    chunk_id_to_text = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith((".txt", ".docx", ".pdf", ".xlsx")):
            document_text = process_file(file_path)
            print(f"[INFO] Processing text from {file_name}")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(document_text)
            chunk_embeddings = encode_passages(chunks, file_name)
            index.add(np.array(chunk_embeddings))
            chunk_id_to_text.update({len(chunk_id_to_text) + i: chunk for i, chunk in enumerate(chunks)})

    # Save index and chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunk_id_to_text, f)

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
    final_prompt = build_prompt(user_query, retrieved)
    response = query_llama_ollama(final_prompt)
    print("\nAnswer:", response)
