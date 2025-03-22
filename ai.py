# pip install sentence-transformers faiss-cpu pip insp python-docx pymupdf pandas requests pickle

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

# === Paths for persistence ===
INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"

# === Load embedding model ===
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
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
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    print("[INFO] Loading FAISS index and chunks from disk...")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunk_id_to_text = pickle.load(f)
else:
    print("[INFO] Building FAISS index from documents...")
    folder_path = "./documents"
    document = load_documents_from_folder(folder_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(document)

    def encode_passages(passages):
        return embedding_model.encode([f"passage: {p}" for p in passages])

    chunk_embeddings = encode_passages(chunks)

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(chunk_embeddings))

    chunk_id_to_text = {i: chunk for i, chunk in enumerate(chunks)}

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

# === Query Ollama ===
def query_ollama(prompt, model_name="gemma3:1b"): #change model here.
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return "An error occurred while querying the model."

# === Ask user ===
while True:
    user_query = input("\nAsk your question (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        break
    retrieved = retrieve_chunks(user_query)
    final_prompt = build_prompt(user_query, retrieved)
    response = query_ollama(final_prompt)
    print("\nAnswer:\n", response)