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

class DocumentProcessor:
    def __init__(self, index_path, chunks_path, processed_files_path, folder_path="./documents"):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.processed_files_path = processed_files_path
        self.folder_path = folder_path
        self.embedding_model = self._load_embedding_model()
        self.index = None
        self.chunk_id_to_text = {}
        self.processed_files = set()
        self._load_or_initialize()

    def _load_embedding_model(self):
        """Load the embedding model."""
        model = SentenceTransformer("intfloat/multilingual-e5-base")
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("[INFO] Using GPU for embeddings.")
        else:
            print("[INFO] Using CPU for embeddings.")
        return model

    def _load_or_initialize(self):
        """Load or initialize the FAISS index and metadata."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
                print("[INFO] Loading FAISS index and chunks from disk...")
                self.index = faiss.read_index(self.index_path)
                with open(self.chunks_path, "rb") as f:
                    self.chunk_id_to_text = pickle.load(f)
                if os.path.exists(self.processed_files_path):
                    with open(self.processed_files_path, "rb") as f:
                        self.processed_files = pickle.load(f)
                else:
                    self._initialize_processed_files()
            else:
                print("[WARNING] FAISS index or chunks not found. Initializing new index...")
                self._initialize_new_index()
        except Exception as e:
            print(f"[ERROR] Failed to load or initialize FAISS index and metadata: {e}")
            print("[INFO] Initializing new FAISS index...")
            self._initialize_new_index()

    def _initialize_new_index(self):
        """Initialize a new FAISS index and metadata."""
        self.index = faiss.IndexFlatL2(self.embedding_model.get_sentence_embedding_dimension())
        self.chunk_id_to_text = {}
        self.processed_files = set()
        self._save_metadata()
        print("[INFO] New FAISS index initialized successfully.")

    def _initialize_processed_files(self):
        """Initialize the processed files set."""
        print("[INFO] Initializing processed files...")
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith((".txt", ".docx", ".pdf", ".xlsx")):
                self.processed_files.add(file_name)
        self._save_processed_files()

    def _save_processed_files(self):
        """Save the processed files set."""
        with open(self.processed_files_path, "wb") as f:
            pickle.dump(self.processed_files, f)
        print("[INFO] Processed files saved successfully.")

    def _save_metadata(self):
        """Save the FAISS index and metadata."""
        print("[INFO] Saving updated FAISS index and metadata...")
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunk_id_to_text, f)
        self._save_processed_files()

    def process_file(self, file_path):
        """Process a single file and return its text content."""
        if file_path.endswith(".txt"):
            return self._read_text_file(file_path)
        elif file_path.endswith(".docx"):
            return self._read_docx_file(file_path)
        elif file_path.endswith(".pdf"):
            return self._read_pdf_with_pymupdf(file_path)
        elif file_path.endswith(".xlsx"):
            return self._read_excel_file(file_path)
        return ""

    def _read_text_file(self, file_path):
        """Read content from a text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _read_docx_file(self, file_path):
        """Read content from a DOCX file."""
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    def _read_pdf_with_pymupdf(self, file_path):
        """Read content from a PDF file using PyMuPDF."""
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text

    def _read_excel_file(self, file_path):
        """Read content from an Excel file."""
        df = pd.read_excel(file_path)
        return df.to_string(index=False)

    def encode_passages(self, passages, file_name, batch_size=32):
        """Encode text passages in batches."""
        embeddings = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            print(f"[LOG] Embedding batch {i // batch_size + 1}/{(len(passages) + batch_size - 1) // batch_size}")
            print(f"Filename: {file_name}, Batch size: {len(batch)}")
            batch_embeddings = self.embedding_model.encode(
                [f"passage: {p}" for p in batch],
                convert_to_tensor=torch.cuda.is_available()
            )
            embeddings.extend(batch_embeddings.cpu().numpy() if torch.cuda.is_available() else batch_embeddings)
        return embeddings

    def process_documents(self):
        """Process new documents and update the index."""
        current_files = set(os.listdir(self.folder_path))
        # Remove embeddings for deleted files
        deleted_files = self.processed_files - current_files
        if deleted_files:
            print(f"[INFO] Removing embeddings for deleted files: {deleted_files}")
            for file_name in deleted_files:
                # Find and remove chunks related to the deleted file
                chunk_ids_to_remove = [
                    chunk_id for chunk_id, text in self.chunk_id_to_text.items()
                    if text.startswith(f"File: {file_name}")
                ]
                if chunk_ids_to_remove:
                    print(f"[INFO] Removing chunks for file: {file_name}")
                    print(f"[DEBUG] Chunk IDs to remove: {chunk_ids_to_remove}")
                    self.index.remove_ids(np.array(chunk_ids_to_remove, dtype=np.int64))
                    for chunk_id in chunk_ids_to_remove:
                        if chunk_id in self.chunk_id_to_text:
                            del self.chunk_id_to_text[chunk_id]
                            print(f"[DEBUG] Removed chunk ID {chunk_id} from metadata.")
                else:
                    print(f"[WARNING] No chunks found for file: {file_name}")
                self.processed_files.remove(file_name)

            # Verify consistency of the index and metadata
            if len(self.chunk_id_to_text) != self.index.ntotal:
                print("[ERROR] Inconsistency detected between FAISS index and metadata. Rebuilding index...")
                self._rebuild_index()

            self._save_metadata()  # Save updated index and metadata after deletion

        # Process new files
        for file_name in current_files:
            file_path = os.path.join(self.folder_path, file_name)
            if file_name in self.processed_files:
                print(f"[INFO] Skipping already processed file: {file_name}")
                continue
            if os.path.isfile(file_path) and file_name.endswith((".txt", ".docx", ".pdf", ".xlsx")):
                print(f"[INFO] Processing new file: {file_name}")
                document_text = self.process_file(file_path)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_text(document_text)
                chunk_embeddings = self.encode_passages(chunks, file_name)
                self.index.add(np.array(chunk_embeddings))
                self.chunk_id_to_text.update({len(self.chunk_id_to_text) + i: f"File: {file_name}\n{chunk}" for i, chunk in enumerate(chunks)})
                self.processed_files.add(file_name)
        self._save_metadata()

    def _rebuild_index(self):
        """Rebuild the FAISS index from the current metadata."""
        print("[INFO] Rebuilding FAISS index...")
        self.index = faiss.IndexFlatL2(self.embedding_model.get_sentence_embedding_dimension())
        for chunk_id, text in self.chunk_id_to_text.items():
            embedding = self.embedding_model.encode([text], convert_to_tensor=False)
            self.index.add(np.array(embedding, dtype=np.float32))
        print("[INFO] FAISS index rebuilt successfully.")
        print(f"[DEBUG] Rebuilt index size: {self.index.ntotal}")

    def retrieve_chunks(self, query, top_k=5):
        """Retrieve top-k chunks for a given query."""
        print(f"[INFO] Retrieving top {top_k} chunks for query: {query}")
        query_vector = self.embedding_model.encode([f"query: {query}"])
        D, I = self.index.search(np.array(query_vector), top_k)
        print(f"[INFO] Retrieved chunk IDs: {I[0]}")
        return [self.chunk_id_to_text[i] for i in I[0]]

    def build_prompt(self, query, retrieved_chunks):
        """Build a prompt using the retrieved chunks."""
        print("[INFO] Building prompt with retrieved chunks...")
        context = "\n\n".join(retrieved_chunks)
        # print(f"[DEBUG] Context sent to LLM:\n{context}")  # Log the context
        return f"""You are a helpful assistant. Use the context below to answer the user's question.

        Context:
        {context}

        Question:
        {query}

        Answer:"""

    def query_llama_ollama(self, prompt, context):
        """Send a prompt to the Ollama model and return the response."""
        try:
            print("[INFO] Sending prompt to Ollama model...")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama2-uncensored", "prompt": prompt, "stream": False}
            )
            print("[INFO] Received response from Ollama model.")
            answer = response.json()['response']
            return f"Context:\n{context}\n\nAnswer:\n{answer}"
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to connect to Ollama model: {e}")
            return f"[ERROR] Unable to process the request due to connection issues.\n\nContext:\n{context}"

# === Main Execution ===
if __name__ == "__main__":
    processor = DocumentProcessor("index.faiss", "chunks.pkl", "documents.pkl")
    processor.process_documents()

    try:
        while True:
            user_query = input("\nQuestion: ")
            retrieved = processor.retrieve_chunks(user_query)
            final_prompt = processor.build_prompt(user_query, retrieved)
            response = processor.query_llama_ollama(final_prompt, "\n\n".join(retrieved))
            print("\n", response)
    except KeyboardInterrupt:
        print("\n[INFO] Exiting the application.")
