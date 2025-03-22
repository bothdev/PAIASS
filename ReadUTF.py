# pip install chromadb ollama sentence-transformers

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB (Persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Load multilingual embedding model (for Khmer support)
embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# Read text file and split into chunks
def load_text_file(file_path, chunk_size=500):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Store text chunks in ChromaDB
def store_text_in_chromadb(file_path):
    text_chunks = load_text_file(file_path)
    for idx, chunk in enumerate(text_chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(ids=[str(idx)], embeddings=[embedding], metadatas=[{"text": chunk}])
    print(f"✅ Stored {len(text_chunks)} chunks in ChromaDB!")

# Retrieve relevant chunks from ChromaDB
def retrieve_relevant_docs(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return [res["text"] for res in results["metadatas"][0]]

# Ask Gemma a question using the retrieved context
def ask_gemma(query):
    # Retrieve relevant docs in Khmer or other languages
    retrieved_docs = retrieve_relevant_docs(query)
    
    # Provide more context to Gemma if necessary
    context = " ".join(retrieved_docs)
    
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    # Send the question and context to Ollama
    response = ollama.chat(model="Gemma3:4b", messages=[{"role": "user", "content": prompt}])
    
    # Return response
    return response["message"]["content"]

# Save the response answer to a file
def save_answer_to_file(answer, filename="response.txt"):
    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"Answer: {answer}\n")
    print(f"✅ Answer saved to {filename}")

# Main function
if __name__ == "__main__":
    file_path = "./documents/data.txt"  # Change this to your text file
    store_text_in_chromadb(file_path)

    while True:
        query = input("\nAsk a question in Khmer (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = ask_gemma(query)
        print("\n🤖 Answer:", answer)
        
        # Save the answer to a file
        save_answer_to_file(answer)
