import chromadb
from chromadb.utils import embedding_functions
import requests
import json

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "event_logs"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3.5:0.8b"

def search_events(query, n_results=5):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results

def generate_answer(query, search_results):
    # Free up VRAM by unloading Moondream
    try:
        requests.post(OLLAMA_URL, json={"model": "moondream", "keep_alive": 0})
    except:
        pass

    # Format context from search results
    context = ""
    for i in range(len(search_results['documents'][0])):
        doc = search_results['documents'][0][i]
        meta = search_results['metadatas'][0][i]
        context += f"{meta['timestamp']}: {doc}\n"

    prompt = f"""
You are a security assistant. Based on the following event logs, answer the user's question.
If the information is not in the logs, say you don't know.

Event Logs:
{context}

Question: {query}

Answer:"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    while True:
        query = input("\nSearch events (or 'q' to quit): ")
        if query.lower() == 'q':
            break

        print("Searching...")
        results = search_events(query)
        
        if not results['documents'][0]:
            print("No relevant events found.")
            continue

        print("\nRelevant Events Found:")
        for i in range(len(results['documents'][0])):
            print(f"- {results['metadatas'][0][i]['timestamp']}: {results['documents'][0][i]}")

        print("\nGenerating AI Summary...")
        answer = generate_answer(query, results)
        print(f"\nAI Answer:\n{answer}")

if __name__ == "__main__":
    main()
