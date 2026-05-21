import os
import time
import json
import requests
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "event_logs"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3.5:0.8b"

def main():
    query = "What vehicles (like forklifts or trucks) are seen in the video, where are they, and what are workers doing near them?"
    print(f"Querying vector database with search query: '{query}'...")
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
    
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    context = ""
    found_events = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        context += f"{meta['timestamp']}: {doc}\n"
        found_events.append(f"{meta['timestamp']}: {doc}")
        
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

    print("Generating AI Summary using Qwen...")
    try:
        # Increased timeout to 120 seconds just in case
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        answer = response.json().get('response', '').strip()
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
        
    # Save results to markdown file
    result_content = f"""# VLM Event Logger & RAG Search Test Results

**Tested Video**: `/home/docketrun/Videos/teshandu.mp4`
**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 1. Retrieved Context Logs (Semantic Search)
Here are the top events matching the query:
"""
    for event in found_events:
        result_content += f"- {event}\n"
        
    result_content += f"""
## 2. Generated RAG Answer (Qwen)
{answer}
"""
    
    result_file = "test_result.md"
    with open(result_file, "w") as f:
        f.write(result_content)
        
    print(f"\n==================================================")
    print(f"RESULTS SAVED TO {result_file}")
    print(f"==================================================")
    print(result_content)

if __name__ == "__main__":
    main()
