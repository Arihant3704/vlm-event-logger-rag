import os
import shutil
import subprocess
import time
import json
import requests
import chromadb
from chromadb.utils import embedding_functions

# Config
VIDEO_PATH = "/home/docketrun/Videos/teshandu.mp4"
LOG_FILE = "event_logs.jsonl"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "event_logs"
OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_MODEL = "moondream"
LLM_MODEL = "qwen3.5:0.8b"

def clean_old_data():
    print("Cleaning old data...")
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print(f"Removed {LOG_FILE}")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Removed database at {CHROMA_PATH}")

def run_logger():
    print(f"Running VLM logger on video: {VIDEO_PATH} at 5 frames per minute...")
    # Run logger.py using subprocess
    # 5 frames per minute means 1 frame every 12 seconds
    proc = subprocess.Popen(["python3", "logger.py", VIDEO_PATH, "5"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Read output in real-time
    while True:
        output = proc.stdout.readline()
        if output == '' and proc.poll() is not None:
            break
        if output:
            print(f"[Logger]: {output.strip()}")
            
    rc = proc.poll()
    print(f"Logger finished with return code {rc}")
    return rc == 0

def run_indexer():
    print("Running indexer to load event logs into vector database...")
    proc = subprocess.run(["python3", "indexer.py"], capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print("Indexer Errors/Warnings:", proc.stderr)
    return proc.returncode == 0

def search_and_generate(query):
    print(f"Querying vector database with search query: '{query}'...")
    
    # Query Chroma
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
    
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    # Unload Moondream from GPU memory first
    print("Unloading Moondream from GPU VRAM...")
    try:
        requests.post(OLLAMA_URL, json={"model": VLM_MODEL, "keep_alive": 0}, timeout=5)
    except Exception as e:
        print("Could not contact Ollama to unload Moondream:", e)
        
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
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        answer = response.json().get('response', '').strip()
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
        
    return found_events, answer

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return
        
    # Start Ollama service verification
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            print("Ollama is not running properly.")
            return
    except Exception as e:
        print("Error connecting to Ollama. Please make sure Ollama is running.")
        return

    clean_old_data()
    
    # 1. Run Logger
    logger_success = run_logger()
    if not logger_success:
        print("Logger failed.")
        return
        
    # 2. Run Indexer
    indexer_success = run_indexer()
    if not indexer_success:
        print("Indexer failed.")
        return
        
    # 3. Search and generate answer
    query = "What vehicles (like forklifts or trucks) are seen in the video, where are they, and what are workers doing near them?"
    found_events, answer = search_and_generate(query)
    
    # Save results to markdown file
    result_content = f"""# VLM Event Logger & RAG Search Test Results

**Tested Video**: `{VIDEO_PATH}`
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
