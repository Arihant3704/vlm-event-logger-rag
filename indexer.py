import chromadb
from chromadb.utils import embedding_functions
import json
import os

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "event_logs"
LOG_FILE = "event_logs.jsonl"

def index_logs():
    # Set up ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Use a default embedding function
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    if not os.path.exists(LOG_FILE):
        print(f"No log file found at {LOG_FILE}")
        return

    print(f"Indexing logs from {LOG_FILE}...")
    
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    ids = []
    documents = []
    metadatas = []

    for i, line in enumerate(lines):
        entry = json.loads(line)
        ids.append(f"id_{i}")
        # We store the description as the document to be embedded
        documents.append(entry["description"])
        # We store the timestamp and any other data as metadata
        metadatas.append({"timestamp": entry["timestamp"]})

    if documents:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully indexed {len(documents)} entries.")
    else:
        print("No new entries to index.")

if __name__ == "__main__":
    index_logs()
