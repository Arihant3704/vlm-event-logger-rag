#!/bin/bash

# VLM Event Logger & RAG Search Starter

# Cleanup function to unload models from Ollama GPU memory on exit or Ctrl+C
cleanup() {
    echo -e "\n🧹 Script exiting. Unloading models from Ollama GPU memory..."
    curl -s -X POST http://localhost:11434/api/generate -d '{"model": "moondream", "keep_alive": 0}' > /dev/null
    curl -s -X POST http://localhost:11434/api/generate -d '{"model": "qwen3.5:0.8b", "keep_alive": 0}' > /dev/null
    echo "✅ GPU memory freed."
}
trap cleanup EXIT INT TERM

# Check and install python dependencies if missing
echo "🔍 Checking Python dependencies..."
python3 -c "import gradio, PIL, chromadb, sentence_transformers, cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Missing dependencies. Installing from requirements.txt..."
    pip3 install -r requirements.txt
else
    echo "✅ Python dependencies are satisfied."
fi

# Check if Ollama is running
echo "🔍 Checking Ollama server..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Error: Ollama is not running. Please start it first (run 'ollama serve' in another terminal)."
    exit 1
fi
echo "✅ Ollama server is running."

# Check and pull models if missing
echo "🔍 Checking local models..."
OLLAMA_TAGS=$(curl -s http://localhost:11434/api/tags)

if ! echo "$OLLAMA_TAGS" | grep -q "moondream"; then
    echo "📥 Downloading 'moondream' VLM model..."
    ollama pull moondream
else
    echo "✅ 'moondream' VLM model is already downloaded."
fi

if ! echo "$OLLAMA_TAGS" | grep -q "qwen3.5:0.8b"; then
    echo "📥 Downloading 'qwen3.5:0.8b' LLM model..."
    ollama pull qwen3.5:0.8b
else
    echo "✅ 'qwen3.5:0.8b' LLM model is already downloaded."
fi

echo "------------------------------------------"
echo "   VLM Event Logger & RAG Search System   "
echo "------------------------------------------"
echo "1) Start Logger (Watch Video/Camera)"
echo "2) Run Indexer (Update Vector DB)"
echo "3) Start RAG Search (Ask questions)"
echo "4) Generate Dummy Logs (For testing)"
echo "5) Launch Gradio Dashboard (Web UI)"
echo "6) Exit"
echo "------------------------------------------"
read -p "Choose an option: " choice

case $choice in
    1)
        read -p "Enter video source (0 for webcam, or path to video file) [0]: " source
        source=${source:-0}
        read -p "Enter number of frames to process PER MINUTE (e.g., 5 means 5 frames spread equally over 1 minute) [5]: " fpm
        fpm=${fpm:-5}
        echo "Starting Logger with source: $source at $fpm frames per minute..."
        python3 logger.py "$source" "$fpm"
        ;;
    2)
        echo "Updating Index..."
        python3 indexer.py
        ;;
    3)
        echo "Starting RAG Search..."
        python3 search.py
        ;;
    4)
        echo "Generating Dummy Logs..."
        python3 create_dummy_logs.py
        echo "Done. Don't forget to run Option 2 to index them."
        ;;
    5)
        echo "Launching Gradio Dashboard Web UI on http://localhost:7865 ..."
        python3 app.py
        ;;
    6)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid option."
        ;;
esac
