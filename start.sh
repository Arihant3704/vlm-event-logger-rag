#!/bin/bash

# VLM Event Logger & RAG Search Starter

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama is not running. Please start it first."
    exit 1
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
        echo "Launching Gradio Dashboard Web UI on http://localhost:7860 ..."
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
