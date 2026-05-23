import gradio as gr
import cv2
import time
import base64
import requests
import json
import os
import threading
import chromadb
from chromadb.utils import embedding_functions

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_MODEL = "moondream"
LLM_MODEL = "qwen3.5:0.8b"
LOG_FILE = "event_logs.jsonl"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "event_logs"

# Global state for logging thread control
logging_active = False
logging_thread = None
latest_frame = None
latest_status = "System Ready. Start the logger to process video frames."
event_list = []
alerts_list = []  # State for security alerts
alert_keywords = ["fire", "accident", "danger", "smoke", "fall", "spill"]
active_video_path = None

# Helper to encode OpenCV frame to base64
def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Helper to get description from local VLM (Moondream)
def get_vlm_description(image_base64):
    payload = {
        "model": VLM_MODEL,
        "prompt": "Describe what is happening in this image briefly but accurately.",
        "images": [image_base64],
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        return f"VLM Error: {str(e)}"

# Helper to parse video timestamps (e.g. "[01:12 Video Time]") to seconds
def parse_timestamp_to_seconds(timestamp):
    try:
        # Extract the time string, e.g. "01:12" from "[01:12 Video Time]"
        clean = timestamp.replace("[", "").replace("]", "").strip()
        time_part = clean.split(" ")[0]  # Get "01:12"
        parts = time_part.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        pass
    return 0

# Background logging worker
def logging_worker(video_source, fpm, alert_rules_str):
    global logging_active, latest_frame, latest_status, event_list, alerts_list, alert_keywords
    
    # Update active alert keywords from input
    alert_keywords = [k.strip().lower() for k in alert_rules_str.split(",") if k.strip()]
    if not alert_keywords:
        alert_keywords = ["fire", "accident", "danger", "smoke", "fall", "spill"]

    # Try parsing video source as integer (webcam index)
    try:
        source = int(video_source)
    except ValueError:
        source = video_source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        latest_status = f"❌ Error: Could not open video source '{video_source}'"
        logging_active = False
        return

    is_live = isinstance(source, int)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    seconds_per_frame = 60.0 / fpm if fpm > 0 else 60.0
    frame_skip = max(1, int(fps * seconds_per_frame))

    latest_status = f"⚡ Running: Processing {fpm} FPM. Source FPS: {fps:.1f}."
    frame_counter = 0
    last_processed_time = 0

    while logging_active:
        ret, frame = cap.read()
        if not ret:
            latest_status = "🏁 Completed: End of video feed."
            break

        should_process = False
        timestamp = ""

        if is_live:
            current_time = time.time()
            if current_time - last_processed_time >= seconds_per_frame:
                should_process = True
                last_processed_time = current_time
                timestamp = time.strftime("[%I:%M %p]", time.localtime(current_time))
        else:
            if frame_counter % frame_skip == 0:
                should_process = True
                video_timestamp_sec = frame_counter / fps
                mins = int(video_timestamp_sec // 60)
                secs = int(video_timestamp_sec % 60)
                timestamp = f"[{mins:02d}:{secs:02d} Video Time]"

        if should_process:
            h, w = frame.shape[:2]
            max_size = 640
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                preview_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                preview_frame = frame.copy()

            # Convert BGR to RGB for Gradio Image Preview
            rgb_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            latest_frame = rgb_frame

            # Get description from Moondream
            image_base64 = encode_image(preview_frame)
            description = get_vlm_description(image_base64)

            log_entry = {
                "timestamp": timestamp,
                "description": description
            }

            # Write to log file
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Add to description logs
            event_list.append(f"{timestamp} - {description}")
            latest_status = f"✅ Logged {timestamp}: {description}"

            # Check Alert Rules
            desc_lower = description.lower()
            triggered_alerts = [kw for kw in alert_keywords if kw in desc_lower]
            for kw in triggered_alerts:
                alert_msg = f"🚨 {timestamp} Alert triggered: '{kw.upper()}' detected! Description: {description}"
                alerts_list.append(alert_msg)

        frame_counter += 1
        time.sleep(0.01)

    cap.release()
    logging_active = False

# Gradio Event: Start Logger
def start_logging(video_source, uploaded_file, fpm, alert_rules):
    global logging_active, logging_thread, latest_status, event_list, alerts_list, active_video_path
    
    if logging_active:
        return "Logger is already running!", None, gr.update(), gr.update()

    # Use uploaded file if present
    source = video_source
    if uploaded_file is not None:
        source = uploaded_file.name
        active_video_path = source  # Update active path for player
    else:
        active_video_path = source if not source.isdigit() else None

    if not source:
        return "❌ Please enter a video source or upload a file.", None, gr.update(), gr.update()

    logging_active = True
    event_list = []
    alerts_list = []  # Clear previous alerts
    
    logging_thread = threading.Thread(target=logging_worker, args=(source, fpm, alert_rules))
    logging_thread.daemon = True
    logging_thread.start()

    # We return the video file path to the video player if it is a local video file
    return "🔄 Initializing logging worker...", None, gr.update(value="Stop Logger", variant="stop"), active_video_path

# Gradio Event: Stop Logger
def stop_logging():
    global logging_active, latest_status
    if logging_active:
        logging_active = False
        latest_status = "⏹️ Logger stopped by user."
        return "⏹️ Stopping logger...", gr.update(value="Start Logger", variant="primary")
    return "Logger is not running.", gr.update(value="Start Logger", variant="primary")

# Gradio Event: Update UI elements (Called by Timer)
def update_ui_status():
    global latest_frame, latest_status, event_list, alerts_list
    
    logs_text = "\n".join(reversed(event_list)) if event_list else "No logs captured yet."
    
    # Format security alerts layout
    if alerts_list:
        alerts_html = "<div style='display:flex; flex-direction:column; gap:8px;'>"
        for alert in reversed(alerts_list):
            alerts_html += f"""
            <div style='background:rgba(239,68,68,0.15); padding:10px; border-radius:8px; border-left:4px solid #ef4444; color:#fca5a5;'>
                {alert}
            </div>
            """
        alerts_html += "</div>"
    else:
        alerts_html = "<div style='color:#94a3b8; font-style:italic;'>No alerts triggered.</div>"

    return latest_frame, latest_status, logs_text, alerts_html

# Gradio Event: Index Logs
def run_indexer():
    if not os.path.exists(LOG_FILE):
        return "❌ Error: 'event_logs.jsonl' does not exist. Run the logger first."

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_func = embedding_functions.DefaultEmbeddingFunction()
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )

        with open(LOG_FILE, "r") as f:
            lines = f.readlines()

        ids = []
        documents = []
        metadatas = []

        for i, line in enumerate(lines):
            entry = json.loads(line)
            ids.append(f"id_{i}")
            documents.append(entry["description"])
            metadatas.append({"timestamp": entry["timestamp"]})

        if documents:
            # Overwrite or add to index
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            return f"🎉 Success: Indexed {len(documents)} event logs into ChromaDB!"
        else:
            return "⚠️ No logs found to index."
    except Exception as e:
        return f"❌ Indexing Error: {str(e)}"

# Gradio Event: RAG Search Query
def perform_search(query):
    if not query:
        return "Please enter a query.", ""

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_func = embedding_functions.DefaultEmbeddingFunction()
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )

        results = collection.query(query_texts=[query], n_results=5)
        
        # Filter by semantic distance threshold (1.45)
        threshold = 1.45
        valid_indices = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                if results['distances'][0][i] <= threshold:
                    valid_indices.append(i)

        if not valid_indices:
            return "No relevant events found in database matching your query.", "<div style='color:#94a3b8; font-style:italic;'>No matches found under relevance threshold.</div>"

        # Format retrieved logs with 'Click to Seek Video' interactive buttons
        retrieved_html = "<div style='display:flex; flex-direction:column; gap:10px;'>"
        context = ""
        for i in valid_indices:
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            context += f"{meta['timestamp']}: {doc}\n"
            
            # Calculate time seek offset in seconds
            seek_seconds = parse_timestamp_to_seconds(meta['timestamp'])
            
            # Render a card with an inline JS onclick handler targeting the gradio HTML5 video player
            retrieved_html += f"""
            <div style='background:rgba(255,255,255,0.03); padding:12px; border-radius:10px; border-left:4px solid #10b981; display:flex; justify-content:space-between; align-items:center;'>
                <div style='flex:1; padding-right:15px;'>
                    <strong style='color:#10b981;'>{meta['timestamp']}</strong>: {doc}
                </div>
                <button onclick="let v = document.querySelector('video'); if(v) {{ v.currentTime = {seek_seconds}; v.play(); }}" 
                        style='background:#10b981; color:white; border:none; padding:6px 12px; border-radius:6px; cursor:pointer; font-weight:600; font-size:0.85rem; transition: background 0.2s;'
                        onmouseover='this.style.background=\"#059669\"'
                        onmouseout='this.style.background=\"#10b981\"'>
                    🎬 Seek Frame
                </button>
            </div>
            """
        retrieved_html += "</div>"

        # 2. Free up VRAM by unloading Moondream
        try:
            requests.post(OLLAMA_URL, json={"model": "moondream", "keep_alive": 0})
        except:
            pass

        # 3. Query local LLM (Qwen)
        prompt = f"""You are a security assistant. Based on the following event logs, answer the user's question.
If the information is not in the logs, say you do not know.

Event Logs:
{context}

Question: {query}

Answer:"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        answer = response.json().get('response', '').strip()

        return answer, retrieved_html
    except Exception as e:
        return f"❌ Search Error: {str(e)} (Make sure database is indexed)", ""

# Custom CSS for UI styling
CSS = """
body {
    background-color: #090d16 !important;
    background-image: radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%), radial-gradient(at 100% 0%, rgba(16, 185, 129, 0.08) 0px, transparent 50%) !important;
}
.gradio-container {
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    background: rgba(13, 20, 35, 0.75) !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4) !important;
    padding: 24px !important;
}
.panel-border {
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.02) !important;
    padding: 16px !important;
}
.title-bar {
    text-align: center;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 15px;
}
.video-preview-holder {
    background: rgba(0,0,0,0.5) !important;
    border-radius: 8px !important;
    overflow: hidden;
}
"""

with gr.Blocks(css=CSS, title="Semantic Surveillance Dashboard") as demo:
    
    with gr.Group(elem_classes=["title-bar"]):
        gr.Markdown(
            """
            # 👁️ Semantic Surveillance & Local RAG Dashboard
            ### Process surveillance feeds locally with Moondream VLM & index to ChromaDB for natural language Q&A.
            """
        )

    with gr.Row():
        # LEFT COLUMN: INGESTION & VIDEO PLAYBACK
        with gr.Column(scale=1.1, elem_classes=["panel-border"]):
            gr.Markdown("### 📹 Video Source & Player")
            
            video_input = gr.Textbox(
                value="0",
                label="Video File Path or Webcam Index",
                placeholder="0 (webcam) or /path/to/video.mp4"
            )
            
            file_uploader = gr.File(
                label="Or Upload Video File",
                file_types=["video"]
            )
            
            fpm_slider = gr.Slider(
                minimum=1,
                maximum=30,
                value=5,
                step=1,
                label="Processing Rate (Frames Per Minute)"
            )
            
            with gr.Row():
                start_btn = gr.Button("Start Logger", variant="primary")
                stop_btn = gr.Button("Stop Logger", variant="secondary")
            
            status_box = gr.Textbox(
                value="System Ready.",
                label="Logger Status",
                interactive=False
            )
            
            # Interactive HTML5 Video Player
            video_player = gr.Video(
                label="Active Video Playback (Jump-to-Seek Target)",
                interactive=False,
                elem_classes=["video-preview-holder"]
            )
            
            # Sub-frame preview (VLM scanner snapshot)
            frame_preview = gr.Image(
                label="Latest VLM Scanner Capture",
                interactive=False,
                height=180
            )

        # MIDDLE COLUMN: EVENT LOGS & ALERTS
        with gr.Column(scale=1.1, elem_classes=["panel-border"]):
            gr.Markdown("### 🚨 Alert Settings & Activity Stream")
            
            alert_rules_input = gr.Textbox(
                value="fire, smoke, accident, danger, vehicle",
                label="Alert Trigger Keywords (Comma separated)",
                placeholder="fire, danger, intrusion"
            )
            
            gr.Markdown("#### 🔴 Active Security Alarms")
            alerts_display = gr.HTML(
                value="<div style='color:#94a3b8; font-style:italic;'>No alerts triggered.</div>"
            )
            
            gr.Markdown("#### 📜 Rolling Descriptions Stream")
            rolling_logs = gr.TextArea(
                value="No logs captured yet.",
                label="Visual Feed Description logs (Newest First)",
                interactive=False,
                lines=10
            )
            
            index_btn = gr.Button("📂 Index Logs into Vector DB", variant="secondary")
            index_status = gr.Textbox(
                label="Indexing Output",
                interactive=False
            )

        # RIGHT COLUMN: RAG SEARCH & INTERACTIVE RETRIEVAL
        with gr.Column(scale=1.3, elem_classes=["panel-border"]):
            gr.Markdown("### 🧠 Semantic AI Search (RAG)")
            
            search_input = gr.Textbox(
                placeholder="Ask about forklift movements, workers, specific colors...",
                label="Search Query"
            )
            
            search_btn = gr.Button("🔍 Query Database", variant="primary")
            
            gr.Markdown("#### 🤖 AI Synthesized Summary")
            ai_answer = gr.Markdown(
                value="*Ask a question above to retrieve events and synthesize answers.*"
            )
            
            gr.Markdown("#### 🎬 Retrieved Events (Click 'Seek Frame' to control Player)")
            retrieved_events = gr.HTML(
                value="<div style='color:#6b7280; font-style:italic;'>No search performed yet.</div>"
            )

    # Timer component to pull status updates while logging is active
    timer = gr.Timer(value=1.0, active=True)
    timer.tick(
        fn=update_ui_status,
        inputs=[],
        outputs=[frame_preview, status_box, rolling_logs, alerts_display]
    )

    # Button click mappings
    start_btn.click(
        fn=start_logging,
        inputs=[video_input, file_uploader, fpm_slider, alert_rules_input],
        outputs=[status_box, frame_preview, start_btn, video_player]
    )
    
    stop_btn.click(
        fn=stop_logging,
        inputs=[],
        outputs=[status_box, start_btn]
    )
    
    index_btn.click(
        fn=run_indexer,
        inputs=[],
        outputs=[index_status]
    )
    
    search_btn.click(
        fn=perform_search,
        inputs=[search_input],
        outputs=[ai_answer, retrieved_events]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)
