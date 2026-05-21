import cv2
import time
import base64
import requests
import json
import os
import sys
from PIL import Image
from io import BytesIO

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "moondream"  # VLM model
OUTPUT_FILE = "event_logs.jsonl"

def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def get_vlm_description(image_base64):
    payload = {
        "model": MODEL_NAME,
        "prompt": "Describe what is happening in this image briefly but accurately.",
        "images": [image_base64],
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main(video_source=0, frames_per_minute=5):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    is_live = isinstance(video_source, int) or str(video_source).isdigit()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    seconds_per_frame = 60.0 / frames_per_minute if frames_per_minute > 0 else 60.0
    frame_skip = max(1, int(fps * seconds_per_frame))

    print(f"Starting logger. Source: {video_source}")
    print(f"Source FPS: {fps:.2f}. Processing {frames_per_minute} frames per minute of video (every {frame_skip} frames).")
    print(f"Logs will be saved to {OUTPUT_FILE}")

    frame_counter = 0
    last_processed_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            should_process = False
            timestamp = ""
            
            if is_live:
                # For live cameras, rely on wall-clock time
                current_time = time.time()
                if current_time - last_processed_time >= seconds_per_frame:
                    should_process = True
                    last_processed_time = current_time
                    timestamp = time.strftime("[%I:%M %p]", time.localtime(current_time))
            else:
                # For video files, rely on video frame counting
                if frame_counter % frame_skip == 0:
                    should_process = True
                    video_timestamp_sec = frame_counter / fps
                    mins = int(video_timestamp_sec // 60)
                    secs = int(video_timestamp_sec % 60)
                    timestamp = f"[{mins:02d}:{secs:02d} Video Time]"

            if should_process:
                # Process frame
                image_base64 = encode_image(frame)
                description = get_vlm_description(image_base64)
                
                log_entry = {
                    "timestamp": timestamp,
                    "description": description
                }
                
                print(f"{timestamp} {description}")
                
                with open(OUTPUT_FILE, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            frame_counter += 1

    except KeyboardInterrupt:
        print("\nStopping logger...")
    finally:
        cap.release()
        # cv2.destroyAllWindows()  # Removed to prevent crash in headless environments

if __name__ == "__main__":
    video_source = 0
    frames_per_minute = 5  # default 5 frames per minute
    
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        if video_source.isdigit():
            video_source = int(video_source)
            
    if len(sys.argv) > 2:
        try:
            frames_per_minute = float(sys.argv[2])
        except ValueError:
            pass
    
    main(video_source, frames_per_minute)
