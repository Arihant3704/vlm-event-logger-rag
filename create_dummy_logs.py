import json

dummy_logs = [
    {"timestamp": "[10:31 PM]", "description": "Person entered restricted zone carrying a suspicious bag."},
    {"timestamp": "[10:33 PM]", "description": "Smoke-like object detected near the server rack."},
    {"timestamp": "[10:34 PM]", "description": "Area cleared. Fire suppression system activated briefly."},
    {"timestamp": "[10:40 PM]", "description": "Maintenance crew arrived at the scene."},
    {"timestamp": "[10:45 PM]", "description": "Everything back to normal. Room is empty."}
]

with open("event_logs.jsonl", "w") as f:
    for entry in dummy_logs:
        f.write(json.dumps(entry) + "\n")

print("Created dummy event_logs.jsonl for testing.")
