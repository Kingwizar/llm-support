import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"

payload = {
    "model": MODEL_NAME,
    "prompt": "Décris l'architecture d'un datacenter moderne.",
    "stream": True
}

with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line.decode("utf-8"))
            print(chunk.get("response", ""), end="", flush=True)