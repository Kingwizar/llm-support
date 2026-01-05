import requests
import json
OLLAMA_URL = "http://ollama:11434/api/generate"


r = requests.post(
    "OLLAMA_URL",
    json={
        "model": "qwen14b_llm",
        "prompt": "Bonjour"
    },
    stream=True
)

for line in r.iter_lines():
    if line:
        data = json.loads(line.decode("utf-8"))
        if "response" in data:
            print(data["response"], end="", flush=True)
