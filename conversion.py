import json
import pandas as pd

rows = []
with open("conversations.json", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # ignorer lignes vides
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print("⚠️ Ligne ignorée:", e)

print(f"{len(rows)} documents chargés")

# Exemple : convertir en tabulaire simplifié
out = []
for doc in rows:
    user_msg  = next((m["text"] for m in doc.get("messages", []) if m["role"]=="user"), None)
    agent_msg = next((m["text"] for m in doc.get("messages", []) if m["role"]=="agent"), None)
    out.append({
        "conversation_id": doc.get("conversation_id"),
        "user_text": user_msg,
        "agent_text": agent_msg,
        "intent": doc.get("labels", {}).get("intent"),
        "category": doc.get("labels", {}).get("category"),
        "tags": " ".join(doc.get("meta", {}).get("tags", []))
    })

df = pd.DataFrame(out)
df.to_csv("dataset_final.csv", index=False)
print("✅ Exporté en dataset_final.csv")
