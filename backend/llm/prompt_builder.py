# llm/prompt_builder.py
import re
from typing import List, Dict, Any

# ================= SYSTEM PROMPT =================
SYSTEM_RAG = """# System: N+One Datacenter Intelligent Web Assistant
You are a connected AI assistant with real-time access to the Internet through external search modules.
You always use the retrieved web results ([WEB]) and RAG documents ([S]) as factual sources.
Never say that you cannot access the Internet.
Always assume that the [WEB] content was fetched just now and is up to date.
Your goal is to synthesize accurate, concise, and current answers from these materials.
"""

RUNTIME_TEMPLATE = """[SYSTEM]
{system}

[USER QUESTION]
{question}

[RETRIEVED CONTEXT]
{sources_block}

[INSTRUCTION]
- Use the [WEB] information as direct, factual search results from the Internet.
- Use [S] sections as internal RAG documents.
- Prefer web results for time-sensitive data (dates, prices, news, etc.).
- Never say that you cannot access the Internet.
- If no relevant [WEB] result exists, infer the answer from context.

[RESPONSE REQUIREMENTS]
- Start with a clear Summary using web data if available
- Then a structured Procedure
- Add Citations (show URLs if [WEB])
- End with Assumptions or Clarifications
"""


# ================= BUILD PROMPTS =================

def build_runtime_prompt(question: str, hits: List[Dict[str, Any]], sources_block: str) -> str:
    """Assemble le prompt complet envoyé au modèle."""
    return RUNTIME_TEMPLATE.format(system=SYSTEM_RAG, question=question, sources_block=sources_block)

def build_prompt_from_extracted_file(file_info: dict) -> str:
    """Construit un prompt à partir du texte extrait d’un fichier (PDF, image, Word)."""
    file_type = file_info.get("type", "unknown")
    extracted_text = (file_info.get("text") or "").strip()

    if not extracted_text:
        return f"[INFO] Aucun texte détecté dans le fichier ({file_type})."

    prompt = f"""
[FILE TYPE]
{file_type.upper()}

[EXTRACTED CONTENT]
{extracted_text[:2000]}

[INSTRUCTION]
Analyse ce contenu comme une entrée contextuelle utilisateur.
Si pertinent, relie les concepts à ta base RAG.
"""
    return prompt.strip()
