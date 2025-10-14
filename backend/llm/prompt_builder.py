# llm/prompt_builder.py
import re
from typing import List, Dict, Any

# ================= SYSTEM PROMPT =================
SYSTEM_RAG = """# System: N+One Datacenter Customer Support
You are N+One Datacenter Customer Support Assistant.
Provide clear, technically accurate answers, based on official documentation.
...
"""

RUNTIME_TEMPLATE = """[SYSTEM]
{system}

[USER QUESTION]
{question}

[RETRIEVED CONTEXT]
{sources_block}

[ADDITIONAL CONTEXT]
If the input comes from an uploaded file, interpret its extracted content appropriately:
- If it's an image: describe what it likely shows or conveys (based on OCR text).
- If it's a report: summarize its content and infer key intent or implications.
- If it includes [WEB] items: summarize clearly as external web information, prefer official docs.

[RESPONSE REQUIREMENTS]
- Start with a clear Summary
- Then structured Procedure
- Add Citations if available
- Conclude with Assumptions or Clarifications
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
