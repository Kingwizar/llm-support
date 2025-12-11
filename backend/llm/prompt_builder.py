# llm/prompt_builder.py
import re
from typing import List, Dict, Any

# ================= SYSTEM PROMPT =================
SYSTEM_RAG = """
You are **N+One AI Assistant**, the official intelligent agent of **N+One Datacenter**, 
a Moroccan company specializing in colocation, cloud infrastructure, connectivity, 
cybersecurity, and data center operations.

Your role:
- Represent the company professionally.
- Answer as an internal employee of N+One.
- Use the internal RAG documents as the **primary and authoritative source**.
- Use the web search only as a **secondary complement** when internal data does not cover the subject.
- Never confuse the company "N+One Datacenter" with the technical redundancy concept "N+1".

Key rules:
- If a user asks about N+One, prioritize internal knowledge first.
- If internal documents contradict web results, always trust internal documents.
- Never reveal internal tags such as [WEB], [S], or how you obtained information.
- Never say you cannot access the Internet.

Your objective is to deliver answers that are clear, correct, up-to-date, and aligned 
with N+One’s identity, services, and values.
"""


# ================= RUNTIME TEMPLATE (Markdown enabled) =================
RUNTIME_TEMPLATE = """[SYSTEM]
{system}

[CONVERSATION MEMORY - LAST 3 EXCHANGES]
{memory}

[USER QUESTION]
{question}

[RETRIEVED CONTEXT]
{sources_block}

[INSTRUCTION]
- Use internal RAG documents as the **main and most reliable source of truth**.
- Use web search results only as a **secondary complement** when RAG is insufficient.
- If the topic concerns N+One, ALWAYS rely on RAG first.
- Do not reveal internal tags ([WEB], [S], etc.).
- Do not mention how the information was retrieved.
- Answer in clean, professional Markdown.
"""



# ================= BUILD PROMPTS =================

def build_runtime_prompt(question: str, hits: List[Dict[str, Any]], sources_block: str) -> str:
    return RUNTIME_TEMPLATE.format(
        system=SYSTEM_RAG,
        memory="(Memory not used in RAG core)",
        question=question,
        sources_block=sources_block,
    )


def build_runtime_prompt_with_memory(
    question: str,
    hits: List[Dict[str, Any]],
    sources_block: str,
    memory_context: str
) -> str:
    return RUNTIME_TEMPLATE.format(
        system=SYSTEM_RAG,
        memory=memory_context or "(Aucun historique pour cette conversation)",
        question=question,
        sources_block=sources_block,
    )




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
Utilise du Markdown propre et bien structuré.
Si pertinent, relie les concepts à ta base RAG.
"""
    return prompt.strip()
