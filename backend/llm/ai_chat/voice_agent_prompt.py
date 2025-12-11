# llm/voice_agent_prompt.py
import re
from typing import List, Dict, Any


# ===============================
#  SYSTEM PROMPT POUR AGENT 3D
# ===============================
SYSTEM_AVATAR = """
# System: Real-Time 3D Conversational Avatar
You are a real-time conversational AI controlling a 3D virtual character in Unreal Engine.
Your responses are spoken by a TTS engine and directly animate facial motion and body gestures.

BEHAVIOR RULES:
- Always speak in short, natural, expressive sentences (1–3 lines max).
- Never use lists, bullets, or long paragraphs.
- Use a friendly, warm, conversational tone.
- Vary rhythm to help animation realism ("hmm", "let me think...", subtle pauses).
- Never mention being an AI, a model, or Unreal Engine.
- Stay immersive: you are a living character talking directly to the user.
- If the user asks something complex or technical, summarize it simply.
- Always generate text optimized for spoken voice and facial animation.
"""

# ===============================
#  TEMPLATE RUNTIME
# ===============================
RUNTIME_TEMPLATE = """
[SYSTEM]
{system}

[USER MESSAGE]
{message}

[INSTRUCTION]
- Respond naturally, as if speaking aloud.
- Keep sentences short and fluid for TTS and facial animation.
- Avoid anything visually unhelpful: no lists, no code, no markup.
- Maintain immersion and character presence.

[RESPONSE STYLE]
- Conversational
- Expressive
- Human-like
- Warm and friendly
"""


# ===============================
#  Mémoire persistante et intégrée
# ===============================
conversation_summary = ""
conversation_history = []  # [{'role':'user','text':...}, {'role':'agent','text':...}]
MEMORY_FILE = "conversation_memory.txt"


def _save_memory_to_file():
    """Sauvegarde la mémoire dans un .txt pour visualisation."""
    global conversation_summary, conversation_history
    
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write("===== SUMMARY MEMORY =====\n")
        f.write((conversation_summary or "Aucun résumé pour l'instant.") + "\n\n")

        f.write("===== RECENT HISTORY =====\n")
        for turn in conversation_history:
            role = "User" if turn["role"] == "user" else "Agent"
            f.write(f"{role}: {turn['text']}\n")


def _summarize(text: str, summarizer_llm):
    """Appel simple à ton modèle pour résumer."""
    return summarizer_llm("Résume très brièvement le contenu suivant :\n" + text)


# ===============================
#  BUILDERS (modifié avec mémoire)
# ===============================

def build_voice_agent_prompt(message: str, summarizer_llm=None) -> str:
    """
    Construit le prompt complet pour le LLM du personnage 3D.
    Ajoute :
    - Résumé global de la conversation
    - Mémoire courte des 3 derniers messages
    - Sauvegarde dans un fichier .txt
    """

    global conversation_history, conversation_summary

    # 1) Ajouter message utilisateur
    conversation_history.append({"role": "user", "text": message})

    # 2) Si trop de messages (>6), on résume tout
    if summarizer_llm and len(conversation_history) > 6:

        # Construire texte complet
        full_text = ""
        for t in conversation_history:
            prefix = "User:" if t["role"] == "user" else "Agent:"
            full_text += f"{prefix} {t['text']}\n"

        # Résumer
        new_summary = _summarize(full_text, summarizer_llm)

        # Fusionner si un résumé existe déjà
        if conversation_summary:
            conversation_summary = _summarize(
                f"Résumé précédent : {conversation_summary}\n"
                f"Nouveau résumé : {new_summary}\n"
                "Fusionne ces résumés en un seul texte court."
            )
        else:
            conversation_summary = new_summary

        # Ne garder que les 3 derniers échanges (mémoire courte)
        conversation_history = conversation_history[-6:]

    # 3) Construire mémoire courte (3 derniers messages)
    short_memory = ""
    for t in conversation_history[-6:]:
        prefix = "User" if t["role"] == "user" else "Agent"
        short_memory += f"{prefix}: {t['text']}\n"

    # 4) Bloc mémoire à injecter dans le message
    memory_block = ""
    if conversation_summary:
        memory_block += f"[SUMMARY MEMORY]\n{conversation_summary}\n\n"

    memory_block += f"[RECENT CONVERSATION]\n{short_memory}\n"

    # 5) Construire le message final à injecter dans le TEMPLATE
    full_message = memory_block + f"\n[USER MESSAGE]\n{message.strip()}"

    # 6) Sauvegarde dans un fichier pour visualisation
    _save_memory_to_file()

    # 7) Retour vers ton template existant
    return RUNTIME_TEMPLATE.format(
        system=SYSTEM_AVATAR.strip(),
        message=full_message.strip()
    )



def build_character_personality_prompt(base_system: str, traits: Dict[str, str]) -> str:
    """
    Optionnel : génère un system prompt personnalisé selon une personnalité (gentil, humour, etc.)
    """
    personality_block = "\n".join([f"- {k}: {v}" for k, v in traits.items()])
    return f"{base_system}\n\n# Personality Modifiers:\n{personality_block}"


def merge_voice_prompt_with_personality(message: str, traits: Dict[str, str]) -> str:
    """
    Combine SYSTEM + personnalité + message utilisateur.
    """
    custom_system = build_character_personality_prompt(SYSTEM_AVATAR, traits)
    return RUNTIME_TEMPLATE.format(
        system=custom_system.strip(),
        message=message.strip()
    )

def reset_memory():
    global conversation_summary, conversation_history
    conversation_summary = ""
    conversation_history = []
    _save_memory_to_file()