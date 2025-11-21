# llm/voice_agent_prompt.py
import re
from typing import List, Dict, Any


# ===============================
# 🔵 SYSTEM PROMPT POUR AGENT 3D
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
# 🔧 TEMPLATE RUNTIME
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
# 🏗️ BUILDERS
# ===============================

def build_voice_agent_prompt(message: str) -> str:
    """
    Construit le prompt complet pour le LLM du personnage 3D.
    """
    return RUNTIME_TEMPLATE.format(
        system=SYSTEM_AVATAR.strip(),
        message=message.strip()
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
