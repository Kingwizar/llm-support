package com.example.npone_llm.data.remote.dto

data class CitationDto(
    val doc: String,
    val score: Double,
    val snippet: String? = null
)

data class ChatResponseDto(
    val summary: String,
    val steps: List<String>,
    val citations: List<CitationDto>,
    val conversation_id: String
)
