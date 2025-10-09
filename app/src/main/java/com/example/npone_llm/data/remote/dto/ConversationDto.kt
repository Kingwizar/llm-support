package com.example.npone_llm.data.remote.dto

data class ConversationDto(
    val id: String,
    val title: String? = "(Sans titre)",
    val messages: List<MessageDto>
)

data class MessageDto(
    val content: String,
    val role: String
)