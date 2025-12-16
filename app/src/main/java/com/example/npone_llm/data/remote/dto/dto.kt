package com.example.npone_llm.data.remote.dto

// Pour le RAG
data class ChatRequestDto(
    val question: String,
    val conv_id: String? = null
)

data class ChatResponseDto(
    val summary: String,
    val steps: List<String>,
    val citations: List<CitationDto>,
    val conversation_id: String
)

data class CitationDto(
    val doc: String,
    val score: Double,
    val snippet: String?
)


// Pour les conversations et messages
data class ConversationDto(
    val id: String,
    val title: String,
    val messages: List<MessageDto> = emptyList()
)

data class ConversationRequestDto(
    val title: String
)
data class ConversationRenameRequestDto(
    val title: String
)


data class MessageDto(
    val _id: String? = null,
    val role: String,
    val content: String?,
    val isUser: Boolean,
    val uploaded_at: String? = null,
    val files: List<FileDto> = emptyList()
)


data class FileDto(
    val file_id: String,
    val file_name: String,
    val file_url: String,
    val uploaded_at: String?
)

data class SendMessageResponseDto(
    val success: Boolean
)

data class RegisterRequestDto(
    val username: String,
    val email: String,
    val password: String
)

data class LoginRequestDto(
    val email: String,
    val password: String
)

data class LoginResponseDto(
    val access_token: String,
    val token_type: String
)

data class UserDto(
    val id: String,
    val username: String,
    val email: String,
    val role: String,
    val created_at: String
)
