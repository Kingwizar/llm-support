package com.example.npone_llm.data

import com.example.npone_llm.data.remote.ApiClient
import com.example.npone_llm.data.remote.dto.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import android.webkit.MimeTypeMap
import com.example.npone_llm.data.remote.ApiSession
import java.io.File
import okhttp3.Interceptor
import okhttp3.Response



class ChatRepository {

    // =========================
    // AUTH
    // =========================

    suspend fun login(email: String, password: String): LoginResponseDto {
        val response = ApiClient.chatApi.login(
            LoginRequestDto(email, password)
        )

        ApiSession.accessToken = response.access_token

        return response
    }


    suspend fun register(username: String, email: String, password: String) {
        ApiClient.chatApi.register(
            RegisterRequestDto(
                username = username,
                email = email,
                password = password
            )
        )
    }

    suspend fun me(): UserDto {
        return ApiClient.chatApi.me()
    }


    // =========================
    // CHAT (RAG)
    // =========================

    suspend fun sendQuestion(
        question: String,
        convId: String? = null
    ): ChatResponseDto {
        return ApiClient.chatApi.chat(
            ChatRequestDto(question = question, conv_id = convId)
        )
    }


    // =========================
    // CONVERSATIONS
    // =========================

    suspend fun getConversations(): List<ConversationDto> {
        return ApiClient.chatApi.getConversations()
    }

    suspend fun createConversation(title: String): ConversationDto {
        return ApiClient.chatApi.createConversation(
            ConversationRequestDto(title = title)
        )
    }

    suspend fun getMessages(conversationId: String): List<MessageDto> {
        return ApiClient.chatApi.getMessages(conversationId)
    }

    suspend fun deleteConversation(conversationId: String): Boolean {
        return ApiClient.chatApi
            .deleteConversation(conversationId)
            .getOrDefault("success", false) as Boolean
    }

    // 🔹 Renommer une conversation
    suspend fun renameConversation(
        conversationId: String,
        newTitle: String
    ): ConversationDto {
        return ApiClient.chatApi.renameConversation(
            conversationId,
            ConversationRenameRequestDto(title = newTitle)
        )
    }





    // =========================
    // MESSAGES (TEXT / FILES)
    // =========================

    suspend fun sendTextMessage(
        conversationId: String,
        text: String
    ): SendMessageResponseDto {
        val textPart = text.toRequestBody("text/plain".toMediaType())
        return ApiClient.chatApi.sendMessage(
            conversationId = conversationId,
            text = textPart,
            files = emptyList()
        )
    }

    suspend fun sendFileMessage(
        conversationId: String,
        text: String,
        files: List<File>
    ): SendMessageResponseDto {

        val textPart = text.toRequestBody("text/plain".toMediaType())

        fun File.getMimeType(): String {
            val ext = extension.lowercase()
            return MimeTypeMap.getSingleton()
                .getMimeTypeFromExtension(ext)
                ?: "application/octet-stream"
        }

        val fileParts = files.map { file ->
            val mime = file.getMimeType()
            val body = file.asRequestBody(mime.toMediaType())
            MultipartBody.Part.createFormData(
                name = "files",
                filename = file.name,
                body = body
            )
        }

        return ApiClient.chatApi.sendMessage(
            conversationId = conversationId,
            text = textPart,
            files = fileParts
        )
    }
}
