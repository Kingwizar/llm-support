package com.example.npone_llm.data

import com.example.npone_llm.data.remote.ApiClient
import com.example.npone_llm.data.remote.dto.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import android.webkit.MimeTypeMap

class ChatRepository {

    // 🔹 1. Envoie une question RAG (chat intelligent)
    suspend fun sendQuestion(question: String, convId: String? = null): ChatResponseDto {
        return ApiClient.chatApi.chat(ChatRequestDto(question, convId))
    }

    // 🔹 2. Récupère toutes les conversations
    suspend fun getConversations() = ApiClient.chatApi.getConversations()

    // 🔹 3. Crée une nouvelle conversation
    suspend fun createConversation(title: String) =
        ApiClient.chatApi.createConversation(ConversationRequestDto(title))

    // 🔹 4. Récupère tous les messages d’une conversation
    suspend fun getMessages(conversationId: String) =
        ApiClient.chatApi.getMessages(conversationId)

    // 🔹 5. Envoie un message texte
    suspend fun sendTextMessage(conversationId: String, text: String): SendMessageResponseDto {
        val textPart = text.toRequestBody("text/plain".toMediaType())
        return ApiClient.chatApi.sendMessage(conversationId, textPart, emptyList())
    }

    // 🔹 6. Envoie un message avec fichier(s)
    suspend fun sendFileMessage(conversationId: String, text: String, files: List<File>): SendMessageResponseDto {
        val textPart = text.toRequestBody("text/plain".toMediaType())


        fun File.getMimeType(): String {
            val ext = extension.lowercase()
            return MimeTypeMap.getSingleton()
                .getMimeTypeFromExtension(ext)
                ?: "application/octet-stream"
        }

        val fileParts = files.map {
            val mime = it.getMimeType()
            val fileBody = it.asRequestBody(mime.toMediaType())
            MultipartBody.Part.createFormData("files", it.name, fileBody)
        }

        return ApiClient.chatApi.sendMessage(conversationId, textPart, fileParts)
    }

    // 🔹 7. Renomme une conversation
    suspend fun renameConversation(id: String, newTitle: String) =
        ApiClient.chatApi.renameConversation(id, mapOf("title" to newTitle))

    // 🔹 8. Supprime une conversation
    suspend fun deleteConversation(id: String) =
        ApiClient.chatApi.deleteConversation(id)
}
