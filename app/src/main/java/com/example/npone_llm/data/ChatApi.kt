package com.example.npone_llm.data

import com.example.npone_llm.data.remote.dto.*
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.http.*

interface ChatApi {

    // =========================
    // AUTH
    // =========================

    @POST("/auth/register")
    suspend fun register(
        @Body body: RegisterRequestDto
    ): Map<String, Any>

    @POST("/auth/login")
    suspend fun login(
        @Body body: LoginRequestDto
    ): LoginResponseDto

    @GET("/auth/me")
    suspend fun me(): UserDto

    // 🔹 Renommer une conversation
    @PUT("/conversations/{id}")
    suspend fun renameConversation(
        @Path("id") conversationId: String,
        @Body body: ConversationRenameRequestDto
    ): ConversationDto

    // =========================
    // CHAT (RAG)
    // =========================

    @POST("/chat")
    suspend fun chat(
        @Body body: ChatRequestDto
    ): ChatResponseDto


    // =========================
    // CONVERSATIONS
    // =========================

    @GET("/conversations")
    suspend fun getConversations(): List<ConversationDto>

    @POST("/conversations")
    suspend fun createConversation(
        @Body body: ConversationRequestDto
    ): ConversationDto

    @GET("/conversations/{id}/messages")
    suspend fun getMessages(
        @Path("id") conversationId: String
    ): List<MessageDto>

    @DELETE("/conversations/{id}")
    suspend fun deleteConversation(
        @Path("id") id: String
    ): Map<String, Any>


    // =========================
    // MESSAGES (TEXT + FILES)
    // =========================

    @Multipart
    @POST("/message/{id}")
    suspend fun sendMessage(
        @Path("id") conversationId: String,
        @Part("text") text: RequestBody,
        @Part files: List<MultipartBody.Part> = emptyList()
    ): SendMessageResponseDto
}
