package com.example.npone_llm.data

import com.example.npone_llm.data.remote.dto.ChatRequestDto
import com.example.npone_llm.data.remote.dto.ChatResponseDto
import com.example.npone_llm.data.remote.dto.ConversationDto
import com.example.npone_llm.data.remote.dto.ConversationRequestDto
import com.example.npone_llm.data.remote.dto.MessageDto
import com.example.npone_llm.data.remote.dto.SendMessageResponseDto
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.http.*

interface ChatApi {

    // 🔹 1. Chat (RAG)
    // FastAPI → POST /chat
    @POST("/chat")
    suspend fun chat(@Body body: ChatRequestDto): ChatResponseDto


    // 🔹 2. Conversations
    // FastAPI → GET /conversations
    @GET("/conversations")
    suspend fun getConversations(): List<ConversationDto>

    // FastAPI → POST /conversations
    @POST("/conversations")
    suspend fun createConversation(@Body body: ConversationRequestDto): ConversationDto


    // 🔹 3. Récupérer les messages d’une conversation
    // FastAPI → GET /conversations/{id}/messages
    @GET("/conversations/{id}/messages")
    suspend fun getMessages(@Path("id") conversationId: String): List<MessageDto>


    // 🔹 4. Envoyer un message (texte + fichiers)
    // FastAPI → POST /message/{id}
    @Multipart
    @POST("/message/{id}")
    suspend fun sendMessage(
        @Path("id") conversationId: String,
        @Part("text") text: RequestBody,
        @Part files: List<MultipartBody.Part> = emptyList()
    ): SendMessageResponseDto


    // 🔹 5. Renommer une conversation
    // FastAPI → PUT /conversations/{id}
    @PUT("/conversations/{id}")
    suspend fun renameConversation(
        @Path("id") id: String,
        @Body body: Map<String, String>
    ): ConversationDto


    // 🔹 6. Supprimer une conversation
    // FastAPI → DELETE /conversations/{id}
    @DELETE("/conversations/{id}")
    suspend fun deleteConversation(@Path("id") id: String): Map<String, Any>
}
