package com.example.npone_llm.data

import com.example.npone_llm.data.remote.dto.ChatRequestDto
import com.example.npone_llm.data.remote.dto.ChatResponseDto
import com.example.npone_llm.data.remote.dto.ConversationDto
import com.example.npone_llm.data.remote.dto.ConversationRequestDto
import com.example.npone_llm.data.remote.dto.MessageRequestDto
import retrofit2.http.Body
import retrofit2.http.DELETE
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Path

interface ChatApi {
    @POST("chat")
    suspend fun chat(@Body body: ChatRequestDto): ChatResponseDto

    @GET("conversations")
    suspend fun getConversations(): List<ConversationDto>

    @POST("conversations")
    suspend fun createConversation(@Body body: ConversationRequestDto): ConversationDto

    @POST("conversations/{id}/messages")
    suspend fun addMessage(
        @Path("id") conversationId: String,
        @Body body: MessageRequestDto
    ): ConversationDto

    @DELETE("conversations/{id}")
    suspend fun deleteConversation(@Path("id") id: String): Map<String, Any>

    @PUT("conversations/{id}")
    suspend fun renameConversation(@Path("id") id: String, @Body body: Map<String, String>): ConversationDto


}
