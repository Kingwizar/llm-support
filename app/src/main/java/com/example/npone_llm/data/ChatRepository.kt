package com.example.npone_llm.data



import com.example.npone_llm.data.remote.ApiClient
import com.example.npone_llm.data.remote.dto.ChatRequestDto
import com.example.npone_llm.data.remote.dto.ChatResponseDto
import com.example.npone_llm.data.remote.dto.ConversationRequestDto
import com.example.npone_llm.data.remote.dto.MessageRequestDto

class ChatRepository {
    suspend fun sendQuestion(question: String): ChatResponseDto {
        return ApiClient.chatApi.chat(ChatRequestDto(question))
    }
    suspend fun getConversations() = ApiClient.chatApi.getConversations()


    suspend fun createConversation(title: String) =
        ApiClient.chatApi.createConversation(ConversationRequestDto(title))

    suspend fun addMessage(convId: String, content: String, role: String) =
        ApiClient.chatApi.addMessage(convId, MessageRequestDto(content, role))

    suspend fun deleteConversation(id: String) =
        ApiClient.chatApi.deleteConversation(id)

    suspend fun renameConversation(id: String, newTitle: String) =
        ApiClient.chatApi.renameConversation(id, mapOf("title" to newTitle))


}
