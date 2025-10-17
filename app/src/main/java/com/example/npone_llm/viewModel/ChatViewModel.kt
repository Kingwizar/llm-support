package com.example.npone_llm.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import com.example.npone_llm.data.ChatRepository
import com.example.npone_llm.data.remote.dto.ChatResponseDto
import com.example.npone_llm.data.remote.dto.ConversationDto
import kotlinx.coroutines.launch
import java.io.File

class ChatViewModel : ViewModel() {
    private val repo = ChatRepository()

    // 🔹 Liste des conversations (historique)
    var conversations = mutableStateListOf<ConversationDto>()
        private set

    // 🔹 Conversation sélectionnée
    var currentConversation = mutableStateOf<ConversationDto?>(null)
        private set

    // 🔹 Dernière réponse du LLM
    var lastResponse = mutableStateOf<ChatResponseDto?>(null)
        private set

    // 🔹 États de chargement / erreur
    var isLoading = mutableStateOf(false)
        private set
    var error = mutableStateOf<String?>(null)
        private set

    // =====================
    // 🔹 Charger les conversations
    // =====================
    fun loadConversations() {
        viewModelScope.launch {
            try {
                isLoading.value = true
                val convs = repo.getConversations()
                conversations.clear()
                conversations.addAll(convs)

                // Si aucune conversation sélectionnée, prendre la première
                if (currentConversation.value == null && convs.isNotEmpty()) {
                    currentConversation.value = convs.first()
                }
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur lors du chargement"
            } finally {
                isLoading.value = false
            }
        }
    }

    // =====================
    // 🔹 Créer une nouvelle conversation
    // =====================
    fun createConversation(title: String) {
        viewModelScope.launch {
            try {
                val newConv = repo.createConversation(title)
                conversations.add(newConv)
                currentConversation.value = newConv
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur création conversation"
            }
        }
    }

    // =====================
    // 🔹 Sélectionner une conversation
    // =====================
    fun selectConversation(id: String) {
        val conv = conversations.find { it.id == id }
        currentConversation.value = conv
    }

    // =====================
    // 🔹 Envoyer un message texte
    // =====================
    fun sendTextMessage(text: String) {
        viewModelScope.launch {
            try {
                val convId = ensureConversation()
                isLoading.value = true

                repo.sendTextMessage(convId, text)

                // 🔁 Recharger la conversation depuis la base
                reloadMessages(convId)

                // 🔹 Envoyer la question au LLM
                val response = repo.sendQuestion(text, convId)
                lastResponse.value = response


                reloadMessages(convId)

                error.value = null
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur envoi message"
            } finally {
                isLoading.value = false
            }
        }
    }

    // =====================
    // 🔹 Envoyer un message avec fichier(s)
    // =====================
    fun sendFileMessage(text: String, files: List<File>) {
        viewModelScope.launch {
            try {
                val convId = ensureConversation()
                isLoading.value = true

                repo.sendFileMessage(convId, text, files)
                reloadMessages(convId)

                error.value = null
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur envoi fichier"
            } finally {
                isLoading.value = false
            }
        }
    }

    // =====================
    // 🔹 Supprimer une conversation
    // =====================
    fun deleteConversation(id: String) {
        viewModelScope.launch {
            try {
                repo.deleteConversation(id)
                conversations.removeAll { it.id == id }
                if (currentConversation.value?.id == id) {
                    currentConversation.value = null
                }
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur suppression conversation"
            }
        }
    }

    // =====================
    // 🔹 Renommer une conversation
    // =====================
    fun renameConversation(id: String, newTitle: String) {
        viewModelScope.launch {
            try {
                val updated = repo.renameConversation(id, newTitle)
                val idx = conversations.indexOfFirst { it.id == id }
                if (idx >= 0) conversations[idx] = updated
                if (currentConversation.value?.id == id) currentConversation.value = updated
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur renommage conversation"
            }
        }
    }

    // =====================
    // 🔹 Utilitaires internes
    // =====================
    private suspend fun ensureConversation(): String {
        val existing = currentConversation.value
        if (existing != null) return existing.id

        val newConv = repo.createConversation("Nouvelle conversation")
        conversations.add(newConv)
        currentConversation.value = newConv
        return newConv.id
    }

    private fun reloadMessages(convId: String) {
        viewModelScope.launch {
            try {
                val msgs = repo.getMessages(convId)
                val conv = currentConversation.value?.copy(messages = msgs)
                currentConversation.value = conv
            } catch (e: Exception) {
                error.value = e.message
            }
        }
    }
}
