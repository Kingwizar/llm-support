package com.example.npone_llm.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import com.example.npone_llm.data.ChatRepository
import com.example.npone_llm.data.remote.dto.ChatResponseDto
import com.example.npone_llm.data.remote.dto.ConversationDto
import kotlinx.coroutines.launch

class ChatViewModel : ViewModel() {
    private val repo = ChatRepository()

    // Liste des conversations (historique)
    var conversations = mutableStateListOf<ConversationDto>()
        private set

    // Conversation actuellement sélectionnée
    var currentConversation = mutableStateOf<ConversationDto?>(null)
        private set

    // Réponse en cours
    var lastResponse = mutableStateOf<ChatResponseDto?>(null)
        private set

    // États de chargement / erreur
    var isLoading = mutableStateOf(false)
        private set
    var error = mutableStateOf<String?>(null)
        private set

    // --- Charger toutes les conversations depuis le backend
    fun loadConversations() {
        viewModelScope.launch {
            try {
                isLoading.value = true
                val convs = repo.getConversations()
                conversations.clear()
                conversations.addAll(convs)
                if (convs.isNotEmpty() && currentConversation.value == null) {
                    currentConversation.value = convs.first()
                }
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur lors du chargement"
            } finally {
                isLoading.value = false
            }
        }
    }

    // --- Créer une nouvelle conversation
    fun createConversation(title: String) {
        viewModelScope.launch {
            try {
                val newConv = repo.createConversation(title)
                conversations.add(newConv)
                currentConversation.value = newConv
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur lors de la création"
            }
        }
    }

    // --- Sélectionner une conversation existante
    fun selectConversation(id: String) {
        val conv = conversations.find { it.id == id }
        currentConversation.value = conv
    }

    // --- Envoyer une question (chat avec LLM)
    fun sendQuestion(question: String) {
        viewModelScope.launch {
            try {
                isLoading.value = true

                // 1) Ajouter le message utilisateur dans la conversation
                val convId = currentConversation.value?.id
                    ?: run {
                        val newConv = repo.createConversation("Nouvelle conversation")
                        conversations.add(newConv)
                        currentConversation.value = newConv
                        newConv.id
                    }

                val updatedConv = repo.addMessage(convId, question, true)
                updateConversationInList(updatedConv)

                // 2) Envoyer la question au backend /chat
                val res = repo.sendQuestion(question)
                lastResponse.value = res

                // 3) Construire un message complet avec summary + steps + citations
                val botMessage = buildString {
                    appendLine(res.summary)
                    appendLine()
                    res.steps.forEachIndexed { i, step ->
                        appendLine("${i + 1}. $step")
                    }
                    appendLine()
                    res.citations.forEach { c ->
                        appendLine("📚 Source: ${c.doc} (score=${c.score})")
                    }
                }

                // Ajouter ce message complet à la conversation
                val updatedWithLLM = repo.addMessage(convId, botMessage, false)
                updateConversationInList(updatedWithLLM)

                error.value = null
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur lors de l'envoi"
            } finally {
                isLoading.value = false
            }
        }
    }

    // --- Utilitaire pour garder la liste des conversations à jour
    private fun updateConversationInList(updated: ConversationDto) {
        val idx = conversations.indexOfFirst { it.id == updated.id }
        if (idx >= 0) {
            conversations[idx] = updated
        } else {
            conversations.add(updated)
        }
        if (currentConversation.value?.id == updated.id) {
            currentConversation.value = updated
        }
    }

    // --- Supprimer une conversation ---
    fun deleteConversation(id: String) {
        viewModelScope.launch {
            try {
                // Supprime côté backend
                repo.deleteConversation(id)
                // Supprime côté local
                conversations.removeAll { it.id == id }
                if (currentConversation.value?.id == id) {
                    currentConversation.value = null
                }
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur lors de la suppression"
            }
        }
    }

    // --- Renommer une conversation ---
    fun renameConversation(id: String, newTitle: String) {
        viewModelScope.launch {
            try {
                val updated = repo.renameConversation(id, newTitle)
                val idx = conversations.indexOfFirst { it.id == id }
                if (idx >= 0) {
                    conversations[idx] = updated
                }
                if (currentConversation.value?.id == id) {
                    currentConversation.value = updated
                }
            } catch (e: Exception) {
                error.value = e.message ?: "Erreur lors du renommage"
            }
        }
    }

}
