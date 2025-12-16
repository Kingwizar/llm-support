package com.example.npone_llm.viewModel

import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.npone_llm.data.ChatRepository
import com.example.npone_llm.data.remote.dto.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.time.Instant

class ChatViewModel : ViewModel() {

    private val repo = ChatRepository()

    // =====================
    // 🔐 AUTH STATE
    // =====================

    var isAuthenticated = mutableStateOf(false)
        private set

    var currentUser = mutableStateOf<UserDto?>(null)
        private set

    var authLoading = mutableStateOf(false)
        private set

    var authError = mutableStateOf<String?>(null)
        private set


    // =====================
    // 💬 CHAT STATE
    // =====================

    var conversations = mutableStateListOf<ConversationDto>()
        private set

    var currentConversation = mutableStateOf<ConversationDto?>(null)
        private set

    var lastResponse = mutableStateOf<ChatResponseDto?>(null)
        private set

    var isLoading = mutableStateOf(false)
        private set

    var error = mutableStateOf<String?>(null)
        private set


    // =====================================================
    // 🔐 AUTH
    // =====================================================

    fun login(email: String, password: String) {
        viewModelScope.launch {
            try {
                authLoading.value = true
                authError.value = null

                withContext(Dispatchers.IO) {
                    repo.login(email, password)
                }

                loadMe()

            } catch (e: Exception) {
                authError.value = e.message
            } finally {
                authLoading.value = false
            }
        }
    }

    fun register(username: String, email: String, password: String) {
        viewModelScope.launch {
            try {
                authLoading.value = true
                authError.value = null

                withContext(Dispatchers.IO) {
                    repo.register(username, email, password)
                }

                login(email, password)

            } catch (e: Exception) {
                authError.value = e.message
            } finally {
                authLoading.value = false
            }
        }
    }

    fun loadMe() {
        viewModelScope.launch {
            try {
                val user = withContext(Dispatchers.IO) {
                    repo.me()
                }

                currentUser.value = user
                isAuthenticated.value = true

                loadConversations()

            } catch (e: Exception) {
                isAuthenticated.value = false
                authError.value = e.message
            }
        }
    }

    fun logout() {
        isAuthenticated.value = false
        currentUser.value = null
        conversations.clear()
        currentConversation.value = null
    }


    // =====================================================
    // 💬 CONVERSATIONS
    // =====================================================

    fun loadConversations() {
        viewModelScope.launch {
            try {
                isLoading.value = true

                val convs = withContext(Dispatchers.IO) {
                    repo.getConversations()
                }

                conversations.clear()
                conversations.addAll(convs)

                if (currentConversation.value == null && convs.isNotEmpty()) {
                    currentConversation.value = convs.first()
                }

            } catch (e: Exception) {
                error.value = e.message
            } finally {
                isLoading.value = false
            }
        }
    }

    fun createConversation(title: String) {
        viewModelScope.launch {
            try {
                val newConv = withContext(Dispatchers.IO) {
                    repo.createConversation(title)
                }

                conversations.add(0, newConv)
                currentConversation.value = newConv

            } catch (e: Exception) {
                error.value = e.message
            }
        }
    }

    fun selectConversation(id: String) {
        currentConversation.value = conversations.find { it.id == id }
    }

    fun renameConversation(id: String, newTitle: String) {
        viewModelScope.launch {
            try {
                val updated = withContext(Dispatchers.IO) {
                    repo.renameConversation(id, newTitle)
                }

                val index = conversations.indexOfFirst { it.id == id }
                if (index >= 0) conversations[index] = updated

                if (currentConversation.value?.id == id) {
                    currentConversation.value = updated
                }

            } catch (e: Exception) {
                error.value = e.message
            }
        }
    }

    fun deleteConversation(id: String) {
        viewModelScope.launch {
            try {
                val success = withContext(Dispatchers.IO) {
                    repo.deleteConversation(id)
                }

                if (success) {
                    conversations.removeAll { it.id == id }
                    if (currentConversation.value?.id == id) {
                        currentConversation.value = conversations.firstOrNull()
                    }
                }

            } catch (e: Exception) {
                error.value = e.message
            }
        }
    }


    // =====================================================
    // ✉️ MESSAGES
    // =====================================================

    fun sendTextMessage(text: String) {
        viewModelScope.launch {
            try {
                val convId = ensureConversation()

                addLocalUserMessage(text)
                isLoading.value = true

                withContext(Dispatchers.IO) {
                    repo.sendTextMessage(convId, text)
                }

                reloadMessages(convId)

                val response = withContext(Dispatchers.IO) {
                    repo.sendQuestion(text, convId)
                }

                lastResponse.value = response
                reloadMessages(convId)

            } catch (e: Exception) {
                error.value = e.message
            } finally {
                isLoading.value = false
            }
        }
    }

    fun sendFileMessage(text: String, files: List<File>) {
        viewModelScope.launch {
            try {
                val convId = ensureConversation()

                addLocalUserMessage(text)
                isLoading.value = true

                withContext(Dispatchers.IO) {
                    repo.sendFileMessage(convId, text, files)
                }

                reloadMessages(convId)

            } catch (e: Exception) {
                error.value = e.message
            } finally {
                isLoading.value = false
            }
        }
    }


    // =====================================================
    // 🔧 INTERNES
    // =====================================================

    private suspend fun ensureConversation(): String {
        currentConversation.value?.let { return it.id }

        val newConv = withContext(Dispatchers.IO) {
            repo.createConversation("Nouvelle conversation")
        }

        conversations.add(0, newConv)
        currentConversation.value = newConv
        return newConv.id
    }

    private fun addLocalUserMessage(text: String) {
        val conv = currentConversation.value ?: return

        val localMessage = MessageDto(
            _id = null,
            role = "user",
            content = text,
            isUser = true,
            uploaded_at = Instant.now().toString(),
            files = emptyList()
        )

        currentConversation.value =
            conv.copy(messages = conv.messages + localMessage)
    }

    fun reloadOnlyMessages(convId: String) {
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

    private fun reloadMessages(convId: String) {
        viewModelScope.launch {
            try {
                val msgs = withContext(Dispatchers.IO) {
                    repo.getMessages(convId)
                }

                currentConversation.value =
                    currentConversation.value?.copy(messages = msgs)

            } catch (e: Exception) {
                error.value = e.message
            }
        }
    }
}
