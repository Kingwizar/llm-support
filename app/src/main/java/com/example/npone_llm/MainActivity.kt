package com.example.npone_llm

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.*
import com.example.npone_llm.screen.ChatApp
import com.example.npone_llm.screen.LoginScreen
import com.example.npone_llm.screen.RegisterScreen
import com.example.npone_llm.ui.theme.Npone_llmTheme
import com.example.npone_llm.viewModel.ChatViewModel

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val vm = ChatViewModel()

        setContent {
            Npone_llmTheme(darkTheme = true) {

                var currentScreen by remember { mutableStateOf("login") }

                // 🔐 Si déjà authentifié → chat direct
                LaunchedEffect(vm.isAuthenticated.value) {
                    if (vm.isAuthenticated.value) {
                        currentScreen = "chat"
                    }
                }

                when (currentScreen) {

                    "login" -> LoginScreen(
                        vm = vm,
                        onGoToRegister = { currentScreen = "register" }
                    )

                    "register" -> RegisterScreen(
                        vm = vm,
                        onBackToLogin = { currentScreen = "login" }
                    )

                    "chat" -> ChatApp(vm)
                }
            }
        }
    }
}
