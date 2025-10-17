package com.example.npone_llm

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.npone_llm.screen.ChatApp
import com.example.npone_llm.screen.ChatScreen
import com.example.npone_llm.ui.theme.Npone_llmTheme
import com.example.npone_llm.viewModel.ChatViewModel


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val vm = ChatViewModel()

        setContent {
            Npone_llmTheme(darkTheme = true) {  // tu peux mettre false pour tester le clair
                ChatApp(vm)
            }
        }
    }
}