package com.example.npone_llm.screen

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.npone_llm.viewModel.ChatViewModel

@Composable
fun RegisterScreen(
    vm: ChatViewModel,
    onBackToLogin: () -> Unit
) {
    var username by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF1E1C14)),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(0.9f),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color(0xFF2A281E)
            ),
            elevation = CardDefaults.cardElevation(8.dp)
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {

                // 🔶 Titre
                Text(
                    text = "Créer un compte",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFFE8C602)
                )

                Spacer(Modifier.height(24.dp))

                // 👤 Username
                OutlinedTextField(
                    value = username,
                    onValueChange = { username = it },
                    label = { Text("Nom d'utilisateur") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Color(0xFFE8C602),
                        unfocusedBorderColor = Color(0xFF757553),
                        focusedLabelColor = Color(0xFFE8C602),
                        unfocusedLabelColor = Color(0xFFBDBB9A),
                        cursorColor = Color(0xFFE8C602)
                    )
                )

                Spacer(Modifier.height(16.dp))

                // 📧 Email
                OutlinedTextField(
                    value = email,
                    onValueChange = { email = it },
                    label = { Text("Email") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Color(0xFFE8C602),
                        unfocusedBorderColor = Color(0xFF757553),
                        focusedLabelColor = Color(0xFFE8C602),
                        unfocusedLabelColor = Color(0xFFBDBB9A),
                        cursorColor = Color(0xFFE8C602)
                    )
                )

                Spacer(Modifier.height(16.dp))

                // 🔑 Mot de passe
                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text("Mot de passe") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Color(0xFFE8C602),
                        unfocusedBorderColor = Color(0xFF757553),
                        focusedLabelColor = Color(0xFFE8C602),
                        unfocusedLabelColor = Color(0xFFBDBB9A),
                        cursorColor = Color(0xFFE8C602)
                    )
                )

                Spacer(Modifier.height(28.dp))

                // ▶️ Bouton inscription
                Button(
                    onClick = {
                        vm.register(
                            username = username.trim(),
                            email = email.trim(),
                            password = password
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp),
                    enabled = !vm.authLoading.value,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFE8C602),
                        contentColor = Color.Black
                    ),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    Text(
                        if (vm.authLoading.value) "Création..." else "Créer le compte",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                }

                Spacer(Modifier.height(12.dp))

                // ❌ Erreur
                vm.authError.value?.let {
                    Text(
                        text = it,
                        color = Color.Red,
                        style = MaterialTheme.typography.bodySmall
                    )
                }

                Spacer(Modifier.height(16.dp))

                // 🔙 Retour login
                TextButton(onClick = onBackToLogin) {
                    Text(
                        "Déjà un compte ? Se connecter",
                        color = Color(0xFFE8C602)
                    )
                }
            }
        }
    }
}
