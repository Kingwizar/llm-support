package com.example.npone_llm.screen

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.npone_llm.data.remote.dto.MessageDto
import com.example.npone_llm.viewModel.ChatViewModel
import kotlinx.coroutines.launch
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.material3.pulltorefresh.rememberPullToRefreshState
import com.example.npone_llm.data.remote.dto.ChatResponseDto
import androidx.compose.material3.*
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.material3.pulltorefresh.rememberPullToRefreshState
import androidx.compose.runtime.*
import kotlinx.coroutines.launch
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.material3.pulltorefresh.rememberPullToRefreshState
import androidx.compose.runtime.*

import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatApp(vm: ChatViewModel) {
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    // Charger les conversations au démarrage
    LaunchedEffect(Unit) {
        vm.loadConversations()
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            DrawerContent(
                vm = vm,
                onSelect = { id ->
                    vm.selectConversation(id)
                    scope.launch { drawerState.close() }
                },
                onAdd = { vm.createConversation(it) },
                onDelete = { vm.deleteConversation(it) },
                onRename = { id, title -> vm.renameConversation(id, title) },
                onCloseDrawer = { scope.launch { drawerState.close() } }
            )

        }
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("LLM Chat") },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(Icons.Default.Menu, contentDescription = "Menu")
                        }
                    }
                )
            }
        ) { padding ->
            // Affichage conditionnel
            val isLoading = vm.isLoading.value
            val error = vm.error.value

            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                contentAlignment = Alignment.Center
            ) {
                when {
                    isLoading -> {
                        CircularProgressIndicator()
                    }
                    error != null -> {
                        Text("Erreur: $error", color = MaterialTheme.colorScheme.error)
                    }
                    else -> {
                        ChatScreen(vm = vm, modifier = Modifier.fillMaxSize())
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: ChatViewModel, modifier: Modifier = Modifier) {
    val conv = vm.currentConversation.value
    val isLoading = vm.isLoading.value
    var input by remember { mutableStateOf("") }

    val refreshState = rememberPullToRefreshState()
    val scope = rememberCoroutineScope()
    var isRefreshing by remember { mutableStateOf(false) }

    // Quand le rafraîchissement est terminé, on arrête l'animation
    LaunchedEffect(isLoading) {
        if (!isLoading) {
            isRefreshing = false
        }
    }

    PullToRefreshBox(
        state = refreshState,
        isRefreshing = isRefreshing,
        onRefresh = {
            scope.launch {
                isRefreshing = true
                vm.loadConversations()
            }
        },
        modifier = Modifier.fillMaxSize()
    ) {
        Column(
            modifier
                .fillMaxSize()
                .padding(8.dp)
        ) {
            // --- Liste des messages ---
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                conv?.messages?.let { msgs ->
                    items(msgs) { msg ->
                        ChatBubble(msg)
                    }
                }
            }

            // --- Champ et bouton d'envoi ---
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it },
                    modifier = Modifier.weight(1f),
                    label = { Text("Écris ta question...") }
                )
                Spacer(Modifier.width(8.dp))
                Button(
                    onClick = {
                        vm.sendQuestion(input)
                        input = ""
                    },
                    enabled = !isLoading && input.isNotBlank()
                ) {
                    Text(if (isLoading) "..." else "Envoyer")
                }
            }
        }
    }
}
@Composable
fun ChatBubble(msg: MessageDto) {
    val isUser = msg.isUser
    Box(
        modifier = Modifier.fillMaxWidth().padding(4.dp),
        contentAlignment = if (isUser) Alignment.CenterEnd else Alignment.CenterStart
    ) {
        Surface(
            color = if (isUser) MaterialTheme.colorScheme.primary else Color(0xFFEFEFEF),
            shape = RoundedCornerShape(16.dp)
        ) {
            Text(
                msg.text,
                color = if (isUser) Color.White else Color.Black,
                modifier = Modifier.padding(12.dp)
            )
        }
    }
}

@Composable
fun DrawerContent(
    vm: ChatViewModel,
    onSelect: (String) -> Unit,
    onAdd: (String) -> Unit,
    onDelete: (String) -> Unit,
    onRename: (String, String) -> Unit,
    onCloseDrawer: () -> Unit // 👈 ajout pour revenir à la conversation
) {
    val conversations = vm.conversations
    var newTitle by remember { mutableStateOf("") }

    var showDeleteDialog by remember { mutableStateOf<String?>(null) }
    var renamingConvId by remember { mutableStateOf<String?>(null) }
    var renameText by remember { mutableStateOf("") }

    // 🌫️ Fond sombre semi-transparent (lisibilité améliorée)
    Box(
        Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.4f))
    ) {
        Surface(
            modifier = Modifier
                .fillMaxHeight()
                .fillMaxWidth(0.85f) // largeur du menu
                .background(Color.White)
                .shadow(8.dp),
            color = Color.White.copy(alpha = 0.95f)
        ) {
            Column(
                Modifier
                    .fillMaxSize()
                    .padding(16.dp)
            ) {
                // 🔹 En-tête fixe
                Text(
                    "Conversations",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.primary
                )

                Spacer(Modifier.height(12.dp))

                // 🔹 Liste déroulante indépendante
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                ) {
                    LazyColumn {
                        items(conversations) { conv ->
                            Row(
                                Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 6.dp)
                                    .background(
                                        MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f),
                                        shape = MaterialTheme.shapes.small
                                    )
                                    .padding(horizontal = 8.dp, vertical = 4.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                if (renamingConvId == conv.id) {
                                    OutlinedTextField(
                                        value = renameText,
                                        onValueChange = { renameText = it },
                                        singleLine = true,
                                        modifier = Modifier.weight(1f),
                                        placeholder = { Text("Nouveau titre") }
                                    )
                                    IconButton(onClick = {
                                        if (renameText.isNotBlank()) {
                                            onRename(conv.id, renameText)
                                            renamingConvId = null
                                            renameText = ""
                                        }
                                    }) {
                                        Icon(Icons.Default.Check, contentDescription = "Valider")
                                    }
                                } else {
                                    Text(
                                        text = conv.title ?: "(Sans titre)",
                                        modifier = Modifier
                                            .weight(1f)
                                            .clickable {
                                                onSelect(conv.id)
                                                onCloseDrawer() // 👈 ferme le drawer
                                            },
                                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                                        style = MaterialTheme.typography.bodyLarge
                                    )
                                    Row {
                                        IconButton(onClick = {
                                            renamingConvId = conv.id
                                            renameText = conv.title ?: ""
                                        }) {
                                            Icon(Icons.Default.Edit, contentDescription = "Renommer")
                                        }
                                        IconButton(onClick = { showDeleteDialog = conv.id }) {
                                            Icon(Icons.Default.Delete, contentDescription = "Supprimer")
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Spacer(Modifier.height(12.dp))

                // 🔹 Zone d’ajout (fixe en bas)
                OutlinedTextField(
                    value = newTitle,
                    onValueChange = { newTitle = it },
                    label = { Text("Nouvelle conversation") },
                    modifier = Modifier.fillMaxWidth()
                )

                Button(
                    onClick = {
                        if (newTitle.isNotBlank()) {
                            onAdd(newTitle)
                            newTitle = ""
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp)
                ) {
                    Text("Ajouter")
                }
            }
        }
    }

    // 🔹 Boîte de dialogue pour suppression
    if (showDeleteDialog != null) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = null },
            title = { Text("Supprimer la conversation ?") },
            text = { Text("Cette action est irréversible.") },
            confirmButton = {
                Button(onClick = {
                    onDelete(showDeleteDialog!!)
                    showDeleteDialog = null
                }) {
                    Text("Supprimer")
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = null }) {
                    Text("Annuler")
                }
            }
        )
    }
}
