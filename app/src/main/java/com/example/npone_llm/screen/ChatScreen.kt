package com.example.npone_llm.screen

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.Environment
import android.webkit.MimeTypeMap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.npone_llm.data.remote.dto.MessageDto
import com.example.npone_llm.viewModel.ChatViewModel
import kotlinx.coroutines.launch
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.material3.pulltorefresh.rememberPullToRefreshState
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

// === MIME utilitaire ===
fun File.getMimeType(): String {
    val ext = extension.lowercase()
    return MimeTypeMap.getSingleton().getMimeTypeFromExtension(ext)
        ?: "application/octet-stream"
}

// === Écran principal ===
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatApp(vm: ChatViewModel) {
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()

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
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                contentAlignment = Alignment.Center
            ) {
                when {
                    vm.isLoading.value -> CircularProgressIndicator()
                    vm.error.value != null -> Text("Erreur: ${vm.error.value}", color = MaterialTheme.colorScheme.error)
                    else -> ChatScreen(vm = vm, modifier = Modifier.fillMaxSize())
                }
            }
        }
    }
}

@Composable
fun ChatBubble(msg: MessageDto, onDownload: (String) -> Unit) {
    val isBot = msg.role == "bot"
    val alignment = if (isBot) Alignment.CenterStart else Alignment.CenterEnd
    val bubbleColor = if (isBot) Color(0xFFE8C602) else Color(0xFF47473A)
    val textColor = if (isBot) Color.White else Color.White

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(4.dp),
        contentAlignment = alignment
    ) {
        Column(
            modifier = Modifier
                .background(bubbleColor, RoundedCornerShape(16.dp))
                .padding(12.dp)
                .widthIn(max = 280.dp)
        ) {
            // 🗣️ Texte principal
            if (!msg.content.isNullOrBlank()) {
                Text(
                    text = msg.content ?: "",
                    color = textColor,
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            // 📎 Fichiers joints
            msg.files?.forEach { file ->
                Spacer(Modifier.height(8.dp))
                Surface(
                    color = if (isBot) Color.Black.copy(alpha = 0.2f) else Color(0xFF757553),
                    shape = RoundedCornerShape(8.dp),
                    tonalElevation = 2.dp
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onDownload(file.file_url) }
                            .padding(horizontal = 8.dp, vertical = 6.dp)
                    ) {
                        Icon(Icons.Default.AttachFile, contentDescription = null, tint = textColor)
                        Spacer(Modifier.width(6.dp))
                        Text(
                            text = file.file_name ?: "Fichier",
                            color = textColor,
                            style = MaterialTheme.typography.bodySmall
                        )
                        Spacer(Modifier.weight(1f))
                        Button(
                            onClick = { onDownload(file.file_url) },
                            colors = ButtonDefaults.buttonColors(
                                containerColor = if (isBot) Color.White.copy(alpha = 0.25f) else Color(0xFFE8C602),
                                contentColor = if (isBot) Color.White else Color.White
                            ),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Text("⬇", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }
        }
    }
}


// === Chat principal ===
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: ChatViewModel, modifier: Modifier = Modifier) {
    val conv = vm.currentConversation.value
    val isLoading = vm.isLoading.value
    val context = LocalContext.current
    var input by remember { mutableStateOf("") }
    val selectedFiles = remember { mutableStateListOf<File>() }

    val refreshState = rememberPullToRefreshState()
    val scope = rememberCoroutineScope()
    var isRefreshing by remember { mutableStateOf(false) }

    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenMultipleDocuments()
    ) { uris: List<Uri> ->
        uris.forEach { uri ->
            val file = copyUriToCache(context, uri)
            if (file != null) selectedFiles.add(file)
        }
    }

    LaunchedEffect(isLoading) {
        if (!isLoading) isRefreshing = false
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
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            ) {
                conv?.messages?.let { msgs ->
                    items(msgs) { msg ->
                        ChatBubble(msg) { url ->
                            val fileName = msg.files?.firstOrNull()?.file_name ?: "fichier.bin"
                            downloadFile(context, url, fileName)
                        }
                    }
                }
            }


            if (selectedFiles.isNotEmpty()) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                ) {
                    selectedFiles.forEach { f ->
                        Text("📎 ${f.name}", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

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
                    label = { Text("Écris ton message...") }
                )
                Spacer(Modifier.width(8.dp))
                IconButton(onClick = { filePicker.launch(arrayOf("*/*")) }) {
                    Icon(Icons.Default.AttachFile, contentDescription = "Ajouter un fichier")
                }
                Spacer(Modifier.width(8.dp))
                Button(
                    onClick = {
                        if (selectedFiles.isNotEmpty()) {
                            vm.sendFileMessage(input, selectedFiles)
                            selectedFiles.clear()
                        } else {
                            vm.sendTextMessage(input)
                        }
                        input = ""
                    },
                    enabled = !isLoading && (input.isNotBlank() || selectedFiles.isNotEmpty())
                ) {
                    Text(if (isLoading) "..." else "Envoyer")
                }
            }
        }
    }
}

// === Bulle de message ===
@Composable
fun MessageItem(message: MessageDto, onDownload: (String) -> Unit) {
    Column(modifier = Modifier.padding(8.dp)) {
        if (!message.content.isNullOrBlank()) {
            Text(
                text = message.content ?: "",
                color = if (message.isUser) Color.Blue else Color.DarkGray
            )
        }

        message.files?.forEach { file ->
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .padding(top = 4.dp)
                    .clickable { onDownload(file.file_url) }
            ) {
                Icon(Icons.Default.AttachFile, contentDescription = null)
                Text(
                    text = file.file_name ?: "Fichier",
                    modifier = Modifier.padding(start = 4.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Button(onClick = { onDownload(file.file_url) }) {
                    Text("Télécharger")
                }
            }
        }
    }
}

// === Menu latéral ===
@Composable
fun DrawerContent(
    vm: ChatViewModel,
    onSelect: (String) -> Unit,
    onAdd: (String) -> Unit,
    onDelete: (String) -> Unit,
    onRename: (String, String) -> Unit,
    onCloseDrawer: () -> Unit
) {
    val conversations = vm.conversations
    var newTitle by remember { mutableStateOf("") }
    var showDeleteDialog by remember { mutableStateOf<String?>(null) }
    var renamingConvId by remember { mutableStateOf<String?>(null) }
    var renameText by remember { mutableStateOf("") }

    Box(
        Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.4f))
    ) {
        Surface(
            modifier = Modifier
                .fillMaxHeight()
                .fillMaxWidth(0.85f)
                .shadow(8.dp),
            color = Color.White
        ) {
            Column(
                Modifier
                    .fillMaxSize()
                    .padding(16.dp)
            ) {
                Text("Conversations", style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.height(12.dp))

                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                ) {
                    items(conversations) { conv ->
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .padding(vertical = 6.dp)
                                .background(
                                    MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.4f),
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
                                            onCloseDrawer()
                                        },
                                    color = MaterialTheme.colorScheme.onSecondaryContainer
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

                Spacer(Modifier.height(12.dp))
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

// === Fichiers utilitaires ===
fun copyUriToCache(context: Context, uri: Uri): File? {
    return try {
        val inputStream: InputStream? = context.contentResolver.openInputStream(uri)
        val ext = getFileExtension(context, uri)
        val fileName = "upload-${System.currentTimeMillis()}${if (ext != null) ".$ext" else ""}"
        val file = File(context.cacheDir, fileName)
        val outputStream = FileOutputStream(file)
        inputStream?.copyTo(outputStream)
        inputStream?.close()
        outputStream.close()
        file
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

fun getFileExtension(context: Context, uri: Uri): String? {
    return try {
        val type = context.contentResolver.getType(uri)
        MimeTypeMap.getSingleton().getExtensionFromMimeType(type)
    } catch (e: Exception) {
        null
    }
}

fun downloadFile(context: Context, url: String, fileName: String) {
    val request = DownloadManager.Request(Uri.parse(url))
        .setTitle(fileName)
        .setDescription("Téléchargement en cours…")
        .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
        .setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, fileName)
        .setAllowedOverMetered(true)

    val manager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
    manager.enqueue(request)
}
