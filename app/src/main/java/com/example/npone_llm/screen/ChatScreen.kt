package com.example.npone_llm.screen

import android.app.DownloadManager
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.webkit.MimeTypeMap
import android.widget.Toast
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
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.material3.pulltorefresh.rememberPullToRefreshState
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import com.example.npone_llm.data.remote.dto.MessageDto
import com.example.npone_llm.viewModel.ChatViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

// =====================================================
// ==================== CHAT APP =======================
// =====================================================
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
                    placeholder = { Text("Entrer un titre") },
                    textStyle = TextStyle(color = Color.Black),
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        unfocusedBorderColor = Color.Black,
                        focusedBorderColor = Color.Black,
                        cursorColor = Color.Black
                    )
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
                onSelect = {
                    vm.selectConversation(it)
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
                            Icon(Icons.Default.Menu, contentDescription = null)
                        }
                    }
                )
            }
        ) { padding ->
            Box(
                Modifier
                    .fillMaxSize()
                    .padding(padding)
            ) {

                ChatScreen(vm = vm)

                if (vm.isLoading.value) {
                    Box(
                        Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(alpha = 0.35f)),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator(color = Color.White)
                    }
                }
            }
        }
    }
}

// =====================================================
// ==================== CHAT UI ========================
// =====================================================

@Composable
fun ChatBubble(msg: MessageDto, onDownload: (String, String) -> Unit) {

    val isBot = msg.role == "bot"
    val alignment = if (isBot) Alignment.CenterStart else Alignment.CenterEnd
    val bubbleColor = if (isBot) Color(0xFFE8C602) else Color(0xFF47473A)

    Box(
        Modifier.fillMaxWidth().padding(4.dp),
        contentAlignment = alignment
    ) {
        Column(
            Modifier
                .background(bubbleColor, RoundedCornerShape(16.dp))
                .padding(12.dp)
                .widthIn(max = 280.dp)
        ) {

            msg.content?.takeIf { it.isNotBlank() }?.let {
                Text(it, color = Color.White)
            }

            msg.files.forEach { file ->
                Spacer(Modifier.height(8.dp))
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = Color.Black.copy(alpha = 0.2f)
                ) {
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .clickable {
                                onDownload(file.file_url, file.file_name)
                            }
                            .padding(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.AttachFile, null, tint = Color.White)
                        Spacer(Modifier.width(6.dp))
                        Text(file.file_name, color = Color.White)
                        Spacer(Modifier.weight(1f))
                        Button(
                            onClick = {
                                onDownload(file.file_url, file.file_name)
                            }
                        ) {
                            Text("⬇")
                        }
                    }
                }
            }
        }
    }
}

// =====================================================
// ==================== CHAT SCREEN ====================
// =====================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: ChatViewModel) {

    val conv = vm.currentConversation.value
    val context = LocalContext.current
    var input by remember { mutableStateOf("") }
    val selectedFiles = remember { mutableStateListOf<File>() }
    val refreshState = rememberPullToRefreshState()
    val scope = rememberCoroutineScope()

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        uris.forEach {
            copyUriToCache(context, it)?.let(selectedFiles::add)
        }
    }

    LaunchedEffect(conv?.id) {
        conv?.id?.let { vm.reloadOnlyMessages(it) }
    }

    LaunchedEffect(vm.lastResponse.value) {
        conv?.id?.let { vm.reloadOnlyMessages(it) }
    }

    PullToRefreshBox(
        state = refreshState,
        isRefreshing = false,
        onRefresh = { scope.launch { vm.loadConversations() } }
    ) {
        Column(Modifier.fillMaxSize()) {

            LazyColumn(Modifier.weight(1f)) {
                conv?.messages?.let {
                    items(it) { msg ->
                        ChatBubble(msg) { url, name ->
                            downloadFile(context, url, name)
                        }
                    }
                }
            }

            Row(Modifier.padding(8.dp)) {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it },
                    modifier = Modifier.weight(1f),
                    label = { Text("Écris ton message...") }
                )
                IconButton(onClick = { picker.launch(arrayOf("*/*")) }) {
                    Icon(Icons.Default.AttachFile, null)
                }
                Button(
                    enabled = input.isNotBlank() || selectedFiles.isNotEmpty(),
                    onClick = {
                        if (selectedFiles.isNotEmpty()) {
                            vm.sendFileMessage(input, selectedFiles)
                            selectedFiles.clear()
                        } else {
                            vm.sendTextMessage(input)
                        }
                        input = ""
                    }
                ) {
                    Text("Envoyer")
                }
            }
        }
    }
}

// =====================================================
// ==================== UTILS ===========================
// =====================================================

fun copyUriToCache(context: Context, uri: Uri): File? =
    try {
        val input = context.contentResolver.openInputStream(uri) ?: return null
        val file = File(context.cacheDir, "upload-${System.currentTimeMillis()}")
        FileOutputStream(file).use { input.copyTo(it) }
        file
    } catch (e: Exception) {
        null
    }

fun downloadFile(context: Context, url: String, fileName: String) {
    CoroutineScope(Dispatchers.IO).launch {
        try {
            val mime = MimeTypeMap.getSingleton()
                .getMimeTypeFromExtension(MimeTypeMap.getFileExtensionFromUrl(url))
                ?: "application/octet-stream"

            if (Build.VERSION.SDK_INT >= 29) {
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, fileName)
                    put(MediaStore.Downloads.MIME_TYPE, mime)
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }
                val uri = context.contentResolver.insert(
                    MediaStore.Downloads.EXTERNAL_CONTENT_URI, values
                ) ?: return@launch

                val conn = URL(url).openConnection() as HttpURLConnection
                conn.inputStream.use { input ->
                    context.contentResolver.openOutputStream(uri)?.use {
                        input.copyTo(it)
                    }
                }
            }
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Erreur téléchargement", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
