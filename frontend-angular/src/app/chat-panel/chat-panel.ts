import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HistoryService } from '../services/history/history';
import { ChatService } from '../services/chat/chat';
import { InputBar } from '../input-bar/input-bar';

interface UploadedFile {
  id: string;
  filename: string;
  content_type: string;
  url: string;
}

@Component({
  selector: 'app-chat-panel',
  standalone: true,
  imports: [CommonModule, InputBar],
  templateUrl: './chat-panel.html',
  styleUrls: ['./chat-panel.css']
})
export class ChatPanelComponent implements OnInit {
  activeConversation: any = null;
  messages: { role: string; content: string; files?: UploadedFile[] }[] = [];

  constructor(private history: HistoryService, private chat: ChatService) {}

  ngOnInit() {
    this.history.activeConversation$.subscribe(c => {
      this.activeConversation = c;
      if (c?._id)
        this.history.getMessages(c._id).subscribe(m => (this.messages = m));
      else this.messages = [];
    });
  }

  onSendMessage(event: { text: string; files: UploadedFile[] }) {
    const { text, files } = event;

    // ➕ Ajout immédiat du message utilisateur
    this.messages.push({ role: 'user', content: text, files });

    // Sauvegarde en BDD
    if (this.activeConversation?._id) {
      this.history.addMessage(this.activeConversation._id, 'user', text).subscribe();
    }

    // Si texte → envoie au LLM
    if (text.trim()) {
  this.history.sendToLLM(text, this.activeConversation._id).subscribe({
    next: (res: any) => {
      const botResponse =
        (res.steps?.length ? res.steps.map((s: string) => s).join('\n') : '') +
        (res.citations?.length
          ? `\n📚 Sources: ${res.citations.map((c: any) => c.doc).join(', ')}`
          : '');

      // Ajoute le message dans la liste locale
      this.messages.push({ role: 'bot', content: botResponse.trim() });

      // Enregistre aussi le message dans la BDD
      this.history.addMessage(this.activeConversation._id, 'bot', botResponse.trim()).subscribe();
    },
    error: (err) => console.error('❌ Erreur API:', err)
  });
    }

  }

  getFileIcon(type?: string) {
    if (!type) return 'assets/icons/file.png';
    if (type.startsWith('image/')) return 'assets/icons/image.png';
    if (type === 'application/pdf') return 'assets/icons/pdf.png';
    return 'assets/icons/file.png';
  }
}
