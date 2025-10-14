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

interface Message {
  _id?: string;
  role: string;
  content: string;
  file_id?: string;
  file_url?: string;
  isUser?: boolean;
  rag_context?: string;
  uploaded_at?: string;
  files?: UploadedFile[];
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
  messages: Message[] = [];

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

    this.messages.push({ role: 'user', content: text, files });

    if (this.activeConversation?._id) {
      this.history.addMessage(this.activeConversation._id, 'user', text).subscribe();
    }

    if (text.trim()) {
      this.history.sendToLLM(text, this.activeConversation._id).subscribe({
        next: (res: any) => {
          const botResponse =
            (res.steps?.length ? res.steps.map((s: string) => s).join('\n') : '') +
            (res.citations?.length
              ? `\n📚 Sources: ${res.citations.map((c: any) => c.doc).join(', ')}`
              : '');
          this.messages.push({ role: 'bot', content: botResponse.trim() });
          this.history.addMessage(this.activeConversation._id, 'bot', botResponse.trim()).subscribe();
        },
        error: (err) => console.error('❌ Erreur API:', err)
      });
    }
  }

  getFileIcon(nameOrType: string): string {
    const name = nameOrType.toLowerCase();
    if (name.endsWith('.pdf')) return 'assets/icons/pdf.png';
    if (name.endsWith('.png') || name.endsWith('.jpg') || name.endsWith('.jpeg')) return 'assets/icons/image.png';
    if (name.endsWith('.doc') || name.endsWith('.docx')) return 'assets/icons/doc.png';
    return 'assets/icons/file.png';
  }

  cleanFilename(content: string): string {
    return content.replace('📎', '').trim();
  }
}
