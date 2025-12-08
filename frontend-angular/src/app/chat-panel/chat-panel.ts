import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HistoryService } from '../services/history/history';
import { ChatService } from '../services/chat/chat';
import { InputBar } from '../input-bar/input-bar';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

interface UploadedFile {
  file_id: string;
  file_name: string;
  file_url: string;
  content_type?: string;
  uploaded_at?: string;
}

interface Message {
  role: string;
  content?: string;
  files?: UploadedFile[];
  isUser?: boolean;
  uploaded_at?: string;
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
    // Quand une conversation devient active
    this.history.activeConversation$.subscribe(conv => {
      this.activeConversation = conv;
      if (conv?.id || conv?._id) {
        this.history.getMessages(conv.id || conv._id).subscribe(msgs => {
          this.messages = msgs;
          console.log("💬 Messages chargés:", msgs.length);
        });
      }
    });
  }
  renderMarkdown(md: string): string {
  const html = marked.parse(md || '') as string;
  return DOMPurify.sanitize(html);
}


  /** Envoi du message + fichiers */
  onSendMessage(event: { text: string; files: File[] }) {
  if (!this.activeConversation?.id && !this.activeConversation?._id) {
    this.chat.pushBotMessage("⚠️ Aucune conversation active");
    return;
  }

  const convId = this.activeConversation.id || this.activeConversation._id;
  const { text, files } = event;

  const formData = new FormData();
  formData.append('text', text || '');
  files.forEach(file => formData.append('files', file));

  console.log("📤 Envoi message:", text, "fichiers:", files.length);

  this.chat.sendMessage(convId, formData).subscribe({
    next: (res: any) => {
      console.log("✅ Message envoyé:", res);

      // Recharge les messages de la base
      this.history.getMessages(convId).subscribe(msgs => {
        this.messages = msgs;
        console.log("🔄 Rechargement complet depuis la base:", msgs.length, "messages");
      });

      // Lance le RAG uniquement si un texte a été envoyé
      if (text.trim()) {
        this.chat.askLLM(text, convId).subscribe({
          next: (resp) => {

            // 🔥 Construction d’un Markdown propre
            const markdownContent = (() => {
              let md = '';

              // 🟦 1. Contenu principal (steps)
              if (resp.steps?.length) {
                md += resp.steps.join('\n\n');   // ✔ double saut de ligne Markdown
              }

              // 🟩 2. Sources au format Markdown propre
              if (resp.citations?.length) {
                md += `\n\n### 📚 Sources\n`;       // ✔ titre Markdown
                md += resp.citations
                  .map(c => `- ${c.doc}`)         // ✔ liste Markdown
                  .join('\n');
              }

              return md.trim();
            })();

            const botMessage = {
              role: 'bot',
              content: markdownContent,
              isUser: false
            };

            this.messages.push(botMessage);
          },
          error: (err) => {
            console.error('❌ Erreur RAG:', err);
            this.chat.pushBotMessage('⚠️ Erreur RAG : ' + err.message);
          }
        });
      }
    },
    error: (err) => {
      console.error("❌ Erreur sendMessage:", err);
      this.chat.pushBotMessage('⚠️ Erreur lors de l’envoi : ' + err.message);
    }
  });
}

  /** Choix de l’icône pour un fichier */
  getFileIcon(nameOrType?: string): string {
    const name = nameOrType?.toLowerCase() || '';
    if (name.endsWith('.pdf')) return 'icon/pdf.png';
    if (name.endsWith('.png') || name.endsWith('.jpg') || name.endsWith('.jpeg')) return 'icon/img.png';
    if (name.endsWith('.doc') || name.endsWith('.docx')) return 'icon/docx.png';
    return 'icon/dossier.png';
  }
}
