import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
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
  _welcome?: boolean;
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

  @ViewChild('messagesContainer') messagesContainer!: ElementRef<HTMLDivElement>;

  constructor(
    private history: HistoryService,
    private chat: ChatService
  ) {}

  ngOnInit() {
    this.history.activeConversation$.subscribe(conv => {
      this.activeConversation = conv;

      if (conv?.id || conv?._id) {
        const convId = conv.id || conv._id;

        this.history.getMessages(convId).subscribe(msgs => {
          this.messages = msgs;

          if (msgs.length === 0) {
            this.messages = [
              {
                role: 'bot',
                content: 'Par quoi commençons-nous ?',
                isUser: false,
                _welcome: true
              }
            ];
          }

          this.scrollToBottom();
        });
      }
    });
  }

  renderMarkdown(md: string): string {
    const html = marked.parse(md || '') as string;
    return DOMPurify.sanitize(html);
  }

  onSendMessage(event: { text: string; files: File[] }) {
    if (!this.activeConversation?.id && !this.activeConversation?._id) {
      this.chat.pushBotMessage('Aucune conversation active');
      return;
    }

    const convId = this.activeConversation.id || this.activeConversation._id;
    const { text, files } = event;

    const formData = new FormData();
    formData.append('text', text || '');
    files.forEach(file => formData.append('files', file));

    this.chat.sendMessage(convId, formData).subscribe({
      next: () => {
        this.history.getMessages(convId).subscribe(msgs => {
          this.messages = msgs;
          this.scrollToBottom();
        });

        if (text.trim()) {
          this.messages = this.messages.filter(m => !m._welcome);

          this.chat.askLLM(text, convId).subscribe({
            next: (resp) => {
              const markdownContent = (() => {
                let md = '';

                if (resp.steps?.length) {
                  md += resp.steps.join('\n\n');
                }

                if (resp.citations?.length) {
                  md += `\n\n### 📚 Sources\n`;
                  md += resp.citations.map(c => `- ${c.doc}`).join('\n');
                }

                return md.trim();
              })();

              this.messages.push({
                role: 'bot',
                content: markdownContent,
                isUser: false
              });

              this.scrollToBottom();
            },
            error: (err) => {
              this.chat.pushBotMessage('Erreur RAG : ' + err.message);
            }
          });
        }
      },
      error: (err) => {
        this.chat.pushBotMessage('Erreur lors de l’envoi : ' + err.message);
      }
    });
  }

  scrollToBottom() {
    setTimeout(() => {
      const el = this.messagesContainer?.nativeElement;
      if (!el) return;
      el.scrollTop = el.scrollHeight;
    }, 0);
  }

  getFileIcon(nameOrType?: string): string {
    const name = nameOrType?.toLowerCase() || '';
    if (name.endsWith('.pdf')) return 'icon/pdf.png';
    if (name.endsWith('.png') || name.endsWith('.jpg') || name.endsWith('.jpeg')) return 'icon/img.png';
    if (name.endsWith('.doc') || name.endsWith('.docx')) return 'icon/docx.png';
    return 'icon/dossier.png';
  }
}
