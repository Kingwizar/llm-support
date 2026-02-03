import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HistoryService } from '../services/history/history';
import { ChatService } from '../services/chat/chat';
import { InputBar } from '../input-bar/input-bar';
import { marked } from 'marked';
import DOMPurify from 'dompurify';


// ======================================================
// DATA INTERFACES
// ======================================================

// Represents a file uploaded by the user and linked to a message
interface UploadedFile {
  file_id: string;
  file_name: string;
  file_url: string;
  content_type?: string;
  uploaded_at?: string;
}

// Represents a chat message (user or assistant)
interface Message {
  role: string;                 // "user" | "bot"
  content?: string;             // Text or markdown content
  files?: UploadedFile[];       // Optional attached files
  isUser?: boolean;             // UI helper flag
  uploaded_at?: string;
  _welcome?: boolean;           // Internal flag for welcome message
}


// ======================================================
// CHAT PANEL COMPONENT
// ======================================================
@Component({
  selector: 'app-chat-panel',
  standalone: true,
  imports: [CommonModule, InputBar],
  templateUrl: './chat-panel.html',
  styleUrls: ['./chat-panel.css']
})
export class ChatPanelComponent implements OnInit {

  // Currently active conversation (provided by HistoryService)
  activeConversation: any = null;

  // List of messages displayed in the chat panel
  messages: Message[] = [];

  // Reference to the DOM container holding chat messages
  // Used for scrolling behavior
  @ViewChild('messagesContainer')
  messagesContainer!: ElementRef<HTMLDivElement>;

  // Flag controlling visibility of the "scroll to bottom" button
  showScrollButton = false;

  constructor(
    private history: HistoryService,
    private chat: ChatService
  ) {}

  // ======================================================
  // COMPONENT INITIALIZATION
  // ======================================================
  ngOnInit() {

    // Subscribe to active conversation changes
    // This observable is updated whenever the user selects
    // or creates a new conversation in the history panel
    this.history.activeConversation$.subscribe(conv => {
      this.activeConversation = conv;

      // Ensure a valid conversation identifier exists
      if (conv?.id || conv?._id) {
        const convId = conv.id || conv._id;

        // Fetch messages associated with the selected conversation
        this.history.getMessages(convId).subscribe(msgs => {
          this.messages = msgs;

          // If the conversation is empty, inject a default welcome message
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

          // Scroll to bottom after loading messages
          this.scrollToBottom();

          // Attach scroll listener for UI feedback
          this.attachScrollListener();
        });
      }
    });
  }

  // ======================================================
  // MARKDOWN RENDERING
  // ======================================================
  // Converts Markdown content into sanitized HTML
  // This is used to safely render LLM responses
  renderMarkdown(md: string): string {
    const html = marked.parse(md || '') as string;
    return DOMPurify.sanitize(html);
  }

  // ======================================================
  // SEND MESSAGE HANDLER
  // ======================================================
  // Triggered when the user sends a message from InputBar
  onSendMessage(event: {
    text: string;
    files: File[];
    useInternet: boolean;
  }) {

    // Guard: no active conversation
    if (!this.activeConversation?.id && !this.activeConversation?._id) {
      this.chat.pushBotMessage('Aucune conversation active');
      return;
    }

    const convId = this.activeConversation.id || this.activeConversation._id;
    const { text, files } = event;

    // Build multipart form data for text + file upload
    const formData = new FormData();
    formData.append('text', text || '');
    files.forEach(file => formData.append('files', file));

    // Step 1: send user message and files to backend
    this.chat.sendMessage(convId, formData).subscribe({
      next: () => {

        // Reload messages from backend to stay in sync
        this.history.getMessages(convId).subscribe(msgs => {
          this.messages = msgs;
          this.scrollToBottom();
        });

        // Step 2: if text is provided, trigger LLM request
        if (text.trim()) {

          // Remove welcome placeholder if present
          this.messages = this.messages.filter(m => !m._welcome);

          // Ask the LLM (RAG pipeline)
          this.chat.askLLM(text, convId, event.useInternet).subscribe({
            next: (resp) => {

              // Build markdown response from LLM output
              const markdownContent = (() => {
                let md = '';

                // Add main answer steps
                if (resp.steps?.length) {
                  md += resp.steps.join('\n\n');
                }

                // Append citations if available
                if (resp.citations?.length) {
                  md += `\n\n### Sources\n`;
                  md += resp.citations
                    .map(c => `- ${c.doc}`)
                    .join('\n');
                }

                return md.trim();
              })();

              // Push assistant message to UI
              this.messages.push({
                role: 'bot',
                content: markdownContent,
                isUser: false
              });

              this.scrollToBottom();
            },
            error: (err) => {
              this.chat.pushBotMessage(
                'RAG error: ' + err.message
              );
            }
          });
        }
      },
      error: (err) => {
        this.chat.pushBotMessage(
          'Send error: ' + err.message
        );
      }
    });
  }

  // ======================================================
  // SCROLL MANAGEMENT
  // ======================================================
  // Attach scroll listener to detect when user scrolls up
  attachScrollListener() {
    setTimeout(() => {
      const el = this.messagesContainer?.nativeElement;
      if (!el) return;

      el.addEventListener('scroll', () => {
        const threshold = 120;

        // Show button if user is far from bottom
        this.showScrollButton =
          el.scrollHeight - el.scrollTop - el.clientHeight > threshold;
      });
    }, 0);
  }

  // Scroll chat container to the bottom
  scrollToBottom(force = false) {
    setTimeout(() => {
      const el = this.messagesContainer?.nativeElement;
      if (!el) return;

      el.scrollTo({
        top: el.scrollHeight,
        behavior: force ? 'smooth' : 'auto'
      });

      this.showScrollButton = false;
    }, 0);
  }

  // ======================================================
  // FILE ICON RESOLUTION
  // ======================================================
  // Returns an icon path based on file extension
  getFileIcon(nameOrType?: string): string {
    const name = nameOrType?.toLowerCase() || '';

    if (name.endsWith('.pdf')) return 'icon/pdf.png';
    if (
      name.endsWith('.png') ||
      name.endsWith('.jpg') ||
      name.endsWith('.jpeg')
    ) return 'icon/img.png';
    if (
      name.endsWith('.doc') ||
      name.endsWith('.docx')
    ) return 'icon/docx.png';

    // Default icon
    return 'icon/dossier.png';
  }
}
