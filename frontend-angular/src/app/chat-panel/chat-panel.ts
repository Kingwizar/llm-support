import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HistoryService } from '../services/history/history';
import { ChatService } from '../services/chat/chat';
import { InputBar } from '../input-bar/input-bar';

@Component({
  selector: 'app-chat-panel',
  standalone: true,
  imports: [CommonModule, InputBar],
  templateUrl: './chat-panel.html',
  styleUrls: ['./chat-panel.css']
})
export class ChatPanelComponent implements OnInit {
  activeConversation: any = null;
  messages: { role: string, content: string }[] = [];

  constructor(
    private historyService: HistoryService,
    private chatService: ChatService
  ) {}

  ngOnInit() {
    // écoute la conversation active
    this.historyService.activeConversation$.subscribe(convo => {
      this.activeConversation = convo;
      if (convo?._id) {
        this.historyService.getMessages(convo._id).subscribe((msgs: any) => {
          this.messages = msgs;
        });
      } else {
        this.messages = [];
      }
    });
  }

  onSendMessage(content: string) {
  console.log("📤 Message utilisateur envoyé:", content);

  if (!this.activeConversation?._id) return;

  // ➡️ Ajout du message utilisateur
  this.messages.push({ role: 'user', content });
  this.historyService.addMessage(this.activeConversation._id, 'user', content).subscribe();

  // ➡️ Appel API pour la réponse bot
  console.log("🚀 Appel API lancé...");
  this.historyService.sendToLLM(content).subscribe({
    next: (res: any) => {
      console.log("✅ Réponse API reçue:", res);

      // On construit une seule réponse bot
      const botResponse =
        (res.steps?.length ? res.steps.map((s: string) =>  s).join("\n") : "") +
        (res.citations?.length ? `\n📚 Sources: ${res.citations.map((c: any) => c.doc).join(", ")}` : "");

      // ➡️ Ajout du message bot, SANS rappel d’onSendMessage !!
      const botMsg = { role: 'bot', content: botResponse.trim() };
      this.messages.push(botMsg);
      this.historyService.addMessage(this.activeConversation._id, 'bot', botMsg.content).subscribe();
    },
    error: (err) => {
      console.error("❌ Erreur API:", err);
    }
  });
}



}
