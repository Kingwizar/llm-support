import { Component, EventEmitter, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../services/chat/chat';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-input-bar',
  standalone: true,
  imports: [FormsModule, CommonModule, HttpClientModule],
  templateUrl: './input-bar.html',
  styleUrls: ['./input-bar.css']
})
export class InputBar {
  @Output() send = new EventEmitter<string>();
  prompt: string = '';

  constructor(private chatService: ChatService) {}

  sendMessage() {
  if (this.prompt.trim() !== '') {
    // 🔹 Envoie seulement le message de l’utilisateur
    this.send.emit(this.prompt);

    // 🔹 Appelle l’API pour récupérer la réponse
    this.chatService.sendQuestion(this.prompt).subscribe({
      next: (res) => {
        // Construit une réponse unique du bot
        const botResponse =
          (res.steps?.length ? res.steps.map((s: string) =>  s).join("\n") : "") +
          (res.citations?.length ? `\n📚 Sources: ${res.citations.map((c: any) => c.doc).join(", ")}` : "");
          

        // ⚠️ Ne pas utiliser `this.send.emit` ici
        // Tu envoies le botResponse directement au chat-panel via un EventEmitter distinct,
        // OU tu laisses chat-panel gérer l'affichage après l'appel API
        this.chatService.pushBotMessage(botResponse.trim());
      },
      error: (err) => {
        this.chatService.pushBotMessage("⚠️ Erreur API : " + err.message);
      }
    });

    this.prompt = '';
  }
}


}
