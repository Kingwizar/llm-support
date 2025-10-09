import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-chat-box',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './chat-box.html',
  styleUrls: ['./chat-box.css']
})
export class ChatBox {
  messages: { sender: 'user' | 'bot', text: string }[] = [
    { sender: 'bot', text: 'Salut, je suis ton assistant 🤖' }
  ];

  addMessage(text: string, sender: 'user' | 'bot' = 'user') {
    this.messages.push({ sender, text });
  }
}