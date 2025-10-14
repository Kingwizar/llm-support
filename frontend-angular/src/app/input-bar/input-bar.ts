import { Component, EventEmitter, Output, Input } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { ChatService } from '../services/chat/chat';

@Component({
  selector: 'app-input-bar',
  standalone: true,
  imports: [FormsModule, CommonModule, HttpClientModule],
  templateUrl: './input-bar.html',
  styleUrls: ['./input-bar.css']
})
export class InputBar {
  @Input() convId?: string;
  @Output() send = new EventEmitter<{ text: string; files: any[] }>();

  prompt: string = '';
  selectedFiles: File[] = [];

  constructor(private chatService: ChatService) {}

  onFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files) return;
    this.selectedFiles.push(...Array.from(input.files));
    input.value = '';
  }

  removeFile(index: number) {
    this.selectedFiles.splice(index, 1);
  }

  sendMessage() {
    if (!this.prompt.trim() && this.selectedFiles.length === 0) return;

    // ✅ Si des fichiers sont sélectionnés
    if (this.selectedFiles.length > 0 && this.convId) {
      const formData = new FormData();
      this.selectedFiles.forEach(file => formData.append('files', file));

      this.chatService.uploadFiles(this.convId, formData).subscribe({
        next: (res) => {
          this.send.emit({ text: this.prompt, files: res.files });
          this.prompt = '';
          this.selectedFiles = [];
        },
        error: (err) =>
          this.chatService.pushBotMessage('⚠️ Erreur upload : ' + err.message)
      });
    } else {
      // Aucun fichier → juste texte
      this.send.emit({ text: this.prompt, files: [] });
      this.prompt = '';
    }
  }
}
