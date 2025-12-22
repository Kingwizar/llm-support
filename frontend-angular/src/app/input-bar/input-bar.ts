import { Component, EventEmitter, Output, Input } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-input-bar',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './input-bar.html',
  styleUrls: ['./input-bar.css']
})
export class InputBar {
  @Input() convId?: string;
  @Output() send = new EventEmitter<{ text: string; files: File[] }>();

  prompt: string = '';
  selectedFiles: File[] = [];

  /** Lorsqu’on sélectionne un ou plusieurs fichiers */
  onFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files) return;
    this.selectedFiles.push(...Array.from(input.files));
    input.value = '';
  }

  /** Supprimer un fichier de la sélection */
  removeFile(index: number) {
    this.selectedFiles.splice(index, 1);
  }

  /** Émet le message vers le parent sans contact API */
  sendMessage() {
    console.log("InputBar → emit vers ChatPanel");
    console.log("Texte:", this.prompt);
    console.log("Fichiers:", this.selectedFiles.length);

    if (!this.prompt.trim() && this.selectedFiles.length === 0) return;

    this.send.emit({ text: this.prompt, files: this.selectedFiles });
    this.prompt = '';
    this.selectedFiles = [];
  }
  onEnter(event: KeyboardEvent) {
  event.preventDefault();
  this.sendMessage();
}

}
