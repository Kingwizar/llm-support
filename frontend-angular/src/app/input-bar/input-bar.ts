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
  @Output() send = new EventEmitter<{
    text: string;
    files: File[];
    useInternet: boolean;
  }>();

  prompt = '';
  selectedFiles: File[] = [];
  useInternet = false;

  toggleInternet() {
    this.useInternet = !this.useInternet;
  }

  sendMessage() {
    if (!this.prompt.trim() && this.selectedFiles.length === 0) return;

    this.send.emit({
      text: this.prompt,
      files: this.selectedFiles,
      useInternet: this.useInternet
    });

    this.prompt = '';
    this.selectedFiles = [];
    this.useInternet = false; // reset après envoi (recommandé)
  }

  onEnter(event: Event) {
    event.preventDefault();
    this.sendMessage();
  }
  removeFile(index: number) {
    this.selectedFiles.splice(index, 1);
  }
  onFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files) return;
    this.selectedFiles.push(...Array.from(input.files));
    input.value = '';
  }
}
