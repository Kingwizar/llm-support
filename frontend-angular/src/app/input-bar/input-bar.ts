import { Component, EventEmitter, Output, Input } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';


// ======================================================
// INPUT BAR COMPONENT
// ======================================================
// This component represents the user input area of the chat.
// It is responsible only for collecting user input
// (text, files, and options) and emitting it upward.
// It does NOT handle any business logic or API calls.
// ======================================================
@Component({
  selector: 'app-input-bar',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './input-bar.html',
  styleUrls: ['./input-bar.css']
})
export class InputBar {

  // Optional conversation identifier (used by parent if needed)
  @Input() convId?: string;

  // Event emitted when the user sends a message
  // The parent component (ChatPanel) handles the actual logic
  @Output() send = new EventEmitter<{
    text: string;
    files: File[];
    useInternet: boolean;
  }>();

  // ======================================================
  // LOCAL STATE
  // ======================================================

  // Current text input (bound to textarea)
  prompt = '';

  // List of files selected by the user
  selectedFiles: File[] = [];

  // Flag indicating whether the user requests web augmentation
  useInternet = false;

  // ======================================================
  // UI ACTIONS
  // ======================================================

  // Toggle the "use internet" option
  // This value is forwarded to the backend through the parent
  toggleInternet() {
    this.useInternet = !this.useInternet;
  }

  // Emit the message to the parent component
  sendMessage() {
    // Prevent sending empty messages
    // unless files are attached
    if (!this.prompt.trim() && this.selectedFiles.length === 0) return;

    // Emit structured payload
    this.send.emit({
      text: this.prompt,
      files: this.selectedFiles,
      useInternet: this.useInternet
    });

    // Reset local state after sending
    this.prompt = '';
    this.selectedFiles = [];
    this.useInternet = false;
  }

  // Handle Enter key press inside the input
  // Prevents newline insertion and triggers send
  onEnter(event: Event) {
    event.preventDefault();
    this.sendMessage();
  }

  // ======================================================
  // FILE MANAGEMENT
  // ======================================================

  // Remove a selected file before sending
  removeFile(index: number) {
    this.selectedFiles.splice(index, 1);
  }

  // Handle file selection from file input
  onFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files) return;

    // Append newly selected files
    this.selectedFiles.push(
      ...Array.from(input.files)
    );

    // Reset file input to allow re-selection
    input.value = '';
  }
}
