import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HistoryService } from '../services/history/history';
import { ChatService } from '../services/chat/chat';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './history.html',
  styleUrls: ['./history.css']
})
export class History implements OnInit {

  // List of all conversations retrieved from the backend
  conversations: any[] = [];

  // Currently selected conversation (active chat context)
  selectedConversation: any = null;

  // Temporary input used when creating a new conversation manually
  newConversationName: string = '';

  // Constructor injects the HistoryService,
  // which acts as the bridge between this component and the backend API
  constructor(private historyService: HistoryService) {}

  // ======================================================
  // COMPONENT INITIALIZATION
  // ======================================================
  ngOnInit() {
    // Load all conversations from the backend on startup
    this.loadConversations();

    // Automatically create a temporary conversation
    // if no conversation is selected shortly after initialization
    setTimeout(() => {
      if (!this.selectedConversation) {
        this.createTemporaryConversation();
      }
    }, 200);
  }

  // ======================================================
  // CLEANUP TEMPORARY CONVERSATION AFTER RELOAD
  // ======================================================
  cleanupTempConversationOnReload() {
    // Look for a conversation explicitly named "Nouvelle conversation"
    const temp = this.conversations.find(c =>
      c.title === "Nouvelle conversation"
    );

    // If no temporary conversation exists, nothing to clean
    if (!temp) return;

    // Fetch messages of the temporary conversation
    this.historyService.getMessages(temp.id).subscribe(msgs => {

      // If the conversation has no messages,
      // it means it was never used and can be safely deleted
      if (msgs.length === 0) {
        console.log("Removing unused temporary conversation after reload");

        this.historyService.deleteConversation(temp.id).subscribe({
          next: () => {
            // Remove it from the frontend state
            this.conversations = this.conversations.filter(c => c.id !== temp.id);

            // Clear temporary conversation reference in the service
            this.historyService.clearTempConversation();
          }
        });
      }
    });
  }

  // ======================================================
  // SORT CONVERSATIONS
  // ======================================================
  sortConversations() {
    // Sort conversations by ID in descending order
    // (MongoDB ObjectId implies chronological order)
    this.conversations = this.conversations.sort((a, b) =>
      b.id.localeCompare(a.id)
    );
  }

  // ======================================================
  // LOAD CONVERSATIONS FROM BACKEND
  // ======================================================
  loadConversations() {
    this.historyService.getConversations().subscribe({
      next: (data) => {
        console.log("Conversations received from backend:", data);

        // Assign and sort conversations (most recent first)
        this.conversations = data;
        this.sortConversations();

        // If no conversation exists at all,
        // automatically create a temporary one
        if (this.conversations.length === 0) {
          this.createTemporaryConversation();
          return;
        }

        // Otherwise, check if an old temporary conversation should be cleaned
        this.cleanupTempConversationOnReload();
      },
      error: (err) =>
        console.error("Error while loading conversations:", err)
    });
  }

  // ======================================================
  // SELECT A CONVERSATION
  // ======================================================
  selectConversation(convo: any) {
    const tempId = this.historyService.getTempConversation();

    // If a temporary conversation exists and user selects another one
    if (tempId && tempId !== convo.id) {

      // Check if the temporary conversation is empty
      this.historyService.isTempConversationEmpty().subscribe(isEmpty => {

        // If empty, delete it to avoid polluting the database
        if (isEmpty) {
          this.historyService.deleteConversation(tempId).subscribe({
            next: () => {
              this.conversations =
                this.conversations.filter(c => c.id !== tempId);
              this.historyService.clearTempConversation();
            }
          });
        }

        // Activate the newly selected conversation
        this.selectedConversation = convo;
        this.historyService.setActiveConversation(convo);
      });

    } else {
      // Normal selection when no temporary cleanup is needed
      this.selectedConversation = convo;
      this.historyService.setActiveConversation(convo);
    }
  }

  // ======================================================
  // CREATE A NEW CONVERSATION (MANUAL)
  // ======================================================
  createConversation() {
    if (this.newConversationName.trim()) {
      this.historyService.createConversation(this.newConversationName).subscribe({
        next: (conv: any) => {

          // Reset input field
          this.newConversationName = "";

          // Add new conversation to the list
          this.conversations.push(conv);
          this.sortConversations();

          // Automatically activate the new conversation
          this.selectedConversation = conv;
          this.historyService.setActiveConversation(conv);
        }
      });
    }
  }

  // ======================================================
  // RENAME A CONVERSATION
  // ======================================================
  renameConversation(convo: any, event: Event) {
    // Prevent click from also selecting the conversation
    event.stopPropagation();

    const newName = prompt('New conversation name:', convo.title);
    if (newName && newName.trim()) {
      this.historyService.renameConversation(convo.id, newName).subscribe({
        next: () => this.loadConversations()
      });
    }
  }

  // ======================================================
  // DELETE A CONVERSATION
  // ======================================================
  deleteConversation(id: string, event: Event) {
    // Prevent click from triggering selection
    event.stopPropagation();

    if (confirm('Delete this conversation?')) {
      this.historyService.deleteConversation(id).subscribe({
        next: () => this.loadConversations()
      });
    }
  }

  // ======================================================
  // CREATE TEMPORARY CONVERSATION
  // ======================================================
  createTemporaryConversation() {
    this.historyService.createConversation("Nouvelle conversation").subscribe({
      next: (conv: any) => {

        // Add temporary conversation to the list
        this.conversations.push(conv);
        this.sortConversations();

        // Activate it immediately
        this.selectedConversation = conv;
        this.historyService.setActiveConversation(conv);

        // Mark it as temporary in the service
        this.historyService.setTempConversation(conv.id);
      }
    });
  }

  // ======================================================
  // HANDLE ENTER KEY ON INPUT
  // ======================================================
  onEnterCreateConversation() {
    if (this.newConversationName.trim()) {
      this.createConversation();
    }
  }
}
