import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HistoryService } from '../services//history/history';


@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './history.html',
  styleUrls: ['./history.css']
})
export class History {
  conversations: any[] = [];
  selectedConversation: any = null;
  newConversationName: string = '';
  http: any;
  apiUrl: any;

  constructor(private historyService: HistoryService) {}

  ngOnInit() {
    this.loadConversations();
  }

  loadConversations() {
    this.historyService.getConversations().subscribe({
  next: (data) => {
    console.log("✅ Conversations reçues depuis backend :", data);
    this.conversations = data;
  },
  error: (err) => console.error("❌ Erreur getConversations Angular :", err)
});

  }

  selectConversation(convo: any) {
  this.selectedConversation = convo;
  this.historyService.setActiveConversation(convo); 
}


  createConversation() {
    if (this.newConversationName.trim()) {
      this.historyService.createConversation(this.newConversationName).subscribe({
        next: () => {
          this.newConversationName = '';
          this.loadConversations();
        }
      });
    }
  }

  renameConversation(convo: any, event: Event) {
    event.stopPropagation(); 
    const newName = prompt('Nouveau nom :', convo.title);
    if (newName && newName.trim()) {
      this.historyService.renameConversation(convo.id, newName).subscribe({
        next: () => this.loadConversations()
      });
    }
  }

  deleteConversation(id: string, event: Event) {
  event.stopPropagation(); 
  if (confirm('Supprimer cette conversation ?')) {
    this.historyService.deleteConversation(id).subscribe({
      next: () => this.loadConversations()
    });
  }
}




}