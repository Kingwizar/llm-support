import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HistoryService } from '../services//history/history';
import { ChatService } from '../services/chat/chat';


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

  // Création automatique d'une nouvelle conversation
  setTimeout(() => {
    if (!this.selectedConversation) {
      this.createTemporaryConversation();
    }
  }, 200);
}
cleanupTempConversationOnReload() {
  // 1️⃣ On cherche une conversation appelée "Nouvelle conversation"
  const temp = this.conversations.find(c => 
    c.title === "Nouvelle conversation"
  );

  if (!temp) return;

  // 2️⃣ On vérifie si elle est vide dans la base
  this.historyService.getMessages(temp.id).subscribe(msgs => {

    if (msgs.length === 0) {
      console.log("🧹 Suppression automatique de la conversation temporaire vide après reload");

      this.historyService.deleteConversation(temp.id).subscribe({
        next: () => {
          // On retire du frontend
          this.conversations = this.conversations.filter(c => c.id !== temp.id);

          // Nettoyage du flag temporaire
          this.historyService.clearTempConversation();
        }
      });
    }
  });
}

sortConversations() {
  this.conversations = this.conversations.sort((a, b) =>
    b.id.localeCompare(a.id)
  );
}


  loadConversations() {
  this.historyService.getConversations().subscribe({
    next: (data) => {
      console.log("Conversations reçues depuis backend :", data);

      // 🔥 TRI DÉCROISSANT : les plus récentes en premier
      this.conversations = data;
      this.sortConversations();



      // 🔥 Si aucune conversation → création automatique
      if (this.conversations.length === 0) {
        this.createTemporaryConversation();
        return;
      }

      // 🔥 Sinon → vérifier si la conversation temporaire est vide
      this.cleanupTempConversationOnReload();
    },
    error: (err) => console.error("Erreur getConversations Angular :", err)
  });
}



  selectConversation(convo: any) {
  const tempId = this.historyService.getTempConversation();

  if (tempId && tempId !== convo.id) {

    // Vérifier vraiment dans la base si la conversation est vide
    this.historyService.isTempConversationEmpty().subscribe(isEmpty => {

      if (isEmpty) {
        // 🔥 SI vide → la supprimer
        this.historyService.deleteConversation(tempId).subscribe({
          next: () => {
            this.conversations = this.conversations.filter(c => c.id !== tempId);
            this.historyService.clearTempConversation();
          }
        });
      }

      // Quoi qu’il arrive, on sélectionne la nouvelle conversation
      this.selectedConversation = convo;
      this.historyService.setActiveConversation(convo);

    });

  } else {

    // Cas normal
    this.selectedConversation = convo;
    this.historyService.setActiveConversation(convo);

  }
}




  createConversation() {
    if (this.newConversationName.trim()) {
      this.historyService.createConversation(this.newConversationName).subscribe({
        next: (conv: any) => {
          this.newConversationName = "";

          // Ajouter dans la liste
          this.conversations.push(conv);

          // 🔥 TRIER immédiatement
          this.sortConversations();
          
          this.selectedConversation = conv;
          this.historyService.setActiveConversation(conv);

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
createTemporaryConversation() {
  this.historyService.createConversation("Nouvelle conversation").subscribe({
    next: (conv: any) => {

      // On l’ajoute dans la liste
      this.conversations.push(conv);

      // 🔥 TRIER immédiatement
      this.sortConversations();

      // On la sélectionne automatiquement
      this.selectedConversation = conv;
      this.historyService.setActiveConversation(conv);

      // On marque cette conversation comme temporaire
      this.historyService.setTempConversation(conv.id);
    }
  });
}
onEnterCreateConversation() {
  if (this.newConversationName.trim()) {
    this.createConversation();
  }
}




}