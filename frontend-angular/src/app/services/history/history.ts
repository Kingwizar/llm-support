import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class HistoryService {
  private apiUrl = 'http://127.0.0.1:3000/conversations';

  // Active conversation (objet complet, pas juste l'id)
  private activeConversation = new BehaviorSubject<any | null>(null);
  activeConversation$ = this.activeConversation.asObservable();

  constructor(private http: HttpClient) {}

  getConversations(id?: string): Observable<any[]> {
  if (id) {
    return this.http.get<any[]>(`${this.apiUrl}/${id}`);
  }
  return this.http.get<any[]>(this.apiUrl);
}

  createConversation(title: string) {
    return this.http.post<any>(this.apiUrl, { title });
  }

  renameConversation(id: string, newtitle: string) {
    return this.http.put<any>(`${this.apiUrl}/${id}`, { title: newtitle });
  }

  addMessage(id: string, role: string, content: string) {
    return this.http.post<any>(`${this.apiUrl}/${id}/messages`, { role, content });
  }

  getMessages(id: string) {
    return this.http.get<any[]>(`${this.apiUrl}/${id}/messages`);
  }

  // 🔥 Nouvelle méthode
  setActiveConversation(convo: any) {
    this.activeConversation.next(convo);
  }
  deleteConversation(id: string) {
  return this.http.delete(`${this.apiUrl}/${id}`);
}
sendToLLM(question: string) {
  return this.http.post<any>('http://127.0.0.1:8000/chat', { question });
}

uploadFile(conversationId: string, formData: FormData) {
  return this.http.post<any>(`${this.apiUrl}/${conversationId}/upload`, formData);
}



}
