import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class HistoryService {
  /** URL du backend Node (proxy vers FastAPI) — ex : http://127.0.0.1:3000 */
  private baseUrl = environment.serverUrl;

  private activeConversation = new BehaviorSubject<any | null>(null);
  activeConversation$ = this.activeConversation.asObservable();

  constructor(private http: HttpClient) {}

  /** 🔹 Récupère toutes les conversations */
 getConversations(): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/conversations`);
  }

  /** 🔹 Récupère les messages d’une conversation */
  getMessages(id: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/api/chat/messages/${id}`);
  }

  /** 🔹 Crée une nouvelle conversation */
  createConversation(title: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/conversations`, { title });
  }

  /** 🔹 Renomme une conversation existante */
  renameConversation(id: string, newTitle: string): Observable<any> {
    return this.http.put(`${this.baseUrl}/api/chat/messages/${id}`, { title: newTitle });
  }

  /** 🔹 Supprime une conversation */
  deleteConversation(id: string): Observable<any> {
    return this.http.delete(`${this.baseUrl}/api/chat/messages/${id}`);
  }

  /** 🔹 Définit la conversation active */
  setActiveConversation(convo: any) {
    this.activeConversation.next(convo);
  }
}
