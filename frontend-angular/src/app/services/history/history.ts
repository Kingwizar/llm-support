import { Injectable } from '@angular/core';
import { BehaviorSubject, catchError, map, Observable, of } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class HistoryService {
  /** URL du backend Node (proxy vers FastAPI) — ex : http://127.0.0.1:3000 */
  private baseUrl = environment.serverUrl;
  private tempConversationId: string | null = null;


  private activeConversation = new BehaviorSubject<any | null>(null);
  activeConversation$ = this.activeConversation.asObservable();

  constructor(private http: HttpClient) {}

  /** 🔹 Récupère toutes les conversations */
 getConversations(): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/conversations`);
  }

  /** 🔹 Récupère les messages d’une conversation */
  getMessages(id: string): Observable<any[]> {
  return this.http
    .get<any[]>(`${this.baseUrl}/api/chat/messages/${id}`)
    .pipe(catchError(() => of([])));
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

    setTempConversation(id: string) {
    this.tempConversationId = id;
  }

  clearTempConversation() {
    this.tempConversationId = null;
  }

  getTempConversation(): string | null {
    return this.tempConversationId;
  }

  isTempConversationEmpty(): Observable<boolean> {
  if (!this.tempConversationId) {
    return of(false);
  }

  // Appeler la base pour vérifier s'il y a des messages
  return this.getMessages(this.tempConversationId).pipe(
    map((messages: any[]) => messages.length === 0),
    catchError(() => of(true)) // en cas d'erreur, on considère que c'est vide
  );
}

  


}
