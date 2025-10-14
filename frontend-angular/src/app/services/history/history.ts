import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class HistoryService {
  private apiUrl = environment.historyApiUrl;
  private activeConversation = new BehaviorSubject<any | null>(null);
  activeConversation$ = this.activeConversation.asObservable();

  constructor(private http: HttpClient) {}

  getConversations(id?: string): Observable<any[]> {
    const url = id ? `${this.apiUrl}/${id}` : this.apiUrl;
    return this.http.get<any[]>(url);
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

  setActiveConversation(convo: any) {
    this.activeConversation.next(convo);
  }

  deleteConversation(id: string) {
    return this.http.delete(`${this.apiUrl}/${id}`);
  }

  sendToLLM(question: string, _id: any) {
    return this.http.post<any>(environment.chatApiUrl, { question });
  }

  uploadFile(conversationId: string, formData: FormData) {
    return this.http.post<any>(`${this.apiUrl}/${conversationId}/upload`, formData);
  }
}
