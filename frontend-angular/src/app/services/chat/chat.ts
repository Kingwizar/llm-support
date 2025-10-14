import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subject } from 'rxjs';
import { environment } from '../../../environments/environment';

interface ChatResponse {
  steps: string[];
  citations: string[];
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private apiUrl = environment.chatApiUrl;      
  private uploadUrl = environment.uploadApiUrl; 
  private botMessage$ = new Subject<string>();
  botMessageObs = this.botMessage$.asObservable();

  constructor(private http: HttpClient) {}

  /** Émet un message bot local (affichage côté front) */
  pushBotMessage(msg: string) {
    this.botMessage$.next(msg);
  }

  /** Envoie la question texte au backend */
  sendQuestion(question: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(this.apiUrl, { question });
  }

  /** Upload de fichiers pour une conversation spécifique */
  uploadFiles(convId: string, formData: FormData): Observable<any> {
    return this.http.post(`${this.uploadUrl}/${convId}`, formData);
  }
}
