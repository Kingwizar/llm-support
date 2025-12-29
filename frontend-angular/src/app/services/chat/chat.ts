import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subject } from 'rxjs';
import { environment } from '../../../environments/environment';

interface ChatResponse {
  summary: string;
  steps: string[];
  citations: { doc: string; score: number; snippet: string }[];
  conversation_id: string;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private baseUrl = environment.serverUrl; // ex: http://127.0.0.1:3000
  private botMessage$ = new Subject<string>();
  botMessageObs = this.botMessage$.asObservable();

  constructor(private http: HttpClient) {}

  /** Permet d'afficher un message bot local */
  pushBotMessage(msg: string) {
    this.botMessage$.next(msg);
  }

  /** Envoi d’un message (texte + fichiers) vers FastAPI via Express */
  sendMessage(convId: string, formData: FormData) {
  return this.http.post(`${this.baseUrl}/api/chat/message/${convId}`, formData);
}


  /** Pose une question au LLM RAG */
  askLLM(question: string, convId?: string, useInternet?: boolean): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.baseUrl}/api/chat`, { question, conv_id: convId, use_web: useInternet });
  }
}
