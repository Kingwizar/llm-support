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
  private botMessage$ = new Subject<string>();
  botMessageObs = this.botMessage$.asObservable();

  constructor(private http: HttpClient) {}

  pushBotMessage(msg: string) {
    this.botMessage$.next(msg);
  }

  sendQuestion(question: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(this.apiUrl, { question });
  }
}
