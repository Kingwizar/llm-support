import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subject } from 'rxjs';

interface ChatResponse {
  steps: string[];
  citations: string[];
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private apiUrl = 'http://127.0.0.1:8000/chat';
  private botMessage$ = new Subject<string>();
botMessageObs = this.botMessage$.asObservable();

pushBotMessage(msg: string) {
  this.botMessage$.next(msg);
}

  constructor(private http: HttpClient) {}

  sendQuestion(question: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(this.apiUrl, { question });
  }
}
