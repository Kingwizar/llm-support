import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subject } from 'rxjs';
import { environment } from '../../../environments/environment';


// ======================================================
// CHAT RESPONSE INTERFACE
// ======================================================
// This interface describes the structure of the response
// returned by the backend after a LLM/RAG inference.
//
// - summary        : short textual summary (optional use)
// - steps          : main generated answer (markdown-ready)
// - citations      : list of documents/sources used by the RAG
// - conversation_id: identifier of the related conversation
// ======================================================
interface ChatResponse {
  summary: string;
  steps: string[];
  citations: { doc: string; score: number; snippet: string }[];
  conversation_id: string;
}


// ======================================================
// CHAT SERVICE
// ======================================================
// This service is responsible for all chat-related actions:
//
// - Sending user messages (text + files)
// - Triggering LLM/RAG inference
// - Broadcasting system/bot messages to the UI
//
// It acts as the main communication layer between
// Angular components and the Express gateway.
// ======================================================
@Injectable({ providedIn: 'root' })
export class ChatService {

  // Base URL of the backend (Express gateway)
  private baseUrl = environment.serverUrl;

  // Subject used to push asynchronous bot/system messages
  // (errors, status messages, fallback responses, etc.)
  private botMessage$ = new Subject<string>();

  // Observable exposed to components
  botMessageObs = this.botMessage$.asObservable();

  constructor(private http: HttpClient) {}


  // ======================================================
  // PUSH BOT MESSAGE (UI FEEDBACK)
  // ======================================================
  // Allows the service to emit a message that can be
  // displayed by the UI without going through the LLM.
  //
  // Example use cases:
  // - Error messages
  // - System notifications
  // - Temporary feedback
  // ======================================================
  pushBotMessage(msg: string) {
    this.botMessage$.next(msg);
  }


  // ======================================================
  // SEND USER MESSAGE (TEXT + FILES)
  // ======================================================
  // Sends a message to the backend for persistence.
  //
  // - convId   : conversation identifier
  // - formData: contains text + uploaded files
  //
  // This endpoint DOES NOT trigger the LLM directly.
  // It only stores the message and indexed documents.
  // ======================================================
  sendMessage(convId: string, formData: FormData) {
    return this.http.post(
      `${this.baseUrl}/api/chat/message/${convId}`,
      formData
    );
  }


  // ======================================================
  // ASK LLM / RAG PIPELINE
  // ======================================================
  // Triggers the LLM inference pipeline on the backend.
  //
  // Parameters:
  // - question    : user input text
  // - convId      : conversation context (memory + RAG)
  // - useInternet : enables/disables web search fallback
  //
  // Backend flow:
  // Angular → Express → FastAPI → RAG + Web → Ollama
  // ======================================================
  askLLM(
    question: string,
    convId?: string,
    useInternet?: boolean
  ): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(
      `${this.baseUrl}/api/chat`,
      {
        question,
        conv_id: convId,
        use_web: useInternet
      }
    );
  }
}
