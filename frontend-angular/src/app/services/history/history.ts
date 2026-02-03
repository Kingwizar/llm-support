import { Injectable } from '@angular/core';
import { BehaviorSubject, catchError, map, Observable, of } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';


// ======================================================
// HISTORY SERVICE
// ======================================================
// This service manages everything related to conversations
// and message history.
//
// Responsibilities:
// - Fetch conversations list
// - Fetch messages for a conversation
// - Create / rename / delete conversations
// - Manage the currently active conversation
// - Handle temporary (auto-created) conversations
//
// It acts as the "state holder" for conversation navigation.
// ======================================================
@Injectable({ providedIn: 'root' })
export class HistoryService {

  // Base URL of the backend (Express gateway)
  private baseUrl = environment.serverUrl;

  // Stores the ID of a temporary conversation
  // (used when the app auto-creates a new empty conversation)
  private tempConversationId: string | null = null;

  // Reactive container holding the currently active conversation
  // Shared across components (History, ChatPanel, etc.)
  private activeConversation = new BehaviorSubject<any | null>(null);

  // Observable exposed to subscribers
  activeConversation$ = this.activeConversation.asObservable();

  constructor(private http: HttpClient) {}


  // ======================================================
  // CONVERSATIONS
  // ======================================================

  // Retrieve the list of all conversations for the user
  getConversations(): Observable<any[]> {
    return this.http.get<any[]>(
      `${this.baseUrl}/conversations`
    );
  }

  // Retrieve all messages for a given conversation
  //
  // If the backend fails or the conversation is empty,
  // an empty array is returned to keep the UI stable.
  getMessages(id: string): Observable<any[]> {
    return this.http
      .get<any[]>(`${this.baseUrl}/api/chat/messages/${id}`)
      .pipe(
        catchError(() => of([]))
      );
  }


  // ======================================================
  // CONVERSATION LIFECYCLE
  // ======================================================

  // Create a new conversation with a given title
  createConversation(title: string): Observable<any> {
    return this.http.post(
      `${this.baseUrl}/conversations`,
      { title }
    );
  }

  // Rename an existing conversation
  renameConversation(id: string, newTitle: string): Observable<any> {
    return this.http.put(
      `${this.baseUrl}/api/chat/messages/${id}`,
      { title: newTitle }
    );
  }

  // Delete a conversation and all its messages
  deleteConversation(id: string): Observable<any> {
    return this.http.delete(
      `${this.baseUrl}/api/chat/messages/${id}`
    );
  }


  // ======================================================
  // ACTIVE CONVERSATION STATE
  // ======================================================

  // Updates the currently active conversation
  // All subscribed components will react automatically
  setActiveConversation(convo: any) {
    this.activeConversation.next(convo);
  }


  // ======================================================
  // TEMPORARY CONVERSATION HANDLING
  // ======================================================
  // Temporary conversations are automatically created
  // when the application starts and no conversation exists.
  //
  // If the user never sends a message, this conversation
  // can be safely deleted.
  // ======================================================

  // Mark a conversation as temporary
  setTempConversation(id: string) {
    this.tempConversationId = id;
  }

  // Clear the temporary conversation flag
  clearTempConversation() {
    this.tempConversationId = null;
  }

  // Retrieve the current temporary conversation ID
  getTempConversation(): string | null {
    return this.tempConversationId;
  }

  // Check if the temporary conversation contains messages
  //
  // Returns:
  // - true  → conversation is empty
  // - false → conversation has messages or does not exist
  isTempConversationEmpty(): Observable<boolean> {
    if (!this.tempConversationId) {
      return of(false);
    }

    return this.getMessages(this.tempConversationId).pipe(
      map((messages: any[]) => messages.length === 0),
      catchError(() => of(true))
    );
  }
}
