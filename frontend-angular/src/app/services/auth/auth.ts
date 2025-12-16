import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, tap } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private baseUrl = environment.serverUrl;
  private tokenKey = 'auth_token';

  private currentUser$ = new BehaviorSubject<any | null>(null);
  user$ = this.currentUser$.asObservable();

  constructor(private http: HttpClient) {}

  login(email: string, password: string) {
    return this.http
      .post<any>(`${this.baseUrl}/auth/login`, { email, password })
      .pipe(
        tap(res => {
          localStorage.setItem(this.tokenKey, res.access_token);
        })
      );
  }

  register(data: { username: string; email: string; password: string }) {
  return this.http.post(`${this.baseUrl}/auth/register`, data);
}


  loadMe() {
    return this.http.get(`${this.baseUrl}/auth/me`).pipe(
      tap(user => this.currentUser$.next(user))
    );
  }

  logout() {
    localStorage.removeItem(this.tokenKey);
    this.currentUser$.next(null);
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  isLoggedIn(): boolean {
    return !!this.getToken();
  }
}
