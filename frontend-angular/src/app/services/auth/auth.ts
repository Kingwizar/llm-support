import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, tap } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private baseUrl = environment.serverUrl;

  // ✅ UNE seule clé pour tous (login local + Auth0)
  private tokenKey = 'auth_token';

  private currentUser$ = new BehaviorSubject<any | null>(null);
  user$ = this.currentUser$.asObservable();

  constructor(private http: HttpClient, private auth0: Auth0Service) {}

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
    // ✅ nettoie proprement
    localStorage.removeItem(this.tokenKey);
    this.currentUser$.next(null);

    // ✅ logout Auth0 (si session Auth0 active)
    this.auth0.logout({
      logoutParams: { returnTo: window.location.origin }
    });
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  isLoggedIn(): boolean {
    return !!this.getToken();
  }

  loginWithAuth0() {
    return this.auth0.loginWithRedirect();
  }

  /** 🔑 Récupère le token Auth0 et le stocke dans la MÊME clé */
  loadAuth0Token() {
    return this.auth0.getAccessTokenSilently().pipe(
      tap(token => {
        localStorage.setItem(this.tokenKey, token);
      })
    );
  }
}

// ✅ configuration Auth0 exportée
export const auth0Config = {
  domain: 'dev-5xqrzsdislhri5jj.us.auth0.com',
  clientId: 'oLJYZBFTXTrIWpY0y9oOxXfBw3dJxlwe',
  authorizationParams: {
    redirect_uri: window.location.origin + '/callback',
    audience: 'https://llm-support-api',
    scope: 'openid profile email'
  }
};

