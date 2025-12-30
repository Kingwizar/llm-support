// app/services/auth/auth.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, tap } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private baseUrl = environment.serverUrl;

  private currentUser$ = new BehaviorSubject<any | null>(null);
  user$ = this.currentUser$.asObservable();

  constructor(private http: HttpClient, private auth0: Auth0Service) {}

  private tokenKey = 'auth_token';

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  isLoggedIn(): boolean {
    return !!this.getToken();
  }

  login(email: string, password: string) {
    return this.http.post(`${this.baseUrl}/auth/login`, { email, password }, { withCredentials: true });
  }

  register(data: { username: string; email: string; password: string }) {
    return this.http.post(`${this.baseUrl}/auth/register`, data, { withCredentials: true });
  }

  loadMe() {
    return this.http.get(`${this.baseUrl}/auth/me`, { withCredentials: true }).pipe(
      tap(user => this.currentUser$.next(user))
    );
  }

  logout() {
    this.currentUser$.next(null);
    this.auth0.logout({ logoutParams: { returnTo: window.location.origin + '/login' } });
  }

  loginWithAuth0() {
    return this.auth0.loginWithRedirect();
  }

  loadAuth0Token() {
    // on ne stocke pas le token en localStorage en mode cookie
    return this.auth0.getAccessTokenSilently().pipe(tap(() => {}));
  }
  
}

// ✅ IMPORTANT : export nommé
export const auth0Config = {
  domain: 'dev-5xqrzsdislhri5jj.us.auth0.com',
  clientId: 'oLJYZBFTXTrIWpY0y9oOxXfBw3dJxlwe',
  authorizationParams: {
    redirect_uri: window.location.origin + '/callback',
    audience: 'https://llm-support-api',
    scope: 'openid profile email'
  }
};
