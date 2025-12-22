import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, tap } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private baseUrl = environment.serverUrl;
  private tokenKey = 'auth_token';

  private currentUser$ = new BehaviorSubject<any | null>(null);
  user$ = this.currentUser$.asObservable();

  constructor(
    private http: HttpClient,
    private auth0: Auth0Service
  ) {}

  // ===== AUTH LOCALE =====
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

  // ===== UTILISATEUR =====
  loadMe() {
  console.log('📡 /auth/me called');
  return this.http.get(`${this.baseUrl}/auth/me`).pipe(
    tap(user => {
      console.log('✅ /auth/me response:', user);
      this.currentUser$.next(user);
    })
  );
}

  

  logout() {
  alert('LOGOUT APPELÉ'); // 🔴 test brutal
  console.log('🚪 Logout appelé');

  localStorage.removeItem(this.tokenKey);
  this.currentUser$.next(null);

  const keycloakLogoutUrl =
    'http://localhost:8080/realms/nplusone/protocol/openid-connect/logout' +
    '?client_id=llm-support-api' +
    '&post_logout_redirect_uri=' +
    encodeURIComponent(window.location.origin + '/login');

  window.location.href = keycloakLogoutUrl;
}



  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  isLoggedIn(): boolean {
    return !!this.getToken();
  }

  // ===== AUTH0 =====
  loginWithAuth0() {
    return this.auth0.loginWithRedirect();
  }

  loadAuth0Token() {
    return this.auth0.getAccessTokenSilently().pipe(
      tap(token => {
        localStorage.setItem(this.tokenKey, token);
      })
    );
  }
}

// ===== CONFIG AUTH0 =====
export const auth0Config = {
  domain: 'dev-5xqrzsdislhri5jj.us.auth0.com',
  clientId: 'oLJYZBFTXTrIWpY0y9oOxXfBw3dJxlwe',
  authorizationParams: {
    redirect_uri: window.location.origin + '/callback',
    audience: 'https://llm-support-api',
    scope: 'openid profile email'
  }
};

// ===== CONFIG KEYCLOAK =====
export const keycloakConfig = {
  url: 'http://localhost:8080',
  realm: 'nplusone',
  clientId: 'llm-support-api'
};
