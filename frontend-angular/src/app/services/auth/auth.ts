// app/services/auth/auth.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, switchMap, tap } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';
import { AuthConfig } from '@auth0/auth0-angular';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private baseUrl = environment.serverUrl;
  private currentUser$ = new BehaviorSubject<any | null>(null);
  user$ = this.currentUser$.asObservable();

  constructor(
    private http: HttpClient,
    private auth0: Auth0Service
  ) {}

  // ===== LOGIN LOCAL =====
  login(email: string, password: string) {
    return this.http.post(
      `${this.baseUrl}/auth/login`,
      { email, password },
      { withCredentials: true }
    ).pipe(
      switchMap(() =>
        this.http.get<{ csrfToken: string }>(
          `${this.baseUrl}/csrf-token`,
          { withCredentials: true }
        )
      ),
      tap(res => localStorage.setItem('csrf_token', res.csrfToken))
    );
  }

  // ===== UTILISATEUR =====
  loadMe() {
  return this.http.get('/auth/me', { withCredentials: true }).pipe(
    tap(user => {
      console.log('🟢 currentUser set', user);
      this.currentUser$.next(user);
    })
  );
}



  logout() {
    this.currentUser$.next(null);

    this.auth0.logout({
      logoutParams: {
        returnTo: window.location.origin + '/login'
      }
    });
  }

  // ===== AUTH0 =====
  loginWithAuth0() {
    return this.auth0.loginWithRedirect();
  }

  loadAuth0Token() {
    // ⚠️ On NE STOCKE PAS le token
    return this.auth0.getAccessTokenSilently();
  }

  isLoggedIn(): boolean {
    return !!this.currentUser$.value;
  }
}


export const auth0Config: AuthConfig = {
  domain: 'dev-5xqrzsdislhri5jj.us.auth0.com',
  clientId: 'oLJYZBFTXTrIWpY0y9oOxXfBw3dJxlwe',

  authorizationParams: {
    redirect_uri: window.location.origin + '/callback',
    audience: 'https://llm-support-api',
    scope: 'openid profile email'
  },

  // 🔴 TYPE STRICT
  cacheLocation: 'localstorage',
  useRefreshTokens: true
};
