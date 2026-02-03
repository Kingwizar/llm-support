// app/services/auth/auth.ts

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, switchMap, tap } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';
import { AuthConfig } from '@auth0/auth0-angular';


// ======================================================
// AUTH SERVICE (ANGULAR)
// ======================================================
// This service centralizes all authentication logic
// for the Angular application.
//
// It supports:
// - Local authentication (email/password)
// - Auth0 authentication (OAuth2 / OpenID Connect)
// - Session-based authentication using HttpOnly cookies
// - CSRF token retrieval and storage
// - Reactive user state management
// ======================================================
@Injectable({ providedIn: 'root' })
export class AuthService {

  // ======================================================
  // CONFIGURATION & STATE
  // ======================================================

  // Base URL of the backend gateway (Express)
  private baseUrl = environment.serverUrl;

  // Internal reactive user state
  // null  -> not authenticated
  // object -> authenticated user
  private currentUser$ = new BehaviorSubject<any | null>(null);

  // Public observable exposed to components
  user$ = this.currentUser$.asObservable();

  constructor(
    // HTTP client for backend communication
    private http: HttpClient,

    // Auth0 SDK service
    private auth0: Auth0Service
  ) {}

  // ======================================================
  // LOCAL AUTHENTICATION (EMAIL / PASSWORD)
  // ======================================================
  // This method performs:
  // 1. POST /auth/login  → backend authentication
  // 2. GET  /csrf-token  → CSRF token retrieval
  // 3. Store CSRF token in localStorage
  //
  // Authentication itself relies on HttpOnly cookies.
  // ======================================================
  login(email: string, password: string) {
    return this.http.post(
      `${this.baseUrl}/auth/login`,
      { email, password },
      {
        // Required to send/receive session cookies
        withCredentials: true
      }
    ).pipe(

      // After successful login, request CSRF token
      switchMap(() =>
        this.http.get<{ csrfToken: string }>(
          `${this.baseUrl}/csrf-token`,
          { withCredentials: true }
        )
      ),

      // Store CSRF token for future protected requests
      tap(res => localStorage.setItem('csrf_token', res.csrfToken))
    );
  }

  // ======================================================
  // LOAD CURRENT USER
  // ======================================================
  // Fetches the authenticated user from backend
  // and updates the reactive user state.
  //
  // This endpoint is protected and requires:
  // - Valid session cookie
  // ======================================================
  loadMe() {
    return this.http.get('/auth/me', { withCredentials: true }).pipe(
      tap(user => {
        console.log('currentUser set', user);
        this.currentUser$.next(user);
      })
    );
  }

  // ======================================================
  // LOGOUT
  // ======================================================
  // Clears local user state and triggers Auth0 logout.
  // This also invalidates the session on Auth0 side.
  // ======================================================
  logout() {
    this.currentUser$.next(null);

    this.auth0.logout({
      logoutParams: {
        returnTo: window.location.origin + '/login'
      }
    });
  }

  // ======================================================
  // AUTH0 AUTHENTICATION
  // ======================================================

  // Redirects user to Auth0 login page
  loginWithAuth0() {
    return this.auth0.loginWithRedirect();
  }

  // Retrieves Auth0 access token silently
  // Used by the callback component to bridge Auth0 → backend session
  loadAuth0Token() {
    return this.auth0.getAccessTokenSilently();
  }

  // ======================================================
  // AUTH STATE HELPER
  // ======================================================
  // Returns true if a user is currently authenticated
  isLoggedIn(): boolean {
    return !!this.currentUser$.value;
  }
}


// ======================================================
// AUTH0 CONFIGURATION
// ======================================================
// This configuration defines the Auth0 client behavior:
// - OAuth2 / OIDC parameters
// - Token storage strategy
// - Audience and scopes
// ======================================================
export const auth0Config: AuthConfig = {

  // Auth0 tenant domain
  domain: 'dev-5xqrzsdislhri5jj.us.auth0.com',

  // Application client ID
  clientId: 'oLJYZBFTXTrIWpY0y9oOxXfBw3dJxlwe',

  authorizationParams: {

    // Redirect URI after successful login
    redirect_uri: window.location.origin + '/callback',

    // API audience used for access token
    audience: 'https://llm-support-api',

    // Requested OpenID scopes
    scope: 'openid profile email'
  },

  // Store tokens in localStorage (needed for refresh tokens)
  cacheLocation: 'localstorage',

  // Enable refresh tokens for long-lived sessions
  useRefreshTokens: true
};
