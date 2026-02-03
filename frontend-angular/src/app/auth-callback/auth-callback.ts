import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';
import { AuthService } from '../services/auth/auth';
import { HttpClient } from '@angular/common/http';


// ======================================================
// AUTH CALLBACK COMPONENT
// ======================================================
// This component is loaded after a successful Auth0 login.
// Its role is to:
// 1. Retrieve the Auth0 access token
// 2. Send it to the backend gateway
// 3. Create a backend session (HttpOnly cookie)
// 4. Load the authenticated user
// 5. Redirect the user to the main application
// ======================================================
@Component({
  standalone: true,
  template: `<p>Signing in...</p>`
})
export class AuthCallbackComponent implements OnInit {

  constructor(
    // Auth0 Angular SDK service
    private auth0: Auth0Service,

    // Application authentication service (backend-based)
    private auth: AuthService,

    // Angular router for navigation
    private router: Router,

    // HTTP client used to call the backend gateway
    private http: HttpClient
  ) {}

  // ======================================================
  // COMPONENT INITIALIZATION
  // ======================================================
  ngOnInit() {

    // Retrieve the Auth0 access token silently
    // This does not trigger any redirect
    this.auth0.getAccessTokenSilently().subscribe({

      next: token => {

        // Step 1: send the Auth0 token to the backend gateway
        // The gateway validates it via FastAPI and converts it
        // into an HttpOnly session cookie
        this.http.post(
          '/auth/auth0',
          {},
          {
            headers: {
              Authorization: `Bearer ${token}`
            },
            withCredentials: true
          }
        ).subscribe({

          // Step 2: backend session successfully created
          next: () => this.finalizeLogin(),

          // Backend rejected the Auth0 token
          error: err => {
            console.error('auth/auth0 failed', err);
            this.router.navigate(['/login']);
          }
        });
      },

      // Failed to retrieve Auth0 token
      error: () => this.router.navigate(['/login'])
    });
  }

  // ======================================================
  // FINALIZE LOGIN
  // ======================================================
  // This method verifies that the backend session is valid
  // by calling /auth/me and loading the authenticated user
  private finalizeLogin() {
    this.auth.loadMe().subscribe({

      // Backend authentication is confirmed
      next: user => {
        console.log('Backend session established:', user);

        // Redirect user to main application
        this.router.navigate(['/']);
      },

      // Backend session invalid or expired
      error: err => {
        console.error('Backend authentication error:', err);
        this.router.navigate(['/login']);
      }
    });
  }
}
