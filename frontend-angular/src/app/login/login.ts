import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../services/auth/auth';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';


// ======================================================
// LOGIN COMPONENT
// ======================================================
// This component provides two authentication mechanisms:
// 1. Local authentication (email + password)
// 2. Federated authentication via Auth0 (Google, Microsoft, GitHub)
//
// The component itself does not store any token.
// All sensitive authentication logic is delegated
// to the backend and Auth0 SDK.
// ======================================================
@Component({
  standalone: true,
  selector: 'app-login',
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './login.html',
  styleUrls: ['./login.css']
})
export class LoginComponent {

  // ======================================================
  // FORM STATE
  // ======================================================

  // User email (bound to input field)
  email = '';

  // User password (bound to input field)
  password = '';

  // Error message displayed in the UI
  error = '';

  constructor(
    // Application authentication service (backend login)
    private auth: AuthService,

    // Angular router for navigation
    private router: Router,

    // Auth0 Angular SDK for federated login
    private auth0: Auth0Service
  ) {}

  // ======================================================
  // LOCAL AUTHENTICATION
  // ======================================================
  // Sends email/password to the backend.
  // On success:
  // - backend creates a session cookie
  // - user profile is loaded
  // - user is redirected to the main application
  login() {
    this.error = '';

    this.auth.login(this.email, this.password).subscribe({
      next: () => {

        // After successful login, retrieve user profile
        this.auth.loadMe().subscribe(() => {

          // Redirect to main application
          this.router.navigate(['/']);
        });
      },
      error: () => {
        // Authentication failure feedback
        this.error = 'Email or password is incorrect';
      }
    });
  }

  // ======================================================
  // AUTH0 FEDERATED AUTHENTICATION
  // ======================================================
  // Redirects the user to Auth0's hosted login page
  // using the selected identity provider
  loginAuth0(provider: 'google' | 'microsoft' | 'github') {

    this.auth0.loginWithRedirect({
      authorizationParams: {

        // Select Auth0 connection dynamically
        connection:
          provider === 'google'
            ? 'google-oauth2'
            : provider === 'microsoft'
              ? 'windowslive'
              : 'github'
      }
    });
  }
}
