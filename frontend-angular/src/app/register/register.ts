import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../services/auth/auth';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';


// ======================================================
// REGISTER COMPONENT
// ======================================================
// This component handles user account creation.
// It communicates directly with the backend gateway
// to register a new user and then prepares the
// authentication context (CSRF token, redirection).
// ======================================================
@Component({
  standalone: true,
  selector: 'app-register',
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './register.html',
  styleUrls: ['./register.css']
})
export class RegisterComponent {

  // ======================================================
  // FORM STATE
  // ======================================================

  // Username chosen by the user
  username = '';

  // User email address
  email = '';

  // User password
  password = '';

  // Error message displayed in the UI
  error = '';

  // Base URL of the backend gateway (from environment)
  private baseUrl = environment.serverUrl;

  constructor(
    // Application authentication service
    private auth: AuthService,

    // Angular HTTP client
    private http: HttpClient,

    // Router used for navigation
    private router: Router
  ) {}

  // ======================================================
  // REGISTRATION FLOW
  // ======================================================
  register() {

    // Reset error state
    this.error = '';

    // Step 1: send registration data to backend
    this.http.post(
      `${this.baseUrl}/auth/register`,
      {
        username: this.username,
        email: this.email,
        password: this.password
      },
      {
        // Required to receive cookies (session, CSRF)
        withCredentials: true
      }
    ).subscribe({

      // Registration successful
      next: () => {

        // Step 2: request a CSRF token for future protected requests
        this.http.get<{ csrfToken: string }>(
          `${this.baseUrl}/csrf-token`,
          { withCredentials: true }
        ).subscribe({

          // Store CSRF token locally and redirect to login
          next: res => {
            localStorage.setItem('csrf_token', res.csrfToken);
            this.router.navigate(['/login']);
          },

          // Even if CSRF retrieval fails, redirect to login
          error: () => {
            this.router.navigate(['/login']);
          }
        });
      },

      // Registration failed
      error: err => {
        this.error =
          err.error?.detail ||
          err.error?.error ||
          'Account creation failed';
      }
    });
  }
}
