import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../services/auth/auth';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';

@Component({
  standalone: true,
  selector: 'app-register',
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './register.html',
  styleUrls: ['./register.css']
})
export class RegisterComponent {
  username = '';
  email = '';
  password = '';
  error = '';

  private baseUrl = environment.serverUrl;

  constructor(
    private auth: AuthService,
    private http: HttpClient,
    private router: Router
  ) {}

  register() {
    this.error = '';

    this.http.post(
      `${this.baseUrl}/auth/register`,
      {
        username: this.username,
        email: this.email,
        password: this.password
      },
      { withCredentials: true }
    ).subscribe({
      next: () => {
        // 🔐 Initialisation CSRF après register
        this.http.get<{ csrfToken: string }>(
          `${this.baseUrl}/csrf-token`,
          { withCredentials: true }
        ).subscribe({
          next: res => {
            localStorage.setItem('csrf_token', res.csrfToken);
            this.router.navigate(['/login']);
          },
          error: () => {
            this.router.navigate(['/login']);
          }
        });
      },
      error: err => {
        this.error =
          err.error?.detail ||
          err.error?.error ||
          'Erreur lors de la création du compte';
      }
    });
  }
}
