import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../services/auth/auth';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';

@Component({
  standalone: true,
  selector: 'app-login',
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './login.html',     // ✅ RELATIF AU DOSSIER login/
  styleUrls: ['./login.css']
})
export class LoginComponent {
  email = '';
  password = '';
  error = '';

  constructor(
    private auth: AuthService,
    private router: Router,
    private auth0: Auth0Service
  ) {}

  

  login() {
    this.error = '';

    this.auth.login(this.email, this.password).subscribe({
      next: () => {
        this.auth.loadMe().subscribe(() => {
          this.router.navigate(['/']);
        });
      },
      error: () => {
        this.error = 'Email ou mot de passe incorrect';
      }
    });
  }

  loginAuth0(provider: 'google' | 'microsoft' | 'github') {
  this.auth0.loginWithRedirect({
    authorizationParams: {
      connection:
        provider === 'google' ? 'google-oauth2'
        : provider === 'microsoft' ? 'windowslive'
        : 'github'
    }
  });
}

}
