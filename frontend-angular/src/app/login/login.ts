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
  templateUrl: './login.html',
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

  loginKeycloak() {
  const loginUrl =
    'http://localhost:8080/realms/nplusone/protocol/openid-connect/auth' +
    '?client_id=llm-support-api' +
    '&redirect_uri=' + encodeURIComponent(window.location.origin + '/callback') +
    '&response_type=code' +
    '&scope=openid' +
    '&kc_idp_hint=google';

  window.location.href = loginUrl;
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