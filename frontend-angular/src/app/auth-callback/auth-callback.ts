import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';
import { AuthService } from '../services/auth/auth';
import { environment } from '../../environments/environment';
import { HttpClient } from '@angular/common/http';

@Component({
  standalone: true,
  template: `<p>Connexion en cours...</p>`
})
export class AuthCallbackComponent implements OnInit {
  constructor(
    private auth0: Auth0Service,
    private auth: AuthService,
    private router: Router,
    private http: HttpClient   // ✅ OBLIGATOIRE
  ) {}

  ngOnInit() {
  console.log('🔵 CALLBACK INIT');
  console.log('URL:', window.location.href);

  this.auth0.getAccessTokenSilently().subscribe({
  next: token => {
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
      next: () => this.finalizeLogin(),
      error: err => {
        console.error('❌ auth/auth0 failed', err);
        this.router.navigate(['/login']);
      }
    });
  },
  error: () => this.router.navigate(['/login'])
});



  
}

private finalizeLogin() {
  this.auth.loadMe().subscribe({
    next: user => {
      console.log('✅ Backend OK:', user);
      this.router.navigate(['/']);
    },
    error: err => {
      console.error('❌ Backend error:', err);
      this.router.navigate(['/login']);
    }
  });
}

}
