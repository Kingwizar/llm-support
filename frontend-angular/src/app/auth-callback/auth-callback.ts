import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';
import { AuthService } from '../services/auth/auth';

@Component({
  standalone: true,
  template: `<p>Connexion en cours...</p>`
})
export class AuthCallbackComponent implements OnInit {
  constructor(
    private auth0: Auth0Service,
    private auth: AuthService,
    private router: Router
  ) {}

  ngOnInit() {
  console.log('🔵 CALLBACK INIT');
  console.log('URL:', window.location.href);

  // ================= AUTH0 (NE PAS TOUCHER) =================
  this.auth0.isAuthenticated$.subscribe(isAuth => {
    if (isAuth) {
      this.auth.loadAuth0Token().subscribe({
        next: () => this.finalizeLogin(),
        error: () => this.router.navigate(['/login'])
      });
    }
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
