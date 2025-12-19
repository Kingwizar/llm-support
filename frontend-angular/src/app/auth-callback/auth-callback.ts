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
    this.auth0.isAuthenticated$.subscribe(isAuth => {
      if (isAuth) {
        this.auth.loadAuth0Token().subscribe({
          next: () => {
            this.auth.loadMe().subscribe({
              next: () => this.router.navigate(['/']),
              error: () => this.router.navigate(['/login'])
            });
          },
          error: () => this.router.navigate(['/login'])
        });
      }
    });
  }
}
