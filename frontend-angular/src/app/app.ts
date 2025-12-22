import { Component, OnInit, inject } from '@angular/core';
import { RouterOutlet, Router } from '@angular/router';
import { AuthService } from './services/auth/auth';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';
import Keycloak from 'keycloak-js'; // ✅ IMPORTANT

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  template: `<router-outlet></router-outlet>`
})
export class App implements OnInit {
  private auth = inject(AuthService);
  private auth0 = inject(Auth0Service);
  private router = inject(Router);

  // ✅ CECI EST LA CLÉ
  private keycloak = inject(Keycloak);

  async ngOnInit() {
    console.log('🟣 APP INIT');

    // ===== Auth0 (NE PAS TOUCHER) =====
    this.auth0.isAuthenticated$.subscribe(isAuth => {
      if (isAuth) {
        this.auth.loadAuth0Token().subscribe();
      }
    });

    // ===== Keycloak =====
    if (this.keycloak.authenticated) {
      console.log('🟢 Keycloak authenticated');

      const token = this.keycloak.token;

      if (!token) {
        console.error('❌ Keycloak token absent');
        return;
      }

      console.log('🟢 Keycloak token récupéré');

      localStorage.setItem('auth_token', token);

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
    } else {
      console.log('ℹ️ Keycloak non authentifié');
    }
  }
}
