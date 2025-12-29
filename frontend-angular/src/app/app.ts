import { Component, OnInit, inject } from '@angular/core';
import { RouterOutlet, Router } from '@angular/router';
import { AuthService } from './services/auth/auth';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  template: `<router-outlet></router-outlet>`
})
export class App implements OnInit {
  private auth = inject(AuthService);
  private auth0 = inject(Auth0Service);
  



  async ngOnInit() {
    console.log('🟣 APP INIT');

    // ===== Auth0 (NE PAS TOUCHER) =====
    this.auth0.isAuthenticated$.subscribe(isAuth => {
      if (isAuth) {
        this.auth.loadAuth0Token().subscribe();
      }
    });

    
  }
}
