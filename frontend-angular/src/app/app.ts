import { Component, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AuthService } from './services/auth/auth';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  template: `<router-outlet></router-outlet>`,
  styleUrls: ['./app.css']
})
export class App implements OnInit {
  constructor(
    private auth: AuthService,
    private auth0: Auth0Service
  ) {}

  ngOnInit() {
    this.auth0.isAuthenticated$.subscribe(isAuth => {
      if (isAuth) {
        this.auth.loadAuth0Token().subscribe({
          next: () => {
            // OK, token stocké
          },
          error: err => console.error(err)
        });
      }
    });
  }
}

