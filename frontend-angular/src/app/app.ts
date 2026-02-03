import { Component, OnInit, inject } from '@angular/core';
import { RouterOutlet, Router } from '@angular/router';
import { AuthService } from './services/auth/auth';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  template: `<router-outlet></router-outlet>`
})
export class App implements OnInit {
  private http = inject(HttpClient);
  private auth0 = inject(Auth0Service);
  private auth = inject(AuthService);

  async ngOnInit() {


    try {
      const data = await firstValueFrom(
        this.http.get<{ csrfToken: string }>('/csrf-token', {
          withCredentials: true
        })
      );
      localStorage.setItem('csrf_token', data.csrfToken);

      console.log(' CSRF token loaded');
    } catch (e) {
      console.warn(' CSRF token load failed', e);
    }

    this.auth0.isAuthenticated$.subscribe(isAuth => {
      if (isAuth) {
        this.auth.loadAuth0Token().subscribe();
      }
    });
  }
}