import { bootstrapApplication } from '@angular/platform-browser';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { provideRouter } from '@angular/router';

import { App } from './app/app';
import { routes } from './app/app.routes';
import { authInterceptor } from './app/services/interceptors/auth-interceptor';

import { provideAuth0 } from '@auth0/auth0-angular';
import { auth0Config } from './app/services/auth/auth';


console.log('🟣 APP BOOTSTRAP');
console.log('🟣 window.location.href =', window.location.href);

bootstrapApplication(App, {
  providers: [
    provideRouter(routes),

    provideHttpClient(
      withInterceptors([authInterceptor])
    ),

    // ===== Auth0 =====
    provideAuth0(auth0Config),
    

  ]
});
