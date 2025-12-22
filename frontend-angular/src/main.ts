import { bootstrapApplication } from '@angular/platform-browser';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { provideRouter } from '@angular/router';

import { App } from './app/app';
import { routes } from './app/app.routes';
import { authInterceptor } from './app/services/interceptors/auth-interceptor';

import { provideAuth0 } from '@auth0/auth0-angular';
import { auth0Config, keycloakConfig } from './app/services/auth/auth';

import { provideKeycloak } from 'keycloak-angular';
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

    // ===== Keycloak (nouvelle API, non deprecated) =====
    provideKeycloak({
  config: keycloakConfig,
  initOptions: {
    onLoad: 'check-sso',
    pkceMethod: 'S256',
    checkLoginIframe: false
  }
})

  ]
});
