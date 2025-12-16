import { bootstrapApplication } from '@angular/platform-browser';
import { provideHttpClient, withInterceptors } from '@angular/common/http';

import { App } from './app/app';
import { authInterceptor } from './app/services/interceptors/auth-interceptor';
import { routes } from './app/app.routes';
import { provideRouter } from '@angular/router';

bootstrapApplication(App, {
  providers: [
    provideRouter(routes),
    provideHttpClient(
      withInterceptors([authInterceptor])
    )
  ]
});
