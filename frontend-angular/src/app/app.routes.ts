import { Routes } from '@angular/router';
import { LoginComponent } from './login/login';
import { RegisterComponent } from './register/register';
import { authGuard } from './services/guards/auth-guard';
import { MainPageComponent } from './main-page/main-page';
import { authGuardFn } from '@auth0/auth0-angular';

export const routes: Routes = [
  { path: 'login', loadComponent: () => import('./login/login').then(m => m.LoginComponent) },
  { path: 'register', loadComponent: () => import('./register/register').then(m => m.RegisterComponent) },
  {
  path: 'callback',
  loadComponent: () =>
    import('./auth-callback/auth-callback').then(m => m.AuthCallbackComponent)
  },


  {
    path: '',
    canActivate: [authGuard], // 👈 TON guard, pas celui d’Auth0
    loadComponent: () => import('./main-page/main-page').then(m => m.MainPageComponent)
  },

  { path: '**', redirectTo: 'login' }
];