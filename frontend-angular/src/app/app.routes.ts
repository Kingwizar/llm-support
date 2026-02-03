import { Routes } from '@angular/router';
import { authGuard } from './services/guards/auth-guard';


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
    canActivate: [authGuard], 
    loadComponent: () => import('./main-page/main-page').then(m => m.MainPageComponent)
  },

  { path: '**', redirectTo: 'login' }
];