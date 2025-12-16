import { Routes } from '@angular/router';
import { LoginComponent } from './login/login';
import { RegisterComponent } from './register/register';
import { authGuard } from './services/guards/auth-guard';
import { MainPageComponent } from './main-page/main-page';

export const routes: Routes = [
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },

  {
    path: '',
    component: MainPageComponent,
    canActivate: [authGuard]
  },

  { path: '**', redirectTo: '' }
];
