import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../services/auth/auth';

@Component({
  standalone: true,
  selector: 'app-login',
  imports: [
    CommonModule,
    FormsModule,
    RouterLink   // ✅ OBLIGATOIRE
  ],
  templateUrl: './login.html',
  styleUrls: ['./login.css']
})
export class LoginComponent {
  email = '';
  password = '';
  error = '';

  constructor(private auth: AuthService, private router: Router) {}

  login() {
    this.error = '';

    this.auth.login(this.email, this.password).subscribe({
      next: () => {
        this.auth.loadMe().subscribe(() => {
          this.router.navigate(['/']);
        });
      },
      error: () => {
        this.error = 'Email ou mot de passe incorrect';
      }
    });
  }
}
