import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../services/auth/auth';

@Component({
  standalone: true,
  selector: 'app-register',
  imports: [CommonModule, FormsModule, RouterLink ],
  templateUrl: './register.html',
  styleUrls: ['./register.css']
})
export class RegisterComponent {
  username = '';
  email = '';
  password = '';
  error = '';

  constructor(private auth: AuthService, private router: Router) {}

 register() {
    this.error = '';

    this.auth.register({
      username: this.username,   // ✅ OBLIGATOIRE
      email: this.email,
      password: this.password
    }).subscribe({
      next: () => {
        this.router.navigate(['/login']);
      },
      error: err => {
        this.error = err.error?.detail || 'Erreur lors de la création du compte';
      }
    });
  }

}
