import { CanActivateFn, Router } from '@angular/router';
import { inject } from '@angular/core';
import { AuthService } from '../auth/auth';

export const authGuard: CanActivateFn = () => {
  const auth = inject(AuthService);
  const router = inject(Router);

  // ✅ si token présent → OK
  if (auth.isLoggedIn()) {
    return true;
  }

  // ❌ sinon → retour login
  router.navigate(['/login']);
  return false;
};
