import { HttpInterceptorFn } from '@angular/common/http';

export const csrfInterceptor: HttpInterceptorFn = (req, next) => {
  const csrf = localStorage.getItem('csrf_token');

  return next(
    req.clone({
      withCredentials: true,
      setHeaders: csrf ? { 'X-CSRF-Token': csrf } : {}
    })
  );
};
