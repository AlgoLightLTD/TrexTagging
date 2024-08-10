import { inject } from '@angular/core';
import { AuthService } from '../services/auth.service';
import { CanActivateFn, Router } from '@angular/router';
import { map, tap } from 'rxjs/operators';

export const denyGuestsGuard: CanActivateFn = (route, state) => {
  return inject(AuthService).isLoggedIn$.pipe(
    tap(isLoggedIn => {
      if (!isLoggedIn) {
        inject(Router).navigate(['']);
      }
    })
  );
};

export const denyUsersGuard: CanActivateFn = (route, state) => {
  return inject(AuthService).isLoggedIn$.pipe(
    map(isLoggedIn => !isLoggedIn),
    tap(notLoggedIn => {
      if (!notLoggedIn) {
        inject(Router).navigate(['/user-panel']);
      }
    })
  );
};