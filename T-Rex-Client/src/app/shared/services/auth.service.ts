import { Injectable } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/compat/auth';
import { Router } from '@angular/router';
import firebase from 'firebase/compat/app';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  user$: Observable<firebase.User | null>;
  isLoggedIn$: Observable<boolean>;
  userID: string | undefined;
  sessionID: string | undefined;

  constructor(private auth: AngularFireAuth, private router: Router) {
    this.user$ = this.auth.authState;
    this.isLoggedIn$ = this.user$.pipe(map(user => !!user));
    this.user$.subscribe(user => {
      if (user) {
        user.getIdToken().then(idToken => {
          this.sessionID = idToken;
        });
      } else {
        this.sessionID = undefined;
      }
    });
    this.user$.subscribe(user => {
      if (user) {
        this.userID = user.uid;
      } else {
        this.userID = undefined;
      }
    });
  }

  async googleSignin() {
    const provider = new firebase.auth.GoogleAuthProvider();
    const credential = await this.auth.signInWithPopup(provider).finally(()=>{
      this.router.navigate(['/user-panel']);
    });
    return credential.user;
  }

  async signOut() {
    await this.auth.signOut().finally(()=>{
      this.router.navigate(['']);
    });
  }
}