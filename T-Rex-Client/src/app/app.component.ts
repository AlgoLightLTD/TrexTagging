import { Component, OnDestroy } from '@angular/core';
import { AuthService } from './shared/services/auth.service';
import { UserDataService } from './shared/services/user-data.service';
import { Subscription } from 'rxjs';
import { ServerService } from './shared/services/server.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  providers: [AuthService]
})
export class AppComponent implements OnDestroy{
  title = 'roog';
  balance: number = 0;
  private subscription: Subscription;

  constructor(
    public authService: AuthService,
    public server: ServerService,
    private userDataService: UserDataService,
    private router: Router) {
    this.subscription = this.userDataService.balanceObservable.subscribe(value => {
      this.balance = value;
    });
  }
  
  ngOnDestroy() {
    this.subscription.unsubscribe();
  }

  moveToHome(){
    this.router.navigate(['/user-panel']);
  }
}