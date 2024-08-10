import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class UserDataService {

  private balance = new BehaviorSubject<number>(0);
  balanceObservable = this.balance.asObservable();

  constructor() { }

  updateBalance(newBalance: number) {
    this.balance.next(newBalance);
  }
}