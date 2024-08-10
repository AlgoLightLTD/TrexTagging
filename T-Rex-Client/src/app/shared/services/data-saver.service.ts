import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DataSaverService {
  private dataSaved = new BehaviorSubject<any | null>(null);

  setData(data: any) {
    this.dataSaved.next(data);
  }

  getData(): Observable<any | null> {
    return this.dataSaved.asObservable();
  }
}
