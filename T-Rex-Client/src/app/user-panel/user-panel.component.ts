import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-user-panel',
  templateUrl: './user-panel.component.html',
  styleUrls: ['./user-panel.component.css']
})
export class UserPanelComponent {
  constructor(private router: Router) { }

  isCurrentRoute(route: string): boolean {
    return this.router.url === route;
  }
}