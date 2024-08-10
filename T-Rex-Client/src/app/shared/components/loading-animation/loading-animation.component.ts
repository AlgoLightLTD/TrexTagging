import { Component, Input } from '@angular/core';
import { ServerService } from '../../services/server.service';

@Component({
  selector: 'app-loading-animation',
  templateUrl: './loading-animation.component.html',
  styleUrls: ['./loading-animation.component.css']
})
export class LoadingAnimationComponent {
  @Input() text: string = 'Loading';
  constructor(public server: ServerService) {}
}