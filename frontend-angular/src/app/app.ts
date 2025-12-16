import { Component } from '@angular/core';
import { MainPageComponent } from './main-page/main-page';

import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  template: `<router-outlet></router-outlet>`,
  styleUrls: ['./app.css']
})
export class App {}
