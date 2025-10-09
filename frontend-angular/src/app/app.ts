import { Component } from '@angular/core';
import { MainPageComponent } from './main-page/main-page';

@Component({
  selector: 'app-root',
  standalone: true,
  template: `<app-main-page></app-main-page>`,
  imports: [MainPageComponent],
  styleUrls: ['./app.css']
})
export class App {}
