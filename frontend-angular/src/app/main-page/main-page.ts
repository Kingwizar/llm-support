import { Component } from '@angular/core';
import { Navbar } from '../navbar/navbar';
import { ChatPanelComponent } from '../chat-panel/chat-panel';
import { History } from '../history/history';

@Component({
  selector: 'app-main-page',
  templateUrl: './main-page.html',
  styleUrls: ['./main-page.css'],
  standalone: true,
  imports: [Navbar, ChatPanelComponent, History]
})
export class MainPageComponent {}
