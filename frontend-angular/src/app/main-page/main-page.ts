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
export class MainPageComponent {
  // version Angular, à placer dans ngAfterViewInit()
ngAfterViewInit() {
  const resizer = document.getElementById("dragMe")!;
  const leftPane = document.querySelector(".history")!;
  const container = document.querySelector(".content")!;

  let x = 0;
  let leftWidth = 0;

  const mouseDownHandler = function (e: MouseEvent) {
    x = e.clientX;
    leftWidth = leftPane.getBoundingClientRect().width;

    document.addEventListener("mousemove", mouseMoveHandler);
    document.addEventListener("mouseup", mouseUpHandler);
  };

  const mouseMoveHandler = function (e: MouseEvent) {
    const dx = e.clientX - x;

    let newWidth = leftWidth + dx;

    // limites
    if (newWidth < 40) newWidth = 0;          // collapse auto
    if (newWidth > 600) newWidth = 600;

    leftPane.setAttribute("style", `width: ${newWidth}px`);
  };

  const mouseUpHandler = function () {
    document.removeEventListener("mousemove", mouseMoveHandler);
    document.removeEventListener("mouseup", mouseUpHandler);
  };

  resizer.addEventListener("mousedown", mouseDownHandler);
}

}

