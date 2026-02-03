import { Component } from '@angular/core';
import { Navbar } from '../navbar/navbar';
import { ChatPanelComponent } from '../chat-panel/chat-panel';
import { History } from '../history/history';


// ======================================================
// MAIN PAGE COMPONENT
// ======================================================
// This component represents the main application layout.
// It composes the global UI using three major blocks:
// - Navbar (top navigation)
// - History panel (left sidebar with conversations)
// - Chat panel (main interaction area)
//
// It also implements a manual horizontal resize system
// allowing the user to adjust the width of the history panel.
// ======================================================
@Component({
  selector: 'app-main-page',
  templateUrl: './main-page.html',
  styleUrls: ['./main-page.css'],
  standalone: true,
  imports: [Navbar, ChatPanelComponent, History]
})
export class MainPageComponent {

  // ======================================================
  // VIEW INITIALIZATION
  // ======================================================
  // DOM-dependent logic is executed after the view
  // has been fully initialized.
  ngAfterViewInit() {

    // Resize handle element
    const resizer = document.getElementById("dragMe")!;

    // Left panel containing the conversation history
    const leftPane = document.querySelector(".history")!;

    // Main container (chat + history)
    const container = document.querySelector(".content")!;

    // Initial mouse position on drag start
    let x = 0;

    // Initial width of the left panel
    let leftWidth = 0;

    // ------------------------------------------------------
    // Mouse down: start resizing
    // ------------------------------------------------------
    const mouseDownHandler = function (e: MouseEvent) {
      x = e.clientX;
      leftWidth = leftPane.getBoundingClientRect().width;

      // Attach listeners on the document to capture
      // mouse movement outside the resizer itself
      document.addEventListener("mousemove", mouseMoveHandler);
      document.addEventListener("mouseup", mouseUpHandler);
    };

    // ------------------------------------------------------
    // Mouse move: update panel width
    // ------------------------------------------------------
    const mouseMoveHandler = function (e: MouseEvent) {
      const dx = e.clientX - x;

      // Compute new width based on mouse movement
      let newWidth = leftWidth + dx;

      // Enforce minimum and maximum width constraints
      if (newWidth < 40) newWidth = 0;
      if (newWidth > 600) newWidth = 600;

      // Apply new width dynamically
      leftPane.setAttribute(
        "style",
        `width: ${newWidth}px`
      );
    };

    // ------------------------------------------------------
    // Mouse up: stop resizing
    // ------------------------------------------------------
    const mouseUpHandler = function () {

      // Clean up event listeners to avoid memory leaks
      document.removeEventListener("mousemove", mouseMoveHandler);
      document.removeEventListener("mouseup", mouseUpHandler);
    };

    // Activate resizing on mouse down
    resizer.addEventListener("mousedown", mouseDownHandler);
  }
}
