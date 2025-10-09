import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatBox } from './chat-box';

describe('ChatBox', () => {
  let component: ChatBox;
  let fixture: ComponentFixture<ChatBox>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatBox]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChatBox);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
