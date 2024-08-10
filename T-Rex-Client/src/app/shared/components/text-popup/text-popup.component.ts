import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';

@Component({
  selector: 'app-text-popup',
  templateUrl: './text-popup.component.html',
  styleUrls: ['./text-popup.component.css']
})
export class TextPopupComponent {
  msg: string = '';
  color: string = '';

  constructor(
    public dialogRef: MatDialogRef<TextPopupComponent>,
    @Inject(MAT_DIALOG_DATA) private data: any
  ) {
    this.msg = this.data.msg;
    this.color = this.data.color;
  }

  closeDialog() {
    this.dialogRef.close();
  }
}