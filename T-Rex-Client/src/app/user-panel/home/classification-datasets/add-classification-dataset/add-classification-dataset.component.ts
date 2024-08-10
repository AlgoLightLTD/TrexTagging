import { Component } from '@angular/core';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';
import { ServerService } from 'src/app/shared/services/server.service';
import { AddClassificationDatasetRequest, AddClassificationDatasetResponse, UploadFileResponse } from 'src/app/shared/ServerCommunication';
import { TextPopupComponent } from 'src/app/shared/components/text-popup/text-popup.component';

@Component({
  selector: 'app-add-classification-dataset',
  templateUrl: './add-classification-dataset.component.html',
  styleUrls: ['./add-classification-dataset.component.css']
})
export class AddClassificationDatasetComponent {
  dataset = {
    title: '',
    description: '',
    file: null as File | null
  };

  constructor(
    public dialog: MatDialog,
    private dialogRef: MatDialogRef<AddClassificationDatasetComponent>,
    private server: ServerService
  ) {}

  onFileSelected(event: any): void {
    const file: File = event.target.files[0];
    if (file) {
      this.dataset.file = file;
    }
  }

  async onSubmit(): Promise<void> {
    if (this.dataset.file) {
      try {
        const response: UploadFileResponse = await this.server.uploadFile(this.dataset.file);
        if (response.code === 0) {
          const req: AddClassificationDatasetRequest = new AddClassificationDatasetRequest();
          req.title = this.dataset.title;
          req.description = this.dataset.description;
          req.id = response.file_id;
  
          this.server.post(AddClassificationDatasetResponse, req).subscribe(
            (res: AddClassificationDatasetResponse) => {
              if (res.code === 0) {
                this.dialogRef.close(true);
              } else {
                this.dialog.open(TextPopupComponent, {
                  data: { msg: res.message, color: "text-red" }
                });
              }
            },
            (error) => {
              console.error('Error adding dataset', error);
            }
          );
        }
      } catch (error) {
        console.error('Error uploading file', error);
      }
    }
  }  

  onCancel(): void {
    this.dialogRef.close(false);
  }
}