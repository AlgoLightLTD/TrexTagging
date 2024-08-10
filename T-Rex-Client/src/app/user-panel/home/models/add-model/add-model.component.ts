import { Component } from '@angular/core';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';
import { ServerService } from 'src/app/shared/services/server.service';
import { AddModelRequest, AddModelResponse, UploadFileResponse } from 'src/app/shared/ServerCommunication';
import { TextPopupComponent } from 'src/app/shared/components/text-popup/text-popup.component';

@Component({
  selector: 'app-add-model',
  templateUrl: './add-model.component.html',
  styleUrls: ['./add-model.component.css']
})
export class AddModelComponent {
  model = {
    title: '',
    description: '',
    file: null as File | null,
    type: 'classification',
    classification_idx_to_class_name: ['']
  };

  constructor(
    public dialog: MatDialog,
    private dialogRef: MatDialogRef<AddModelComponent>,
    private server: ServerService
  ) {}

  onFileSelected(event: any): void {
    const file: File = event.target.files[0];
    if (file) {
      this.model.file = file;
    }
  }

  addClassName(): void {
    this.model.classification_idx_to_class_name.push('');
  }

  removeClassName(index: number): void {
    if (this.model.classification_idx_to_class_name.length > 1) {
      this.model.classification_idx_to_class_name.splice(index, 1);
    }
  }

  async onSubmit(): Promise<void> {
    if (this.model.file) {
      try {
        const response: UploadFileResponse = await this.server.uploadFile(this.model.file);
        if (response.code === 0) {
          const req: AddModelRequest = new AddModelRequest();
          req.title = this.model.title;
          req.description = this.model.description;
          req.id = response.file_id;
          req.model_type = this.model.type;
          if (this.model.type === 'classification') {
            // req.classification_idx_to_class_name is a dict, index as key (number), class name as value
            req.classification_idx_to_class_name = [];
            this.model.classification_idx_to_class_name.forEach((class_name, index) => {
              req.classification_idx_to_class_name.push(class_name);
            });
            console.log(req.classification_idx_to_class_name);
          }

          this.server.post(AddModelResponse, req).subscribe(
            (res: AddModelResponse) => {
              if (res.code === 0) {
                console.log('Model added successfully', res);
                this.dialogRef.close(true);
              } else {
                this.dialog.open(TextPopupComponent, {
                  data: { msg: res.message, color: "text-red" }
                });
              }
            },
            (error) => {
              console.error('Error adding model', error);
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

  trackByIndex(index: number, item: any): any {
    return index;
  }
}