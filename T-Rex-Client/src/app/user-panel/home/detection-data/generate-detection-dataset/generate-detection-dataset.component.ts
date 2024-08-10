import { Component, Inject, OnInit } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialog, MatDialogRef } from '@angular/material/dialog';
import { GenerateDetectionDatasetFromDetectionDataRequest, GenerateDetectionDatasetFromDetectionDataResponse, GetModelsRequest, GetModelsResponse, Model, UploadFileResponse, AddReferenceImageRequest, AddReferenceImageResponse } from 'src/app/shared/ServerCommunication';
import { TextPopupComponent } from 'src/app/shared/components/text-popup/text-popup.component';
import { ServerService } from 'src/app/shared/services/server.service';

@Component({
  selector: 'app-generate-detection-dataset',
  templateUrl: './generate-detection-dataset.component.html',
  styleUrls: ['./generate-detection-dataset.component.css']
})
export class GenerateDetectionDatasetComponent implements OnInit {
  dataset = {
    title: '',
    description: '',
    detectionModelId: '',
    classificationModelId: '',
    referenceImages: [{ file: null as File | null, previewUrl: '', boundingBoxes: [] as number[][] }]
  };
  detection_data_id: string = "";
  classification_models: Model[] = [];
  detection_models: Model[] = [];

  constructor(
    public dialog: MatDialog,
    private dialogRef: MatDialogRef<GenerateDetectionDatasetComponent>,
    private server: ServerService,
    @Inject(MAT_DIALOG_DATA) public data: { detection_data_id: string }
  ) {
    this.detection_data_id = data.detection_data_id;
  }

  ngOnInit(): void {
    const req: GetModelsRequest = new GetModelsRequest();
    this.server.post(GetModelsResponse, req).subscribe(
      (res: GetModelsResponse) => {
        if (res.code === 0) {
          this.classification_models = res.models.filter(model => model.model_type === 'classification');
          this.detection_models = res.models.filter(model => model.model_type === 'detection');
        } else {
          this.dialog.open(TextPopupComponent, {
            data: { msg: res.message, color: "text-red" }
          });
        }
      },
      (error) => {
        console.error('Error fetching models', error);
      }
    );
  }

  onFileSelected(event: any, index: number): void {
    const file: File = event.target.files[0];
    if (file) {
      this.dataset.referenceImages[index].file = file;
      this.dataset.referenceImages[index].previewUrl = URL.createObjectURL(file);
    }
  }

  addReferenceImage(): void {
    this.dataset.referenceImages.push({ file: null, previewUrl: '', boundingBoxes: [] });
  }

  removeReferenceImage(index: number): void {
    if (this.dataset.referenceImages.length > 1) {
      this.dataset.referenceImages.splice(index, 1);
    } else {
      this.dataset.referenceImages = [{ file: null, previewUrl: '', boundingBoxes: [] }];
    }
  }

  onBoundingBoxesChange(boundingBoxes: number[][], index: number): void {
    this.dataset.referenceImages[index].boundingBoxes = boundingBoxes;
  }

  async onSubmit(): Promise<void> {
    if (this.dataset.detectionModelId || this.dataset.referenceImages.some(ref => ref.file)) {
      try {
        const req: GenerateDetectionDatasetFromDetectionDataRequest = new GenerateDetectionDatasetFromDetectionDataRequest();
        req.title = this.dataset.title;
        req.description = this.dataset.description;
        req.data_id = this.detection_data_id;
        req.detection_model_id = this.dataset.detectionModelId;
        req.classification_model_id = this.dataset.classificationModelId;
        req.reference_image_ids = [];

        // Upload reference images and add reference image requests
        for (const refImage of this.dataset.referenceImages) {
          if (refImage.file) {
            const response: UploadFileResponse = await this.server.uploadFile(refImage.file);
            if (response.code === 0) {
              const addRefReq: AddReferenceImageRequest = new AddReferenceImageRequest();
              addRefReq.id = response.file_id;
              addRefReq.bounding_boxes = refImage.boundingBoxes;
        
              try {
                const res: AddReferenceImageResponse = await this.server.post(AddReferenceImageResponse, addRefReq).toPromise() as AddReferenceImageResponse;
                if (res.code === 0) {
                  req.reference_image_ids.push(addRefReq.id);
                } else {
                  this.dialog.open(TextPopupComponent, {
                    data: { msg: res.message, color: "text-red" }
                  });
                  return;
                }
              } catch (error) {
                console.error('Error adding reference image', error);
                this.dialog.open(TextPopupComponent, {
                  data: { msg: 'Error adding reference image', color: "text-red" }
                });
                return;
              }
            } else {
              this.dialog.open(TextPopupComponent, {
                data: { msg: response.message, color: "text-red" }
              });
              return;
            }
          }
        }

        this.server.post(GenerateDetectionDatasetFromDetectionDataResponse, req).subscribe(
          (res: GenerateDetectionDatasetFromDetectionDataResponse) => {
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
      } catch (error) {
        console.error('Error uploading file', error);
      }
    } else {
      this.dialog.open(TextPopupComponent, {
        data: { msg: 'You must select at least one detection model or upload one reference image.', color: "text-red" }
      });
    }
  }

  onCancel(): void {
    this.dialogRef.close(false);
  }
}
