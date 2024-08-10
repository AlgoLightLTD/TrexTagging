import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { TextPopupComponent } from 'src/app/shared/components/text-popup/text-popup.component';
import { ClassificationDataset, DetectionDataset, GetClassificationDatasetsRequest, GetClassificationDatasetsResponse, GetDetectionDatasetsRequest, GetDetectionDatasetsResponse, StartTrainingRequest, StartTrainingResponse } from 'src/app/shared/ServerCommunication';
import { DataSaverService } from 'src/app/shared/services/data-saver.service';
import { ServerService } from 'src/app/shared/services/server.service';

@Component({
  selector: 'app-train-new-model',
  templateUrl: './train-new-model.component.html',
  styleUrls: ['./train-new-model.component.css']
})
export class TrainNewModelComponent implements OnInit {
  selected_model_type: string = '';
  selected_dataset_id: string = '';
  model_name: string = '';
  model_description: string = '';
  yolo_base_model: string = '';

  detection_datasets: DetectionDataset[] = [];
  classification_datasets: ClassificationDataset[] = [];
  yolo_base_models: string[] = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']; // Add your available YOLO base models here

  constructor(public dialog: MatDialog, private server: ServerService, private router: Router, private DataSaver: DataSaverService) {}

  ngOnInit(): void {
    this.refresh();
  }
  
  refresh() {
    const req: GetDetectionDatasetsRequest = new GetDetectionDatasetsRequest();
    this.server.getting = true;
    this.server.post(GetDetectionDatasetsResponse, req).subscribe(res => {
      this.detection_datasets = res.detection_datasets;
      this.server.getting = false;
    });

    const req2: GetClassificationDatasetsRequest = new GetClassificationDatasetsRequest();
    this.server.getting = true;
    this.server.post(GetClassificationDatasetsResponse, req2).subscribe(res => {
      this.classification_datasets = res.classification_datasets;
      this.server.getting = false;
    });
  }

  startTraining(): void {
    const req: StartTrainingRequest = new StartTrainingRequest();
    req.dataset_id = this.selected_dataset_id;
    req.model_type = this.selected_model_type;
    req.final_model_name = this.model_name;
    req.final_model_description = this.model_description;
    req.yolo_base_model = this.yolo_base_model + ".pt";
    this.server.post(StartTrainingResponse, req).subscribe(res => {
      if (res.code != 0) {
        this.dialog.open(TextPopupComponent, {
          data: { msg: res.message, color: "text-red" }
        });
      }
      else{
        this.router.navigate(['/user-panel/home/models']);
      }
    }); 
  }

  isFormValid(): boolean {
    // check if all fields are filled, yolo_base_model and class_names can be empty when classification
    if (this.selected_model_type === 'classification') {
      return !!this.selected_dataset_id && !!this.model_name && !!this.model_description;
    }
    return !!this.selected_dataset_id && !!this.model_name && !!this.model_description && !!this.yolo_base_model;
  }

  trackByIndex(index: number, item: any): any {
    return index;
  }
}
