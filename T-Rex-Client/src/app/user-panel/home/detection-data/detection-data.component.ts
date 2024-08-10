import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { DeleteDetectionDataRequest, DeleteDetectionDataResponse, DetectionData, GenerateDetectionDatasetFromDetectionDataRequest, GenerateDetectionDatasetFromDetectionDataResponse, GetDetectionDataRequest, GetDetectionDataResponse } from 'src/app/shared/ServerCommunication';
import { TextPopupComponent } from 'src/app/shared/components/text-popup/text-popup.component';
import { DataSaverService } from 'src/app/shared/services/data-saver.service';
import { ServerService } from 'src/app/shared/services/server.service';
import { AddDetectionDataComponent } from './add-detection-data/add-detection-data.component';
import { GenerateDetectionDatasetComponent } from './generate-detection-dataset/generate-detection-dataset.component';

@Component({
  selector: 'app-detection-data',
  templateUrl: './detection-data.component.html',
  styleUrl: './detection-data.component.css',
  providers: [ServerService],
})
export class DetectionDataComponent implements OnInit{
  detection_data_units: DetectionData[] = [];

  constructor(public dialog: MatDialog, private server: ServerService, private router: Router, private DataSaver: DataSaverService) {}

  ngOnInit(): void {
    this.refresh();
  }
  
  refresh(){
    let req: GetDetectionDataRequest = new GetDetectionDataRequest();
      this.server.getting = true;
      this.server.post(GetDetectionDataResponse, req).subscribe(res => {
        this.detection_data_units = res.detection_data_units;
        this.server.getting = false;
    });
  }

  addDetectionData() {
    let dialogRef = this.dialog.open(AddDetectionDataComponent,{});
    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.refresh();
      }
    });
  }

  sendDetectionDataToDatasetProcess(DetectionData_idx: number){
    let detetction_data_id_value: string = this.detection_data_units[DetectionData_idx].id;
    let dialogRef = this.dialog.open(GenerateDetectionDatasetComponent, {
      data: { detection_data_id: detetction_data_id_value },
      width: '80vw',
      maxHeight: '90vh',
    });    
    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        // Handle the case when a new dataset is successfully added
      }
    });
  }

  editDetectionData(DetectionData_idx: number) {

  }

  copyDetectionData(DetectionData_idx: number) {
    // let dialogRef = this.dialog.open(CopyDetectionDataComponent,{});
  }

  showDetectionDataMetadata(DetectionData_idx: number) {
  }

  deleteDetectionData(DetectionData_idx: number) {
    let req: DeleteDetectionDataRequest = new DeleteDetectionDataRequest();
    req.dataset_id = this.detection_data_units[DetectionData_idx].id;
    this.server.getting = true;
    this.server.post(DeleteDetectionDataResponse, req).subscribe(res => {
      this.server.getting = false;

      if(res.code == 0){
        this.refresh();
      }
      else{
        let dialogRef = this.dialog.open(TextPopupComponent,{
          data:{
            msg: res.message,
            color: "text-red"
          }
        });
      }
    });
  }
}