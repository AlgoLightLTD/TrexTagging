import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { DeleteDetectionDatasetRequest, DeleteDetectionDatasetResponse, DetectionData, DetectionDataset, GetDetectionDatasetsRequest, GetDetectionDatasetsResponse } from 'src/app/shared/ServerCommunication';
import { TextPopupComponent } from 'src/app/shared/components/text-popup/text-popup.component';
import { DataSaverService } from 'src/app/shared/services/data-saver.service';
import { ServerService } from 'src/app/shared/services/server.service';

@Component({
  selector: 'app-detection-datasets',
  templateUrl: './detection-datasets.component.html',
  styleUrl: './detection-datasets.component.css',
  providers: [ServerService],
})
export class DetectionDatasetsComponent implements OnInit{
  detection_datasets: DetectionDataset[] = [];

  constructor(public dialog: MatDialog, private server: ServerService, public router: Router) {
  }

  ngOnInit(): void {
    this.refresh();
  }
  
  refresh(){
    let req: GetDetectionDatasetsRequest = new GetDetectionDatasetsRequest();
      this.server.getting = true;
      this.server.post(GetDetectionDatasetsResponse, req).subscribe(res => {
        this.detection_datasets = res.detection_datasets;
        this.server.getting = false;
    });
  }

  editDetectionDataset(DetectionDatasets_idx: number) {

  }

  copyDetectionDataset(DetectionDatasets_idx: number) {
    // let dialogRef = this.dialog.open(CopyDetectionDatasetsComponent,{});
  }

  showDetectionDatasetMetadata(DetectionDatasets_idx: number) {
  }

  deleteDetectionDataset(DetectionDatasets_idx: number) {
    let req: DeleteDetectionDatasetRequest = new DeleteDetectionDatasetRequest();
    req.dataset_id = this.detection_datasets[DetectionDatasets_idx].id;
    this.server.getting = true;
    this.server.post(DeleteDetectionDatasetResponse, req).subscribe(res => {
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