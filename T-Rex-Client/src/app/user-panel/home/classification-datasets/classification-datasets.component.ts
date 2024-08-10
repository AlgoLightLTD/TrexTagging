import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { ClassificationDataset, DeleteClassificationDatasetRequest, DeleteClassificationDatasetResponse, GetClassificationDatasetsRequest, GetClassificationDatasetsResponse } from 'src/app/shared/ServerCommunication';
import { TextPopupComponent } from 'src/app/shared/components/text-popup/text-popup.component';
import { DataSaverService } from 'src/app/shared/services/data-saver.service';
import { ServerService } from 'src/app/shared/services/server.service';
import { AddClassificationDatasetComponent } from './add-classification-dataset/add-classification-dataset.component';
import { CopyClassificationDatasetComponent } from './copy-classification-dataset/copy-classification-dataset.component';

@Component({
  selector: 'app-classification-datasets',
  templateUrl: './classification-datasets.component.html',
  styleUrl: './classification-datasets.component.css',
  providers: [ServerService],
})
export class ClassificationDatasetsComponent implements OnInit{
  classification_datasets: ClassificationDataset[] = [];

  constructor(public dialog: MatDialog, private server: ServerService, private router: Router, private DataSaver: DataSaverService) {}

  ngOnInit(): void {
    this.refresh();
  }
  
  refresh(){
    let req: GetClassificationDatasetsRequest = new GetClassificationDatasetsRequest();
      this.server.getting = true;
      this.server.post(GetClassificationDatasetsResponse, req).subscribe(res => {
        this.classification_datasets = res.classification_datasets;
        this.server.getting = false;
    });
  }

  addClassificationDataset() {
    let dialogRef = this.dialog.open(AddClassificationDatasetComponent,{});
    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.refresh();
      }
    });
  }

  editClassificationDataset(classificationDataset_idx: number) {

  }

  copyClassificationDataset(classificationDataset_idx: number) {
    let dialogRef = this.dialog.open(CopyClassificationDatasetComponent,{});
  }

  showClassificationDatasetMetadata(classificationDataset_idx: number) {
  }

  deleteClassificationDataset(classificationDataset_idx: number) {
    let req: DeleteClassificationDatasetRequest = new DeleteClassificationDatasetRequest();
    req.dataset_id = this.classification_datasets[classificationDataset_idx].id;
    this.server.getting = true;
    this.server.post(DeleteClassificationDatasetResponse, req).subscribe(res => {
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