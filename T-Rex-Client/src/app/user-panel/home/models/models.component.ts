import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { Model, DeleteModelRequest, DeleteModelResponse, GetModelsRequest, GetModelsResponse } from 'src/app/shared/ServerCommunication';
import { TextPopupComponent } from 'src/app/shared/components/text-popup/text-popup.component';
import { DataSaverService } from 'src/app/shared/services/data-saver.service';
import { ServerService } from 'src/app/shared/services/server.service';
import { AddModelComponent } from './add-model/add-model.component';

@Component({
  selector: 'app-models',
  templateUrl: './models.component.html',
  styleUrl: './models.component.css'
})
export class ModelsComponent implements OnInit{
  models: Model[] = [];

  constructor(public dialog: MatDialog, private server: ServerService, private router: Router, private DataSaver: DataSaverService) {}

  ngOnInit(): void {
    this.refresh();
  }
  
  refresh(){
    let req: GetModelsRequest = new GetModelsRequest();
      this.server.getting = true;
      this.server.post(GetModelsResponse, req).subscribe(res => {
        this.models = res.models;
        this.server.getting = false;
    });
  }

  addModel() {
    let dialogRef = this.dialog.open(AddModelComponent,{});
    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.refresh();
        // Handle the case when a new dataset is successfully added
      }
    });
  }

  showModelMetadata(Models_idx: number) {
  }

  deleteModel(Models_idx: number) {
    let req: DeleteModelRequest = new DeleteModelRequest();
    req.model_id = this.models[Models_idx].id;
    this.server.getting = true;
    this.server.post(DeleteModelResponse, req).subscribe(res => {
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