import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { LoadingAnimationComponent } from './shared/components/loading-animation/loading-animation.component';
import { LoginComponent } from './login/login.component';
import { UserPanelComponent } from './user-panel/user-panel.component';
import { AngularFireModule } from '@angular/fire/compat';
import { AngularFireAuthModule } from '@angular/fire/compat/auth';
import { environment } from 'src/environment';
import { HomeComponent } from './user-panel/home/home.component';
import { TextPopupComponent } from './shared/components/text-popup/text-popup.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { DetectionDataComponent } from './user-panel/home/detection-data/detection-data.component';
import { ClassificationDatasetsComponent } from './user-panel/home/classification-datasets/classification-datasets.component';
import { DetectionDatasetsComponent } from './user-panel/home/detection-datasets/detection-datasets.component';
import { AddClassificationDatasetComponent } from './user-panel/home/classification-datasets/add-classification-dataset/add-classification-dataset.component';
import { AddDetectionDataComponent } from './user-panel/home/detection-data/add-detection-data/add-detection-data.component';
import { GenerateDetectionDatasetComponent } from './user-panel/home/detection-data/generate-detection-dataset/generate-detection-dataset.component';
import { AddModelComponent } from './user-panel/home/models/add-model/add-model.component';
import { ModelsComponent } from './user-panel/home/models/models.component';
import { TrainNewModelComponent } from './user-panel/home/train-new-model/train-new-model.component';
import { BoundingBoxEditorComponent } from './shared/components/bounding-box-editor/bounding-box-editor.component';
import { FileUrlPipe } from './shared/pipes/file-url.pipe';

@NgModule({
    declarations: [
        AppComponent,
        LoadingAnimationComponent,
        LoginComponent,
        UserPanelComponent,
        HomeComponent,
        TextPopupComponent,
        ClassificationDatasetsComponent,
        DetectionDataComponent,
        AddClassificationDatasetComponent,
        DetectionDatasetsComponent,
        AddDetectionDataComponent,
        GenerateDetectionDatasetComponent,
        ModelsComponent,
        AddModelComponent,
        TrainNewModelComponent,
        BoundingBoxEditorComponent,
        FileUrlPipe
    ],
    providers: [],
    bootstrap: [AppComponent],
    imports: [
        BrowserModule,
        AppRoutingModule,
        FormsModule,
        HttpClientModule,
        AngularFireModule.initializeApp(environment.firebaseConfig),
        AngularFireAuthModule,
        BrowserAnimationsModule
    ]
})
export class AppModule { }
