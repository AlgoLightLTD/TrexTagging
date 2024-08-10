import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { LoginComponent } from './login/login.component';
import { denyGuestsGuard, denyUsersGuard } from './shared/guards/auth.guard';
import { LandingPageComponent } from './landing-page/landing-page.component';
import { UserPanelComponent } from './user-panel/user-panel.component';
import { HomeComponent } from './user-panel/home/home.component';
import { DetectionDataComponent } from './user-panel/home/detection-data/detection-data.component';
import { ModelsComponent } from './user-panel/home/models/models.component';
import { TrainNewModelComponent } from './user-panel/home/train-new-model/train-new-model.component';
import { ClassificationDatasetsComponent } from './user-panel/home/classification-datasets/classification-datasets.component';
import { DetectionDatasetsComponent } from './user-panel/home/detection-datasets/detection-datasets.component';

const routes: Routes = [
  { 
    path: '',
    component: LandingPageComponent,
    canActivate: [denyUsersGuard]
  },
  {
    path: 'login',
    component: LoginComponent,
    canActivate: [denyUsersGuard]
  },
  {
    path: 'user-panel',
    component: UserPanelComponent,
    canActivate: [denyGuestsGuard],
    children:[
      {
        path: '',
        redirectTo: 'home',
        pathMatch: 'full'
      },
      {
        path: 'home',
        component: HomeComponent,
        canActivate: [denyGuestsGuard],
        children:[
          {
            path: '',
            redirectTo: 'classification-datasets',
            pathMatch: 'full',
          },
          {
            path: 'classification-datasets',
            component: ClassificationDatasetsComponent,
            canActivate: [denyGuestsGuard],
          },
          {
            path: 'detection-data',
            component: DetectionDataComponent,
            canActivate: [denyGuestsGuard],
          },
          {
            path: 'detection-datasets',
            component: DetectionDatasetsComponent,
            canActivate: [denyGuestsGuard],
          },
          {
            path: 'models',
            component: ModelsComponent,
            canActivate: [denyGuestsGuard],
          },
          {
            path: 'train-new-model',
            component: TrainNewModelComponent,
            canActivate: [denyGuestsGuard],
          },
        ]
      },
    ]
  },  
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
