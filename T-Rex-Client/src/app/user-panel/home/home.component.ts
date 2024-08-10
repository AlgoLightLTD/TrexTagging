import { Component, OnInit } from '@angular/core';
import { Router, NavigationEnd, ActivatedRoute } from '@angular/router';
import { filter } from 'rxjs/operators';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {
  selectedCategoryIndex: number = 0;
  categoryList: { title: string, path: string }[] = [
    { title: "Models", path: "models" },
    { title: "Classification Data", path: "classification-datasets" },
    { title: "Detection Data", path: "detection-data" },
    { title: "Detection Datasets", path: "detection-datasets" },
    { title: "Train New Model", path: "train-new-model" },
  ];

  constructor(private router: Router, private activatedRoute: ActivatedRoute) { }

  ngOnInit(): void {
    // Listen to route changes
    this.router.events.pipe(filter(event => event instanceof NavigationEnd)).subscribe(() => {
      this.updateSelectedCategoryIndex();
    });

    // Initialize the selected index on component load
    this.updateSelectedCategoryIndex();
  }

  updateSelectedCategoryIndex(): void {
    const currentPath = this.router.url.split('/').pop();
    this.selectedCategoryIndex = this.categoryList.findIndex(category => category.path === currentPath);
  }

  selectPart(index: number, path: string) {
    this.selectedCategoryIndex = index;
    this.router.navigate(['/user-panel/home/' + path]);
  }
}
