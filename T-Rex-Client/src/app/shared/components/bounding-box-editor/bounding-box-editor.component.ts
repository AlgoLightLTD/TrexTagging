import { Component, ElementRef, Input, Output, EventEmitter, ViewChild, AfterViewInit } from '@angular/core';

@Component({
  selector: 'app-bounding-box-editor',
  templateUrl: './bounding-box-editor.component.html',
  styleUrls: ['./bounding-box-editor.component.css']
})
export class BoundingBoxEditorComponent implements AfterViewInit {
  @Input() imageUrl: string = '';
  @Output() boundingBoxesChange = new EventEmitter<number[][]>();
  @ViewChild('baseCanvas', { static: false }) baseCanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('overlayCanvas', { static: false }) overlayCanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('canvasWrapper', { static: false }) canvasWrapperRef!: ElementRef<HTMLDivElement>;

  private baseCtx!: CanvasRenderingContext2D;
  private overlayCtx!: CanvasRenderingContext2D;
  private isDrawing = false;
  private startX = 0;
  private startY = 0;
  private boundingBoxes: number[][] = [];
  private hoveredBoxIndex: number | null = null;

  ngAfterViewInit(): void {
    if (this.baseCanvasRef && this.overlayCanvasRef && this.canvasWrapperRef) {
      this.loadImage();
    } else {
      console.error('ViewChild references are not available');
    }
  }

  private loadImage() {
    const img = new Image();
    img.src = this.imageUrl;
    img.onload = () => {
      this.drawImageOnCanvas(img);
    };
    img.onerror = (error) => {
      console.error('Error loading image:', error);
    };
  }

  private drawImageOnCanvas(img: HTMLImageElement) {
    const baseCanvas = this.baseCanvasRef.nativeElement;
    const overlayCanvas = this.overlayCanvasRef.nativeElement;

    baseCanvas.width = img.width;
    baseCanvas.height = img.height;
    overlayCanvas.width = img.width;
    overlayCanvas.height = img.height;

    this.baseCtx = baseCanvas.getContext('2d')!;
    this.overlayCtx = overlayCanvas.getContext('2d')!;
    if (!this.baseCtx || !this.overlayCtx) {
      console.error('Failed to get canvas 2D context');
      return;
    }

    this.baseCtx.drawImage(img, 0, 0, img.width, img.height);

    // Adjust the container height to match the image height
    this.canvasWrapperRef.nativeElement.style.height = `${img.height}px`;
  }

  onMouseDown(event: MouseEvent): void {
    const clickedBoxIndex = this.getHoveredBoxIndex(event.offsetX, event.offsetY);
    if (clickedBoxIndex !== null) {
      const [x, y, width, height] = this.boundingBoxes[clickedBoxIndex];
      if (event.offsetX >= x + width - 10 && event.offsetY <= y + 10) {
        this.removeBoundingBox(clickedBoxIndex);
        return;
      }
    }
    this.isDrawing = true;
    this.startX = event.offsetX;
    this.startY = event.offsetY;
  }

  onMouseMove(event: MouseEvent): void {
    if (!this.isDrawing) {
      this.hoveredBoxIndex = this.getHoveredBoxIndex(event.offsetX, event.offsetY);
      this.drawExistingBoundingBoxes();
      return;
    }

    const currentX = event.offsetX;
    const currentY = event.offsetY;
    const rectWidth = currentX - this.startX;
    const rectHeight = currentY - this.startY;

    this.overlayCtx.clearRect(0, 0, this.overlayCanvasRef.nativeElement.width, this.overlayCanvasRef.nativeElement.height);
    this.drawExistingBoundingBoxes();
    this.overlayCtx.strokeStyle = 'red';
    this.overlayCtx.lineWidth = 2;
    this.overlayCtx.strokeRect(this.startX, this.startY, rectWidth, rectHeight);
  }

  onMouseUp(event: MouseEvent): void {
    if (!this.isDrawing) return;

    this.isDrawing = false;
    const endX = event.offsetX;
    const endY = event.offsetY;
    const rectWidth = endX - this.startX;
    const rectHeight = endY - this.startY;
    this.boundingBoxes.push([this.startX, this.startY, rectWidth, rectHeight]);
    this.boundingBoxesChange.emit(this.boundingBoxes);
    this.drawExistingBoundingBoxes();
  }

  private getHoveredBoxIndex(mouseX: number, mouseY: number): number | null {
    for (let i = 0; i < this.boundingBoxes.length; i++) {
      const [x, y, width, height] = this.boundingBoxes[i];
      if (mouseX >= x && mouseX <= x + width && mouseY >= y && mouseY <= y + height) {
        return i;
      }
    }
    return null;
  }

  private removeBoundingBox(index: number) {
    this.boundingBoxes.splice(index, 1);
    this.boundingBoxesChange.emit(this.boundingBoxes);
    this.drawExistingBoundingBoxes();
  }

  private drawExistingBoundingBoxes() {
    this.overlayCtx.clearRect(0, 0, this.overlayCanvasRef.nativeElement.width, this.overlayCanvasRef.nativeElement.height);
    this.boundingBoxes.forEach((box, index) => {
      const [x, y, width, height] = box;
      this.overlayCtx.strokeStyle = 'red';
      this.overlayCtx.lineWidth = 2;
      this.overlayCtx.strokeRect(x, y, width, height);

      if (this.hoveredBoxIndex === index) {
        // Draw the 'X' for deletion
        this.overlayCtx.fillStyle = 'rgba(255, 0, 0, 0.5)';
        this.overlayCtx.fillRect(x + width - 10, y, 10, 10);
        this.overlayCtx.fillStyle = 'white';
        this.overlayCtx.fillText('X', x + width - 8, y + 8);
      }
    });
  }
}
