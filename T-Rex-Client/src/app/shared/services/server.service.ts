import { HttpClient } from '@angular/common/http';
import { Injectable, NgZone } from '@angular/core';
import { SERVER } from 'src/environment';
import { AuthService } from './auth.service';
import { plainToClass } from 'class-transformer';
import { catchError, map } from 'rxjs/operators';
import { Observable, of } from 'rxjs';
import { CreateFileRequest, CreateFileResponse, UploadFileInChunksRequest, UploadFileInChunksResponse, UploadFileResponse } from '../ServerCommunication';

@Injectable({
  providedIn: 'root'
})
export class ServerService {
  baseURL: string[] = SERVER;
  getting: boolean = false;

  constructor(private http: HttpClient, private auth: AuthService, private ngZone: NgZone) {}

  post<T>(type: new () => T, body: any) {
    const url: string = this.baseURL[0] + body.endpoint + this.baseURL[1];
    body.user_id = this.auth.userID;
    body.session_id = this.auth.sessionID;
    return this.http.post(url, body).pipe(
      map(data => plainToClass(type, data)),
      catchError(err => {
        let to_ret_data: T = { code: 1, message: err.message } as T;
        let to_ret: Observable<T> = of(to_ret_data);
        return to_ret;
      })
    );
  }

  async uploadFile(file: File): Promise<UploadFileResponse> {
    this.getting = true;
    const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB per chunk
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

    let createFileRequest = new CreateFileRequest();
    createFileRequest.file_extension = file.name.split('.').pop() || '';
    const response = await this.post(CreateFileResponse, createFileRequest).toPromise();
    
    if (!response || response.code !== 0) {
      this.getting = false;
      return { file_id: '', message: 'File upload failed', code: 1 } as UploadFileResponse;
    }

    const file_id = response.file_id;
    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
      let currentRequest = new UploadFileInChunksRequest();
      const start = chunkIndex * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);

      const chunkBase64 = await this.blobToBase64(chunk);

      currentRequest.file_id = file_id;
      currentRequest.chunk_index = chunkIndex;
      currentRequest.total_chunks = totalChunks;
      currentRequest.chunk = chunkBase64;
      currentRequest.file_extension = file.name.split('.').pop() || '';

      try {
        let currentResp = await this.post(UploadFileInChunksResponse, currentRequest).toPromise();
        if (currentResp && currentResp.code !== 0) {
            this.getting = false;
            return { file_id, message: currentResp.message, code: currentResp.code } as UploadFileResponse;
        }
        else if (!currentResp) {
            this.getting = false;
            return { file_id, message: 'File upload failed', code: 1 } as UploadFileResponse;
        }
      } catch (err) {
        this.getting = false;
        return { file_id, message: 'File upload failed', code: 1 } as UploadFileResponse;
      }
    
    }
    
    this.getting = false;
    return { file_id, message: 'File uploaded successfully', code: 0 } as UploadFileResponse;
  }

  private blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = () => {
        const base64data = reader.result as string;
        resolve(base64data.split(',')[1]);
      };
      reader.onerror = reject;
    });
  }
}