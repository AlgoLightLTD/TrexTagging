interface ClassCountEntry {
    class_name: string;
    count: number;
}

export interface BaseDocument {
    owner_id: string;
    id: string;
}

export interface BaseData extends BaseDocument {
    title: string;
    description: string;
    id: string;
}
  
export interface ClassificationDataset extends BaseData {
    ClassCount?: ClassCountEntry[];
}

export interface DetectionData extends BaseData {
}

export interface DetectionDataset extends BaseData {
    status: string;
}

export interface Model extends BaseData {
    model_type: string;
    status: string;
}


// Bas requesnt and response
export interface BaseRequest {
    endpoint: string;
    user_id?: string;
    session_id?: string;
}

export class BaseResponse{
    public message: string = '';
    public code: number = 0;
}

// User requests and responses

// create file request and response
export class CreateFileRequest implements BaseRequest {
    public endpoint = 'CreateFile';
    public file_extension: string = '';
}
  
export class CreateFileResponse extends BaseResponse {
    public file_id: string = '';
}
  
export class UploadFileResponse extends BaseResponse {
    public file_id: string = '';
}  

export class UploadFileInChunksRequest implements BaseRequest {
    endpoint = 'UploadFileInChunks';
    public chunk: string = '';
    public file_id: string = '';
    public chunk_index: number = 0;
    public total_chunks: number = 0;
    public file_extension: string = '';
}
  
export class UploadFileInChunksResponse extends BaseResponse {
    public file_id: string = '';
}  

// Classification dataset requests and responses
export class GetClassificationDatasetsRequest implements BaseRequest{
    endpoint = 'GetClassificationDatasets';
}
export class GetClassificationDatasetsResponse extends BaseResponse{
    public classification_datasets: ClassificationDataset[] = [];
}

export class DeleteClassificationDatasetRequest implements BaseRequest{
    endpoint = 'DeleteClassificationDataset';

    public dataset_id: string =  "";
}
export class DeleteClassificationDatasetResponse extends BaseResponse{
}

export class AddClassificationDatasetRequest implements BaseRequest{
    endpoint = 'AddClassificationDataset';

    public title: string =  "";
    public description: string =  "";
    public id: string =  "";
}
export class AddClassificationDatasetResponse extends BaseResponse{
}


// Detection data requests and responses
export class GetDetectionDataRequest implements BaseRequest{
    endpoint = 'GetDetectionData';
}
export class GetDetectionDataResponse extends BaseResponse{
    public detection_data_units: DetectionData[] = [];
}

export class DeleteDetectionDataRequest implements BaseRequest{
    endpoint = 'DeleteDetectionData';

    public dataset_id: string =  "";
}
export class DeleteDetectionDataResponse extends BaseResponse{
}

export class AddDetectionDataRequest implements BaseRequest{
    endpoint = 'AddDetectionData';

    public title: string =  "";
    public description: string =  "";
    public id: string =  "";
}
export class AddDetectionDataResponse extends BaseResponse{
}


// Models request response
export class GetModelsRequest implements BaseRequest{
    endpoint = 'GetModels';
}
export class GetModelsResponse extends BaseResponse{
    public models: Model[] = [];
}

export class DeleteModelRequest implements BaseRequest{
    endpoint = 'DeleteModel';

    public model_id: string =  "";
}
export class DeleteModelResponse extends BaseResponse{
}

export class AddModelRequest implements BaseRequest{
    endpoint = 'AddModel';

    public title: string =  "";
    public description: string =  "";
    public id: string =  "";
    public model_type: string =  "";
    // dict style object, relates between classification index to class name
    public classification_idx_to_class_name: string[] = [];
}
export class AddModelResponse extends BaseResponse{
}

// Generate detection dataset from detection data requests and responses
//add reference image request and response
// Add reference image request and response
export class AddReferenceImageRequest implements BaseRequest {
    public endpoint = 'AddReferenceImage';
    public id: string = '';
    public bounding_boxes: number[][] = []; // Changed to a list of 4 int values
}
export class AddReferenceImageResponse extends BaseResponse {
}

export class GenerateDetectionDatasetFromDetectionDataRequest implements BaseRequest {
    endpoint = 'GenerateDetectionDatasetFromDetectionData';

    public title: string =  "";
    public description: string =  "";
    public data_id: string =  "";
    public detection_model_id: string =  "";
    public classification_model_id: string =  "";
    public reference_image_ids: string[] = [];
}
export class GenerateDetectionDatasetFromDetectionDataResponse extends BaseResponse {
}

// Detection dataset requests and responses
export class GetDetectionDatasetsRequest implements BaseRequest{
    endpoint = 'GetDetectionDatasets';
}
export class GetDetectionDatasetsResponse extends BaseResponse{
    public detection_datasets: DetectionDataset[] = [];
}

export class DeleteDetectionDatasetRequest implements BaseRequest{
    endpoint = 'DeleteDetectionDataset';

    public dataset_id: string =  "";
}
export class DeleteDetectionDatasetResponse extends BaseResponse{
}

export class GetDetectionDatasetStatisticsRequest implements BaseRequest{
    endpoint = 'GetDetectionDatasetStatistics';

    public dataset_id: string =  "";
}
export class GetDetectionDatasetStatisticsResponse extends BaseResponse{
    public title: string = '';
    public description: string = '';
    public class_count: ClassCountEntry[] = [];
}

// Training requests and responses
export class StartTrainingRequest implements BaseRequest {
    endpoint = 'StartTraining';

    public dataset_id: string = '';
    public model_type: string = '';
    public final_model_name: string = '';
    public final_model_description: string = '';
    public yolo_base_model: string = '';
}
export class StartTrainingResponse extends BaseResponse {
}
