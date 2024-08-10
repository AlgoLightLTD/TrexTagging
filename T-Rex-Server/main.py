from glob import glob
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
import uuid
import base64

from config import *
from server_utils import add_general_data, add_general_file, delete_paths, get_all_json_files, get_datasets

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/Base")
async def Base(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    
    return JSONResponse(content={"code": 0, "message": ""})

@app.post("/CreateFile")
async def CreateFile(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    file_extension = request_json.get('file_extension')
    
    # Perform your logic here
    file_id = str(uuid.uuid4())

    user_folder_path = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_folder_path, exist_ok=True)

    # create empty file
    with open(os.path.join(user_folder_path, file_id+f".{file_extension}"), "w+") as final_file:
        pass
    return JSONResponse(content={"file_id": file_id, "code": 0, "message": "File created"})

@app.post("/UploadFileInChunks")
async def UploadFileInChunks(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    file_id = request_json.get('file_id')
    chunk_index = request_json.get('chunk_index')
    total_chunks = request_json.get('total_chunks')
    chunk = request_json.get('chunk')
    file_extension = request_json.get('file_extension')
    
    chunks_dir = os.path.join(UPLOAD_DIR, user_id, file_id, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    chunk_file_path = os.path.join(chunks_dir, f"chunk_{chunk_index}")

    chunk_bytes = base64.b64decode(chunk)
    
    with open(chunk_file_path, "wb") as chunk_file:
        chunk_file.write(chunk_bytes)
    
    if chunk_index + 1 == total_chunks:
        user_folder_path = os.path.join(UPLOAD_DIR, user_id)
        # get file path with name file_id in user folder path
        final_file_path = glob(os.path.join(user_folder_path, file_id + '.' + file_extension))
        if not final_file_path:
            return JSONResponse(content={"file_id": file_id, "code": 1, "message": "File not found"})

        with open(final_file_path[0], "wb") as final_file:
            for i in range(total_chunks):
                with open(os.path.join(chunks_dir, f"chunk_{i}"), "rb") as chunk_file:
                    final_file.write(chunk_file.read())
        delete_paths([chunks_dir])
    
    return JSONResponse(content={"file_id": file_id, "code": 0, "message": "Chunk uploaded successfully"})


#### Classification Dataset ####
@app.post("/GetClassificationDatasets")
async def GetClassificationDatasets(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')

    user_dir = os.path.join(UPLOAD_DIR, user_id)
    datasets = get_datasets(user_dir, CLASSIFICATION_DATASET_PREFIX)

    return JSONResponse(content={"code": 0, "message": "", "classification_datasets": datasets})

@app.post("/AddClassificationDataset")
async def AddClassificationDataset(request: Request):
    request_json = await request.json()
    
    return add_general_data(CLASSIFICATION_DATASET_PREFIX, request_json)

@app.post("/DeleteClassificationDataset")
async def DeleteClassificationDataset(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    dataset_id = request_json.get('dataset_id')

    res = delete_paths([os.path.join(UPLOAD_DIR, user_id, f"{CLASSIFICATION_DATASET_PREFIX}{dataset_id}")])
    if res=="":
        return JSONResponse(content={"code": 0, "message": "Dataset deleted successfully"})
    return JSONResponse(content={"code": 1, "message": "Dataset not found"})


#### Detection Data ####
@app.post("/GetDetectionData")
async def GetDetectionData(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    
    user_dir = os.path.join(UPLOAD_DIR, user_id)
    datasets = get_datasets(user_dir, DETECTION_DATA_PREFIX)
    
    return JSONResponse(content={"code": 0, "message": "", "detection_data_units": datasets})

@app.post("/AddDetectionData")
async def AddDetectionData(request: Request):
    request_json = await request.json()
    
    return add_general_data(DETECTION_DATA_PREFIX, request_json)

@app.post("/DeleteDetectionData")
async def DeleteDetectionData(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    dataset_id = request_json.get('dataset_id')

    # check if the detection data folder exists using glob
    detection_data_folder_path = os.path.join(UPLOAD_DIR, user_id, f"*{DETECTION_DATA_PREFIX}{dataset_id}")
    detection_data_folder_path = glob(detection_data_folder_path)
    if not detection_data_folder_path:
        return JSONResponse(content={"code": 1, "message": "Detection data folder not found"})

    res = delete_paths(detection_data_folder_path)
    if res=='':
        return JSONResponse(content={"code": 0, "message": "Dataset deleted successfully"})
    return JSONResponse(content={"code": 1, "message": "Dataset not found"})

#### Models ####
@app.post("/GetModels")
async def GetModels(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    
    user_dir = os.path.join(UPLOAD_DIR, user_id)
    classification_models_path = os.path.join(user_dir, CLASSIFICATION_MODELS_FOLDER_NAME)
    detection_models_path = os.path.join(user_dir, DETECTION_MODELS_FOLDER_NAME)

    # get classification models
    classification_models = get_all_json_files(classification_models_path)
    # add model_type "classification" to all classification models
    for model in classification_models:
        model["model_type"] = "classification"
    
    # get detection models
    detection_models = get_all_json_files(detection_models_path)
    # add model_type "classification" to all classification models
    for model in detection_models:
        model["model_type"] = "detection"
    
    return JSONResponse(content={"code": 0, "message": "", "models": classification_models+detection_models})

@app.post("/AddModel")
async def AddModel(request: Request):
    request_json = await request.json()
    model_type = request_json.get('model_type')
    if model_type == "classification":
        save_folder_name = CLASSIFICATION_MODELS_FOLDER_NAME
        extra_data = [request_json.get('classification_idx_to_class_name'), "classification"]
        extra_data_key = ["classification_idx_to_class_name", "model_type"]
        return add_general_file(save_folder_name, request_json, extra_data, extra_data_key)
    
    elif model_type == "detection":
        save_folder_name = DETECTION_MODELS_FOLDER_NAME
        extra_data = ["detection"]
        extra_data_key = ["model_type"]
        return add_general_file(save_folder_name, request_json)
    
    return JSONResponse(content={"code": 1, "message": "Invalid model type"})

@app.post("/DeleteModel")
async def DeleteModel(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    model_id = request_json.get('model_id')

    # get all files with the model_id
    model_files = glob(os.path.join(UPLOAD_DIR, user_id, "**", f"{model_id}*"), recursive=True)
    if not model_files:
        return JSONResponse(content={"code": 1, "message": "Model not found"})
    
    res = delete_paths(model_files)
    if res=="":
        return JSONResponse(content={"code": 0, "message": "Model deleted successfully"})
    return JSONResponse(content={"code": 1, "message": "Model not found"})

#### Detection Dataset Process ####
# add reference image
@app.post("/AddReferenceImage")
async def AddReferenceImage(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    file_id = request_json.get('id')
    bounding_boxes = request_json.get('bounding_boxes')
    
    # Ensure the request has the necessary data
    if not user_id or not file_id or not bounding_boxes:
        return JSONResponse(content={"code": 1, "message": "Missing required fields"})

    # Create user directory if it doesn't exist
    user_folder_path = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_folder_path, exist_ok=True)

    # Verify the file exists
    file_path_pattern = os.path.join(user_folder_path, f"{file_id}.*")
    file_path = glob(file_path_pattern)
    if not file_path:
        return JSONResponse(content={"code": 1, "message": "File not found"})
    
    file_path = file_path[0]

    # Create reference images directory
    ref_images_folder_path = os.path.join(user_folder_path, REFERENCE_IMAGES_FOLDER_NAME)
    os.makedirs(ref_images_folder_path, exist_ok=True)

    # Move the file to the reference images folder
    new_file_path = os.path.join(ref_images_folder_path, os.path.basename(file_path))
    shutil.move(file_path, new_file_path)

    # Save bounding boxes to a JSON file in the same folder
    bounding_boxes_data = {"boxes": bounding_boxes}
    json_path = os.path.splitext(new_file_path)[0] + '.json'
    with open(json_path, 'w') as json_file:
        json.dump(bounding_boxes_data, json_file, indent=4)

    return JSONResponse(content={"code": 0, "message": "Reference image added and moved successfully"})

@app.post("/GenerateDetectionDatasetFromDetectionData")
async def GenerateDetectionDatasetFromDetectionData(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    title = request_json.get('title')
    description = request_json.get('description')
    data_id = request_json.get('data_id')
    classification_model_id = request_json.get('classification_model_id')
    detection_model_id = request_json.get('detection_model_id')
    reference_image_ids = request_json.get('reference_image_ids')

    yolo_model_path = os.path.join(UPLOAD_DIR, user_id, DETECTION_MODELS_FOLDER_NAME, f"{detection_model_id}.pt")
    classification_model_path = os.path.join(UPLOAD_DIR, user_id, CLASSIFICATION_MODELS_FOLDER_NAME, f"{classification_model_id}.pt")
    classification_idx_to_class_name = json.load(open(os.path.join(UPLOAD_DIR, user_id, CLASSIFICATION_MODELS_FOLDER_NAME, f"{classification_model_id}.json"), "r"))["classification_idx_to_class_name"]
    reference_images_data = []
    for reference_image_id in reference_image_ids:
        # reference_images_data (list): list which contains data about each reference image (dict with the keys "image_path" and "boxes" which represents the bounding boxes of the images).
        reference_image_path = os.path.join(UPLOAD_DIR, user_id, f"{REFERENCE_IMAGES_FOLDER_NAME}", f"{reference_image_id}.*")
        reference_image_path = glob(reference_image_path)
        if not reference_image_path:
            return JSONResponse(content={"code": 1, "message": f"Reference image not found {reference_image_id}"})
        
        # take first not json file
        for ref_path in reference_image_path:
            if not ref_path.endswith(".json"):
                reference_image_path = ref_path
                break

        reference_image_data_path = os.path.join(UPLOAD_DIR, user_id, f"{REFERENCE_IMAGES_FOLDER_NAME}", f"{reference_image_id}.json")
        if not os.path.exists(reference_image_data_path):
            return JSONResponse(content={"code": 1, "message": f"Reference image data not found {reference_image_id}"})
        
        reference_image_data = json.load(open(reference_image_data_path, "r"))
        reference_image_data["image_path"] = reference_image_path
        reference_images_data.append(reference_image_data)
    
    # check if models exist
    if not os.path.exists(yolo_model_path) and len(reference_images_data)==0:
        return JSONResponse(content={"code": 1, "message": "YOLO model not found and no reference images found"})
    
    if not os.path.exists(classification_model_path):
        return JSONResponse(content={"code": 1, "message": "Classification model not found"})
    
    detection_data_folder_path = os.path.join(UPLOAD_DIR, user_id, f"*{DETECTION_DATA_PREFIX}{data_id}")
    detection_data_folder_path = glob(detection_data_folder_path)
    if not detection_data_folder_path:
        return JSONResponse(content={"code": 1, "message": "Detection data folder not found"})
    
    detection_data_folder_path = detection_data_folder_path[0]

    if f"_{DETECTION_DATA_PREFIX}" in detection_data_folder_path:
        return JSONResponse(content={"code": 1, "message": "Detection data already processed or being processed"})
    
    # save a process_pipeline_data.json to the data folder with the relevant data (yolo_model_path, classification_model_path, classification_idx_to_class_name, reference_images_data)
    process_pipeline_data = {
        "title": title,
        "description": description,
        "yolo_model_path": yolo_model_path,
        "classification_model_path": classification_model_path,
        "classification_idx_to_class_name": classification_idx_to_class_name,
        "reference_images_data": reference_images_data
    }
    with open(os.path.join(detection_data_folder_path, 'process_pipeline_data.json'), 'w') as f:
        json.dump(process_pipeline_data, f, indent=4)

    processing_data_folder_path = os.path.join(UPLOAD_DIR, user_id, f"{WAITING_FOR_PROCESSING_PREFIX}{DETECTION_DATA_PREFIX}{data_id}")
    os.rename(detection_data_folder_path, processing_data_folder_path)
    
    return JSONResponse(content={"code": 0, "message": ""})

@app.post("/GetDetectionDatasets")
async def GetDetectionDatasets(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    
    user_dir = os.path.join(UPLOAD_DIR, user_id)
    datasets = get_datasets(user_dir, DETECTION_DATASET_PREFIX)
    # add status for the dataset ("")
    for dataset in datasets:
        dataset["status"] = ""
    
    waiting_datasets = get_datasets(user_dir, WAITING_FOR_PROCESSING_PREFIX+DETECTION_DATA_PREFIX, "process_pipeline_data")
    # add status for the datasets ("waiting")
    for dataset in waiting_datasets:
        dataset["status"] = "waiting"

    processing_datasets = get_datasets(user_dir, PROCESSING_PREFIX+DETECTION_DATA_PREFIX, "process_pipeline_data")
    # add status for the datasets ("processing")
    for dataset in processing_datasets:
        dataset["status"] = "processing"
    
    return JSONResponse(content={"code": 0, "message": "", "detection_datasets": datasets+waiting_datasets+processing_datasets})

@app.post("/DeleteDetectionDataset")
async def DeleteDetectionDataset(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    dataset_id = request_json.get('dataset_id')

    # check if the detection data folder exists using glob
    detection_data_folder_path = os.path.join(UPLOAD_DIR, user_id, f"*{DETECTION_DATASET_PREFIX}{dataset_id}")
    detection_data_folder_path = glob(detection_data_folder_path)
    if not detection_data_folder_path:
        return JSONResponse(content={"code": 1, "message": "Detection data folder not found"})
    
    res = delete_paths(detection_data_folder_path)
    if res=='':
        return JSONResponse(content={"code": 0, "message": "Dataset deleted successfully"})
    return JSONResponse(content={"code": 1, "message": "Dataset not found"})

#### Model Training ####
@app.post("/StartTraining")
async def StartTraining(request: Request):
    request_json = await request.json()
    
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    dataset_id = request_json.get('dataset_id')
    model_type = request_json.get('model_type')
    final_model_name = request_json.get('final_model_name')
    final_model_description = request_json.get('final_model_description')
    yolo_base_model = request_json.get('yolo_base_model')

    #### create .json file with the relevant data ####
    model_id = str(uuid.uuid4())
    model_info = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "yolo_base_model": yolo_base_model,
        "title": final_model_name,
        "description": final_model_description,
        "status": "waiting for training"
    }

    if model_type == "classification":
        save_folder_name = CLASSIFICATION_MODELS_FOLDER_NAME
    elif model_type == "detection":
        save_folder_name = DETECTION_MODELS_FOLDER_NAME
    
    with open(os.path.join(UPLOAD_DIR, user_id, save_folder_name, f"{WAITING_FOR_TRAINING_PREFIX}{model_id}.json"), "w") as model_info_file:
        json.dump(model_info, model_info_file, indent=4)
    
    return JSONResponse(content={"code": 0, "message": ""})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
