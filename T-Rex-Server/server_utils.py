from glob import glob
import json
import os
import shutil

from fastapi.responses import JSONResponse

from config import *

def delete_paths(paths: list):
    for path in paths:
        if os.path.exists(path):
            # check if folder or files, delete accordicngly
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        else:
            return "Path not found"
    return ""

def get_dataset_info(info_file_path: str):
    if os.path.exists(info_file_path):
        with open(info_file_path, "r") as info_file:
            info = json.load(info_file)
    return info

def get_datasets(user_dir: str, prefix: str, json_filename:str = "info"):
    if not os.path.exists(user_dir):
        return []

    datasets = []
    for dataset_dir in glob(os.path.join(user_dir, f"*{prefix}*")):
        info_file_path = os.path.join(dataset_dir, f"{json_filename}.json")
        info = get_dataset_info(info_file_path)
        info['id'] = os.path.basename(dataset_dir).split("_")[-1]
        datasets.append(info)

    return datasets

def extract_data_zip_file(zip_file_path, base_data_dir, title: str = None, description: str = None):
    if not zip_file_path or not os.path.exists(zip_file_path) \
            or not os.path.isfile(zip_file_path) or not zip_file_path.endswith(".zip"):
        return "Dataset .zip file not found"
    
    data_dir = os.path.join(base_data_dir, "dataset")
    os.makedirs(data_dir, exist_ok=True)
    shutil.unpack_archive(zip_file_path, data_dir)

    delete_paths([zip_file_path])

    # save title and description in a JSON file
    if title or description:
        json_file_path = os.path.join(base_data_dir, "info.json")
        with open(json_file_path, "w") as info_file:
            json.dump({"title": title, "description": description}, info_file, indent=4)
    
    return None

def add_general_data(prefix: str, request_json: dict):
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    title = request_json.get('title')
    description = request_json.get('description')
    id = request_json.get('id')

    original_file_path = glob(os.path.join(UPLOAD_DIR, user_id, id+".*")) # should be path to zip file
    base_data_dir = os.path.join(UPLOAD_DIR, user_id, f"{prefix}{id}")

    to_ret = extract_data_zip_file(original_file_path[0], base_data_dir, title, description)

    if not to_ret:
        return JSONResponse(content={"code": 0, "message": ""})
    return JSONResponse(content={"code": 1, "message": to_ret})

def add_general_file(save_folder_name: str, request_json: dict, extra_data: list = None, extra_data_keys: list = None):
    user_id = request_json.get('user_id')
    session_id = request_json.get('session_id')
    title = request_json.get('title')
    description = request_json.get('description')
    id = request_json.get('id')

    original_file_path = glob(os.path.join(UPLOAD_DIR, user_id, id+".*")) # should be path to the file
    base_save_folder_path = os.path.join(UPLOAD_DIR, user_id, save_folder_name)
    os.makedirs(base_save_folder_path, exist_ok=True)
    new_file_path = os.path.join(base_save_folder_path, os.path.basename(original_file_path[0]))
    # move the file to the new location
    shutil.move(original_file_path[0], new_file_path)

    # save title and description in a JSON file
    json_file_path = os.path.join(base_save_folder_path, f"{id}.json")
    with open(json_file_path, "w") as info_file:
        extra_data_dict = {}
        if extra_data and extra_data_keys:
            for extra_data_key, extra_data in zip(extra_data_keys, extra_data):
                extra_data_dict[extra_data_key] = extra_data
        json.dump({"title": title, "description": description} | extra_data_dict, info_file, indent=4)

    return JSONResponse(content={"code": 0, "message": ""})

def get_all_json_files(folder_path):
    to_ret = []
    if os.path.exists(folder_path):
        json_files = glob(os.path.join(folder_path, "*.json"))
        for json_file in json_files:
            with open(json_file, "r") as f:
                json_data = json.load(f)
                to_ret.append(json_data | {"id": os.path.basename(json_file).split(".")[0]})
    return to_ret