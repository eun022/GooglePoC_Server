import os
import json
import base64
from PIL import Image
import io

def save_chart_json(folder_name: str, uuid: str, content_json: dict):
    base_path = "static"  # 상위 기본 경로
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{uuid}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(content_json, f, ensure_ascii=False, indent=2)
    return 

def save_image(request_id: str, image):
    folder = "static/img"
    os.makedirs(folder, exist_ok=True)  # 폴더 없으면 자동 생성
    save_path = os.path.join(folder, f"{request_id}.png")
    image.save(save_path, format="PNG")
    return 

def save_init_json(request_id, content_json):
    save_chart_json("chart_data", request_id, content_json)
    save_chart_json("chartQA_data", request_id, content_json)
    
    content_json = {"highlight_mode": "all"}
    save_chart_json("QA", request_id, content_json)
    return 

def read_file_json(folder_name: str, uuid: str):
    with open(f"static/{folder_name}/{uuid}.json", "r", encoding="utf-8") as f:
        output_data = json.load(f)
    return output_data

def write_file_json(folder_name: str, uuid: str, data: dict):

    file_path = f"static/{folder_name}/{uuid}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def read_png2b64(folder_name: str, uuid: str):
    with open(f"static/{folder_name}/{uuid}.png", "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode('ascii')
    return b64

def read_png2rgb(folder_name: str, uuid: str):
    with open(f"static/{folder_name}/{uuid}.png", "rb") as f:
        data = f.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    return image

async def read_file2img(file):
    data = await file.read()
    file.file.seek(0)
    image = Image.open(file.file).convert("RGB")
    b64 = base64.b64encode(data).decode('ascii')
    return {"b64": b64, "rgb":image}


