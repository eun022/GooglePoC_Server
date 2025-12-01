from fastapi import UploadFile, File, HTTPException, APIRouter, Form, Depends
import uuid
from datetime import datetime
import numpy as np
import json
import ast
import numpy as np
from PIL import Image
#from config.aimodels import mainModel
import ai.utils as utils
import uuid
from database.conn import get_db
from chart_def import get_draw
from sqlalchemy.orm import Session
from repository.ChartDAO import ChartDAO
from repository.QuestionDAO import QuestionDAO
from templates.file_manage import  read_file2img, read_png2b64, read_png2rgb, save_init_json, save_image, save_chart_json, read_file_json
from ai.services import get_chart_structure_by_text, get_chart_structure_by_IMG, chart_image_descriptor, scatter_chat, analyze_finger_positions, general_chart_chat
from dot_api import translate_to_japanese_braille
import requests
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, Depends




chart_type = ""
router = APIRouter()
request_id = ""


# 메모리 저장소 (실제 환경에서는 데이터베이스 사용)
uploaded_images = {}
conversations = {}

router = APIRouter()


@router.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "message": "FastAPI 서버가 정상적으로 실행 중입니다.",
        "timestamp": datetime.now().isoformat()
    }

async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: func(*args, **kwargs)
    )


BASE = "https://dev-saas.dotincorp.com"

@router.get("/load_file")
async def load_file_api(fileKey: str):
    print("📌 /load_file 요청 수신:", fileKey)

    url = f"{BASE}/drive-app/v1/dtm/images/{fileKey}/device/300/to-dtms"
    res = requests.get(url, verify=False)
    data = json.loads(res.text)

    items = data.get("DTMS_JSON", {}).get("items", [])

    pages = [
        {
            "page": x.get("page"),
            "name": x.get("graphic", {}).get("name", ""),
            "data": x.get("graphic", {}).get("data", ""),
            "plain": x.get("text", {}).get("plain", ""),
            "imageAttachNo": x.get("imageAttachNo")   # ✅ 추가된 부분
        }
        for x in items
    ]

    return {
        "type": "load_file_result",
        "pages": pages
    }


import base64
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@router.get("/get_image_base64")
def get_image_base64(no: int):
    url = f"https://dev-saas.dotincorp.com/sys-app/v1/sites/S220727001/attach/{no}/files/1"

    try:
        res = requests.get(url, verify=False)  # ★ SSL 검증 비활성화

        if res.status_code != 200:
            return {"error": f"이미지 요청 실패: {res.status_code}"}

        encoded = base64.b64encode(res.content).decode('utf-8')

        return {
            "base64": f"data:image/jpeg;base64,{encoded}"
        }

    except Exception as e:
        return {"error": str(e)}



import time



@router.post("/sa2va")
async def analyze_image(
    file: UploadFile = File(...),
    ours_or_baseline: str = Form(...),
    userid: str = Form(...),
    db: Session = Depends(get_db)
):
    T0 = time.time()

    # UUID 생성
    request_id = uuid.uuid4().hex[:8]

    # 파일 읽기 (await OK)
    file_dict = await read_file2img(file)
    b64 = file_dict['b64']
    image = file_dict['rgb']


    text = await run_blocking(get_chart_structure_by_IMG, b64, request_id)


    if text is None:
        print("텍스트 없음")
        return {"error": "분석 실패"}

    # 아래는 기존 순서 그대로 실행 (순서 절대 안 깨짐)
    content_json = json.loads(text)
    save_init_json(request_id, content_json)

    output_data = content_json
    chart_type = output_data.get("chart_type", {}).get("type", "None")

    chart_data = {
        'userid': userid,
        "uid": request_id,
        "type": ours_or_baseline,
        "chart_type": chart_type,
        "query_count": 0,
    }
    # ChartDAO.create(db, chart_data)

    # 그리기 (전부 순서 유지)
    if chart_type == "scatter":
        resized_list = ...
    else:
        resize_function = get_draw().get(chart_type)
        resized_mask = resize_function(request_id)
        resized_list = np.round(resized_mask).astype(np.int32).tolist()

    legend = output_data.get("legend") or {}
    grid = utils.draw_legend_on_grid(legend)

    return {
        "resized_list": resized_list,
        "type": utils.chart_type_to_korean(chart_type),
        "legend": grid,
        "uuid": request_id,
    }




@router.post("/TextTactile")
async def TextTactile(
    ours_or_baseline: str = Form(...),
    userid: str = Form(...),
    db: Session = Depends(get_db),
    DataText: str = Form(...)):

    request_id = uuid.uuid4().hex[:8]


    text = get_chart_structure_by_text( DataText, request_id)
   
    if text is None:
        print("텍스트 없음")
    else:
        content_json = json.loads(text)
        save_init_json(request_id, content_json)
    output_data = content_json

    chart_type = output_data.get("chart_type", {}).get("type", "None")
    
    resize_function = get_draw().get(chart_type)
    resized_mask = resize_function(request_id)
    resized_list = np.round(resized_mask).astype(np.int32).tolist()

    legend = output_data.get("legend") or {}
    grid = utils.draw_legend_on_grid(legend)


    b64 = read_png2b64("img",request_id)
    Descript = await run_blocking(chart_image_descriptor, b64)

    return {
                "resized_list": resized_list,
                "type": utils.chart_type_to_korean(chart_type),
                "legend":grid,
                "uuid": request_id,
                "DS": Descript
            }
    

@router.post("/QArag")
async def QArag(
    ours_or_baseline: str = Form(...),
    userid: str = Form(...),
    db: Session = Depends(get_db),
    DataText: str = Form(...)):
    request_id = uuid.uuid4().hex[:8]

    answer, sources, api = qaSystem.answer_question(DataText)
    rag = answer

    text = get_chart_structure_by_text( answer, request_id)
    if text is None:
        print("텍스트 없음")
    else:
        content_json = json.loads(text)
        save_init_json(request_id, content_json)
    output_data = content_json


    chart_type = output_data.get("chart_type", {}).get("type", "None")

    resize_function = get_draw().get(chart_type)
    resized_mask = resize_function(request_id)
    resized_list = np.round(resized_mask).astype(np.int32).tolist()

    legend = output_data.get("legend") or {}
    grid = utils.draw_legend_on_grid(legend)


    b64 = read_png2b64("img", request_id)
    Descript = await run_blocking(chart_image_descriptor, b64)
    

    return {
                "resized_list": resized_list,
                "type": utils.chart_type_to_korean(chart_type),
                "legend":grid,
                "uuid": request_id,
                "DS": utils.clean_text(Descript),
                "rag": utils.clean_text(rag), 
            }


@router.post("/imageDS")
async def imageDS(file: UploadFile = File(...)):
    file_dict = await read_file2img(file)
    b64 = file_dict["b64"]
    Descript = await run_blocking(chart_image_descriptor, b64)

    return Descript


@router.post("/translate")
def translate_A(text: str = Form(...)):
    text_hex = translate_to_japanese_braille(text)


    result = {
                "resized_list": "",
                "text": text_hex,
            }
    return result

@router.post("/F3")
async def F3(request_id: str = Form(...), 
                   payload: str = Form(...),
                   messege: str = Form(...) ):
    
    payload_dict = json.loads(payload) 
    Descript = analyze_finger_positions(payload_dict, request_id, messege)

    result = utils.clean_text(Descript)
    # 언어 바꾼 후에 이 부분 수정 필요
    if  "정확한" not in result:
        items = [x.strip() for x in result.split(",")]
        braille =items[-1]
    else:
        braille = "다시 시도하세요"

    return {
            "text": result,
            "braille": braille
            }



@router.post("/chat")
async def chat_with_ai(file: UploadFile = File(None), message: str = Form(...), request_id: str = Form(), db: Session = Depends(get_db), state: str = Form(...)):
    """
    Chat 할때 사용
    """
    if state == "text":
        image = read_png2rgb("img", request_id)
        b64 = read_png2b64("img", request_id)
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="file is required when state='img'")
        file_dict = await read_file2img(file)
        b64 = file_dict["b64"]
        image = file_dict["rgb"]

    

    # model = mainModel.vlm
    # vlm = model['model']
    # tokenizer = model['token']



    


    chart_data = read_file_json("chartQA_data", request_id)
    chart_type = chart_data.get("chart_type", {}).get("type", "None")

    

    if chart_type == "scatter":
        # Axis Segmentation
        None
    else:
        # GPT-4o Image Chat
        text = await run_blocking(general_chart_chat, message, b64, request_id)
        print(message, text)
        chart_data = read_file_json("chartQA_data", request_id)
        chart_type = chart_data.get("chart_type", {}).get("type", "None")

        # # 함수 선택
        if chart_type not in get_draw():
            print(f"Unknown chart type: {chart_type}")
            #raise HTTPException(status_code=400, detail=f"Unknown chart type: {chart_type}")
        print(f"📊 차트 타입: {chart_type}")

        resize_function = get_draw().get(chart_type)
        resized_mask = resize_function(request_id)
        resized_list = np.round(resized_mask).astype(np.int32).tolist()

    # DB에 질문/답변 저장
    #chart_obj = ChartDAO.get(db, request_id)
    # if chart_obj:
    #     question = QuestionDAO.insert_by_chart_id(
    #         db,
    #         chart_id=chart_obj.id,
    #         content=message,      # 질문
    #         answer_content=text,  # 답변
    #         create_date=datetime.now(),

    #     )
    #     print(f"질문/답변 저장 완료: {question.id}")
    

    return {
                "resized_list": resized_list,
                "text": utils.clean_text(text),
                "type": utils.chart_type_to_korean(chart_type),

    }



@router.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """대화 기록 조회"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="대화 기록을 찾을 수 없습니다.")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]
    }
