
from fastapi import APIRouter
from fastapi import UploadFile, File, Form
from PIL import Image
import numpy as np
from fastapi import Body, FastAPI, UploadFile, File, HTTPException, APIRouter, Form, Depends

#from config.aimodels import mainModel
from database.conn import get_db
from sqlalchemy.orm import Session
from domain.timer_log import TimerLogCreate
from repository.ChartDAO import ChartDAO
from repository.QuestionDAO import QuestionDAO
from repository.TimeLogDAO import TimeLogDAO
from datetime import datetime


router = APIRouter()

# @router.post("/chartqa")
# async def chartqa_endpoint(file: UploadFile = File(...), message: str = Form(...)):
#     """
#     ChartQA 엔드포인트
#     """
#     # 여기에 ChartQA 관련 로직을 추가하세요.

#     data = await file.read()
#     print("📥chat 받은 파일 이름:", message)
#     image = Image.open(file.file).convert("RGB")
#     output = mainModel.chartQA.generate(instruction=message, image=image)
#     print(output)

#     return {"message": "ChartQA 엔드포인트에 접근했습니다."}

# @router.post("/timer")
# async def timer(request_id: str = Form(), timeSpent: float = Form(), db: Session = Depends(get_db)):
#     """대화 기록 조회"""
#     chart_obj = ChartDAO.get(db, request_id)
#     chart_obj.total_time = timeSpent
#     db.commit()
#     db.refresh(chart_obj)

#     return {
#         "time_spent": chart_obj.total_time
#     }


@router.post("/timer")
async def timer(
    timer_create: TimerLogCreate, db: Session = Depends(get_db),
):
    """시간 로그 저장"""
    chart_obj = ChartDAO.get(db, timer_create.request_id)
    if chart_obj:
        print("성공")
        # TimeLog 생성
        time_log_data = {
            "chart_id": timer_create.request_id,
            "step_name": timer_create.step_name,
            "elapsed_time": timer_create.elapsed_time,
        }
        time_log = TimeLogDAO.create(db, time_log_data)
        print(f"시간 로그 저장 완료: {time_log.id}")
        
        return {
            "time_log_id": time_log.id,
        }
    else:
        raise HTTPException(status_code=404, detail="차트를 찾을 수 없습니다.")