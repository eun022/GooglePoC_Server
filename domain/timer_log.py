from pydantic import BaseModel
from typing import Optional, List


class TimerLogCreate(BaseModel):
    request_id : str
    step_name : str  # ISO format datetime string
    elapsed_time:  float    # ISO format datetime string


# # 요청/응답 모델
# class ChatRequest(BaseModel):
#     message: str
#     image_id: Optional[str] = None
#     conversation_id: Optional[str] = None


# class ChatResponseItem(BaseModel):
#     file_name: str
#     reply: str

# class ChatResponse(BaseModel):
#     responses: List[ChatResponseItem]


# class AnalysisRequest(BaseModel):
#     image_id: str

# class AnalysisResponse(BaseModel):
#     objects: list[str]
#     colors: list[str]
#     characteristics: str
#     confidence: float

# class StemContentRequest(BaseModel):
#     image_id: str
#     subject: Optional[str] = None
#     difficulty: str = "medium"