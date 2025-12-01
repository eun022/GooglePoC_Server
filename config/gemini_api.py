import os
from dotenv import load_dotenv

# 로컬 환경에서만 .env 로드
if os.environ.get("ENV") != "cloud":
    load_dotenv()

MODEL: str = "gemini-2.5-flash-preview-09-2025"
MODEL_RAG: str = "gemini-2.5-flash-preview-09-2025"

API_KEY: str | None = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY 환경변수가 설정되지 않았습니다. "
        "Cloud Run 콘솔 또는 로컬 .env 파일에서 GEMINI_API_KEY를 설정하세요."
    )

__all__ = ["MODEL", "MODEL_RAG", "API_KEY"]
