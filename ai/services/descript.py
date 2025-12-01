from google.genai import types
from google import genai
from templates.prompt import get_image_descript
from ai.models.clients.gemini_client import client
from config.gemini_api import MODEL

def chart_image_descriptor(base64_image: str) -> str:
    DS_prompt= get_image_descript()
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part(text=DS_prompt),  # 텍스트 프롬프트
            types.Part(                  # base64 이미지
                inline_data=types.Blob(
                    mime_type="image/png",
                    data=base64_image
                )
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0
        )
    )
    return response.text.strip()
    return  """ 今日はいい天気ですね。
コーヒーを一杯ください。
明日は休みです。
これは私のお気に入りです。
少し手伝ってくれますか？"""

