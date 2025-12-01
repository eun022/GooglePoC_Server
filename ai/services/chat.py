import json
import time
from google.genai import types
from google import genai
from templates.prompt import get_conversion, get_general_chart_QA, get_highlight, get_scatter_QA, get_dict
from templates.file_manage import read_file_json, save_chart_json, write_file_json  
from ai.models.clients.gemini_client import client
from config.gemini_api import MODEL

def scatter_chat(text: str, base64_image: str, rgb: str):


    system_prompt = get_scatter_QA(rgb)

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            # 시스템 프롬프트
            types.Part(text=system_prompt),
            # 유저 입력 텍스트
            types.Part(text=text),
            # 유저 입력 이미지
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/png",
                    data=base64_image
                )
            ),
        ],
        config=types.GenerateContentConfig(
          temperature=0.5
        ),
    )

    return response.text

import json
import re

def update_chart_type(request_id: str, new_chart_json_text: str):
    """
    변환된 JSON 또는 순수 문자열을 받아 chartQA_data["chart_type"]["type"]을 업데이트.
    new_chart_json_text: 모델이 반환한 response_json0.text
    """

    raw = new_chart_json_text.strip()

    # 1) Markdown 코드블록 제거
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    # 2) 양끝이 "문자열" 형태이면 제거
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    # 3) raw가 JSON처럼 보이면 JSON 파싱 시도
    # 예: {"chart_type": {"type": "treemap"}}
    if raw.startswith("{") and raw.endswith("}"):
        try:
            new_chart_data = json.loads(raw)
            new_type = new_chart_data.get("chart_type", {}).get("type")
            if not new_type:
                print("❌ JSON 내부에 chart_type.type 없음:", new_chart_data)
                return False
        except Exception as e:
            print("❌ JSON 파싱 실패:", e, raw)
            return False

    else:
        # 4) JSON이 아닌 경우 → raw 자체가 차트 타입 문자열이라고 간주
        # 예: treemap / "violin" / violin
        new_type = raw.strip()

        # 다만 문자열 길이가 비정상적으로 긴 경우는 잘못된 응답으로 간주
        if len(new_type) > 30:
            print("❌ JSON도 아니고, 타입 문자열로도 보기 어려움")
            print("입력값:", new_type)
            return False

    # 5) 기존 chartQA_data 로드 후 업데이트
    chartQA_data = read_file_json("chartQA_data", request_id)
    chartQA_data["chart_type"]["type"] = new_type

    # 6) 저장
    write_file_json("chartQA_data", request_id, chartQA_data)

    print(f"✅ chartQA_data 업데이트 완료 → 새 타입: {new_type}")
    return True




def general_chart_chat(text: str, base64_image: str, request_id: str):

    chart_json = read_file_json("chart_data", request_id)
    chartQA_data = read_file_json("chartQA_data", request_id)
    chart_type = chartQA_data["chart_type"]["type"]

    QA_prompt = get_general_chart_QA(chartQA_data, chart_type)

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            # system 역할
            types.Part(text=QA_prompt),
            # user text
            types.Part(text=text),
            # user image
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/png",  # 필요 시 image/jpeg
                    data=base64_image
                )
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0
        )
    )

    text = response.text.replace(" ", "")
    if any(keyword in text for keyword in ["conversion", "변환", "変換"]):
      system_prompt = get_dict("convert_chart_type")

      response_json0 = client.models.generate_content(
                model=MODEL,
                contents=[
                    # system 역할
                    types.Part(text=system_prompt),
                    # user text
                    types.Part(text=text),
                    # user image
                    
                ],
                  config=types.GenerateContentConfig(
                  temperature=0.0,
                  response_mime_type="application/json"  # JSON 강제
                  )
          )

      print("---conversion---", response_json0.text)
      update_chart_type(request_id, response_json0.text)


    #   system_prompt = get_conversion(text, chart_type, chart_json, response_json0.text)
      
    #   max_retries = 4
    #   for attempt in range(max_retries):
    #     try:

    #       response_json1 = client.models.generate_content(
    #             model=MODEL,
    #             contents=[
    #                 # system 역할
    #                 types.Part(text=system_prompt),
    #                 # user text
    #                 types.Part(text=text),
    #                 # user image
                    
    #             ],
    #               config=types.GenerateContentConfig(
    #               temperature=0.0,
    #               response_mime_type="application/json"  # JSON 강제
    #               )
    #       )


    #       if response_json1 is None or not response_json1.text:
    #             raise ValueError("응답이 None이거나 비어 있음")

    #         # JSON 파싱
    #       content_json = json.loads(response_json1.text)

    #         # 저장
    #       print(content_json)
    #       save_chart_json("chartQA_data", request_id, content_json)

    #       break

    #     except Exception as e:
    #         print(f"[재시도 {attempt+1}/{max_retries}] 오류: {e}")
    #         time.sleep(1) 



    print("---highlight---")
    system_prompt = get_highlight(chartQA_data, text)

    max_retries = 4
    for attempt in range(max_retries):
        try:
            response_json = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part(text=system_prompt),          # system 역할
                    types.Part(text=response.text.strip()),  # user 입력
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json"    # JSON 강제
                )
            )
            if response_json is None or not response_json:
                raise ValueError("응답이 None이거나 비어 있음")

            # JSON 파싱
            content_json = json.loads(response_json.text)

            # 저장
            save_chart_json("QA", request_id, content_json)

            break  # ✅ 성공했으니 루프 종료

        except Exception as e:
            print(f"[재시도 {attempt+1}/{max_retries}] 오류: {e}")
            time.sleep(1)




      # JSON 파일로 저장

    print(response.text)
      # 파일이 완전히 저장될 때까지 잠시 대기
    return response.text 
