import time
from templates.prompt import get_chart_type_template_image, get_chart_type_template_text, get_dict
from google.genai import types
from google import genai
from ai.models.clients.gemini_client import client
from config.gemini_api import MODEL


def get_chart_structure_by_text(data_text: str, request_id: str) -> str:
    system_prompt = get_chart_type_template_text()
    max_retries = 4
    text = None
    for attempt in range(max_retries):
        try:
          response = client.models.generate_content(
              model=MODEL,
              contents=[
                  # 시스템 프롬프트 역할
                  types.Part(text=system_prompt),
                  types.Part(text=data_text), 
              ],
              config=types.GenerateContentConfig(
                  temperature=0.0,
                  response_mime_type="application/json"  # JSON 강제
              )
          )


          if response is None or not response.text:
                continue

          print(f"Saved to {request_id}.json")
          text = response.text
          
          break  
          

        except Exception as e:
            print(f"[재시도 {attempt+1}/{max_retries}] 오류: {e}")
            time.sleep(1) 

    return text




def extract_chart_data(base64_image, chart_type) -> dict:    # step2
    if chart_type in ["pie","treemap"]:
        system_prompt = get_dict(chart_type)
    else:
        system_prompt = get_dict("public") + get_dict(chart_type)
    print(system_prompt)
    max_retries = 4
    text = None
    
    for attempt in range(max_retries):
        try:
          response = client.models.generate_content(
              model=MODEL,
              contents=[
                  # 시스템 프롬프트 역할
                  types.Part(text=system_prompt),

                  # 유저 프롬프트
                
                  types.Part(
                      inline_data=types.Blob(
                          mime_type="image/png",
                          data=base64_image.split(",")[1] if base64_image.startswith("data:") else base64_image
                      )
                  ),
              ],
              config=types.GenerateContentConfig(
                  temperature=0.0,
                  response_mime_type="application/json"  # JSON 강제
              )
          )
          
          if response is None or not response.text:
                continue
          

          text = response.text
          break
        
        except Exception as e:
            print(f"[재시도 {attempt+1}/{max_retries}] 오류: {e}")
            time.sleep(1) 
        print(text)
    return text




def get_chart_structure_by_IMG(base64_image: str, request_id: str) -> str:

    system_prompt = get_dict("chart_type")
    max_retries = 4
    text = None
    
    for attempt in range(max_retries):
        try:
          response = client.models.generate_content(
              model=MODEL,
              contents=[
                  # 시스템 프롬프트 역할
                  types.Part(text=system_prompt),

                  # 유저 프롬프트
                  types.Part(text=""),  
                  types.Part(
                      inline_data=types.Blob(
                          mime_type="image/png",
                          data=base64_image.split(",")[1] if base64_image.startswith("data:") else base64_image
                      )
                  ),
              ]
              
          )
          
          if response is None or not response.text:
                continue
          

          print(f"Saved to {request_id}.json")
          print(response.text)
          text = extract_chart_data(base64_image, response.text) 
          break
        
        except Exception as e:
            print(f"[재시도 {attempt+1}/{max_retries}] 오류: {e}")
            time.sleep(1) 
    
    return text

# def get_chart_structure_by_IMG(base64_image: str, request_id: str) -> str:

#     system_prompt = get_chart_type_template_image()
#     max_retries = 4
#     text = None
    
#     for attempt in range(max_retries):
#         try:
#           response = client.models.generate_content(
#               model=MODEL,
#               contents=[
#                   # 시스템 프롬프트 역할
#                   types.Part(text=system_prompt),

#                   # 유저 프롬프트
#                   types.Part(text=""),  
#                   types.Part(
#                       inline_data=types.Blob(
#                           mime_type="image/png",
#                           data=base64_image.split(",")[1] if base64_image.startswith("data:") else base64_image
#                       )
#                   ),
#               ],
#               config=types.GenerateContentConfig(
#                   temperature=0.0,
#                   response_mime_type="application/json"  # JSON 강제
#               )
#           )
          
#           if response is None or not response.text:
#                 continue
          

#           print(f"Saved to {request_id}.json")
#           text = response.text
#           break
        
#         except Exception as e:
#             print(f"[재시도 {attempt+1}/{max_retries}] 오류: {e}")
#             time.sleep(1) 
    
#     return text
