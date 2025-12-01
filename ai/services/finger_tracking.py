import ai.utils as utils
from google.genai import types
from google import genai

from templates.prompt import get_finger_positions
from templates.file_manage import read_file_json

from ai.models.clients.gemini_client import client
from config.gemini_api import MODEL

def analyze_finger_positions(payload_dict: dict, request_id: str, messege:str):
    chart_json = read_file_json("chart_data", request_id)
    system_prompt = get_finger_positions(chart_json)
    
    img1 = utils.file_to_data_url(f"static/img/{request_id}.png")
    img2 = utils.mark_point_on_image(payload_dict, request_id)

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part(text=system_prompt),  
            types.Part(text=messege),
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/png", 
                    data=img1.split(",")[1] if img1.startswith("data:") else img1
                )
            ),
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/png",
                    data=img2.split(",")[1] if img2.startswith("data:") else img2
                )
            ),
        ],
        
    )

    return response.text 