from fastapi import APIRouter, WebSocket
import requests
import json
from starlette.websockets import WebSocketState

router = APIRouter()

BASE = "https://dev-saas.dotincorp.com"
files = "/drive-app/v1/dtms/groups?COMP_NO=C251117001&PARENT_GROUP_NO=G251118095713224&DRIVER_KIND=P&USER_NO=U251114061831005"





@router.websocket("/ws/file_list")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        res = requests.get(f"{BASE}{files}", verify=False)
        data = json.loads(res.text)

        # 이름 + key 함께 전달
        items = [
            {
                "name": item.get("FILE_NAME"),
                "key": item.get("FILE_KEY")
            }
            for item in data.get("items", [])
        ]

        print("File List:", items)

        result = {
            "type": "load_groups_result",
            "items": items
        }

        await websocket.send_json(result)

    except Exception as e:
        print("WebSocket Error:", e)




