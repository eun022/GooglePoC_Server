import requests
from dot_api.config import BRAILLE_API_URL

url = BRAILLE_API_URL


def translate_to_japanese_braille(text: str) -> str:
    """
    일본어 TEXT → 점자로 변환하는 함수
    API를 호출하고 결과를 유니코드 점자로 반환한다.
    """
    
    data = {
        "LANGUAGE": "japanese",
        "OPTION": "1",
        "CELL": "20",
        "TEXT": text
    }

    response = requests.post(url, json=data)
    res_json = response.json()

    # BRAILLE_RESULT 가져오기
    braille_hex = res_json.get("BRAILLE_RESULT", "")
    braille_hex = braille_hex.replace(" ", "")
    print("TEXT:", text)
    #print("braille:", res_json)
    print("HEX:", braille_hex)


    return braille_hex
