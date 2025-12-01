import numpy as np
import cv2
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import json
import os
import numpy as np
from glob import glob
import re
from typing import Optional, Tuple, List
from collections import Counter
from PIL import Image, ImageDraw
import io
from scipy.stats import gaussian_kde
import squarify  
import math
import random
#from config.aimodels import mainModel
import os
from typing import Dict, Optional
import base64, mimetypes, pathlib
from dot_api import translate_to_japanese_braille


def file_to_data_url(path: str) -> str:
    p = pathlib.Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def mark_point_on_image(
    payload_dict: Dict,
    request_id: str,
    base_dir: str = "static/binary",
    save_dir: str = "static/IMGdot",
    radius: Optional[int] = None,
    color: tuple = (255, 0, 0),  # 빨강
    outline: Optional[tuple] = None
) -> str:
    """
    payload_dict = {'x': 152, 'y': 86, 'roi_w': 391, 'roi_h': 256}
    request_id   = '...'
    base_dir     = 'static/binary'
    save_dir     = 'static/IMGdot'
    """
    # 원본 이미지 경로 확인
    img_path = os.path.join(base_dir, f"{request_id}.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

    # 파라미터 파싱
    try:
        x = int(payload_dict["x"])
        y = int(payload_dict["y"])
        roi_w = int(payload_dict["roi_w"])
        roi_h = int(payload_dict["roi_h"])
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"payload_dict 형식 오류: {e}")

    if roi_w <= 0 or roi_h <= 0:
        raise ValueError("roi_w, roi_h는 양의 정수여야 합니다.")

    # 출력 폴더 보장
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{request_id}.png")

    # 원본 이미지 열기 + 점 그리기
    with Image.open(img_path) as im:
        im = im.convert("RGBA")
        W, H = im.size
        draw = ImageDraw.Draw(im, "RGBA")

        # 좌표 변환
        scale_x = W / float(roi_w)
        scale_y = H / float(roi_h)
        X = round(x * scale_x)
        Y = round(y * scale_y)

        # 경계 보정
        X = max(0, min(W - 1, X))
        Y = max(0, min(H - 1, Y))

        # 점 크기
        if radius is None:
            radius = max(3, min(12, math.ceil(max(W, H) * 0.03)))

        # 점 그리기
        x0, y0 = X - radius, Y - radius
        x1, y1 = X + radius, Y + radius
        draw.ellipse([x0, y0, x1, y1], fill=color + (255,), outline=outline)

        # 파일로 저장
        im.save(save_path, format="PNG")

        # 메모리 버퍼에도 저장 → Data URL 반환
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        buffer.seek(0)

    # Data URL 변환
    mime = "image/png"
    b64 = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"



def braille_to_hex(s: str) -> str:
    """
    점자(⠿) 문자열을 유니코드 U+2800~U+28FF 코드포인트에서
    8비트 값(0~255)으로 변환 후 16진수 문자열로 반환.
    """
    out = []
    for ch in s:
        o = ord(ch)
        if 0x2800 <= o <= 0x28FF:
            val = o - 0x2800   # 이게 dot 비트 0~255
            out.append(f"{val:02X}")  # 2자리 hex
        else:
            out.append("00")   # 점자 아니면 00
    return "".join(out)


def braille_char_to_matrix(cell: str) -> np.ndarray:
    """유니코드 점자(U+2800~U+28FF)를 3x2(6점) 매트릭스로 변환"""
    cp = ord(cell)
    if cp < 0x2800 or cp > 0x28FF:
        return np.zeros((3, 2), dtype=int)

    bits = cp - 0x2800
    # 도트 비트: 1..6만 사용 (7,8은 필요 시 확장)
    d1 = 1 if (bits & (1 << 0)) else 0
    d2 = 1 if (bits & (1 << 1)) else 0
    d3 = 1 if (bits & (1 << 2)) else 0
    d4 = 1 if (bits & (1 << 3)) else 0
    d5 = 1 if (bits & (1 << 4)) else 0
    d6 = 1 if (bits & (1 << 5)) else 0

    return np.array([
        [d1, d4],
        [d2, d5],
        [d3, d6],
    ], dtype=int)



def braille_text_to_matrices_3x2(text: str):
    """
    점자 문자열을 '단어 단위' 매트릭스들로 변환.
    - 공백 전까지의 문자 점자(각 3x2)를 가로로 붙여 (3 x (2N + (N-1))) 하나의 매트릭스로 만듦
      (문자 사이에 3x1의 빈 칼럼을 1칸씩 삽입)
    - 공백(스페이스/개행/탭 등)에서 끊어 새로운 매트릭스로 시작
    - 연속 공백은 무시(빈 단어는 추가하지 않음)
    - braille_char_to_matrix(ch)는 3x2 numpy 배열을 반환한다고 가정
    """
    import numpy as np

    mats = []
    cur_cells = []  # 단어(공백 사이) 내부의 3x2 셀들

    def flush_current():
        nonlocal cur_cells
        if not cur_cells:
            return
        # 문자 사이에 3x1 zero spacer 삽입
        parts = []
        for i, cell in enumerate(cur_cells):
            M = np.array(cell, dtype=int)
            if M.shape != (3, 2):
                try:
                    M = M.reshape(3, 2)
                except Exception:
                    continue
            parts.append(M)
            if i != len(cur_cells) - 1:  # 마지막 문자 뒤에는 간격 X
                parts.append(np.zeros((3, 1), dtype=int))  # 1칸 간격
        if parts:
            word = np.hstack(parts) if len(parts) > 1 else parts[0]
            mats.append(word)
        cur_cells = []

    for ch in text:
        if ch.isspace():        # 공백류(스페이스/개행/탭 등)
            flush_current()     # 지금까지 쌓인 단어를 하나로 묶어 넣기
            continue
        M = braille_char_to_matrix(ch)
        cur_cells.append(M)

    # 마지막 단어 flush
    flush_current()

    return mats

def hex_to_braille_unicode(hex_str: str) -> str:
    """
    공백 없는 HEX 문자열일 경우 2자리씩 분리하여
    각각을 U+2800 점자로 변환
    """
    result = []

    # 🔥 공백 제거
    hex_str = hex_str.replace(" ", "")

    # 🔥 2자리씩 토큰화
    tokens = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]

    for tok in tokens:
        try:
            val = int(tok, 16)
        except ValueError:
            val = 0

        cp = 0x2800 + val
        result.append(chr(cp))

    return "".join(result)


def hex_to_matrix(hex_str: str) -> list:
    """
    공백 없는 HEX 문자열일 경우 2자리씩 분리하여
    각각을 3x2 점자 매트릭스로 변환
    """
    import numpy as np

    matrices = []

    # 🔥 공백 제거
    hex_str = hex_str.replace(" ", "")

    # 🔥 2자리씩 분할
    tokens = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]

    for tok in tokens:
        try:
            val = int(tok, 16)
        except ValueError:
            val = 0

        cp = 0x2800 + val  # 점자 코드포인트

        # 범위 밖이면 빈 매트릭스
        if cp < 0x2800 or cp > 0x28FF:
            matrices.append(np.zeros((3, 2), dtype=int))
            continue

        bits = val

        d1 = 1 if (bits & (1 << 0)) else 0
        d2 = 1 if (bits & (1 << 1)) else 0
        d3 = 1 if (bits & (1 << 2)) else 0
        d4 = 1 if (bits & (1 << 3)) else 0
        d5 = 1 if (bits & (1 << 4)) else 0
        d6 = 1 if (bits & (1 << 5)) else 0

        mat = np.array([
            [d1, d4],
            [d2, d5],
            [d3, d6],
        ], dtype=int)

        matrices.append(mat)

    return matrices



def draw_legend_on_grid(
    legend,
    H: int = 40,
    W: int = 60,
    top_row: int = 0,
    left_col: int = 0,              # 왼쪽에서 5칸 띄우려면 호출 시 left_col=5 로 전달
    col_gap_after_pattern: int = 3, # 패턴(2열) 뒤 점자 시작까지 간격
    char_gap: int = 1,              # 단어 블록 간 간격
    row_gap_between_items: int = 1, # 같은 블록 내 항목 간 세로 간격
    series_to_cats_gap: int = 2,    # 시리즈 줄에서 카테고리 첫 줄까지 간격
    category_indent_cols: int = 4   # 카테고리 들여쓰기(패턴 시작 열 기준)
):
    """
    - legend: dict 또는 list[dict]
      * dict: {"이름":[비트...], ...}  → 한 줄씩 나열
      * list[dict]: [{ 시리즈:[], 카테1:[], 카테2:[], ... }, {...}, ...]
                    → 각 dict의 '첫 key'는 시리즈, 나머지는 카테고리(들여쓰기)로 출력
    - 같은 key가 여러 번 나와도 '처음 등장한 패턴'을 고정해서 재사용.
    - 점자 변환은 모든 이름을 '단어 단위(공백으로 분할)' 3x2N 매트릭스 리스트로 전처리.
      => 단어들은 한 줄에서 이어 찍고, 가로가 모자라면 그때만 다음 줄로 내려간다.
    """
    import numpy as np

    grid = np.zeros((H, W), dtype=np.uint8)
    r = top_row

    # ---------------------- 공통 유틸 ----------------------
    def _flatten_bits(b):
        out = []
        def _rec(x):
            if isinstance(x, (list, tuple)):
                for xx in x: _rec(xx)
            else:
                out.append(x)
        _rec(b)
        return out

    def _ensure_even_bits(bits):
        # 중첩 허용 + int 변환 + 짝수 보정 (비면 기본값)
        flat = []
        for v in _flatten_bits(bits or []):
            try:
                flat.append(int(v))
            except:
                pass
        if not flat:
            flat = [0, 0, 1, 1]
        if len(flat) % 2 == 1:
            flat.append(0)
        return flat

    def _put_2xN_bits(r0, c0, bits):
        """2열 패턴(세로 N행)을 (r0,c0)부터 채운다."""
        b = _ensure_even_bits(bits)
        rows = len(b) // 2
        m = np.array(b, dtype=int).reshape(rows, 2)
        for rr in range(rows):
            for cc in range(2):
                rr_abs = r0 + rr
                cc_abs = c0 + cc
                if 0 <= rr_abs < H and 0 <= cc_abs < W and m[rr, cc]:
                    grid[rr_abs, cc_abs] = 1
        return rows  # 그린 세로높이(행)

    def _put_braille_sequence(r0, start_col, word_mats, char_gap_local=1):
        """
        word_mats: braille_text_to_matrices_3x2(text)가 반환한 '단어 단위' 매트릭스 리스트
                   - 각 원소는 (3 x 2N) ndarray
        정책:
          * 같은 줄에서 이어서 찍는다.
          * 남은 가로 여백이 부족할 때만 같은 항목 내에서 줄바꿈한다.
          * 줄바꿈 시 row += h + 1, c = start_col 로 이동.
        반환: 실제 사용 높이(행)
        """
        c = start_col
        row = r0
        used_top = r0
        used_bottom = r0  # exclusive

        for M in (word_mats or []):
            M = np.array(M, dtype=int)
            if M.size == 0:
                continue
            h, w = M.shape

            # 현재 줄에 이 단어가 안 들어가면, 같은 항목 내에서만 줄바꿈
            if c + w > W:
                row = row + h + 1   # 한 줄 비우고 개행
                c = start_col

            # 세로 초과면 중단
            if row + h > H:
                break

            # 단어 블록 찍기
            for rr in range(h):
                rr_abs = row + rr
                if 0 <= rr_abs < H:
                    for cc in range(w):
                        cc_abs = c + cc
                        if 0 <= cc_abs < W and M[rr, cc]:
                            grid[rr_abs, cc_abs] = 1

            # 같은 줄에서 계속 오른쪽으로 진행
            c += w + char_gap_local
            used_bottom = max(used_bottom, row + h)

        # 최소 높이 3 보장(점자 기본 높이)
        return max(3, used_bottom - used_top)

    # ------------------ ① 이름→'고정 패턴' 사전 구성 ------------------
    def _collect_name_bits_pairs(legend_obj):
        pairs = []  # [(name, bits)] in order
        if isinstance(legend_obj, dict):
            for k, b in legend_obj.items():
                pairs.append((str(k), b))
        elif isinstance(legend_obj, list):
            for block in legend_obj:
                if isinstance(block, dict) and block:
                    for k, b in block.items():  # 순서 유지(첫 key가 시리즈)
                        pairs.append((str(k), b))
        return pairs

    pairs = _collect_name_bits_pairs(legend)
    first_bits_for_name = {}  # name -> frozen(처음 본) 패턴
    for name, bits in pairs:
        if name not in first_bits_for_name:
            first_bits_for_name[name] = _ensure_even_bits(bits)

    # ------------------ ② 이름 전처리: 점자(단어 단위) 1회 변환 ------------------
    unique_names = []
    if isinstance(legend, dict):
        unique_names = [str(k) for k in legend.keys()]
    elif isinstance(legend, list):
        for block in legend:
            if isinstance(block, dict) and block:
                for k in block.keys():
                    unique_names.append(str(k))

    # 중복 제거(순서 유지)
    seen = set()
    ordered_names = []
    for nm in unique_names:
        if nm not in seen:
            seen.add(nm)
            ordered_names.append(nm)

    name_to_braille = {}
    for nm in ordered_names:
        # 외부 제공: mainModel.translate(nm) → 점자 문자열
        # braille_text_to_matrices_3x2: 공백 단위(단어)로 3x2N 매트릭스 리스트 생성
        btxt = translate_to_japanese_braille(nm)                 # 정확히 1회만 호출
        name_to_braille[nm] = hex_to_matrix(btxt)
        print("hex_to_braille_unicode", hex_to_braille_unicode(btxt))

    # ---------------- ③ 그리기 (고정 패턴 + 전처리된 점자 사용) ---------------
    # case A) legend: dict → 단순 나열
    if isinstance(legend, dict):
        for key, _orig_bits in legend.items():
            name = str(key)
            frozen_bits = first_bits_for_name.get(name, _ensure_even_bits(_orig_bits))

            pat_h = _put_2xN_bits(r, left_col, frozen_bits)
            start_c = left_col + 2 + col_gap_after_pattern

            text_h = _put_braille_sequence(
                r, start_c, name_to_braille.get(name, []), char_gap_local=char_gap
            )

            r += max(pat_h, text_h) + row_gap_between_items

        return grid.astype(int).tolist()

    # case B) legend: list[dict] → 첫 key=시리즈, 나머지=카테고리(들여쓰기)
    if isinstance(legend, list):
        for block in legend:
            if not isinstance(block, dict) or len(block) == 0:
                continue
            items = list(block.items())  # 순서 유지

            # (a) 시리즈 1줄
            series_name, series_bits_orig = items[0]
            series_name = str(series_name)
            series_bits = first_bits_for_name.get(series_name, _ensure_even_bits(series_bits_orig))

            pat_h = _put_2xN_bits(r, left_col, series_bits)
            start_c = left_col + 2 + col_gap_after_pattern

            text_h = _put_braille_sequence(
                r, start_c, name_to_braille.get(series_name, []), char_gap_local=char_gap
            )
            r += max(pat_h, text_h) + series_to_cats_gap

            # (b) 카테고리 n줄 (들여쓰기)
            for cat_name, cat_bits_orig in items[1:]:
                cat_name = str(cat_name)
                cat_bits = first_bits_for_name.get(cat_name, _ensure_even_bits(cat_bits_orig))

                pat_col = left_col + category_indent_cols
                pat_h = _put_2xN_bits(r, pat_col, cat_bits)

                start_c = pat_col + 2 + col_gap_after_pattern
                text_h = _put_braille_sequence(
                    r, start_c, name_to_braille.get(cat_name, []), char_gap_local=char_gap
                )

                r += max(pat_h, text_h) + row_gap_between_items

            # 블록 간 여백(시리즈 사이 2칸)
            r += 2

        return grid.astype(int).tolist()

    # 그 외 타입은 빈 그리드
    return grid.astype(int).tolist()


def chart_type_to_korean(chart_type: str) -> str:
    """
    차트 종류(영문) → 한국어 이름 변환
    """
    mapping = {
        "bar": "棒グラフ",
        "scatter": "散布図",
        "line": "折れ線グラフ",
        "pie": "円グラフ",
        "boxplot": "箱ひげ図",
        "violin": "バイオリンプロット",
        "treemap": "ツリーマップ",
        "mixed": "複合グラフ",
    }

    return mapping.get(chart_type, chart_type)  # 없는 경우 원래 문자열 반환

def clean_text(text: str, remove_chars="*-&\"") -> str:
    """
    text에서 remove_chars에 지정된 문자들을 제거하고 반환
    """
    return ''.join(ch for ch in text if ch not in remove_chars)


def extract_chart_type(text: str):
    # 찾을 차트 타입 목록
    chart_types = ['bar', 'line', 'pie', 'scatter']

    # 텍스트 끝에 있는 차트 타입 찾기
    pattern = r'(' + '|'.join(chart_types) + r')\s*$'
    match = re.search(pattern, text.strip())

    if match:
        chart_type = match.group(1)
        cleaned_text = text[:match.start()].strip()
    else:
        chart_type = None
        cleaned_text = text.strip()
    print('cleaned_text',  cleaned_text, 'chart_type', chart_type)
    return cleaned_text, chart_type


#---------------------------------------------------------------


