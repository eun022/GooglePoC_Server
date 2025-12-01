# from scipy.ndimage import binary_opening, binary_closing, binary_dilation, binary_erosion
# import numpy as np
# import cv2
# from sklearn.cluster import KMeans

# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import json
# import os
# import numpy as np
# from glob import glob
# import re
# from typing import Optional, Tuple, List
# from collections import Counter
# from PIL import Image, ImageDraw
# import io
# from scipy.stats import gaussian_kde
# import squarify  
# import math
# import random
# #from config.aimodels import mainModel
# import os
# from typing import Dict, Optional
# import base64, mimetypes, pathlib


# def parse_wrapped_json(wrapper):
#     if "content" not in wrapper:
#         raise ValueError("content 필드가 없습니다.")
#     raw = wrapper["content"].strip()
#     # ```json\n ... \n``` 제거
#     cleaned = re.sub(r"^```json\n|```$", "", raw.strip(), flags=re.MULTILINE)
#     return json.loads(cleaned)

# def resize_keep_all(mask, target_shape):
#         H, W = mask.shape
#         target_H, target_W = target_shape

#         # 긴 쪽을 target에 맞추면 잘리므로, 짧은 쪽 기준으로 리사이즈
#         scale = min(target_H / H, target_W / W)
#         new_H, new_W = int(H * scale), int(W * scale)

#         # 리사이즈 (Nearest Interpolation)
#         resized = cv2.resize(mask.astype(np.uint8), (new_W, new_H), interpolation=cv2.INTER_NEAREST)

#         # 패딩 (여백을 균형 잡아 가운데 정렬)
#         pad_top = (target_H - new_H) // 2
#         pad_bottom = target_H - new_H - pad_top
#         pad_left = (target_W - new_W) // 2
#         pad_right = target_W - new_W - pad_left

#         final_mask = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)

#         return final_mask


# #산점도
# def parse_rgb_from_text(text: str) -> Tuple[str, List[Tuple[int, int, int]]]:
#     # 모든 RGB 패턴 찾기
#     matches = re.findall(r'\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)', text)

#     if not matches:
#         # RGB 표현이 없으면 원본 텍스트와 빈 리스트 반환
#         return text, []

#     # RGB 튜플 리스트로 변환
#     rgb_tuples = [tuple(map(int, match)) for match in matches]

#     # 텍스트에서 모든 RGB 패턴 제거
#     text_without_rgb = re.sub(r'\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\)', '', text)

#     # 공백과 구두점 정리
#     text_without_rgb = text_without_rgb.strip().rstrip('.,')

#     print("text_without_rgb:", text_without_rgb)
#     print("rgb_tuples:", rgb_tuples)

#     return text_without_rgb, rgb_tuples

# def visualize_color_similarity_from_json(request_id: str, targets: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
#     with open(f"static/color/{request_id}.json", 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     colors = [tuple(c) for c in data.get("colors", [])]

#     def cosine_similarity(a, b):
#         a = np.array(a, dtype=float)
#         b = np.array(b, dtype=float)
#         return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#     best_colors = []

#     if not targets:
#         # targets가 없으면 모든 색상을 best_colors에 추가
#         best_colors = colors.copy()
#     else:
#         # targets가 있으면 가장 유사한 색상 찾기
#         for target in targets:
#             similarities = [cosine_similarity(c, target) for c in colors]
#             most_similar_idx = int(np.argmax(similarities))
#             best_colors.append(colors[most_similar_idx])
#     print("best_colors: ", best_colors)
#     return best_colors

# def rgb_to_hex(color):
#     """RGB 색상을 16진수 코드로 변환"""
#     return ''.join(f"{v:02X}" for v in color)


# def extract_and_save_colors(image_rgb, request_id):
#     """
#     단일 이미지에서 주요 색상을 추출하고 JSON 파일로 저장 (고속 버전)
#     """
#     # HSV 변환 및 유효 마스크 생성 (흰색, 검은색, 회색 제거)
#     image_hsv = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2HSV)
#     h, s, v = cv2.split(image_hsv)
#     valid_mask = (s > 30) & (v > 50)

#     # 유효 픽셀만 추출 및 리스트 변환 (튜플화)
#     image_rgb = np.array(image_rgb)
#     masked_pixels = image_rgb[valid_mask]
#     if len(masked_pixels) == 0:
#         print(f":경고: 유효한 픽셀이 없습니다.")
#         return []

#     # 색상 빈도수 카운트 (Counter로 대체)
#     pixel_colors = [tuple(color) for color in masked_pixels]
#     color_counter = Counter(pixel_colors)

#     total_valid_pixels = len(masked_pixels)
#     min_pixel_count = total_valid_pixels * 0.01  # 1% 기준

#     # 1% 이상 색상만 추출
#     significant_colors = [color for color, count in color_counter.items() if count >= min_pixel_count]

#     if not significant_colors:
#         print(f":경고: 1% 이상을 차지하는 색상이 없습니다.")
#         return []

#     print(f":예술:: 대표 색상 {len(significant_colors)}개 추출")

#     # JSON 파일로 저장
#     color_list = [[int(c) for c in color] for color in significant_colors]
#     json_data = {
#         "colors": color_list
#     }
#     os.makedirs(os.path.dirname(f"static/color/{request_id}.json"), exist_ok=True)
#     with open(f"static/color/{request_id}.json", 'w', encoding='utf-8') as f:
#         json.dump(json_data, f, indent=2, ensure_ascii=False)
#     print(f"   :흰색_확인_표시: JSON 저장: {request_id}.json")

#     return significant_colors

# def convert_keys_to_str(mapping: dict) -> dict:
#     return {f"{k[0]},{k[1]},{k[2]}": v for k, v in mapping.items()}


# def visualize_and_resize_mask_dot(final_colors: list[tuple[int,int,int]],
#                                   request_id: str,
#                                       image_rgb,
#                                        mask_axis: np.ndarray,
#                                        color_distance_thresh: int = 30,
#                                        resize_shape=(40, 60)) -> np.ndarray:
#     image_rgb = np.array(image_rgb)
#     if not final_colors:
#         final_colors = extract_and_save_colors(image_rgb, request_id)
#     H, W = image_rgb.shape[:2]
#     pixels = image_rgb.reshape(-1, 3).astype(np.float32)
#     combined = np.zeros(len(pixels), dtype=bool)

    
#     for rgb in final_colors:
#         ref = np.array(rgb, dtype=np.float32)
#         mask_flat = np.linalg.norm(pixels - ref, axis=1) < color_distance_thresh
#         combined |= mask_flat

#     # STEP 2.5: 합쳐진 컬러 마스크
#     color_mask = combined.reshape(H, W)
    
#     # STEP 3: 원래 축 마스크와 병합
#     merged = np.logical_or(mask_axis, color_mask).astype(np.uint8)

#     # STEP 4: 후처리 (Opening → Closing → Dilation)
#     smooth = binary_opening(binary_closing(merged, np.ones((3,3))), np.ones((2,2)))
#     smooth = binary_dilation(smooth, np.ones((9,1))).astype(np.uint8)
#     smooth = binary_dilation(smooth, np.ones((1,7))).astype(np.uint8)

#     # STEP 5: 리사이즈
#     padded = np.pad(smooth, 2)
#     tensor = torch.tensor(padded, dtype=torch.float32)[None,None]
#     resized = F.interpolate(tensor, size=resize_shape, mode='nearest')[0,0].numpy().astype(np.uint8)


#     plt.figure(figsize=(6, 4))
#     plt.imshow(resized, cmap='gray', interpolation='nearest')
#     plt.axis("off")

#         # PNG로 저장
#     plt.savefig(f"static/binary/{request_id}.png", dpi=300, bbox_inches='tight', pad_inches=0)
#     return resized
