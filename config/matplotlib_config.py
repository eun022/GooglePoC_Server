import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# NOTO 폰트 경로 자동검색
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# Matplotlib에 폰트 강제 등록
font_manager.fontManager.addfont(font_path)
rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

rcParams["axes.unicode_minus"] = False



# from matplotlib import font_manager, rcParams

# font_candidates = ["NanumGothic", "Noto Sans CJK KR", "UnDotum", "Malgun Gothic", "AppleGothic"]

# for font_name in font_candidates:
#     try:
#         path = font_manager.findfont(font_name, fallback_to_default=False)
#         if path and font_name in path:
#             rcParams["font.family"] = font_name
#             break
#     except Exception:
#         continue
# else:
#     rcParams["font.family"] = "DejaVu Sans"

# rcParams["axes.unicode_minus"] = False
