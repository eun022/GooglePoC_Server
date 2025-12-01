import os, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import Rectangle
from PIL import Image
import squarify  # pip install squarify
import config.matplotlib_config #절대 지우기 금지


# ─────────────────────────────────────────────────────────────────────────────
# 하이라이트 유틸 (pie와 동일 규칙: all / custom / series)
# ─────────────────────────────────────────────────────────────────────────────
def compute_treemap_layout(series_names, categories_per_series, values_per_series,
                           W=60, H=40, series_gap=3, category_inset=1):
    import squarify
    layout = {"series_boxes": [], "cells": []}
    totals = [sum(vs) for vs in values_per_series]
    if sum(totals) <= 0:
        return layout

    series_rects = squarify.squarify(
        squarify.normalize_sizes(totals, W, H), 0, 0, W, H
    )

    for si, (r, sname, cats, vals) in enumerate(zip(series_rects, series_names, categories_per_series, values_per_series)):
        sx0 = max(0, int(r["x"]) + series_gap)
        sy0 = max(0, int(r["y"]) + series_gap)
        sx1 = min(W, int(r["x"] + r["dx"]) - series_gap)
        sy1 = min(H, int(r["y"] + r["dy"]) - series_gap)
        if sx1 <= sx0 or sy1 <= sy0: 
            continue

        layout["series_boxes"].append({"si": si, "name": sname, "x0": sx0, "y0": sy0, "x1": sx1, "y1": sy1})

        n = min(len(cats), len(vals))
        if n == 0 or sum(vals[:n]) <= 0:
            continue

        sub = squarify.squarify(
            squarify.normalize_sizes(vals[:n], sx1 - sx0, sy1 - sy0),
            sx0, sy0, sx1 - sx0, sy1 - sy0
        )
        for ci, (sr, cname, v) in enumerate(zip(sub, cats[:n], vals[:n])):
            x0 = max(0, int(sr["x"]) + category_inset)
            y0 = max(0, int(sr["y"]) + category_inset)
            x1 = min(W, int(sr["x"] + sr["dx"]) - category_inset)
            y1 = min(H, int(sr["y"] + sr["dy"]) - category_inset)
            if x1 <= x0 or y1 <= y0: 
                continue
            layout["cells"].append({"si": si, "ci": ci, "series": sname, "category": cname, "value": float(v),
                                    "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return layout


def make_highlight_fn(highlight_cfg, series_names, categories_per_series):
    """
    highlight_cfg:
      {"highlight_mode":"all"}
      {"highlight_mode":"custom","custom_indices":[{"series":"S","category":"C"}, ...]}
      {"highlight_mode":"series","custom_indices":[{"series":"S1"}, {"series":"S2"}]}
    """
    if not isinstance(highlight_cfg, dict):
        return lambda s, c: True

    mode = highlight_cfg.get("highlight_mode", "all")
    if mode == "all":
        return lambda s, c: True

    if mode == "custom":
        lst = highlight_cfg.get("custom_indices", []) or []
        want = {(d.get("series",""), d.get("category","")) for d in lst}
        return lambda s, c: (s, c) in want

    if mode == "series":
        want_series = { (d.get("series","")) for d in (highlight_cfg.get("custom_indices", []) or []) }
        return lambda s, c: s in want_series

    return lambda s, c: True


def build_pie_highlight_mask(series_names, categories_per_series, gpt):
    """시리즈/카테고리별 True/False 마스크 생성."""
    mode = (gpt or {}).get("highlight_mode", "all")
    mask = {s: {c: True for c in (categories_per_series[i] if i < len(categories_per_series) else [])}
            for i, s in enumerate(series_names)}

    if mode == "all":
        return mask

    if mode == "custom":
        mask = {s: {c: False for c in (categories_per_series[i] if i < len(categories_per_series) else [])}
                for i, s in enumerate(series_names)}
        for item in (gpt.get("custom_indices", []) or []):
            s = item.get("series", "")
            c = item.get("category", "")
            if s in mask and c in mask[s]:
                mask[s][c] = True
        return mask

    if mode == "series":
        mask = {s: {c: False for c in (categories_per_series[i] if i < len(categories_per_series) else [])}
                for i, s in enumerate(series_names)}
        want_series = { (item.get("series") or "") for item in (gpt.get("custom_indices", []) or []) }
        for i, s in enumerate(series_names):
            if s in want_series:
                for c in categories_per_series[i]:
                    mask[s][c] = True
        return mask

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Treemap 스펙 정규화
# ─────────────────────────────────────────────────────────────────────────────
def normalize_treemap_spec(data: dict):
    """
    입력 스키마:
    {
      "chart_type": {"type":"treemap"},
      "series":[{"name":"S1","categories":[{"name":"A","value":10}, ...]}, ...],
      "legend":[ {"S1":[bits], "A":[bits], ...}, {"S2":[bits], ...} ]
    }
    반환:
      series_names, categories_per_series, values_per_series, series_bits
    """
    assert data.get("chart_type", {}).get("type") == "treemap", "treemap만 지원합니다."

    series_defs = data.get("series", []) or []
    series_names = []
    categories_per_series = []
    values_per_series = []

    for s in series_defs:
        sname = s.get("name", "")
        series_names.append(sname)
        cats = s.get("categories")
        if not cats:
            # children → categories 변환
            ch = s.get("children")
            if isinstance(ch, list):
                cats = [{"name": c.get("name", ""), "value": c.get("value", 0)} for c in ch]
            else:
                cats = []
        # 단일 객체 보정: {"name":..,"value":..}
        if isinstance(cats, dict):
            cats = [cats]
        categories_per_series.append([c.get("name", "") for c in cats])
        values_per_series.append([float(c.get("value", 0.0)) for c in cats])

    # legend에서 "시리즈명" 패턴만 사용(트리맵 내부 채움용)
    series_bits = {}
    for leg in (data.get("legend") or []):
        if isinstance(leg, dict):
            for key, bits in leg.items():
                if key in series_names and isinstance(bits, list):
                    series_bits[key] = list(map(int, bits))

    return series_names, categories_per_series, values_per_series, series_bits


# ─────────────────────────────────────────────────────────────────────────────
# 패턴(2열 타일) 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _bits_to_tile(bits):
    if not isinstance(bits, (list, tuple)) or len(bits) == 0:
        bits = [1, 0]
    rows = max(1, len(bits)//2)
    use = list(bits[:rows*2])
    return np.array(use, dtype=np.uint8).reshape(rows, 2)

def _pattern_value(tile: np.ndarray, i: int, j: int) -> int:
    rows = max(1, tile.shape[0])
    return int(tile[i % rows, j % 2])


# ─────────────────────────────────────────────────────────────────────────────
# 60x40 트리맵 격자(0/1) 생성
# ─────────────────────────────────────────────────────────────────────────────
def build_raster_grid_treemap_bitmap(
    series_names, categories_per_series, values_per_series, series_bits, request_id,
    W: int = 60, H: int = 40,
    series_gap: int = 2,        # 서로 다른 시리즈 간 3칸
    category_inset: int = 1,    # 같은 시리즈 내 사각형은 양쪽 1칸(= 실제 간격 2칸)
    highlight_cfg: dict | None = None,
    highlight_mask: dict | None = None
):
    grid = np.zeros((H, W), dtype=np.uint8)

    layout = compute_treemap_layout(
        series_names, categories_per_series, values_per_series,
        W=W, H=H, series_gap=series_gap, category_inset=category_inset
    )

    if not layout["cells"] and not layout["series_boxes"]:
        return grid

    hi_fn = make_highlight_fn(highlight_cfg or {}, series_names, categories_per_series)

    # 셀 채우기(시리즈 패턴) + 테두리
    for cell in layout["cells"]:
        sname = cell["series"]; cname = cell["category"]
        x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]

        bits = series_bits.get(sname, [1, 0, 1, 0])
        tile = _bits_to_tile(bits)

        is_hi = (highlight_mask and highlight_mask.get(sname, {}).get(cname, False)) or hi_fn(sname, cname)
        if is_hi:
            grid[y0:y1, x0:x1] = 1 

        # 테두리는 항상
        grid[y0:y1, x0] = 1; grid[y0:y1, x1-1] = 1
        grid[y0, x0:x1] = 1; grid[y1-1, x0:x1] = 1

    # (선택) 시리즈 박스 윤곽만 그리고 싶다면 여기서 layout["series_boxes"] 이용 가능
    plt.figure(figsize=(6, 4))
    plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.axis("off")

        # PNG로 저장
    plt.savefig(f"static/binary/{request_id}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    return grid

# ─────────────────────────────────────────────────────────────────────────────
# PNG 저장(사람이 보기 쉽게 색상/라벨 포함) — 레전드 박스는 출력하지 않음
# ─────────────────────────────────────────────────────────────────────────────

def save_treemap_png(
    series_names, categories_per_series, values_per_series,
    mpl_png_path: str,
    W: int = 60, H: int = 40,
    series_gap: int = 3, category_inset: int = 1,
    cmap_name: str = "tab20", scale: float = 6.0,
    # 글자 크게!
    title_fs: float | None = None,
    empty_series_fs: float | None = None,
    label_size_factor: float = 1.6,  # 박스 짧은 변의 배율(↑)
    label_fs_max: float = 52,        # 상한(↑)
    label_fs_min: float = 18,        # 하한(↑)
    area_threshold: float = 3        # 매우 작은 칸만 생략
):
    """
    squarify로 배치(60x40 좌표) → Matplotlib 저장.
    - 사각형 테두리 없음(edgecolor='none')
    - 글자: 검정색, 크게. 칸을 넘치면 자동 축소.
    - 같은 시리즈는 같은 색.
    """
    import os
    import squarify
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def _wrap_label(s, max_chars=12):
        s = str(s)
        return s if len(s) <= max_chars else s[:max_chars] + "\n" + s[max_chars:]

    if title_fs is None:        title_fs = 22 * scale / 3
    if empty_series_fs is None: empty_series_fs = 22 * scale / 3

    totals = [sum(vs) for vs in values_per_series]
    if sum(totals) <= 0:
        fig, ax = plt.subplots(figsize=(3*scale, 2*scale))
        ax.axis("off"); plt.tight_layout()
        os.makedirs(os.path.dirname(mpl_png_path), exist_ok=True)
        plt.savefig(mpl_png_path, dpi=300); plt.close()
        return

    # ── 레이아웃(격자와 동일 좌표) ───────────────────────────────
    def compute_layout(W=60, H=40, series_gap=3, category_inset=1):
        layout = {"series_boxes": [], "cells": []}
        series_rects = squarify.squarify(
            squarify.normalize_sizes([sum(vs) for vs in values_per_series], W, H),
            0, 0, W, H
        )
        for si, (r, sname, cats, vals) in enumerate(zip(series_rects, series_names, categories_per_series, values_per_series)):
            sx0 = max(0, int(r["x"]) + series_gap)
            sy0 = max(0, int(r["y"]) + series_gap)
            sx1 = min(W, int(r["x"] + r["dx"]) - series_gap)
            sy1 = min(H, int(r["y"] + r["dy"]) - series_gap)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            layout["series_boxes"].append({"si": si, "name": sname, "x0": sx0, "y0": sy0, "x1": sx1, "y1": sy1})

            n = min(len(cats), len(vals))
            if n == 0 or sum(vals[:n]) <= 0:
                continue
            sub = squarify.squarify(
                squarify.normalize_sizes(vals[:n], sx1 - sx0, sy1 - sy0),
                sx0, sy0, sx1 - sx0, sy1 - sy0
            )
            for ci, (sr, cname, v) in enumerate(zip(sub, cats[:n], vals[:n])):
                x0 = max(0, int(sr["x"]) + category_inset)
                y0 = max(0, int(sr["y"]) + category_inset)
                x1 = min(W, int(sr["x"] + sr["dx"]) - category_inset)
                y1 = min(H, int(sr["y"] + sr["dy"]) - category_inset)
                if x1 <= x0 or y1 <= y0:
                    continue
                layout["cells"].append({
                    "si": si, "ci": ci, "series": sname, "category": cname, "value": float(v),
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1
                })
        return layout

    layout = compute_layout(W=W, H=H, series_gap=series_gap, category_inset=category_inset)

    # ── Figure ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3*scale, 2*scale))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # y축 아래로 증가(격자와 일치)
    ax.set_aspect("equal"); ax.axis("off")

    # 시리즈별 단일 색
    cmap = plt.get_cmap(cmap_name)
    S = max(1, len(series_names))
    series_colors = [cmap(int(round((i + 0.5) * (cmap.N / S))) % cmap.N) for i in range(S)]

    # 텍스트 자동 피팅(검정 글자, 박스 없음)
    def _draw_fitted_label(x0, y0, w, h, txt):
        p0 = ax.transData.transform((x0, y0))
        p1 = ax.transData.transform((x0 + w, y0 + h))
        max_w_px = abs(p1[0] - p0[0]) * 0.96
        max_h_px = abs(p1[1] - p0[1]) * 0.92

        fs = min(label_fs_max, max(label_fs_min, label_size_factor * min(w, h)))
        while fs >= label_fs_min:
            t = ax.text(x0 + w*0.5, y0 + h*0.5, txt,
                        ha="center", va="center",
                        fontsize=fs, color="black", fontweight="bold",
                        linespacing=1.0)
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            if bb.width <= max_w_px and bb.height <= max_h_px:
                return
            t.remove()
            fs *= 0.93
        ax.text(x0 + w*0.5, y0 + h*0.5, txt,
                ha="center", va="center",
                fontsize=label_fs_min, color="black", fontweight="bold",
                linespacing=1.0)

    # 사각형(테두리 없음) + 라벨
    for cell in layout["cells"]:
        x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
        w, h = (x1 - x0), (y1 - y0)
        if w <= 0 or h <= 0:
            continue
        face = series_colors[cell["si"]]
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=face, edgecolor="none", linewidth=0))
        if w * h >= area_threshold:
            _draw_fitted_label(x0, y0, w, h, f"{_wrap_label(cell['category'], 16)}\n{cell['value']:g}")

    # 시리즈 타이틀(검정 텍스트, 박스/테두리 없음)
    for sb in layout["series_boxes"]:
        sx0, sy0, sx1, sy1 = sb["x0"], sb["y0"], sb["x1"], sb["y1"]
        ax.text(sx0 + 1, sy1 - 1.2, sb["name"],
                ha="left", va="top",
                fontsize=title_fs, color="black", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(mpl_png_path), exist_ok=True)
    plt.savefig(mpl_png_path, dpi=300)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 엔드포인트: JSON 읽어 60×40 격자 + PNG 생성
# ─────────────────────────────────────────────────────────────────────────────
def treemap_single_highlight(request_id: str):
    """
    입력:
      - static/chartQA_data/{id}.json  (Treemap 데이터)
      - static/QA/{id}.json            (하이라이트 규칙: all/custom/series)
    동작:
      - 하이라이트 규칙 반영(강조=채움, 비강조=윤곽만) 60×40 격자 생성
      - PNG 저장: static/img/{id}.png (색상+라벨)
    반환:
      - 60×40 격자 (list[list[int]])
    """
    chart_fp = f"static/chartQA_data/{request_id}.json"
    qa_fp    = f"static/QA/{request_id}.json"
    mpl_out  = f"static/img/{request_id}.png"

    # 0) 로드
    try:
        with open(chart_fp, "r", encoding="utf-8") as f:
            chart_data = json.load(f)
    except Exception as e:
        print("⚠️ JSON 파일 로드 실패:", e)
        return []

    try:
        with open(qa_fp, "r", encoding="utf-8") as f:
            gpt = json.load(f)
    except Exception:
        gpt = {"highlight_mode": "all"}

    # 1) 스펙 정규화
    try:
        series_names, categories_per_series, values_per_series, series_bits = normalize_treemap_spec(chart_data)
    except AssertionError as e:
        print("⚠️ 스펙 오류:", e)
        return []

    # 2) 하이라이트 마스크
    hi_mask = build_pie_highlight_mask(series_names, categories_per_series, gpt)

    # 3) 격자 생성
    grid = build_raster_grid_treemap_bitmap(
        series_names, categories_per_series, values_per_series, series_bits,request_id,
        W=60, H=40,
        series_gap=1,
        category_inset=1,
        highlight_cfg=gpt,
        highlight_mask=hi_mask
    )

    # 4) PNG 저장(색상/라벨)
    save_treemap_png(
        series_names, categories_per_series, values_per_series,
        mpl_png_path=mpl_out,
        W=60, H=40,
        series_gap=1, category_inset=1,
        cmap_name="tab20", scale=6.0
    )

    arr = np.asarray(grid, dtype=np.uint8)
    print("shape:", arr.shape)  # (40, 60)
    return arr.astype(int).tolist()
