# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import config.matplotlib_config #절대 지우기 금지

# ─────────────────────────────────────────────────────────
# 0) 박스플롯 입력 표준화
# ─────────────────────────────────────────────────────────
def normalize_box_spec(data: dict):
    assert data.get("chart_type", {}).get("type", "") == "boxplot", "boxplot만 지원합니다."

    series = list(data.get("series", [])) or ["1"]
    categories = list(data.get("categories", [])) or ["1"]
    raw = data.get("data", {}) or {}
    legend = data.get("legend", {}) or {}

    # 축 파싱: axes.Y/axes.y → 없으면 range/interval 폴백
    ax = data.get("axes", {}) or {}
    if isinstance(ax, dict) and ("Y" in ax or "y" in ax):
        y_ax = ax.get("Y") or ax.get("y") or {}
        rng = y_ax.get("range", [0, 100])
        vstep = y_ax.get("interval", 10)
    else:
        rng = ax.get("range", [0, 100])
        vstep = ax.get("interval", 10)

    # 가드
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
        rng = [0, 100]
    vmin, vmax = float(rng[0]), float(rng[1])
    try:
        vstep = int(vstep)
    except Exception:
        vstep = 10

    # eff_data: 시리즈명 → (카테고리별 stats 리스트) 구조로 통일
    # 카테고리가 1개인 경우에도 리스트로 맞춤
    eff_data = {s: [] for s in series}
    # raw는 통상 s별이 아니라 s키를 최상위에 두는 구조(질문 예시) → 표준화
    # 예: {"male": {...}, "female": {...}}
    if set(raw.keys()) >= set(series):
        for s in series:
            stats = raw.get(s, {})
            eff_data[s] = [stats for _ in categories]
    else:
        # 혹시 카테고리 키 기반으로 들어올 가능성도 고려
        for s in series:
            row = []
            for c in categories:
                stats = (raw.get(c, {}) or {}).get(s, {})
                row.append(stats)
            eff_data[s] = row

    return categories, series, eff_data, legend, vmin, vmax, vstep


# ─────────────────────────────────────────────────────────
# 1) 하이라이트 마스크 (series×category)
# ─────────────────────────────────────────────────────────
def build_highlight_mask(series, categories, gpt_response: dict | None):
    """시리즈×카테고리 boolean 마스크 생성."""
    S, C = len(series), len(categories)
    mask = {s: [False]*C for s in series}
    mode = (gpt_response or {}).get("highlight_mode")

    if mode == "series":
        targets = {it.get("series") for it in (gpt_response.get("custom_indices") or [])}
        for s in series:
            if s in targets:
                mask[s] = [True]*C

    elif mode == "category":
        cats = set(gpt_response.get("categories") or [])
        for j, c in enumerate(categories):
            if c in cats:
                for s in series:
                    mask[s][j] = True

    elif mode == "custom":
        for it in (gpt_response.get("custom_indices") or []):
            s, c = it.get("series"), it.get("category")
            if (s in mask) and (c in categories):
                j = categories.index(c)
                mask[s][j] = True

    elif mode == "all":
        for s in series:
            mask[s] = [True]*C

    # 아무것도 없으면 all 폴백
    if not any(any(row) for row in mask.values()):
        for s in series:
            mask[s] = [True]*C
    return mask


# ─────────────────────────────────────────────────────────
# 2) 60×40 격자에 박스플롯 그리기
#    - 강조: 박스 채움 + 수염, median 줄은 0
#    - 비강조: 박스 테두리만, 수염 없음, median 줄은 1
# ─────────────────────────────────────────────────────────
def build_raster_grid_boxplot(
    categories, series, eff_data, legend, request_id,
    vmin: float, vmax: float, vstep: int,
    W: int = 60, H: int = 40,
    highlight_mask: dict | None = None
) -> np.ndarray:
    import numpy as np

    grid = np.zeros((H, W), dtype=np.uint8)

    # ── 레이아웃/축
    y_axis_col = 0
    x_axis_row = H - 6
    plot_top, plot_left, plot_right = 2, y_axis_col + 4, W - 1
    plot_bottom = x_axis_row
    grid[plot_top:x_axis_row+1, y_axis_col] = 1
    grid[x_axis_row, :] = 1

    # ── 눈금
    usable_height = (plot_bottom - plot_top + 1)
    usable_width  = (plot_right - plot_left + 1)
    # 기존: row_mark = min(H-1, x_axis_row + 1)
    row_mark_bottom = min(H-1, x_axis_row + 1)  # X축 '아래' 점
    row_mark_top    = max(0,    x_axis_row - 1)  # X축 '위'   점  ← 추가

    def clamp(x, lo, hi): return max(lo, min(hi, x))

    tick_vals = list(range(int(vmin)+vstep, int(vmax)+1, vstep))
    tick_rows = []
    for tv in tick_vals:
        t = (tv - vmin) / (vmax - vmin) if vmax > vmin else 0
        r = int(round(plot_bottom - t * usable_height))
        tick_rows.append(r)
    if plot_top not in tick_rows: tick_rows.append(plot_top)
    tick_rows = sorted({r for r in tick_rows if plot_top <= r <= x_axis_row - 1})
    for r in tick_rows:
        c0, c1 = y_axis_col + 1, min(y_axis_col + 2, plot_right)
        grid[r, c0:c1+1] = 1

    # 값→y
    def value_to_row(v: float) -> int:
        if vmax <= vmin: return plot_bottom
        t = (v - vmin) / (vmax - vmin)
        r = int(round(plot_bottom - t * usable_height))
        return clamp(r, plot_top, plot_bottom)

    # 헬퍼
    def hline(y, x0, x1, val=1):
        y = clamp(y, plot_top, plot_bottom)
        x0, x1 = clamp(x0, plot_left, plot_right), clamp(x1, plot_left, plot_right)
        if x0 <= x1: grid[y, x0:x1+1] = val
    def vline(x, y0, y1, val=1):
        x = clamp(x, plot_left, plot_right)
        y0, y1 = clamp(y0, plot_top, plot_bottom), clamp(y1, plot_top, plot_bottom)
        if y0 <= y1: grid[y0:y1+1, x] = val
    def box_fill(y0, y1, x0, x1, val=1):
        y0, y1 = clamp(y0, plot_top, plot_bottom), clamp(y1, plot_top, plot_bottom)
        x0, x1 = clamp(x0, plot_left, plot_right), clamp(x1, plot_left, plot_right)
        if y0 <= y1 and x0 <= x1: grid[y0:y1+1, x0:x1+1] = val
    def box_outline(y0, y1, x0, x1, val=1):
        hline(y0, x0, x1, val); hline(y1, x0, x1, val)
        vline(x0, y0, y1, val); vline(x1, y0, y1, val)

    # 배치
    C, S = len(categories), len(series)
    group_gap, inner_gap, min_slot_w = 3, 1, 3
    usable = usable_width - max(0, (C - 1) * group_gap)
    if usable <= C: return grid
    cat_slot = max(1, usable // C)
    ser_slot = max(min_slot_w, (cat_slot - max(0, (S - 1) * inner_gap)) // max(1, S))
    half_box = max(1, ser_slot // 3)
    half_box = min(half_box, max(1, (ser_slot // 2) - 1))

    # 하단 패턴
    def draw_bottom_pattern(center_col, bits):
        if not isinstance(bits, (list, tuple)): bits = [0,0,1,1]
        rows = 3 if len(bits) >= 6 else 2
        arr = np.array(list(bits[:rows*2]), dtype=np.uint8).reshape(rows, 2)
        patt_top = max(0, H - 3)
        for rr in range(rows):
            for cc in range(2):
                r = patt_top + rr
                c = center_col - 1 + cc
                if plot_left <= c <= plot_right and r < H and arr[rr, cc]:
                    grid[r, c] = 1

    # 수집용
    pattern_centers = []   # 모든 박스의 (cx, series_name) → 패턴은 여기에 대해 전부 찍음
    xdot_positions = []    # X축 점: C==1이면 각 박스, C>1이면 카테고리 그룹 중앙

    # 메인 루프 (모든 시리즈를 그림)
    x_cursor = plot_left
    for ci, cat in enumerate(categories):
        cat_left = x_cursor

        # 다중일 때 카테고리 그룹 중앙(밑점용)
        group_width  = S*ser_slot + (S-1)*inner_gap
        group_center = clamp(cat_left + group_width//2, plot_left, plot_right)
        if C > 1:
            xdot_positions.append(group_center)

        for si, s in enumerate(series):
            stats = eff_data.get(s, [None]*C)[ci] if ci < len(eff_data.get(s, [])) else None
            if not isinstance(stats, dict): continue
            try:
                v_min = float(stats["min"]); v_q1 = float(stats["Q1"])
                v_med = float(stats["median"]); v_q3 = float(stats["Q3"])
                v_max = float(stats["max"])
            except Exception:
                continue

            cx = cat_left + si*(ser_slot + inner_gap) + ser_slot//2
            x0, x1 = cx - half_box, cx + half_box
            y_min = value_to_row(v_min); y_q1 = value_to_row(v_q1)
            y_med = value_to_row(v_med); y_q3 = value_to_row(v_q3)
            y_max = value_to_row(v_max)
            yb0, yb1 = sorted((y_q1, y_q3))
            yw0, yw1 = sorted((y_min, y_max))

            # 패턴은 모든 박스에 찍기 위해 수집
            pattern_centers.append((clamp(cx, plot_left, plot_right), s))

            is_hi = True
            if highlight_mask and s in highlight_mask and ci < len(highlight_mask[s]):
                is_hi = bool(highlight_mask[s][ci])

            if is_hi:
                # 수염 세로선(박스 내부 통과 금지)
                if yw0 <= yb0-1: vline(cx, yw0, yb0-1, 1)
                if yb1+1 <= yw1: vline(cx, yb1+1, yw1, 1)
                # 박스 채움 + 중앙선 0
                box_fill(yb0, yb1, x0, x1, 1)
                hline(y_med, x0, x1, 0)
                # 캡
                cap_half = max(1, (x1 - x0) // 3)
                hline(yw0, cx - cap_half, cx + cap_half, 1)
                hline(yw1, cx - cap_half, cx + cap_half, 1)
            else:
                # 비강조: 테두리, 중앙선=1, 수염 = '세로선+캡'을 박스 바깥만
                box_outline(yb0, yb1, x0, x1, 1)
                hline(y_med, x0, x1, 1)
                cap_half = max(1, (x1 - x0) // 3)
                if yw0 <= yb0-1: vline(cx, yw0, yb0-1, 1)
                if yb1+1 <= yw1: vline(cx, yb1+1, yw1, 1)
                hline(yw0, cx - cap_half, cx + cap_half, 1)
                hline(yw1, cx - cap_half, cx + cap_half, 1)

            # 이상치(공통)
            for o in (stats.get("outliers") or []):
                if isinstance(o, (int, float)) and np.isfinite(o):
                    yo = value_to_row(float(o))
                    if plot_top <= yo <= plot_bottom:
                        grid[yo, clamp(cx, plot_left, plot_right)] = 1

        x_cursor = cat_left + cat_slot + group_gap

    # ── 패턴: 모든 박스에 찍기
    for x, s in pattern_centers:
        draw_bottom_pattern(x, legend.get(s, [0,0,1,1]))

    # ── X축 밑 점: 단일이면 시리즈별, 다중이면 카테고리별
    if C == 1:
        for x, _ in pattern_centers:
            grid[row_mark_bottom, x] = 1     # 아래 점 (기존)
            grid[row_mark_top,    x] = 1     # 위   점 (추가)
    else:
        for x in xdot_positions:
            grid[row_mark_bottom, x] = 1     # 아래 점 (기존)
            grid[row_mark_top,    x] = 1     # 위   점 (추가)

    # 안전망: X축
    grid[x_axis_row, :] = 1
    plt.figure(figsize=(6, 4))
    plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.axis("off")

        # PNG로 저장
    plt.savefig(f"static/binary/{request_id}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    return grid

# ─────────────────────────────────────────────────────────
# 3) 참고용 Matplotlib 박스플롯 저장
#    (시리즈×카테고리 → 카테고리별 그룹, 시리즈 배치)
# ─────────────────────────────────────────────────────────


def save_matplotlib_boxplot(categories, series, eff_data,
                            vmin: float, vmax: float, vstep: int,
                            out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    C, S = len(categories), len(series)

    gap_between_categories = 1.0       # 카테고리 간 간격
    width_total = 0.8                  # 카테고리 내 전체 폭
    box_width = width_total / S        # 시리즈별 박스 폭
    cap_half = box_width * 0.35        # 캡 절반폭

    colors = plt.cm.tab10.colors  # 10개 색상 팔레트

    positions = []
    labels = []

    for ci, cat in enumerate(categories):
        cat_center = ci * (width_total + gap_between_categories) + 1.0
        for si, s in enumerate(series):
            stats = eff_data.get(s, [None]*C)[ci]
            if not isinstance(stats, dict):
                continue

            try:
                q1, q3 = float(stats["Q1"]), float(stats["Q3"])
                med    = float(stats["median"])
                mn, mx = float(stats["min"]), float(stats["max"])
            except Exception:
                continue

            # 위치 계산: 카테고리 중앙에서 시리즈 위치로 이동
            pos = cat_center + (si - (S-1)/2) * box_width

            # 박스(Q1~Q3)
            ax.add_patch(plt.Rectangle((pos - box_width/2, q1),
                                       box_width, q3 - q1,
                                       facecolor=colors[si % len(colors)],
                                       edgecolor="black",
                                       linewidth=1.2,
                                       alpha=0.6,
                                       label=s if ci == 0 else None))  # 범례는 첫 카테고리에서만 추가

            # 수염
            ax.vlines(pos, mn, mx, colors=colors[si % len(colors)], linewidth=1.2)

            # 캡
            ax.hlines([mn, mx], pos - cap_half, pos + cap_half,
                      colors=colors[si % len(colors)], linewidth=1.2)

            # 중앙값
            ax.hlines(med, pos - box_width/2, pos + box_width/2,
                      colors="black", linewidth=1.2)

            # 이상치
            outs = [o for o in (stats.get("outliers") or [])
                    if isinstance(o, (int, float)) and np.isfinite(o)]
            if outs:
                ax.scatter([pos]*len(outs), outs,
                           s=15, c=[colors[si % len(colors)]],
                           edgecolors="black")

    # 카테고리 라벨
    cat_centers = [ci * (width_total + gap_between_categories) + 1.0 for ci in range(C)]
    ax.set_xticks(cat_centers)
    ax.set_xticklabels(categories)

    # y축 범위 및 눈금
    ax.set_ylim(vmin, vmax)
    if isinstance(vstep, (int, float)) and vstep > 0:
        yt = np.arange(vmin, vmax + 1e-9, vstep)
        if len(yt) == 0 or yt[-1] < vmax:
            yt = np.append(yt, vmax)
        ax.set_yticks(yt)

    # 스타일
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 범례
    ax.legend(title="Series")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────
# 4) 파이프라인: 파일 읽고 격자 리스트 반환 + 참고 PNG 저장
# ─────────────────────────────────────────────────────────
def boxplot_single_highlight(request_id: str):
    """
    입력:
      - static/chartQA_data/{id}.json  (박스플롯 데이터)
      - static/QA/{id}.json            (하이라이트 규칙)
    동작:
      - 하이라이트 규칙 반영한 60×40 격자 생성
      - 참고용 박스플롯 PNG를 static/img/{id}.png 저장
    반환:
      - 60×40 격자 리스트 (list[list[int]])
    """
    chart_fp = f"static/chartQA_data/{request_id}.json"
    qa_fp    = f"static/QA/{request_id}.json"
    png_out  = f"static/img/{request_id}.png"

    try:
        with open(chart_fp, "r", encoding="utf-8") as f:
            chart = json.load(f)
    except Exception as e:
        print("⚠️ JSON 파일 로드 실패:", e)
        return []

    try:
        with open(qa_fp, "r", encoding="utf-8") as f:
            gpt = json.load(f)
    except Exception:
        gpt = {"highlight_mode": "all"}

    # 1) 표준화
    cats, sers, eff_data, legend, vmin, vmax, vstep = normalize_box_spec(chart)

    # 2) 하이라이트 마스크
    hi_mask = build_highlight_mask(sers, cats, gpt)

    # 3) 격자 생성
    grid = build_raster_grid_boxplot(
        cats, sers, eff_data, legend,  
        request_id,
        vmin, vmax, vstep,
        W=60, H=40,
        highlight_mask=hi_mask
    )

    # 4) 참고 PNG
    save_matplotlib_boxplot(cats, sers, eff_data, vmin, vmax, vstep, png_out)

    arr = np.asarray(grid)
    print("shape:", arr.shape)                 # (40, 60)
    print("x-axis row idx:", 40-6)
    print("x-axis last 5:", arr[40-6, -5:])
    return grid.astype(int).tolist()
