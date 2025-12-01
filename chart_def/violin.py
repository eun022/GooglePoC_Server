# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import Patch
import config.matplotlib_config #절대 지우기 금지


# ─────────────────────────────────────────────────────────
# 0) 박스플롯 입력 표준화
# ─────────────────────────────────────────────────────────
def normalize_box_spec(data: dict):
    assert data.get("chart_type", {}).get("type", "") == "violin", "boxplot만 지원합니다."

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
def build_raster_grid_violin(
    categories, series, eff_data, legend, request_id,
    vmin: float, vmax: float, vstep: int,
    W: int = 60, H: int = 40,
    highlight_mask: dict | None = None
) -> np.ndarray:
    import numpy as np

    grid = np.zeros((H, W), dtype=np.uint8)

    # ── 레이아웃/축/눈금 (박스플롯과 동일)
    y_axis_col = 0
    x_axis_row = H - 6
    plot_top, plot_left, plot_right = 2, y_axis_col + 4, W - 1
    plot_bottom = x_axis_row
    grid[plot_top:x_axis_row+1, y_axis_col] = 1
    grid[x_axis_row, :] = 1

    usable_height = (plot_bottom - plot_top + 1)
    usable_width  = (plot_right - plot_left + 1)
    # 기존: row_mark = min(H-1, x_axis_row + 1)
    row_mark_bottom = min(H-1, x_axis_row + 1)  # X축 아래 점
    row_mark_top    = max(0,    x_axis_row - 1)  # X축 위   점  ← 추가

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

    # 값↔행 변환
    def value_to_row(v: float) -> int:
        if vmax <= vmin: return plot_bottom
        t = (v - vmin) / (vmax - vmin)
        r = int(round(plot_bottom - t * usable_height))
        return clamp(r, plot_top, plot_bottom)

    def row_to_value(r: int) -> float:
        # value_to_row의 역함수(근사)
        t = (plot_bottom - r) / max(1, usable_height)
        return vmin + t * (vmax - vmin)

    # 그리기 헬퍼
    def hline(y, x0, x1, val=1):
        y = clamp(y, plot_top, plot_bottom)
        x0, x1 = clamp(x0, plot_left, plot_right), clamp(x1, plot_left, plot_right)
        if x0 <= x1: grid[y, x0:x1+1] = val

    # 하단 패턴(그대로 재사용)
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

    def draw_line(x0, y0, x1, y1, val=1):
        x0 = clamp(x0, plot_left, plot_right)
        x1 = clamp(x1, plot_left, plot_right)
        y0 = clamp(y0, plot_top, plot_bottom)
        y1 = clamp(y1, plot_top, plot_bottom)
        dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            grid[y, x] = val
            if x == x1 and y == y1: break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy


    # 배치 폭 계산 (박스플롯과 동일)
    C, S = len(categories), len(series)
    group_gap, inner_gap, min_slot_w = 3, 1, 5
    usable = usable_width - max(0, (C - 1) * group_gap)
    if usable <= C: return grid
    cat_slot = max(1, usable // C)
    ser_slot = max(min_slot_w, (cat_slot - max(0, (S - 1) * inner_gap)) // max(1, S))
    max_half = max(1, ser_slot//2 - 1)      # 바이올린 최대 반폭
    mid_half = max(1, int(max_half*0.70))   # Q1~Q3 경계에서의 반폭(부드럽게)

    # 수집용
    pattern_centers = []   # (cx, series)
    xdot_positions  = []   # C>1이면 카테고리 그룹 중앙

    # 메인 루프
    x_cursor = plot_left
    for ci, cat in enumerate(categories):
        cat_left = x_cursor

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
            cx = clamp(cx, plot_left, plot_right)
            pattern_centers.append((cx, s))

            # 하이라이트 여부 (True=비우기, False=채우기)
            is_hi = True
            if highlight_mask and s in highlight_mask and ci < len(highlight_mask[s]):
                is_hi = bool(highlight_mask[s][ci])

            # y 범위 별 반폭 프로필(선형-삼각형 형태)
            y0 = value_to_row(v_max)
            y1 = value_to_row(v_min)
            prev_xL = None
            prev_xR = None
            for y in range(min(y0,y1), max(y0,y1)+1):
                v = row_to_value(y)
                if v < v_min or v > v_max: 
                    continue

                if v <= v_q1:   # min→Q1 : 0 → mid_half
                    w = 0.0 if v_q1==v_min else (v - v_min) / (v_q1 - v_min)
                    half = int(round((0.05 + 0.95*w) * mid_half))
                elif v >= v_q3: # Q3→max : mid_half → 0
                    w = 0.0 if v_max==v_q3 else (v_max - v) / (v_max - v_q3)
                    half = int(round((0.05 + 0.95*w) * mid_half))
                else:
                    # Q1→Q3 : 중간에서 최대(max_half), Q1/Q3에서 mid_half
                    span = (v_q3 - v_q1) if v_q3 > v_q1 else 1.0
                    w = 1.0 - 2.0*abs(v - v_med)/span    # -1..1 → 0..1
                    w = max(0.0, w)
                    half = int(round(mid_half + (max_half - mid_half) * w))

                half = max(1, min(half, max_half))
                xL, xR = cx - half, cx + half

    
                if is_hi:
                    
                    hline(y, xL, xR, 1)
                else:
                                    
                    profile = []  # [(y, xL_c, xR_c)]
                    y0 = value_to_row(v_max)
                    y1 = value_to_row(v_min)
                    y_top = min(y0, y1)
                    y_bot = max(y0, y1)

                    for y in range(y_top, y_bot + 1):
                        v = row_to_value(y)
                        # half 계산은 위와 동일하게:
                        if v <= v_q1:
                            w = 0.0 if v_q1 == v_min else (v - v_min) / (v_q1 - v_min)
                            half = int(round((0.05 + 0.95*w) * mid_half))
                        elif v >= v_q3:
                            w = 0.0 if v_max == v_q3 else (v_max - v) / (v_max - v_q3)
                            half = int(round((0.05 + 0.95*w) * mid_half))
                        else:
                            span = (v_q3 - v_q1) if v_q3 > v_q1 else 1.0
                            w = 1.0 - 2.0*abs(v - v_med)/span
                            w = max(0.0, w)
                            half = int(round(mid_half + (max_half - mid_half) * w))

                        half = max(1, min(half, max_half))
                        xL, xR = cx - half, cx + half
                        xL_c, xR_c = clamp(xL, plot_left, plot_right), clamp(xR, plot_left, plot_right)

                        # 채워서 실루엣을 만들고(연속성 확보)
                        hline(y, xL_c, xR_c, 1)
                        # 나중에 내부 비우기 위해 경계 저장
                        profile.append((y, xL_c, xR_c))

                    # 2) 내부만 깎아내기(빈 윤곽선 만들기)
                    #    thickness_inner = 1 이면 정확히 1픽셀 윤곽, 2 이상이면 좀 더 두껍게
                    thickness_inner = 2
                    for y, xL_c, xR_c in profile:
                        inner_left  = xL_c + thickness_inner
                        inner_right = xR_c - thickness_inner
                        if inner_left <= inner_right:
                            # 내부를 0으로 지워서 빈 윤곽선으로
                            grid[y, inner_left:inner_right+1] = 0

            # 이상치(점)
            for o in (stats.get("outliers") or []):
                if isinstance(o, (int, float)) and np.isfinite(o):
                    yy = value_to_row(float(o))
                    if plot_top <= yy <= plot_bottom:
                        grid[yy, cx] = 1

        x_cursor = cat_left + cat_slot + group_gap

    # ── 패턴: 모든 데이터 아래에 찍기
    for x, s in pattern_centers:
        draw_bottom_pattern(x, legend.get(s, [0,0,1,1]))

    # ── X축 밑 점: 단일=시리즈별, 다중=카테고리별
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

def save_matplotlib_violin(categories, series, eff_data,
                           vmin: float, vmax: float, vstep: int,
                           out_path: str):

    fig, ax = plt.subplots(figsize=(6, 4))
    C, S = len(categories), len(series)

    # 합성 샘플 생성 함수: min~Q1, Q1~Q3, Q3~max에 서로 다른 가중치
    def synth_samples(st, n=400):
        mn, q1, md, q3, mx = st["min"], st["Q1"], st["median"], st["Q3"], st["max"]
        n1, n2, n3 = int(n*0.25), int(n*0.5), int(n*0.25)
        a = np.random.uniform(mn, q1, size=max(1, n1))
        b = np.random.uniform(q1, q3, size=max(1, n2))
        c = np.random.uniform(q3, mx, size=max(1, n3))
        # 중앙값 근처 조금 더 모으기
        b = np.concatenate([b, np.random.normal(md, (q3-q1)/8.0 if q3>q1 else 1.0, size=max(1, n//8))])
        return np.clip(np.concatenate([a,b,c]), mn, mx)

    # 위치 계산
    gap_between_categories = 1.0
    width_total = 0.8
    box_width = width_total / max(1, S)

    colors = plt.cm.tab10.colors
    legend_handles = []

    for si, s in enumerate(series):
        if si < len(colors):
            legend_handles.append(Patch(facecolor=colors[si], edgecolor='black', alpha=0.6, label=s))

    for ci, cat in enumerate(categories):
        cat_center = ci * (width_total + gap_between_categories) + 1.0
        for si, s in enumerate(series):
            stats = eff_data.get(s, [None]*C)[ci]
            if not isinstance(stats, dict): continue
            st = {k: float(stats[k]) for k in ("min","Q1","median","Q3","max")}
            samples = synth_samples(st)

            pos = cat_center + (si - (S-1)/2) * box_width

            parts = ax.violinplot([samples], positions=[pos], widths=box_width*0.95,
                                  showmeans=False, showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[si % len(colors)])
                pc.set_edgecolor('black')
                pc.set_alpha(0.6)

            # 중앙선 표시
            ax.hlines(st["median"], pos - box_width*0.45, pos + box_width*0.45, colors='black', linewidth=1.2)

            # 이상치가 있다면 점으로
            outs = [o for o in (stats.get("outliers") or []) if isinstance(o,(int,float)) and np.isfinite(o)]
            if outs:
                ax.scatter([pos]*len(outs), outs, s=15, c=[colors[si % len(colors)]], edgecolors='black')

    # x/y축 설정
    cat_centers = [ci * (width_total + gap_between_categories) + 1.0 for ci in range(C)]
    ax.set_xticks(cat_centers)
    ax.set_xticklabels(categories)

    ax.set_ylim(vmin, vmax)
    if isinstance(vstep,(int,float)) and vstep>0:
        yt = np.arange(vmin, vmax + 1e-9, vstep)
        if len(yt)==0 or yt[-1]<vmax: yt = np.append(yt, vmax)
        ax.set_yticks(yt)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if legend_handles:
        ax.legend(handles=legend_handles, title="Series")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────
# 4) 파이프라인: 파일 읽고 격자 리스트 반환 + 참고 PNG 저장
# ─────────────────────────────────────────────────────────
def violin_single_highlight(request_id: str):
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

    cats, sers, eff_data, legend, vmin, vmax, vstep = normalize_box_spec(chart)
    hi_mask = build_highlight_mask(sers, cats, gpt)

    grid = build_raster_grid_violin(
        cats, sers, eff_data, legend, request_id,
        vmin, vmax, vstep,
        W=60, H=40,
        highlight_mask=hi_mask
    )

    save_matplotlib_violin(cats, sers, eff_data, vmin, vmax, vstep, png_out)

    arr = np.asarray(grid)
    print("shape:", arr.shape)
    print("x-axis row idx:", 40-6)
    print("x-axis last 5:", arr[40-6, -5:])
    return grid.astype(int).tolist()
