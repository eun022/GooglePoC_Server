# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import config.matplotlib_config #절대 지우기 금지



# ─────────────────────────────────────────────────────────────────────────────
# 0) 입력 스펙 정규화: bar 전용
# ─────────────────────────────────────────────────────────────────────────────
def normalize_bar_spec(data: dict):
    assert data.get("chart_type", {}).get("type", "bar") == "bar", "bar만 지원합니다."

    series = list(data.get("series", []))
    categories = list(data.get("categories", []))
    raw_data = data.get("data", {})
    legend = data.get("legend", {})

    ax = data.get("axes", {}) or {}
    # ✅ 축 파싱: axes.Y 또는 axes.y 우선, 없으면 flat 형태(range/interval) 시도
    if isinstance(ax, dict) and ("Y" in ax or "y" in ax):
        y_ax = ax.get("Y") or ax.get("y") or {}
        rng = y_ax.get("range", [0, 100])
        vstep = y_ax.get("interval", 10)
    else:
        rng = ax.get("range", [0, 100])
        vstep = ax.get("interval", 10)

    # 가드: rng 길이/형 체크 및 캐스팅
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
        rng = [0, 100]
    vmin, vmax = int(rng[0]), int(rng[1])
    vstep = int(vstep) if isinstance(vstep, (int, float)) else 10
    if vstep <= 0:
        vstep = max(1, (vmax - vmin) // 5)  # 동적 기본값
        if vstep <= 0:
            vstep = 1


    single_mode = (len(categories) == 1 and categories[0] in ("1", "")) and len(series) >= 1
    if single_mode:
        eff_categories = series[:]  # X축 라벨 = 원래 series
        eff_series = ["1"]
        eff_data = {"1": [(raw_data[s][0] if raw_data.get(s) else None) for s in series]}
    else:
        eff_categories = categories[:]
        eff_series = series[:]
        eff_data = {s: raw_data.get(s, [None]*len(eff_categories)) for s in eff_series}

    return eff_categories, eff_series, eff_data, legend, vmin, vmax, vstep, single_mode



# ─────────────────────────────────────────────────────────────────────────────
# 1) 60×40 격자 생성(축/눈금/막대/2×2패턴 + 하이라이트 반영)
# ─────────────────────────────────────────────────────────────────────────────
def build_raster_grid(
    eff_categories, eff_series, eff_data, legend,  request_id ,
    vmin: int, vmax: int, vstep: int, single_mode: bool,
    W: int = 60, H: int = 40,
    highlight_mask: dict | None = None,
    original_series: list | None = None,
    original_categories: list | None = None
) -> np.ndarray:
    import numpy as np
    grid = np.zeros((H, W), dtype=np.uint8)

    # ── 레이아웃 고정값
    y_axis_col = 0
    x_axis_row = H - 6
    plot_top   = 2
    plot_left  = y_axis_col + 4
    plot_right = W - 1
    plot_bottom = x_axis_row

    # 축
    grid[plot_top:x_axis_row+1, y_axis_col] = 1
    grid[x_axis_row, :] = 1

    # 눈금(짧게)
    usable_height = (plot_bottom - plot_top + 1)
    usable_width  = (plot_right - plot_left + 1)

    row_mark = x_axis_row + 1 if x_axis_row + 1 < H else H - 1

    tick_vals = list(range(vmin + vstep, vmax + 1, vstep))
    tick_rows = []
    for tv in tick_vals:
        t = (tv - vmin) / (vmax - vmin) if vmax > vmin else 0
        r = int(round(plot_bottom - t * usable_height))
        tick_rows.append(r)
    if plot_top not in tick_rows:
        tick_rows.append(plot_top)
    tick_rows = sorted({r for r in tick_rows if plot_top <= r <= x_axis_row - 1})
    for r in tick_rows:
        c0, c1 = y_axis_col + 1, min(y_axis_col + 2, plot_right)
        grid[r, c0:c1+1] = 1

    # ── 헬퍼
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    def fill_rect(r0, r1, c0, c1, val=1):
        r0, r1 = clamp(r0, plot_top, plot_bottom), clamp(r1, plot_top, plot_bottom)
        c0, c1 = clamp(c0, plot_left, plot_right), clamp(c1, plot_left, plot_right)
        if r0 <= r1 and c0 <= c1:
            grid[r0:r1+1, c0:c1+1] = val

    def value_to_height(v):
        if v is None or vmax == vmin: return 0
        v = clamp(v, vmin, vmax)
        if v <= vmin: return 0
        return max(1, int(np.ceil((v - vmin) / (vmax - vmin) * usable_height)))

    def draw_bar_filled(r0, r1, c0, c1):
        if r1 >= r0 and c1 >= c0:
            fill_rect(r0, r1, c0, c1, 1)

    def draw_bar_outline(r0, r1, c0, c1):
        if r1 < r0 or c1 < c0:
            return
        grid[r0, c0:c1+1] = 1
        grid[r1, c0:c1+1] = 1
        grid[r0:r1+1, c0] = 1
        grid[r0:r1+1, c1] = 1

    # ── 하이라이트 판정 (S==1일 때도 안전하게 동작)
    C, S = len(eff_categories), len(eff_series)
    bar_centers = []

    def is_highlight_for_single(cat_index: int) -> bool:
        """
        S == 1에서의 하이라이트 판정.
        - single_mode=True: 카테고리 == 원본 시리즈명 매핑, 해당 시리즈의 [0]만 본다.
        - single_mode=False: 유일한 시리즈명에 대해 카테고리 인덱스를 본다.
        기본값은 무조건 False (마스크 없거나 키/인덱스 없으면 False).
        """
        if not highlight_mask:
            return False

        if single_mode:
            # 카테고리 → 원본 시리즈명 매핑
            if original_series and 0 <= cat_index < len(original_series):
                s_name = original_series[cat_index]
            elif 0 <= cat_index < len(eff_categories):
                s_name = eff_categories[cat_index]
            else:
                return False

            row = highlight_mask.get(s_name)
            return bool(row and len(row) > 0 and row[0])

        else:
            # 시리즈는 1개, 카테고리 인덱스별 판정
            if not eff_series:
                return False
            s0 = eff_series[0]
            row = highlight_mask.get(s0)
            return bool(row and 0 <= cat_index < len(row) and row[cat_index])

    def is_highlight_for_group(s_name, cat_index):
        """
        S > 1에서의 하이라이트 판정.
        기본값은 False (마스크 없거나 키/인덱스 없으면 False).
        """
        if not highlight_mask:
            return False
        row = highlight_mask.get(s_name)
        return bool(row and 0 <= cat_index < len(row) and row[cat_index])

    # ── 배치 파라미터
    G = max(1, C)
    inner_gap = 1
    min_bar_w = 2

    group_gap = 1
    numerator = usable_width - (G+1)*group_gap - G*(S-1)*inner_gap
    if numerator < G*S:
        group_gap = 0
        numerator = usable_width - (G+1)*group_gap - G*(S-1)*inner_gap
    bar_w = max(min_bar_w, numerator // (G*S) if numerator > 0 else min_bar_w)

    total_used = G*(S*bar_w + (S-1)*inner_gap) + (G+1)*group_gap
    slack = max(0, usable_width - total_used)
    base_extra = slack // (G+1)
    extra_rem  = slack %  (G+1)
    gaps = [group_gap + base_extra + (1 if i < extra_rem else 0) for i in range(G+1)]
    cur = plot_left + gaps[0]

    # ── 막대 그리기
    if S == 1:
        # ✅ 단일 시리즈일 때 값/하이라이트/범례 키를 일관되게 처리
        s_key = eff_series[0] if eff_series else "1"
        vals = eff_data.get(s_key)
        if vals is None:
            # single_mode=True 인 경우 normalize_bar_spec가 "1" 키를 만듭니다.
            vals = eff_data.get("1", [None] * C)
        # 길이 안전망
        if len(vals) < C:
            vals = list(vals) + [None] * (C - len(vals))

        for ci in range(C):
            c0 = cur
            c1 = min(c0 + bar_w - 1, plot_right)
            v  = vals[ci]
            h  = value_to_height(v)
            if h > 0:
                r1, r0 = plot_bottom, plot_bottom - h + 1
                (draw_bar_filled if is_highlight_for_single(ci) else draw_bar_outline)(r0, r1, c0, c1)
            center_col = (c0 + c1) // 2
            grid[row_mark, clamp(center_col, plot_left, plot_right)] = 1
            if x_axis_row - 1 >= 0:  # 안전 체크
                grid[x_axis_row - 1, clamp(center_col, plot_left, plot_right)] = 1 
            bar_centers.append((c0 + c1)//2)
            cur = c1 + 1 + gaps[ci+1]
    else:
        for ci in range(C):
            gleft = cur
            start_idx = len(bar_centers)
            for si, s in enumerate(eff_series):
                c0 = gleft + si * (bar_w + inner_gap)
                c1 = min(c0 + bar_w - 1, plot_right)
                h  = value_to_height((eff_data.get(s) or [None]*C)[ci])
                if h > 0:
                    r1, r0 = plot_bottom, plot_bottom - h + 1
                    (draw_bar_filled if is_highlight_for_group(s, ci) else draw_bar_outline)(r0, r1, c0, c1)
                
                bar_centers.append((c0 + c1)//2)
            if len(bar_centers) >= start_idx + S:
                left_center  = bar_centers[start_idx]
                right_center = bar_centers[start_idx + S - 1]
                group_center = (left_center + right_center) // 2
                grid[row_mark, clamp(group_center, plot_left, plot_right)] = 1
                if x_axis_row - 1 >= 0:
                    grid[x_axis_row - 1, clamp(group_center, plot_left, plot_right)] = 1
            group_total_w = S*bar_w + (S-1)*inner_gap
            cur = gleft + group_total_w + gaps[ci+1]

    # ── 2×2 패턴 (막대 중앙 아래)



    def draw_2x2(center_col, bits):
        if not isinstance(bits, (list, tuple)):
            bits = [0, 0, 1, 1]
        if len(bits) in (4, 6):
            rows = len(bits) // 2
            use = bits
        else:
            rows = max(1, len(bits) // 2)
            use = list(bits[:rows * 2])
        arr = np.array(use, dtype=np.uint8).reshape(rows, 2)
        patt_top = max(0, H - 3)          # 항상 하단 3행 영역의 최상단
        bottom_band_top = max(0, H - 3)   # 가시 영역 체크용 기준

        for rr in range(rows):
            for cc in range(2):
                r = patt_top + rr
                c = center_col - 1 + cc
                # 하단 3행에만 찍히도록 보장
                if plot_left <= c <= plot_right and bottom_band_top <= r < H:
                    if arr[rr, cc]:
                        grid[r, c] = 1
    
    if S == 1:
        if single_mode:
            for i, cc in enumerate(bar_centers):
                draw_2x2(cc, legend.get(eff_categories[i], [0,0,1,1]))
        else:
            s0 = eff_series[0] if eff_series else None
            bits = legend.get(s0, [0,0,1,1])
            for cc in bar_centers:
                draw_2x2(cc, bits)
    else:
        idx = 0
        for _ci in range(C):
            for s in eff_series:
                draw_2x2(bar_centers[idx], legend.get(s, [0,0,1,1]))
                idx += 1
                
 

    # 안전망: X축 다시 덮어쓰기
    grid[x_axis_row, :] = 1
    plt.figure(figsize=(6, 4))
    plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.axis("off")

        # PNG로 저장
    plt.savefig(f"static/binary/{request_id}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    return grid

# ─────────────────────────────────────────────────────────────────────────────
# 2) 하이라이트 마스크 생성
# ─────────────────────────────────────────────────────────────────────────────
def build_highlight_mask(series, categories, values, gpt_response):
    # values 길이 보정
    for s in series:
        if s not in values:
            values[s] = [0] * len(categories)
        elif len(values[s]) != len(categories):
            if len(values[s]) > len(categories):
                values[s] = values[s][:len(categories)]
            else:
                values[s] = values[s] + [0] * (len(categories) - len(values[s]))

    mask = {s: [False] * len(categories) for s in series}
    mode = (gpt_response or {}).get("highlight_mode")

    if mode == "series":
        target_series = {it.get("series") for it in (gpt_response.get("custom_indices") or [])}
        for s in series:
            if s in target_series:
                mask[s] = [True] * len(categories)

    elif mode == "category":
        target_cats = set(gpt_response.get("categories") or [])
        for j, c in enumerate(categories):
            if c in target_cats:
                for s in series:
                    mask[s][j] = True

    elif mode == "custom":
        for it in (gpt_response.get("custom_indices") or []):
            s, c = it.get("series"), it.get("category")
            if s in mask and c in categories:
                j = categories.index(c)
                mask[s][j] = True

    elif mode == "all":
        for s in series:
            mask[s] = [True] * len(categories)

    # 아무것도 못 찍으면 all로 폴백
    if not any(any(row) for row in mask.values()):
        for s in series:
            mask[s] = [True] * len(categories)

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 3) 참고용 Matplotlib 막대그래프 저장(이미지 경로만 지정)
# ─────────────────────────────────────────────────────────────────────────────
def save_matplotlib_bar(
    eff_categories, eff_series, eff_data,
    vmin: int, vmax: int, vstep: int,
    mpl_png_path: str
):
    fig, ax = plt.subplots(figsize=(6, 4))
    S = len(eff_series)
    C = len(eff_categories)

    if S == 1:
        # ✅ 단일 시리즈에서도 "1" 또는 실제 시리즈명(예: "서울") 모두 지원
        s_key = eff_series[0] if eff_series else "1"
        y = eff_data.get(s_key)
        if y is None:
            y = eff_data.get("1", [0] * C)  # single_mode(True) 대비 폴백
        if len(y) < C:
            y = list(y) + [0] * (C - len(y))

        x = np.arange(C)
        ax.bar(x, y, label=s_key)
        ax.set_xticks(x)
        ax.set_xticklabels(eff_categories)

    else:
        # ✅ 다중 시리즈(예: "서울","대구")
        x = np.arange(C)
        width = max(0.1, 0.8 / max(1, S))
        for si, s in enumerate(eff_series):
            offs = (si - (S - 1) / 2) * width
            y = eff_data.get(s, [0] * C)
            if len(y) < C:
                y = list(y) + [0] * (C - len(y))
            ax.bar(x + offs, y, width=width, label=s)

        ax.set_xticks(x)
        ax.set_xticklabels(eff_categories)
    ax.legend()

    # ✅ 축/눈금 안전 설정
    ax.set_ylim(vmin, vmax)
    if isinstance(vstep, (int, float)) and vstep > 0:
        yt = np.arange(vmin, vmax + 1, vstep)
        # vmax가 눈금에 없으면 추가
        if len(yt) == 0 or yt[-1] != vmax:
            yt = np.append(yt, vmax)
        ax.set_yticks(yt)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(mpl_png_path, dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4) 최종: 파일 읽고 격자 리스트 반환 + 참고 PNG 저장
# ─────────────────────────────────────────────────────────────────────────────
def bar_single_highlight(request_id: str):
    """
    입력:
      - static/chart_data/{id}.json (차트 데이터)
      - static/QA/{id}.json         (하이라이트 규칙)
    동작:
      - 하이라이트 반영 60×40 격자를 만들어 '리스트'로 반환
      - 참고용 Matplotlib 차트를 api-server/static/img/{id}.png 로 저장
    반환:
      - 60×40 격자 리스트 (list[list[int]])
    """
    chart_fp = f"static/chartQA_data/{request_id}.json"
    qa_fp    = f"static/QA/{request_id}.json"
    mpl_out  = f"static/img/{request_id}.png"

    try:
        with open(chart_fp, "r", encoding="utf-8") as f:
            chart_data = json.load(f)
        try:
            with open(qa_fp, "r", encoding="utf-8") as f:
                response = json.load(f)
        except Exception:
            # 규칙 파일이 없거나 파싱 실패 시 전체 하이라이트
            response = {"highlight_mode": "all"}
    except Exception as e:
        print("⚠️ JSON 파일 로드 실패:", e)
        return []

    # 1) 정규화
    eff_categories, eff_series, eff_data, legend, vmin, vmax, vstep, single_mode = normalize_bar_spec(chart_data)

    # 2) 하이라이트 마스크 (원본 라벨 기반)
    orig_series = chart_data.get("series", [])
    orig_categories = chart_data.get("categories", [])
    values = chart_data.get("data", {})
    highlight_mask = build_highlight_mask(orig_series, orig_categories, values, response)

    # 3) 60×40 격자 생성
    grid = build_raster_grid(
        eff_categories, eff_series, eff_data, legend, request_id,
        vmin, vmax, vstep, single_mode,
        W=60, H=40,
        highlight_mask=highlight_mask,
        original_series=orig_series,
        original_categories=orig_categories
    )

    # 4) 참고용 Matplotlib PNG 저장
    os.makedirs(os.path.dirname(mpl_out), exist_ok=True)
    save_matplotlib_bar(eff_categories, eff_series, eff_data, vmin, vmax, vstep, mpl_out)

    arr = np.asarray(grid)
    print("shape:", arr.shape)           # 기대: (40, 60)
    print("right5 col sums:", arr[:, -5:].sum(axis=0))  # 마지막 5열의 합
    print("x-axis last 5:", arr[40-6, -5:])  
    
    # 5) 격자 리스트로 반환
    return grid.astype(int).tolist()
