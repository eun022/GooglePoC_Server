# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import config.matplotlib_config #절대 지우기 금지



# ─────────────────────────────────────────────────────────────────────────────
# 0) 입력 스펙 정규화: bar 전용
# ─────────────────────────────────────────────────────────────────────────────
def normalize_line_spec(data: dict):
    # bar 전용 assert 제거
    series = list(data.get("series", []))
    categories = list(data.get("categories", []))
    raw_data = data.get("data", {})
    legend = data.get("legend", {})

    ax = data.get("axes", {}) or {}
    if isinstance(ax, dict) and ("Y" in ax or "y" in ax):
        y_ax = ax.get("Y") or ax.get("y") or {}
        rng = y_ax.get("range", [0, 100])
        vstep = y_ax.get("interval", 10)
    else:
        rng = ax.get("range", [0, 100])
        vstep = ax.get("interval", 10)

    if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
        rng = [0, 100]
    vmin = float(rng[0])
    vmax = float(rng[1])
    vstep = float(vstep) if isinstance(vstep, (int, float)) else 10.0


    # 각 시리즈 길이 보정
    eff_categories = categories[:]
    C = len(eff_categories)
    eff_series = series[:]
    eff_data = {}
    for s in eff_series:
        vals = list(raw_data.get(s, []))
        if len(vals) < C:
            vals = vals + [None] * (C - len(vals))
        elif len(vals) > C:
            vals = vals[:C]
        eff_data[s] = vals

    single_mode = False  # 라인에서는 사용하지 않음
    return eff_categories, eff_series, eff_data, legend, vmin, vmax, vstep, single_mode


def build_line_raster_grid(
    eff_categories, eff_series, eff_data, legend, request_id,
    vmin: int, vmax: int, vstep: int,
    W: int = 60, H: int = 40,
    highlight_mask: dict | None = None,
    right_margin: int = 4
) -> np.ndarray:
    import numpy as np

    grid = np.zeros((H, W), dtype=np.uint8)

    # ── 레이아웃 & 축 ─────────────────────────────────────────
    y_axis_col = 0
    x_axis_row = H - 6
    plot_top   = 2
    plot_left  = y_axis_col + 4
    plot_right = (W - 1) - max(0, right_margin)
    plot_bottom = x_axis_row

    # 축(우측 마진 구역은 그리지 않음)
    grid[plot_top:x_axis_row+1, y_axis_col] = 1
    grid[x_axis_row, :plot_right+1] = 1

    # 눈금
    usable_height = (plot_bottom - plot_top + 1)
    usable_width  = (plot_right - plot_left + 1)
    if vstep == 0:
        vstep = 1
    tick_vals = np.arange(vmin + vstep, vmax + vstep/2, vstep)
    tick_vals = list(tick_vals)
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

    # ── 헬퍼 ────────────────────────────────────────────────
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    def value_to_row(v):
        if v is None or vmax == vmin: return plot_bottom
        v = clamp(v, vmin, vmax)
        t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
        return int(round(plot_bottom - t * usable_height))

    # 카테고리별 X 좌표
    C, S = len(eff_categories), len(eff_series)
    if C <= 1:
        x_cols = [plot_left + usable_width // 2]
    else:
        x_cols = [int(round(c)) for c in np.linspace(plot_left, plot_right, C)]

    # X축 아래/위 포인트(보호 대상)
    if x_axis_row + 1 < H:
        for cc in x_cols:
            if 0 <= cc < W:
                grid[x_axis_row + 1, cc] = 1
    if x_axis_row - 1 >= 0:
        for cc in x_cols:
            if 0 <= cc < W:
                grid[x_axis_row - 1, cc] = 1

    # ── 보호 마스크 ────────────────────────────────────────
    protected = np.zeros_like(grid, dtype=bool)
    protected[plot_top:x_axis_row+1, y_axis_col] = True           # Y축
    protected[x_axis_row, :plot_right+1] = True                   # X축
    for r in tick_rows:
        c0, c1 = y_axis_col + 1, min(y_axis_col + 2, plot_right)  # 눈금
        protected[r, c0:c1+1] = True
    if x_axis_row + 1 < H:
        for cc in x_cols: protected[x_axis_row + 1, cc] = True
    if x_axis_row - 1 >= 0:
        for cc in x_cols: protected[x_axis_row - 1, cc] = True

    def in_plot(r, c):
        return (plot_top <= r <= plot_bottom) and (plot_left <= c <= plot_right)

    # ── 시리즈 라스터(halo 없이 두께 2px) ──────────────────
    def rasterize_series_mask(pts):
        m = np.zeros_like(grid, dtype=np.uint8)

        def place(r, c):
            if in_plot(r, c):
                m[r, c] = 1
                if r+1 <= plot_bottom:
                    m[r+1, c] = 1  # 세로 2px 보강

        def draw_line_no_halo(r0, c0, r1, c1):
            dr, dc = r1 - r0, c1 - c0
            steps = max(abs(dr), abs(dc))
            if steps == 0:
                place(r0, c0); return
            for i in range(steps + 1):
                rr = int(round(r0 + dr * i / steps))
                cc = int(round(c0 + dc * i / steps))
                place(rr, cc)

        for j in range(len(pts) - 1):
            if pts[j] is None or pts[j+1] is None:
                continue
            r0, c0 = pts[j]
            r1, c1 = pts[j + 1]
            draw_line_no_halo(r0, c0, r1, c1)
        return m

    # ── halo carve: 나중 선의 주변을 0으로 파되 보호는 건드리지 않음 ──
    def carve_around_mask(canvas, mask, radius=1):
        rr_idx, cc_idx = np.where(mask == 1)
        for r, c in zip(rr_idx, cc_idx):
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    if dr == 0 and dc == 0:
                        continue
                    r2, c2 = r + dr, c + dc
                    if in_plot(r2, c2) and not protected[r2, c2]:
                        canvas[r2, c2] = 0

    # ── 활성 시리즈 결정(하이라이트 지정 시 그 시리즈만) ─────────────
    active_series = list(eff_series)
    if highlight_mask:
        picked = [s for s in eff_series if any(highlight_mask.get(s, []))]
        if picked:
            active_series = picked

    # ── 1) 각 시리즈 포인트/마스크 미리 생성 ────────────────────────
    pts_per_series = {}
    mask_per_series = {}
    for s in active_series:
        vals = eff_data.get(s, [None] * C)
        pts  = [(value_to_row(vals[j]), x_cols[j]) if vals[j] is not None else None
                for j in range(C)]
        pts_per_series[s] = pts
        mask_per_series[s] = rasterize_series_mask(pts)

    # ── 2) 합성: "나중에 그려지는 선"이 기준이 되도록 순차 carve+draw ──
    canvas = grid.copy()  # 축/눈금/마커가 이미 있음
    for s in active_series:               # 뒤쪽 시리즈일수록 더 나중에 그려짐
        m = mask_per_series[s]
        carve_around_mask(canvas, m, radius=1)   # 기존(이전) 선 주변을 파고
        canvas[m == 1] = 1                       # 현재(나중) 선을 올림
    grid[:, :] = canvas

    # ── 레전드(2×3, 우측 끝 고정) ─────────────────────────
    def draw_legend_2x3_flush_right(r_ref, bits):
        if not isinstance(bits, (list, tuple)):
            bits = [1]*6
        flat = list(bits[:6]) + [1]*max(0, 6 - len(bits))
        arr = np.array(flat, dtype=np.uint8).reshape(3, 2)

        Lh, Lw = 3, 2
        margin_left  = plot_right + 1
        margin_right = W - 1
        left = max(margin_left, margin_right - (Lw - 1))
        top  = clamp(r_ref - 1, plot_top, plot_bottom - (Lh - 1))

        for rr in range(Lh):
            for cc in range(Lw):
                if arr[rr, cc]:
                    R = top + rr
                    Cc = left + cc
                    if 0 <= R < H and 0 <= Cc < W:
                        grid[R, Cc] = 1

    # ── 하이라이트 패턴(주변 carve 없음) ─────────────────────────
    HILITE_TALL = [
        [1,1,1],
        [1,0,1],
        [1,0,1],
        [1,1,1],
    ]
    Hh, Hw = 4, 3

    def draw_hilite_center(r, c):
        top  = r - (Hh // 2 + (Hh % 2 == 0)) + 1  # r-2
        left = c - (Hw // 2)                      # c-1
        top  = clamp(top,  plot_top,    plot_bottom - (Hh - 1))
        left = clamp(left, plot_left,   plot_right  - (Hw - 1))
        for rr in range(Hh):
            for cc in range(Hw):
                R = top + rr
                Cc= left + cc
                if in_plot(R, Cc):
                    if HILITE_TALL[rr][cc]:
                        grid[R, Cc] = 1
                    else:
                        grid[R, Cc] = 0  # 내부 0은 유지(요구와 동일)

    # ── 3) 레전드 → 4) 하이라이트(마지막) ─────────────────────────
    for s in active_series:
        pts = pts_per_series[s]
        last_idx = next((k for k in range(C-1, -1, -1) if pts[k] is not None), None)
        if last_idx is not None:
            r_last, _ = pts[last_idx]
            draw_legend_2x3_flush_right(r_last, legend.get(s, [1]*6))

    if highlight_mask:
        for s in active_series:
            pts = pts_per_series[s]
            mask_row = highlight_mask.get(s, [])
            for j in range(C):
                if pts[j] is None:
                    continue
                if j < len(mask_row) and mask_row[j]:
                    r, c = pts[j]
                    draw_hilite_center(r, c)

    # 렌더/저장
    plt.figure(figsize=(6, 4))
    plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.axis("off")
    plt.savefig(f"static/binary/{request_id}.png", dpi=300, bbox_inches='tight', pad_inches=0)

    return grid


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
    # if not any(any(row) for row in mask.values()):
    #     for s in series:
    #         mask[s] = [True] * len(categories)

    return mask



def save_matplotlib_line(
    eff_categories, eff_series, eff_data,
    vmin: int, vmax: int, vstep: int,
    mpl_png_path: str
):
    fig, ax = plt.subplots(figsize=(6, 4))
    C = len(eff_categories)
    x = np.arange(C)

    for s in eff_series:
        y = eff_data.get(s, [None] * C)
        ax.plot(x, y, marker="o", label=s)

    ax.set_xticks(x); ax.set_xticklabels(eff_categories, rotation=0)
    ax.set_ylim(vmin, vmax)

    if isinstance(vstep, (int, float)) and vstep > 0:
        yt = np.arange(vmin, vmax + 1, vstep)
        if len(yt) == 0 or yt[-1] != vmax:
            yt = np.append(yt, vmax)
        ax.set_yticks(yt)

    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout(); plt.savefig(mpl_png_path, dpi=150); plt.close()


def line_single_highlight(request_id: str):
    """
    입력:
      - static/chartQA_data/{id}.json (차트 데이터: series/categories/data/axes/legend)
      - static/QA/{id}.json           (하이라이트 규칙)
    동작:
      - 꺾은선 그래프를 60×40 격자로 렌더링
      - 하이라이트된 점에 3×4 패턴(1111/1001/1111) 표시
      - 참고용 PNG 저장
    반환:
      - 60×40 격자 (list[list[int]])
    """
    chart_fp = f"static/chartQA_data/{request_id}.json"
    qa_fp    = f"static/QA/{request_id}.json"
    mpl_out  = f"static/img/{request_id}.png"

    try:
        with open(chart_fp, "r", encoding="utf-8") as f:
            chart_data = json.load(f)
        try:
            with open(qa_fp, "r", encoding="utf-8") as f:
                gpt_response = json.load(f)
        except Exception:
            gpt_response = {"highlight_mode": "all"}
    except Exception as e:
        print("⚠️ JSON 로드 실패:", e); return []

    # 1) 정규화
    eff_categories, eff_series, eff_data, legend, vmin, vmax, vstep, _ = normalize_line_spec(chart_data)

    # 2) 하이라이트 마스크
    orig_series = chart_data.get("series", [])
    orig_categories = chart_data.get("categories", [])
    values = chart_data.get("data", {})
    highlight_mask = build_highlight_mask(orig_series, orig_categories, values, gpt_response)

    # 3) 라인 격자 생성
    grid = build_line_raster_grid(
        eff_categories, eff_series, eff_data, legend, request_id,
        vmin, vmax, vstep,
        W=60, H=40,
        highlight_mask=highlight_mask
    )

    # 4) 참고 PNG
    os.makedirs(os.path.dirname(mpl_out), exist_ok=True)
    save_matplotlib_line(eff_categories, eff_series, eff_data, vmin, vmax, vstep, mpl_out)

    return np.asarray(grid).astype(int).tolist()










