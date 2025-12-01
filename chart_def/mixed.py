# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import config.matplotlib_config #절대 지우기 금지

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

    # 기본값: 전부 비강조(False)
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
        # ✅ all: 아무 것도 강조하지 않음(전부 False 유지)
        pass

    # 🔎 폴백 정책 조정:
    # - 이전에는 "아무 것도 없으면 all"로 강제 True를 채웠지만,
    #   이제는 '비강조로만 그리기'가 정상 케이스일 수 있으므로 폴백 제거.
    #   (필요하면 아래 조건부 폴백처럼 mode가 all이 아닐 때만 적용)
    #
    # if mode not in ("all", None) and not any(any(row) for row in mask.values()):
    #     # 매치가 전혀 없을 때의 안전장치가 필요하면 여기서 결정하세요.
    #     # 예: 그대로 두고(모두 False) 렌더러가 '그리지 않음' 처리하게 둘 수도 있음.
    #     pass

    return mask






# ─────────────────────────────────────────────────────────────────────────────
# 0) 입력 스펙 정규화: mixed (line + bar)
#   - chart_type.components의 각 요소(type: "line"/"bar", series[], y_axis: "left"/"right")
#   - y_axes.left/right(range, interval) 사용
# ─────────────────────────────────────────────────────────────────────────────
def normalize_mixed_spec(data: dict):
    ct = (data.get("chart_type") or {}).get("type", "mixed")
    assert ct == "mixed", "mixed만 지원합니다."

    components = (data.get("chart_type") or {}).get("components", [])
    categories = list(data.get("categories", []))
    C = len(categories)

    # 전체 series/데이터/레전드
    all_series = list(data.get("series", []))
    raw_data   = data.get("data", {})
    legend     = data.get("legend", {})

    # 좌/우 축 파라미터
    y_axes = data.get("axes") or data.get("y_axes") or {}

    left_ax  = y_axes.get("left",  {}) or {}
    right_ax = y_axes.get("right", {}) or {}

    def parse_axis(ax, default_range=(0, 100), default_step=10):
        rng = ax.get("range", list(default_range))
        step = ax.get("interval", default_step)
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            rng = list(default_range)
        vmin, vmax = int(rng[0]), int(rng[1])
        vstep = int(step) if isinstance(step, (int, float)) else default_step
        return vmin, vmax, vstep

    vmin_l, vmax_l, vstep_l = parse_axis(left_ax)
    vmin_r, vmax_r, vstep_r = parse_axis(right_ax)

    # 컴포넌트 분리
    line_series = []
    bar_series  = []
    series_to_axis = {}
    for comp in components:
        ctype = comp.get("type")
        ss    = list(comp.get("series", []))
        
        side  = comp.get("y_axis", "left")
        side  = "left" if str(side).lower() == "left" else "right"
        for s in ss:
            series_to_axis[s] = side
        if ctype == "line":
            line_series.extend(ss)
        elif ctype == "bar":
            bar_series.extend(ss)

    # 시리즈별 데이터 길이 보정
    eff_data = {}
    for s in all_series:
        vals = list(raw_data.get(s, []))
        if len(vals) < C:
            vals = vals + [None] * (C - len(vals))
        elif len(vals) > C:
            vals = vals[:C]
        eff_data[s] = vals

    return (
       categories, line_series, bar_series, eff_data, legend,
       (vmin_l, vmax_l, vstep_l), (vmin_r, vmax_r, vstep_r),
       series_to_axis
   )


# ─────────────────────────────────────────────────────────────────────────────
# 1) 복합차트 60×40 격자 생성
#    - 좌측 Y축: line(Price), 우측 Y축: bar(Demand)
#    - X축/눈금: 플롯 영역(우측 마진 제외)만 그리기
#    - 선 하이라이트: 4×3(세로형) 패턴, 형태보존(패턴 박스 단위 클램프)
#    - 막대 하이라이트: 채움(True)/외곽선(False) 방식
#    - 레전드: line → 우측 끝 2×3, bar → 막대 중앙 아래 2×2
# ─────────────────────────────────────────────────────────────────────────────
def build_centers_int(left_col: int, right_col: int, C: int, prefer_gap: int = 2) -> list[int]:
    """
    정수 격자 중심열 생성:
    - 가능한 한 균등 배치
    - 최소 간격 보장(불가 시 1까지 완화)
    - [left_col, right_col] 경계 내 강제
    """
    width = right_col - left_col
    if C <= 1 or width <= 0:
        return [left_col + max(0, width)//2]

    # 우선 균등 실수 배치 → 정수화
    ideal = np.linspace(left_col, right_col, C)
    x_cols = [int(round(v)) for v in ideal]

    # 최소 간격 목표
    min_gap_possible = max(1, width // max(1, C - 1))
    gap = max(1, min(prefer_gap, min_gap_possible))

    # 좌→우 증가/간격 보장
    for i in range(1, C):
        if x_cols[i] <= x_cols[i-1] + gap:
            x_cols[i] = x_cols[i-1] + gap

    # 우측 경계 초과 시 일괄 이동
    overflow = x_cols[-1] - right_col
    if overflow > 0:
        x_cols = [x - overflow for x in x_cols]

    # 좌측 경계 미만 시 일괄 이동
    under = left_col - x_cols[0]
    if under > 0:
        x_cols = [x + under for x in x_cols]

    # 마지막 안전 점검: 그래도 넘치면 gap을 1까지 낮추며 재보정
    if x_cols[-1] > right_col:
        for g in range(gap-1, 0, -1):
            xs = [x_cols[0]]
            for i in range(1, C):
                xs.append(xs[-1] + g)
            if xs[-1] <= right_col:
                x_cols = xs
                break
        if x_cols[-1] > right_col:
            x_cols = [int(round(v)) for v in np.linspace(left_col, right_col, C)]
    return x_cols


# ─────────────────────────────────────────────────────────────────────────────

def build_mixed_raster_grid(
    categories, line_series, bar_series, eff_data, legend, request_id,
    left_axis, right_axis,
    W: int = 60, H: int = 40,
    right_margin: int = 0,     # (사용 안 함) 호환용
    highlight_mask: dict | None = None,
    force_deemph_if_no_match: bool = False,  # (이제 사용 안 함) 호환용
    series_to_axis: dict | None = None,
    **_: dict
) -> np.ndarray:
    """
    - 데이터는 [3,53] 열 범위에만 그림(오른쪽 축과 물리적 이격)
    - 카테고리 그룹 폭/간격을 전역에서 계산하여 겹침 방지
    - 카테고리 간 최소 1칸 공백 보장, 남는 폭은 균등 분배
    - 선/막대 중심 완전 일치
    - 양쪽 Y축 눈금은 1칸만 안쪽으로
    - X축은 열 0부터 오른쪽 축까지

    하이라이트 동작(요청사항):
      • ALL(=어떤 타입에도 강조 없음) → 선/막대 모두 '비강조'로 그림
      • LINE 강조만 존재 → 선만 그림(기본 라인 + 강조 지점 표시), 막대는 그리지 않음
      • BAR  강조만 존재 → 막대만 그림(강조=채움, 비강조=외곽선), 선은 그리지 않음
      • 둘 다 강조 존재   → 두 타입 모두 위 규칙대로 그림
    """
    import numpy as np

    grid = np.zeros((H, W), dtype=np.uint8)

    # ── 레이아웃/축
    y_axis_col_left = 0
    x_axis_row      = H - 6
    plot_top        = 2

    RIGHT_AXIS_FROM_RIGHT = 4
    y_axis_col_right = W - RIGHT_AXIS_FROM_RIGHT

    DATA_LEFT_COL  = 3
    DATA_RIGHT_COL = 53
    L, R = DATA_LEFT_COL, DATA_RIGHT_COL
    plot_bottom = x_axis_row
    C = len(categories)

    # 축 스케일
    vmin_l, vmax_l, vstep_l = left_axis
    vmin_r, vmax_r, vstep_r = right_axis

    # ── 축 그리기
    grid[plot_top:plot_bottom+1, y_axis_col_left]  = 1
    grid[plot_top:plot_bottom+1, y_axis_col_right] = 1
    grid[x_axis_row, 0:y_axis_col_right+1] = 1  # X축은 0열부터

    # ── 눈금(1칸만 튀게)
    def tick_rows_for_axis(vmin, vmax, vstep):
        if not (isinstance(vstep, (int, float)) and vstep > 0) or vmax == vmin:
            return []
        rows = []
        for tv in range(int(vmin + vstep), int(vmax) + 1, int(vstep)):
            t = (tv - vmin) / (vmax - vmin) if vmax > vmin else 0
            r = int(round(plot_bottom - t * (plot_bottom - plot_top + 1)))
            rows.append(r)
        if plot_top not in rows:
            rows.append(plot_top)
        return sorted({r for r in rows if plot_top <= r <= x_axis_row - 1})

    for r in tick_rows_for_axis(vmin_l, vmax_l, vstep_l):
        c = y_axis_col_left + 1
        if c <= R: grid[r, c] = 1

    for r in tick_rows_for_axis(vmin_r, vmax_r, vstep_r):
        c = y_axis_col_right - 1
        if c >= L: grid[r, c] = 1

    # ── 도우미
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    def in_data_bounds(r, c): return (plot_top <= r <= plot_bottom) and (L <= c <= R)

    def value_to_row_left(v):
        if v is None or vmax_l == vmin_l: return plot_bottom
        v = clamp(v, vmin_l, vmax_l)
        t = (v - vmin_l) / (vmax_l - vmin_l)
        return int(round(plot_bottom - t * (plot_bottom - plot_top + 1)))

    def value_to_row_right(v):
        if v is None or vmax_r == vmin_r: return plot_bottom
        v = clamp(v, vmin_r, vmax_r)
        t = (v - vmin_r) / (vmax_r - vmin_r)
        return int(round(plot_bottom - t * (plot_bottom - plot_top + 1)))
    def value_to_height_left(v):
        if v is None or vmax_l == vmin_l: return 0
        v = clamp(v, vmin_l, vmax_l)
        if v <= vmin_l: return 0
        usable_h = (plot_bottom - plot_top + 1)
        return max(1, int(np.ceil((v - vmin_l) / (vmax_l - vmin_l) * usable_h)))

    # 축 매핑 기본값(이전 동작 유지): line→left, bar→right
    if series_to_axis is None:
        series_to_axis = {s: "left" for s in line_series}
        series_to_axis.update({s: "right" for s in bar_series})

    # ── 전역 그룹 배치 (겹침 방지의 핵심)
    Sbar = len(bar_series)
    data_width = R - L + 1
    min_gap = 1                        # 카테고리 최소 간격(빈 칸) 1칸
    inner_gap = 1 if Sbar >= 2 else 0  # 시리즈 간 간격
    min_group_w = max(1, Sbar*1 + inner_gap*(Sbar-1))  # 각 그룹의 최소 폭(막대 1칸씩)

    # 가능한 조합 찾기: group_w 최대화, 안되면 gap↓, inner_gap↓
    while True:
        cap = (data_width - (C-1)*min_gap) // C if C > 0 else data_width
        if cap >= min_group_w:
            group_w = cap
            break
        if min_gap > 0:
            min_gap -= 1
            continue
        if inner_gap > 0:
            inner_gap = 0
            min_group_w = max(1, Sbar)  # 재계산
            continue
        group_w = max(1, cap)  # 최후 보정
        break

    # 남는 폭(slack) 균등 분배: [좌여백] + [C-1개의 사이 간격] + [우여백]
    spaces = [0] + [min_gap]*(max(0, C-1)) + [0]
    total_used = C*group_w + (C-1)*min_gap
    slack = max(0, data_width - total_used)
    # 라운드로빈으로 분배(가장자리에만 몰리지 않도록)
    idx = 0
    for _ in range(slack):
        spaces[idx] += 1
        idx = (idx + 1) % len(spaces)

    # gleft 배열 생성 (겹침 불가, 경계 내 보장)
    gleft = []
    cur = L + spaces[0]
    for ci in range(C):
        gleft.append(cur)
        if ci < C-1:
            cur = cur + group_w + spaces[ci+1]

    # 최종 중심열: 막대/선 공통
    x_cols = [gl + group_w//2 for gl in gleft]
    if x_axis_row + 1 < H:
        for cc in x_cols:
            if 0 <= cc < W:
                grid[x_axis_row + 1, cc] = 1

    # ── 그리기 유틸
    def draw_line(r0, c0, r1, c1):
        dr, dc = r1 - r0, c1 - c0
        steps = max(abs(dr), abs(dc))
        if steps == 0:
            if in_data_bounds(r0, c0): grid[r0, c0] = 1
            return
        for i in range(steps + 1):
            rr = int(round(r0 + dr * i / steps))
            cc = int(round(c0 + dc * i / steps))
            if in_data_bounds(rr, cc):
                grid[rr, cc] = 1
                if rr + 1 <= plot_bottom: grid[rr+1, cc] = 1  # 1px 보강

    HILITE_TALL = [[1,1,1],[1,0,1],[1,0,1],[1,1,1]]
    Hh, Hw = 4, 3
    def draw_hilite_center(r, c):
        top  = clamp(r - (Hh // 2 + (Hh % 2 == 0)) + 1, plot_top, plot_bottom - (Hh - 1))
        left = clamp(c - (Hw // 2), L, R - (Hw - 1))
        for rr in range(Hh):
            for cc in range(Hw):
                grid[top + rr, left + cc] = 1 if HILITE_TALL[rr][cc] else 0

    def fill_rect(r0, r1, c0, c1, val=1):
        r0, r1 = clamp(r0, plot_top, plot_bottom), clamp(r1, plot_top, plot_bottom)
        c0, c1 = clamp(c0, L, R), clamp(c1, L, R)
        if r0 <= r1 and c0 <= c1:
            grid[r0:r1+1, c0:c1+1] = val

    def draw_bar_outline(r0, r1, c0, c1):
        if r1 < r0 or c1 < c0: return
        r0, r1 = clamp(r0, plot_top, plot_bottom), clamp(r1, plot_top, plot_bottom)
        c0, c1 = clamp(c0, L, R), clamp(c1, L, R)
        grid[r0, c0:c1+1] = 1; grid[r1, c0:c1+1] = 1
        grid[r0:r1+1, c0] = 1; grid[r0:r1+1, c1] = 1

    def value_to_height_right(v):
        if v is None or vmax_r == vmin_r: return 0
        v = clamp(v, vmin_r, vmax_r)
        if v <= vmin_r: return 0
        usable_h = (plot_bottom - plot_top + 1)
        return max(1, int(np.ceil((v - vmin_r) / (vmax_r - vmin_r) * usable_h)))

    def draw_3x2_below_bar(center_col, bits):
        if not isinstance(bits, (list, tuple)): bits = [1]*6
        use = list(bits[:6]) + [1]*max(0, 6 - len(bits))
        arr = np.array(use, dtype=np.uint8).reshape(3, 2)
        patt_h, patt_w = 3, 2
        # 그리드 아래 고정
        patt_top = max(0, H - patt_h)
        left = int(np.clip(center_col - 1, L, R - (patt_w - 1)))
        for rr in range(patt_h):
            for cc in range(patt_w):
                if arr[rr, cc]:
                    r = patt_top + rr; c = left + cc
                    if (x_axis_row + 1) <= r < H and L <= c <= R:
                        grid[r, c] = 1

    def draw_legend_2x3_flush_right(r_ref, bits):
        if not isinstance(bits, (list, tuple)): bits = [1]*6
        flat = list(bits[:6]) + [1]*max(0, 6 - len(bits))
        arr = np.array(flat, dtype=np.uint8).reshape(3, 2)
        Lh, Lw = 3, 2
        margin_left  = y_axis_col_right + 1
        margin_right = W - 1
        left = max(margin_left, margin_right - (Lw - 1))
        top  = clamp(r_ref - 1, plot_top, plot_bottom - (Lh - 1))
        for rr in range(Lh):
            for cc in range(Lw):
                if arr[rr, cc]:
                    grid[top + rr, left + cc] = 1

    # ── 하이라이트 마스크: 기본은 전부 False (ALL 판정 위해)
    if not highlight_mask:
        highlight_mask = {s:[False]*C for s in (line_series + bar_series)}

    # 어떤 타입에 강조가 있는지
    has_line_focus = any(any(highlight_mask.get(s, [])) for s in line_series)
    has_bar_focus  = any(any(highlight_mask.get(s, [])) for s in bar_series)

    # ALL 모드: 둘 다 강조 없음 → 둘 다 '비강조'로 그림
    draw_all = (not has_line_focus) and (not has_bar_focus)

    # ── 선(좌축)
    pts_per_line = {}
    if draw_all or has_line_focus:
        for s in line_series:
            vals = eff_data.get(s, [None]*C)
            row_fn = value_to_row_left if series_to_axis.get(s, "left") == "left" else value_to_row_right
            pts  = [(row_fn(vals[j]), x_cols[j]) if vals[j] is not None else None
                    for j in range(C)]
            pts_per_line[s] = pts

            # 기본 라인: ALL이든 LINE강조든 '기본 라인'은 깔아준다
            for j in range(C-1):
                if pts[j] is None or pts[j+1] is None:
                    continue
                r0, c0 = pts[j]; r1, c1 = pts[j+1]
                draw_line(r0, c0, r1, c1)

    # ── 막대(우측)
    bar_centers = []
    if (draw_all or has_bar_focus) and Sbar > 0 and C > 0:
        # group_w에 맞춰 막대폭 산출
        bw = (group_w - inner_gap*(Sbar-1)) // max(1, Sbar)
        if bw < 1:
            inner_gap = 0
            bw = max(1, group_w // max(1, Sbar))

        for ci in range(C):
            gl = gleft[ci]
            for si, s in enumerate(bar_series):
                c0 = gl + si * (bw + inner_gap)
                c1 = min(c0 + bw - 1, R)
                v  = (eff_data.get(s) or [None]*C)[ci]
                if series_to_axis.get(s, "right") == "left":
                    h = value_to_height_left(v)
                else:
                    h = value_to_height_right(v)
                if h > 0:
                    r1, r0 = plot_bottom, plot_bottom - h + 1
                    hi_row = (highlight_mask.get(s) or [False]*C)
                    hi = (ci < len(hi_row) and bool(hi_row[ci]))

                    if draw_all:
                        # ALL → 전부 비강조(외곽선)
                        draw_bar_outline(r0, r1, c0, c1)
                    else:
                        # BAR 강조 모드 → 강조=채움 / 비강조=외곽선
                        if hi:
                            fill_rect(r0, r1, c0, c1, 1)
                        else:
                            draw_bar_outline(r0, r1, c0, c1)
                bar_centers.append((c0 + c1)//2)

    # ── 레전드(해당 타입을 그릴 때만)
    if draw_all or has_line_focus:
        for s in line_series:
            pts = pts_per_line.get(s, [])
            if not pts: 
                continue
            last_idx = next((k for k in range(C-1, -1, -1) if k < len(pts) and pts[k] is not None), None)
            if last_idx is not None:
                r_last, _ = pts[last_idx]
                draw_legend_2x3_flush_right(r_last, legend.get(s, [1]*6))

    if (draw_all or has_bar_focus) and Sbar > 0 and len(bar_centers) > 0:
        idx = 0
        for _ci in range(C):
            for s in bar_series:
                if idx < len(bar_centers):
                    draw_3x2_below_bar(bar_centers[idx], legend.get(s, [1,1,1,1,1,1]))
                idx += 1

    # ── 라인 "강조 지점"은 LINE 강조가 있을 때만 찍는다 (ALL에선 안 찍음)
    if has_line_focus:
        for s in line_series:
            pts = pts_per_line.get(s, [])
            mask_row = highlight_mask.get(s, [False]*C)
            for j in range(C):
                if j < len(pts) and pts[j] is not None and j < len(mask_row) and mask_row[j]:
                    r, c = pts[j]
                    draw_hilite_center(r, c)
    plt.figure(figsize=(6, 4))
    plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.axis("off")

        # PNG로 저장
    plt.savefig(f"static/binary/{request_id}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    return grid

# ─────────────────────────────────────────────────────────────────────────────
# 2) Matplotlib 참고 이미지 저장(혼합)
# ─────────────────────────────────────────────────────────────────────────────
def save_matplotlib_mixed(
    categories, line_series, bar_series, eff_data, left_axis, right_axis, png_path: str, series_to_axis: dict | None = None
):
    import matplotlib.pyplot as plt
    C = len(categories)
    x = np.arange(C)

    vmin_l, vmax_l, vstep_l = left_axis
    vmin_r, vmax_r, vstep_r = right_axis

    fig, ax_l = plt.subplots(figsize=(6,4))
    ax_r = ax_l.twinx()

    if series_to_axis is None:
        series_to_axis = {s: "left" for s in line_series}
        series_to_axis.update({s: "right" for s in bar_series})

    # 막대(우측축)
    if len(bar_series) >= 1:
        width = 0.8 / max(1, len(bar_series))
        for si, s in enumerate(bar_series):
            offs = (si - (len(bar_series)-1)/2) * width
            y = eff_data.get(s, [None]*C)
            y = [float(v) if v is not None else np.nan for v in y]
            ax = ax_l if series_to_axis.get(s, "right") == "left" else ax_r
            ax.bar(x + offs, y, width=width, label=s, alpha=0.5)

    # 라인(좌측축)
    for s in line_series:
        y = eff_data.get(s, [None]*C)
        y = [float(v) if v is not None else np.nan for v in y]
        ax = ax_l if series_to_axis.get(s, "left") == "left" else ax_r
        ax.plot(x, y, marker="o", label=s)

    ax_l.set_xticks(x); ax_l.set_xticklabels(categories)

    ax_l.set_ylim(vmin_l, vmax_l)
    if isinstance(vstep_l, (int, float)) and vstep_l > 0:
        yt = np.arange(vmin_l, vmax_l + 1, vstep_l)
        if len(yt) == 0 or yt[-1] != vmax_l: yt = np.append(yt, vmax_l)
        ax_l.set_yticks(yt)

    ax_r.set_ylim(vmin_r, vmax_r)
    if isinstance(vstep_r, (int, float)) and vstep_r > 0:
        yt = np.arange(vmin_r, vmax_r + 1, vstep_r)
        if len(yt) == 0 or yt[-1] != vmax_r: yt = np.append(yt, vmax_r)
        ax_r.set_yticks(yt)

    ax_l.spines['top'].set_visible(False); ax_r.spines['top'].set_visible(False)
    ax_l.spines['right'].set_visible(False)  # 오른쪽은 twin 축이 표시
    # 범례는 두 축의 라벨을 합쳐서
    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_l.legend(h1+h2, l1+l2, loc="best")

    plt.tight_layout(); plt.savefig(png_path, dpi=150); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3) 최종 호출 함수
#    - static/chartQA_data/{id}.json : mixed 데이터(JSON)
#    - static/QA/{id}.json           : 하이라이트 규칙(JSON)
#    - build_highlight_mask(series, categories, values, gpt_response) 재사용
# ─────────────────────────────────────────────────────────────────────────────
def mixed_single_highlight(request_id: str, build_highlight_mask_fn=None):
    """
    build_highlight_mask_fn: 주입 안 하면, 모듈 내부 build_highlight_mask 사용
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

    # 정규화
    (categories, line_series, bar_series, eff_data, legend,
    left_axis, right_axis, series_to_axis) = normalize_mixed_spec(chart_data)

    if build_highlight_mask_fn is None:
        build_highlight_mask_fn = build_highlight_mask

    orig_series     = chart_data.get("series", [])
    orig_categories = chart_data.get("categories", [])
    values          = chart_data.get("data", {})

    highlight_mask  = build_highlight_mask_fn(orig_series, orig_categories, values, gpt_response)

    # ▶ 하이라이트 요청 여부 플래그
    had_req = bool(gpt_response) and (gpt_response.get("highlight_mode") != "all")

    grid = build_mixed_raster_grid(
        categories, line_series, bar_series, eff_data, legend, request_id,
        left_axis, right_axis,
        W=60, H=40, right_margin=4,
        highlight_mask=highlight_mask,
        # ▶ 새 인자
        force_deemph_if_no_match=had_req,
        series_to_axis=series_to_axis
    )

    # 참고 PNG
    os.makedirs(os.path.dirname(mpl_out), exist_ok=True)
    save_matplotlib_mixed(categories, line_series, bar_series, eff_data,
                          left_axis, right_axis, mpl_out,
                          series_to_axis=series_to_axis)
    return np.asarray(grid).astype(int).tolist()
