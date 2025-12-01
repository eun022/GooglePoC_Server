# -*- coding: utf-8 -*-
import json, os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import io, numpy as np, matplotlib.pyplot as plt
from PIL import Image
import cv2, math
import config.matplotlib_config #절대 지우기 금지


# ─────────────────────────────────────────────────────────────────────────────
# 0) 입력 스펙 정규화 (pie + 새 legend 스키마 지원)
#    legend: [ { "<시리즈명>": [bits], "<카테고리명>": [bits], ... }, ... ]
# ─────────────────────────────────────────────────────────────────────────────
def normalize_pie_spec(data: dict):
    assert data.get("chart_type", {}).get("type") == "pie", "pie만 지원합니다."

    series_defs = data.get("series", []) or []
    raw_legend  = data.get("legend", []) or []

    series_names = []
    categories_per_series = []
    values_per_series     = []
    for s in series_defs:
        sname = s.get("name", "")
        series_names.append(sname)
        cats = s.get("categories", []) or []
        categories_per_series.append([c.get("name","") for c in cats])
        values_per_series.append([float(c.get("value", 0.0)) for c in cats])

    category_bits_global = {}
    series_bits          = {}

    for leg in raw_legend:
        if not isinstance(leg, dict):
            continue
        for key, bits in leg.items():
            if key in series_names:
                if isinstance(bits, list):
                    series_bits[key] = list(map(int, bits))
            else:
                if isinstance(bits, list):
                    category_bits_global.setdefault(key, list(map(int, bits)))

    category_bits_per_series = {}
    for si, sname in enumerate(series_names):
        cnames = categories_per_series[si]
        c_map = {}
        for cname in cnames:
            c_map[cname] = category_bits_global.get(cname, [0,0,1,1])  # 폴백
        category_bits_per_series[sname] = c_map

    return series_names, categories_per_series, values_per_series, series_bits, category_bits_per_series


# ─────────────────────────────────────────────────────────────────────────────
# 1) 유틸
# ─────────────────────────────────────────────────────────────────────────────
def make_highlight_fn(highlight_cfg, series_names, categories_per_series):
    """
    highlight_cfg 예:
      {"highlight_mode":"all"}
      {"highlight_mode":"custom",
       "custom_indices":[{"series":"상품별 매출","category":"노트북"}, ...]}
    반환: is_hi(sname, cname) -> bool
    """
    # 기본: 모두 강조
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

    # 알 수 없는 모드는 그냥 모두 강조
    return lambda s, c: True


def _estimate_max_cat_rows(category_bits_per_series):
    max_rows = 1
    if not isinstance(category_bits_per_series, dict):
        return max_rows
    for cmap in category_bits_per_series.values():
        if not isinstance(cmap, dict):
            continue
        for bits in cmap.values():
            if isinstance(bits, (list, tuple)) and len(bits) > 0:
                rows = max(1, len(bits)//2)
                if rows > max_rows:
                    max_rows = rows
    return max_rows


def build_pie_highlight_mask(series_names, categories_per_series, gpt):
    """
    반환 형태:
      mask: Dict[str, Dict[str, bool]]
        mask[series_name][category_name] = True/False
    gpt 예시:
      {"highlight_mode":"all"}
      {"highlight_mode":"custom",
       "custom_indices":[{"series":"S1","category":"A"}, {"series":"S1","category":"B"}]}
    """
    mode = (gpt or {}).get("highlight_mode", "all")
    mask = {s: {c: True for c in (categories_per_series[i] if i < len(categories_per_series) else [])}
            for i, s in enumerate(series_names)}

    if mode == "all":
        return mask

    if mode == "custom":
        # 기본은 모두 False로 시작
        mask = {s: {c: False for c in (categories_per_series[i] if i < len(categories_per_series) else [])}
                for i, s in enumerate(series_names)}
        for item in (gpt.get("custom_indices", []) or []):
            s = item.get("series", "")
            c = item.get("category", "")
            if s in mask and c in mask[s]:
                mask[s][c] = True
        return mask
    
    if mode == "series":
        # 기본은 모두 False
        mask = {s: {c: False for c in (categories_per_series[i] if i < len(categories_per_series) else [])}
                for i, s in enumerate(series_names)}
        want_series = { (item.get("series") or "") for item in (gpt.get("custom_indices", []) or []) }
        for i, s in enumerate(series_names):
            if s in want_series:
                for c in categories_per_series[i]:
                    mask[s][c] = True
        return mask

    # 알 수 없는 모드는 all 취급
    return mask


def _draw_bits_block(grid, top, left, bits):
    """bits를 2열로 찍고 배치 박스 반환 (r0,c0,r1,c1)."""
    if not isinstance(bits, (list, tuple)) or len(bits) == 0:
        bits = [0,0,1,1]
    rows = max(1, len(bits)//2)
    use  = list(bits[:rows*2])
    arr  = np.array(use, dtype=np.uint8).reshape(rows, 2)

    H, W = grid.shape
    r0, c0 = top, left
    r1, c1 = min(H-1, top+rows-1), min(W-1, left+1)
    rr = 0
    for y in range(r0, r1+1):
        cc = 0
        for x in range(c0, c1+1):
            if arr[rr, cc]:
                grid[y, x] = 1
            cc += 1
        rr += 1
    return (r0, c0, r1, c1)


def _draw_bits_block_outline(grid, top, left, bits):
    """2열 블록의 외곽선(윤곽)만 그립니다."""
    if not isinstance(bits, (list, tuple)) or len(bits) == 0:
        bits = [0,0,1,1]
    rows = max(1, len(bits)//2)
    cols = 2
    H, W = grid.shape
    r0, c0 = top, left
    r1, c1 = min(H-1, top+rows-1), min(W-1, left+cols-1)

    # 테두리만 1
    for y in range(r0, r1+1):
        if 0 <= c0 < W: grid[y, c0] = 1
        if 0 <= c1 < W: grid[y, c1] = 1
    for x in range(c0, c1+1):
        if 0 <= r0 < H: grid[r0, x] = 1
        if 0 <= r1 < H: grid[r1, x] = 1
    return (r0, c0, r1, c1)


def _draw_bits_near_angle_safe(grid, cy, cx, r, angle_deg, bits,
                               base_offset=2,
                               top_reserved_rows=0,
                               bottom_reserved_rows=3,
                               left_edge=0, right_edge=None,
                               forbidden_boxes=None,
                               max_tries=12,
                               extra_up_shift=3):
    """겹침/경계/예약 고려해 안전 배치하고, 박스 반환. 실패 시 None."""
    H, W = grid.shape
    if right_edge is None:
        right_edge = W - 1
    if forbidden_boxes is None:
        forbidden_boxes = []

    rows = max(1, len(bits)//2) if isinstance(bits, (list, tuple)) else 2
    cols = 2

    def _overlap_box(b0, b1):
        (a0, a1, a2, a3) = b0
        (b0_, b1_, b2_, b3_) = b1
        return not (a2 < b0_ or b2_ < a0 or a3 < b1_ or b3_ < a1)

    def _ok_place(top, left):
        r0, c0 = top, left
        r1, c1 = top + rows - 1, left + cols - 1
        if r0 < top_reserved_rows: return False
        if r1 > H - 1 - bottom_reserved_rows: return False
        if c0 < left_edge: return False
        if c1 > right_edge: return False
        for fb in forbidden_boxes:
            if _overlap_box((r0,c0,r1,c1), fb):
                return False
        if grid[r0:r1+1, c0:c1+1].any():
            return False
        return True

    rad = math.radians(angle_deg)
    def _pref_left(px):  # 좌우에 따라 left 보정
        return px if math.cos(rad) >= 0 else px - 1

    # 1차: 바깥으로 늘려가며 시도
    for t in range(max_tries):
        off   = base_offset + t
        px    = int(round(cx + (r + off) * math.cos(rad)))
        py    = int(round(cy + (r + off) * math.sin(rad)))
        left  = max(left_edge, min(right_edge - (cols-1), _pref_left(px)))
        top   = max(top_reserved_rows, min(H - bottom_reserved_rows - rows, py - rows//2))
        if _ok_place(top, left):
            return _draw_bits_block(grid, top, left, bits)

    # 2차: 위로 약간 올려서
    base_off = base_offset
    for dy in range(1, extra_up_shift+1):
        px = int(round(cx + (r + base_off) * math.cos(rad)))
        py = int(round(cy + (r + base_off) * math.sin(rad))) - dy
        left = max(left_edge, min(right_edge - (cols-1), _pref_left(px)))
        top  = max(top_reserved_rows, min(H - bottom_reserved_rows - rows, py - rows//2))
        if _ok_place(top, left):
            return _draw_bits_block(grid, top, left, bits)

    return None


def _paste_binary_clipped(dst, src, top, left,
                          row_min, row_max_excl,
                          col_min, col_max_excl):
    """src 비트맵을 dst에 안전 합성(클리핑)."""
    H, W = dst.shape
    sh, sw = src.shape
    r0 = max(row_min, top)
    c0 = max(col_min, left)
    r1 = min(row_max_excl, top + sh)
    c1 = min(col_max_excl, left + sw)
    if r0 >= r1 or c0 >= c1:
        return
    sr0 = r0 - top
    sc0 = c0 - left
    sr1 = sr0 + (r1 - r0)
    sc1 = sc0 + (c1 - c0)
    dst[r0:r1, c0:c1] = np.clip(dst[r0:r1, c0:c1] + src[sr0:sr1, sc0:sc1], 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2) 범례 패턴(시리즈/카테고리)
# ─────────────────────────────────────────────────────────────────────────────
def draw_series_pattern_fixed_bottom(grid, cx, bits, right_reserved_cols=0):
    """아래서 1~3번째 줄에 2열 패턴(정확히 3행=6비트)."""
    H, W = grid.shape
    band_top = max(0, H - 3)
    use_bits = (bits or [0,0,1,1,1,1])[:6]
    use_bits += [0] * (6 - len(use_bits))
    max_x = max(0, W - 2 - right_reserved_cols)
    left = max(0, min(max_x, cx - 1))
    _draw_bits_block(grid, band_top, left, use_bits)


# ─────────────────────────────────────────────────────────────────────────────
# 3) 본체: 60×40 파이차트 격자 생성
# ─────────────────────────────────────────────────────────────────────────────
def build_raster_grid_pie_bitmap(
    series_names, categories_per_series, values_per_series,
    series_bits, category_bits_per_series,
    request_id,
    W: int = 60, H: int = 40,
    gap: float = 0.10,
    dpi: int = 220,
    threshold: int = 240,
    edge_frac: float = 0.035,
    close_kernel: int = 0,
    desired_cell_gap: int = 2,
    legend_offset: int = 2,
    edge_margin_cols: int = 0,
    highlight_cfg: dict | None = None,
    highlight_mask: dict | None = None,
    slice_gap_cells: int = 1          # ← 추가: 조각 사이 간격(셀)
):

    """
    핵심 규칙
    - 전체 가로폭을 시리즈 개수 S로 균등 분할 → 각 시리즈는 자기 구역 안에서만 그림.
      예) S=2면 30열+30열, S=3면 20열씩.
    - 파이는 각 구역의 정확한 중앙(cx)·수직 중앙(cy 인 근처)에 배치.
    - 레전드는 각 웨지의 중앙각 바깥쪽에 2열 블록으로 찍되:
        * 같은 "행"에 두 레전드가 겹치지 않도록 전역 rows_used로 한 줄도 겹침 금지
        * 자기 구역 [L..R] 밖으로는 절대 배치하지 않음
    - 브로드캐스트 에러 방지 위해 항상 클리핑 합성 사용.
    - 하이라이트: 강조=채움, 비강조=윤곽만
    """
    import io, numpy as np, matplotlib.pyplot as plt
    from PIL import Image
    import cv2, math

    grid = np.zeros((H, W), dtype=np.uint8)
    S = max(1, len(series_names))

    # 하이라이트 판정기(백업용)
    highlight_fn = make_highlight_fn(highlight_cfg, series_names, categories_per_series)

    # ── 상/하단 예약
    def _estimate_max_cat_rows(bits_per_series):
        mx = 1
        if isinstance(bits_per_series, dict):
            for m in bits_per_series.values():
                if isinstance(m, dict):
                    for b in m.values():
                        if isinstance(b, (list, tuple)) and len(b) > 0:
                            mx = max(mx, len(b)//2 or 1)
        return mx

    top_reserved_rows    = max(1, _estimate_max_cat_rows(category_bits_per_series) + 1)  # 레전드 행 스냅용 여유
    bottom_reserved_rows = 3  # 아래 3줄은 시리즈 패턴

    # ── 수직 위치/반지름 상한
    usable_top    = top_reserved_rows
    usable_bottom = H - bottom_reserved_rows
    usable_h      = max(8, usable_bottom - usable_top)
    cy            = usable_top + usable_h // 2

    # 세로 제약으로 반지름 상한 결정(위쪽 레전드 공간 고려)
    top_room    = (cy - usable_top) - legend_offset - 1
    bottom_room = (usable_bottom - cy) - 1
    max_r_by_h  = max(4, min(top_room, bottom_room))

    # 전역 "행 사용" 마스크: 같은 행에 레전드 겹치기 금지
    rows_used = np.zeros(H, dtype=bool)

    # ---- 내부 유틸 -----------------------
    def _render_local(vals, local_side, radius):
        if not vals:
            return np.zeros((local_side, local_side), dtype=np.uint8)
        explode_base = max(
            gap,
            float(desired_cell_gap) / max(1, radius),
            float(slice_gap_cells) / max(1, radius)   # ← 추가
        )
        explode = [explode_base]*len(vals)
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
        wedges, _ = ax.pie(
            [max(0.0, float(v)) for v in vals],
            labels=None, startangle=90, explode=explode,
            wedgeprops=dict(edgecolor="none", linewidth=0.0, antialiased=True),
            counterclock=False
        )
        ax.set(aspect="equal"); ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close(fig); buf.seek(0)
        arr = np.array(Image.open(buf).convert("L"))
        bin_img = (arr < threshold).astype(np.uint8)
        small = Image.fromarray((bin_img*255).astype(np.uint8)).resize((local_side, local_side), Image.NEAREST)
        bin_local = (np.array(small) > 127).astype(np.uint8)
        edge_px = max(1, int(round(local_side * edge_frac)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_px*2+1), (edge_px*2+1))
        # 위 한 줄은 오타 우려 → 아래 줄로 대체
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_px*2+1, edge_px*2+1))
        eroded = cv2.erode(bin_local.astype(np.uint8), k, iterations=1)
        inner_ring = np.clip(bin_local - eroded, 0, 1).astype(np.uint8)
        out = np.clip(bin_local + inner_ring, 0, 1).astype(np.uint8)
        if close_kernel and close_kernel > 1:
            kk = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kk, iterations=1)
        return out

    def _render_local_with_outlines(vals, highlights, local_side, r, threshold, edge_frac, close_kernel, dpi=220):


        # 전체 파이 이진화 + 윤곽 추출
        def _render_binary():
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
            wedges, _ = ax.pie(
                [max(0.0, float(v)) for v in (vals or [1.0])],
                labels=None, startangle=90,
                explode=[0]*len(vals),  # 여기선 0 (조각 분리는 slice_mask erosion으로 처리)
                wedgeprops=dict(edgecolor="none", linewidth=0.0, antialiased=True),
                counterclock=False
            )
            ax.set(aspect="equal"); ax.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="white")
            plt.close(fig); buf.seek(0)
            arr = np.array(Image.open(buf).convert("L"))
            bin_img = (arr < threshold).astype(np.uint8)
            small = Image.fromarray((bin_img*255).astype(np.uint8)).resize((local_side, local_side), Image.NEAREST)
            bin_local = (np.array(small) > 127).astype(np.uint8)

            edge_px = max(1, int(round(local_side * edge_frac)))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_px*2+1, edge_px*2+1))
            eroded = cv2.erode(bin_local.astype(np.uint8), k, iterations=1)
            inner_ring = np.clip(bin_local - eroded, 0, 1).astype(np.uint8)

            if close_kernel and close_kernel > 1:
                kk = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
                bin_local = cv2.morphologyEx(bin_local, cv2.MORPH_CLOSE, kk, iterations=1)
            return bin_local, inner_ring

        bin_local, inner_ring = _render_binary()

        # 각도 기반 슬라이스 마스크
        h, w = local_side, local_side
        yy, xx = np.mgrid[0:h, 0:w]
        cy = h // 2; cx = w // 2
        dy = yy - cy; dx = xx - cx
        ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        rr  = np.sqrt(dx*dx + dy*dy)
        within_circle = (rr <= (r + 1)).astype(np.uint8)

        total = float(sum(max(0.0, float(v)) for v in (vals or [])) or 1.0)
        start_deg = 270.0
        cur = start_deg
        out = np.zeros_like(bin_local, dtype=np.uint8)

        # 조각 경계 사이에 **셀 단위 틈**을 만들기 위한 kernel
        gap_k = None
        if slice_gap_cells and slice_gap_cells > 0:
            g = int(slice_gap_cells)
            gap_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*g+1, 2*g+1))

        for i, v in enumerate(vals or []):
            frac = max(0.0, float(v)) / total
            span = 360.0 * frac
            if span <= 0: 
                continue
            a0 = cur % 360.0
            a1 = (cur + span) % 360.0
            cur += span

            if a0 <= a1: ang_mask = ((ang >= a0) & (ang < a1))
            else:        ang_mask = ((ang >= a0) | (ang < a1))
            slice_mask = (ang_mask & (within_circle == 1)).astype(np.uint8)

            # ★★ 핵심: 슬라이스 마스크를 erosion 해서 **이웃 조각과 떨어뜨림**
            if gap_k is not None:
                slice_mask = cv2.erode(slice_mask, gap_k, iterations=1)

            is_hi = bool(highlights[i]) if i < len(highlights) else True
            part_fill  = (bin_local  & slice_mask)
            part_ring  = (inner_ring & slice_mask)

            part = part_fill if is_hi else part_ring  # 강조=채움, 비강조=윤곽만
            out = np.clip(out + part, 0, 1)
        

# 시각화 및 저장

        return out


    def _fits_row_block(t, rows, tmin, tmax):
        if t < tmin or t+rows-1 > tmax: return False
        return not rows_used[t:t+rows].any()

    def _reserve_rows(t, rows):
        rows_used[t:t+rows] = True

    def _place_legend(bits, ang_deg, cx, cy, r, L, R, tmin, tmax, outline=False):
        """중앙각 바깥쪽 선호행 → 그 행이 비어있으면 [L..R] 내 좌→우 스캔으로 자리 잡기.
           비어있지 않으면 가까운 빈 행 찾음."""
        rows = max(1, len(bits)//2) if isinstance(bits, (list, tuple)) else 2
        cols = 2
        rad = math.radians(ang_deg)

        # 선호 행(top) 계산
        px = int(round(cx + (r + legend_offset) * math.cos(rad)))
        py = int(round(cy + (r + legend_offset) * math.sin(rad)))
        pref_top = max(tmin, min(tmax - rows + 1, py - rows//2))

        # 가까운 빈 행 탐색 (위/아래로 확장)
        cand_tops = [pref_top]
        for d in range(1, max(1, (tmax - tmin) // 2) + 1):
            up = pref_top - d
            down = pref_top + d
            if tmin <= up <= tmax - rows + 1:   cand_tops.append(up)
            if tmin <= down <= tmax - rows + 1: cand_tops.append(down)

        for top in cand_tops:
            if not _fits_row_block(top, rows, tmin, tmax):
                continue
            # 좌→우 스캔 (구역 내부에서만)
            # 중앙각 방향에 맞춰 시작 열을 대략 맞추고(좌측/우측에 따라), 없으면 전체 스캔
            start_left = px if math.cos(rad) >= 0 else px - 1
            start_left = max(L, min(R - (cols-1), start_left))
            order = list(range(start_left, R - (cols-1) + 1)) + list(range(L, start_left))
            for left in order:
                if not grid[top:top+rows, left:left+cols].any():
                    if outline:
                        _draw_bits_block_outline(grid, top, left, bits)
                    else:
                        _draw_bits_block(grid, top, left, bits)
                    _reserve_rows(top, rows)
                    return True
        return False
    # ------------------------------------------------------------------------

    # ── 가로를 S개 구역으로 "정확히" 균등 분할
    left_margin  = edge_margin_cols + 1
    right_margin = edge_margin_cols + 1
    full_left  = left_margin
    full_right = W - 1 - right_margin
    span = max(4, full_right - full_left + 1)

    series_sep_cols = 1 if S >= 2 else 0  # 시리즈 간 1칸 공백
    span_eff = max(1, span - series_sep_cols * (S - 1))  # 공백 제외한 실제 가용 폭


    base_w = span // S
    extra  = span - base_w * S  # 앞쪽부터 1칸씩 더 줌

    regions, centers, radii = [], [], []
    cur = full_left
    for i in range(S):
        w = base_w + (1 if i < extra else 0)
        L = cur
        R = L + w - 1
        # 다음 시리즈 시작점: 현재 구역 뒤 + 공백 1칸
        cur = R + 1 + series_sep_cols

        cx = (L + R) // 2
        # 가로 제약(좌우 1칸 여유)과 세로 제약 동시 고려
        max_r_by_w = max(4, (w - 2) // 2)
        Rcap = min(max_r_by_h, max_r_by_w)
        radius = max(4, Rcap)

        regions.append((L, R))
        centers.append(cx)
        radii.append(radius)

    # ── 시리즈별: 파이 합성 + 카테고리 레전드 + 시리즈 패턴
    for si, sname in enumerate(series_names):
        L, R = regions[si]
        cx   = int(centers[si])
        r    = int(radii[si])
        local_side = max(2*r + 3, 16)

        vals = values_per_series[si] if si < len(values_per_series) else []
        cats = categories_per_series[si] if si < len(categories_per_series) else []

        # 각 카테고리 강조 여부 생성
        if highlight_mask and sname in highlight_mask:
            cat_hi = [bool(highlight_mask[sname].get(cn, True)) for cn in cats]
        else:
            cat_hi = [bool(highlight_fn(sname, cn)) for cn in cats]

        # 1) 파이 비트맵 — 강조=채움, 비강조=윤곽만 + 조각 분리(erosion)
        local = _render_local_with_outlines(
            vals, cat_hi, local_side, r,
            threshold=threshold, edge_frac=edge_frac, close_kernel=close_kernel, dpi=dpi
        )
        top   = cy - local_side // 2
        left  = cx - local_side // 2
        _paste_binary_clipped(
            grid, local, top, left,
            row_min=usable_top, row_max_excl=usable_bottom,
            col_min=L,          col_max_excl=R+1
        )

        # 2) 카테고리 레전드(각 웨지 중앙각 바깥쪽) — 비강조는 윤곽만
        total = sum(max(0.0, float(v)) for v in (vals or [0])) or 1.0
        start_deg = -90.0
        cum = start_deg
        mid_angles = []
        for v in (vals or []):
            frac = max(0.0, float(v)) / total
            span_deg = 360.0 * frac
            mid_angles.append(cum + span_deg/2.0)
            cum += span_deg

        cmap = category_bits_per_series.get(sname, {}) if isinstance(category_bits_per_series, dict) else {}
        bits_list = [cmap.get(cn, [0,0,1,1]) for cn in cats]

        for ang, bits, cn, is_hi in zip(mid_angles, bits_list, cats, cat_hi):
            _ = _place_legend(
                bits, ang, cx, cy, r,
                L, R,
                tmin=top_reserved_rows, tmax=usable_bottom-1,
                outline=False
            )
            # 실패해도 그냥 넘어감

        # 3) 시리즈 패턴(아래 3줄 고정)
        s_bits = series_bits.get(sname, [0,0,1,1])
        draw_series_pattern_fixed_bottom(grid, cx, s_bits, right_reserved_cols=0)
    plt.figure(figsize=(6, 4))
    plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.axis("off")

        # PNG로 저장
    plt.savefig(f"static/binary/{request_id}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# 4) 참고용 Matplotlib 파이 이미지 저장 (라벨 비표시)
# ─────────────────────────────────────────────────────────────────────────────
def save_matplotlib_pie(
    series_names,
    categories_per_series,
    values_per_series,
    mpl_png_path: str,
    gap_frac: float = 0.12,          # 조각 사이 간격(폭 바깥으로 살짝)
    separator_width: float = 2.0,    # 조각 경계선 두께
    separator_color: str = "white",  # 조각 경계선 색(배경색과 맞추면 틈처럼 보임)
    donut_width: float = None,       # 도넛(0~1). 예: 0.6  -> 가운데가 뚫림
    scale: float = 6.0               # 해상도 스케일(크게 출력하려면 키우세요)
):
    """
    전체 플롯 비율을 항상 60:40(3:2)로 고정.
    시리즈 개수와 상관없이 3:2 캔버스에 모두 배치.
    글자는 파이 내부에는 퍼센트만, 카테고리명은 하단 레전드(캔버스 내부)로 정리.
    """
    n = max(1, len(series_names))

    # ---- 3:2 고정 캔버스 ----
    fig_w, fig_h = 3*scale, 2*scale
    fig = plt.figure(figsize=(fig_w, fig_h))

    # 상단/하단 여백을 미리 잡아 하단 레전드가 "캔버스 안"에 들어오도록 함
    top, bottom = 0.88, 0.20       # 하단 레전드 영역 확보
    left, right = 0.05, 0.95
    gs = fig.add_gridspec(
        1, n,
        left=left, right=right, top=top, bottom=bottom,
        wspace=0.28  # 파이들 사이 간격
    )
    axs = [fig.add_subplot(gs[0, i]) for i in range(n)]

    # 하단 레전드용 핸들/라벨 모으기(캔버스 안 하단에 배치)
    legend_handles = []
    legend_labels = []

    for i, ax in enumerate(axs):
        labels = categories_per_series[i] if i < len(categories_per_series) else []
        vals   = values_per_series[i]     if i < len(values_per_series)     else []

        explode = [gap_frac]*len(vals) if gap_frac and len(vals) > 0 else None
        wedgeprops = dict(linewidth=separator_width, edgecolor=separator_color)
        if donut_width is not None:
            # matplotlib pie의 width는 "반지름 두께"
            wedgeprops["width"] = 1 - donut_width

        wedges, _texts, autotexts = ax.pie(
            vals,
            labels=None,                 # 라벨은 파이 내부에 두지 않음(퍼센트만)
            explode=explode,
            startangle=90,
            counterclock=False,
            autopct="%1.1f%%",          # 퍼센트만 내부에
            pctdistance=0.72,           # 퍼센트 텍스트가 파이 내부로 들어오게
            wedgeprops=wedgeprops
        )

        # 폰트 크기 스케일링(캔버스가 커져도 보기 좋게)
        for t in autotexts:
            t.set_fontsize(10*scale/3)

        # 제목(시리즈명)은 캔버스 안 위쪽
        ax.set_title(series_names[i], pad=8, fontsize=11*scale/3)
        ax.set_aspect("equal")

        # 하단 레전드 항목(캔버스 내부)에 사용할 핸들/라벨 수집
        legend_handles.extend(wedges)
        legend_labels.extend(labels)

    # ---- 하단 레전드(캔버스 안) ----
    if legend_handles and legend_labels:
        ncol = min(max(len(legend_labels), 1), 6)
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.06),   # bottom=0.20 안쪽에 들어오도록
            ncol=ncol,
            frameon=False,
            fontsize=10*scale/3,
            handlelength=1.2,
            handletextpad=0.6,
            columnspacing=1.2
        )

    os.makedirs(os.path.dirname(mpl_png_path), exist_ok=True)
    plt.savefig(mpl_png_path, dpi=300)  # 큰 해상도
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5) 엔드포인트 (기본/하이라이트 적용)
# ─────────────────────────────────────────────────────────────────────────────

def pie_single_highlight(request_id: str):
    """
    입력:
      - static/chartQA_data/{id}.json  (파이차트 데이터; normalize_pie_spec 사용)
      - static/QA/{id}.json            (하이라이트 규칙)
    동작:
      - 하이라이트 규칙 반영(강조=채움, 비강조=윤곽만) 60×40 격자 생성
      - 참고용 파이 PNG를 static/img/{id}.png 저장
    반환:
      - 60×40 격자 리스트 (list[list[int]])
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

    # 1) 표준화
    try:
        series_names, categories_per_series, values_per_series, series_bits, category_bits_per_series = normalize_pie_spec(chart_data)
    except AssertionError as e:
        print("⚠️ 스펙 오류:", e)
        return []

    # 2) 하이라이트 마스크
    hi_mask = build_pie_highlight_mask(series_names, categories_per_series, gpt)

        # 3) 격자 생성 (비강조는 윤곽-only)
    grid = build_raster_grid_pie_bitmap(
        series_names, categories_per_series, values_per_series,
        series_bits, category_bits_per_series, request_id = request_id,
        W=60, H=40,
        highlight_cfg=gpt,
        highlight_mask=hi_mask,
        slice_gap_cells=1   # ← 1~3 정도 추천. 붙지 않게 확실히 하려면 2~3
    )


    # 4) 참고 PNG
    save_matplotlib_pie(series_names, categories_per_series, values_per_series, mpl_out)

    arr = np.asarray(grid)
    print("shape:", arr.shape)                 # (40, 60)
    print("bottom 3 rows sum:", arr[-3:, :].sum())
    return grid.astype(int).tolist()
