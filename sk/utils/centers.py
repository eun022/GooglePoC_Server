import cv2
import numpy as np
import argparse
import sys
import time




# ---------- Drawing helpers ----------
def draw_cross(img, pt, size=8, color=(0,255,0), thickness=2):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)

def compute_center(corners: np.ndarray) -> tuple:
    c = corners.reshape(-1, 2)  # (4,2)
    cx, cy = c.mean(axis=0)
    return float(cx), float(cy)

def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
    order = np.argsort(-angles)  # clockwise
    return pts[order]

def order_points_rect(points: np.ndarray) -> np.ndarray:
    """
    4개의 점을 (Top-Left, Top-Right, Bottom-Right, Bottom-Left) 순서로 정렬
    """
    pts = points.astype(np.float32)
    s = pts.sum(axis=1)              # x+y
    d = (pts[:,0] - pts[:,1])        # x-y

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

# 파일 상단 어딘가에 추가
def order_points_by_ids(centers, id_map=(0,1,2,3)):
    """
    centers: [(id, (cx,cy)), ...]
    id_map: (TL, TR, BL, BR) 에 해당하는 마커 IDs
            기본: 0=TL, 1=TR, 2=BL, 3=BR
    return: np.float32 [[TL],[TR],[BR],[BL]]  (주의: dst로 갈 때 BR, BL 순서 필요)
    """
    d = {mid: np.array(cxy, np.float32) for mid, cxy in centers}
    if all(mid in d for mid in id_map):
        TL = d[id_map[0]]
        TR = d[id_map[1]]
        BL = d[id_map[2]]
        BR = d[id_map[3]]
        # 우리 homography는 [TL,TR,BR,BL] 순으로 기대
        return np.array([TL, TR, BR, BL], dtype=np.float32)
    return None


def is_down_key(key: int) -> bool:
    # 다양한 백엔드 방향키 하위호환 + 보조 'j'
    return key in (0x280000, 2621440, 65364, ord('j'))

# === 4점 → 직사각형 Homography ===
def quad_to_rect_homography(quad_pts: np.ndarray, target_wh=None):
    w_top  = np.linalg.norm(quad_pts[1] - quad_pts[0])
    w_bot  = np.linalg.norm(quad_pts[2] - quad_pts[3])
    h_left = np.linalg.norm(quad_pts[3] - quad_pts[0])
    h_right= np.linalg.norm(quad_pts[2] - quad_pts[1])

    W = int(round(max(w_top, w_bot)))
    H = int(round(max(h_left, h_right)))
    if target_wh is not None:
        W, H = target_wh
    W, H = max(50, W), max(50, H)

    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    H_mat = cv2.getPerspectiveTransform(quad_pts.astype(np.float32), dst)
    return H_mat, (W, H), dst

# ---------- ArUco detection ----------
def detect_aruco(frame, aruco_dict_name="DICT_4X4_50", expect_count=4, draw=True):
    if not hasattr(cv2, "aruco"):
        return frame, [], None
    if not hasattr(cv2.aruco, aruco_dict_name):
        raise ValueError(f"Unsupported ArUco dict: {aruco_dict_name}")

    if hasattr(cv2.aruco, "DetectorParameters_create"):
        detector_params = cv2.aruco.DetectorParameters_create()
    else:
        detector_params = cv2.aruco.DetectorParameters()

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = detector.detectMarkers(gray)

    centers, ordered_quad = [], None
    if ids is not None and len(ids) > 0:
        if draw:
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)
        # 마커 중심점으로 ROI 계산
        # for corners, id_ in zip(corners_list, ids.flatten()):
        #     cxy = compute_center(corners)
        #     centers.append((int(id_), cxy))
        #     if draw:
        #         draw_cross(frame, cxy, 10, (0,255,0), 2)
        #         cv2.putText(frame, f"ID {int(id_)}", (int(cxy[0])+8, int(cxy[1])-8),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)

        # --- (B) 추가: "모서리 기반 ROI"용 4점 구성 ---
        # OpenCV는 각 마커 코너를 TL, TR, BR, BL 순서로 반환합니다: corners[0][0..3]
        # id → 해당 마커에서 사용할 코너 인덱스 매핑(예시)
        #   TL 마커(ID 0): 그 마커의 TR 코너(1) 같은 식으로 원하는 조합 가능
        #   여기서는 간단히: TL=0, TR=1, BR=2, BL=3 각 마커의 '같은 번호' 코너를 사용한다고 가정
        # id_to_corners = {int(i): c.reshape(4,2).astype(np.float32) for c, i in zip(corners_list, ids.flatten())}

        # if all(mid in id_to_corners for mid in (0,1,2,3)):  # 0,1,2,3이 다 보일 때
        #     # 각 마커에서 같은 인덱스의 코너를 뽑아 TL,TR,BR,BL 구성
        #     TL = id_to_corners[0][2]  # BL
        #     TR = id_to_corners[1][3]  # BR
        #     BR = id_to_corners[3][1]  # TR
        #     BL = id_to_corners[2][0]  # TL
        #     ordered_quad = np.array([TL, TR, BR, BL], dtype=np.float32)

        # --- (B') 고정 ID 매핑으로 '안쪽 모서리'를 선택 (ID 확정 배치일 때만) ---
        id_to_corners = {int(i): c.reshape(4,2).astype(np.float32) 
                        for c, i in zip(corners_list, ids.flatten())}

        ordered_quad = None
        inner_index_map = {  # 각 마커(ID)의 '안쪽' 코너 인덱스
            0: 2,  # TL marker -> BR corner
            1: 3,  # TR marker -> BL corner
            2: 1,  # BL marker -> TR corner
            3: 0,  # BR marker -> TL corner
        }
        id_map = (0, 1, 3, 2)  # ROI 순서(TL,TR,BR,BL)로 사용할 마커 ID 나열 (환경에 맞게 조정)

        if all(mid in id_to_corners for mid in id_map):
            TL = id_to_corners[id_map[0]][ inner_index_map[id_map[0]] ]
            TR = id_to_corners[id_map[1]][ inner_index_map[id_map[1]] ]
            BR = id_to_corners[id_map[2]][ inner_index_map[id_map[2]] ]
            BL = id_to_corners[id_map[3]][ inner_index_map[id_map[3]] ]
            ordered_quad = np.array([TL, TR, BR, BL], dtype=np.float32)

        # (선택) 디버그 드로잉
        if draw and ordered_quad is not None:
            for i in range(4):
                p1 = tuple(ordered_quad[i].astype(int))
                p2 = tuple(ordered_quad[(i+1)%4].astype(int))
                cv2.line(frame, p1, p2, (255,0,0), 2, cv2.LINE_AA)

        centers.sort(key=lambda x: x[0])
        if len(centers) >= expect_count:
            # ordered_quad = order_points_clockwise(np.array([c for _, c in centers[:expect_count]], np.float32))
            # ordered_quad = order_points_rect(np.array([c for _, c in centers[:expect_count]], np.float32))
            # ❶ ID 고정 매핑 우선
            ordered_quad = order_points_by_ids(centers, id_map=(0,1,2,3))
            # ❷ 혹시 0~3이 아닐 때만 안전한 백업 휴리스틱
            if ordered_quad is None:
                ordered_quad = order_points_rect(
                    np.array([c for _, c in centers[:expect_count]], np.float32)
                )
            if draw:
                for i in range(4):
                    p1 = tuple(ordered_quad[i].astype(int))
                    p2 = tuple(ordered_quad[(i+1)%4].astype(int))
                    cv2.line(frame, p1, p2, (255,0,0), 2, cv2.LINE_AA)
                centroid = ordered_quad.mean(axis=0)
                draw_cross(frame, centroid, 12, (0,0,255), 2)
                cv2.putText(frame, "FOUR-CENTER", (int(centroid[0])+8, int(centroid[1])-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

    return frame, centers, ordered_quad


# ---------- MediaPipe Index Tip detection ----------
def detect_index_tip_bgr(frame_bgr, hands_ctx=None, draw=True, MP_AVAILABLE=True):
    if not MP_AVAILABLE or hands_ctx is None:
        return False, (None, None)
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands_ctx.process(rgb)
    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        tip = hand.landmark[8]
        x_px, y_px = tip.x * w, tip.y * h
        if draw:
            draw_cross(frame_bgr, (x_px, y_px), 10, (0,200,255), 2)
            cv2.putText(frame_bgr, f"IndexTip ({int(x_px)}, {int(y_px)})",
                        (int(x_px)+8, int(y_px)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2, cv2.LINE_AA)
        return True, (float(x_px), float(y_px))
    return False, (None, None)

# ----------- 여러 손의 검지 TIP 탐지 -----------
def detect_index_tips_bgr(frame_bgr, hands_ctx=None, draw=True, MP_AVAILABLE=True):
    """
    여러 손의 '검지 끝' 픽셀 좌표들을 반환.
    반환: (found_any: bool, tips_px: List[(x, y)])
    """
    if not MP_AVAILABLE or hands_ctx is None:
        return False, []

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands_ctx.process(rgb)

    tips = []
    if res.multi_hand_landmarks:
        for i, hand in enumerate(res.multi_hand_landmarks):
            tip = hand.landmark[8]  # INDEX_FINGER_TIP
            x_px, y_px = float(tip.x * w), float(tip.y * h)
            tips.append((x_px, y_px))
            if draw:
                draw_cross(frame_bgr, (x_px, y_px), 10, (0, 200, 255), 2)
                cv2.putText(frame_bgr,
                            f"Tip{i+1} ({int(x_px)},{int(y_px)})",
                            (int(x_px)+8, int(y_px)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2, cv2.LINE_AA)
    return (len(tips) > 0), tips

def save_roi_with_tip(last_clean_roi, last_quad, roi_target_wh, last_index_tip_px, radius=3):
    """
    ROI 이미지에 손가락 끝 점을 표시하고 저장합니다.

    Args:
        last_clean_roi (np.ndarray): ROI 이미지
        last_quad (np.ndarray): 4점 쿼드 좌표
        roi_target_wh (tuple): ROI target width, height
        last_index_tip_px (tuple): 손가락 끝 좌표 (x, y)
        radius (int): 표시할 원의 반지름
    Returns:
        str or None: 저장된 파일명, 준비되지 않았으면 None
    """
    if (last_clean_roi is None) or (last_quad is None) or (roi_target_wh is None) or (last_index_tip_px is None):
        print("[SAVE] 실패: ROI/quad/tip 중 일부가 준비되지 않았습니다. 손을 치우고 ROI가 한 번 스냅되게 하세요.")
        return None

    H_mat, (W, H), _ = quad_to_rect_homography(last_quad, roi_target_wh)
    pt = cv2.perspectiveTransform(np.array([[last_index_tip_px]], np.float32), H_mat)[0, 0]

    canvas = last_clean_roi.copy()
    cv2.circle(canvas, (int(round(pt[0])), int(round(pt[1]))), radius, (0, 0, 255), -1, cv2.LINE_AA)

    fname = f"roi_with_point_{int(time.time())}.png"
    cv2.imwrite(fname, canvas)
    print(f"[SAVE] {fname} (ROI {canvas.shape[1]}x{canvas.shape[0]})")

    return fname