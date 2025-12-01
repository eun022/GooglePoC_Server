from fastapi import APIRouter
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import mediapipe as mp

from sk.utils.centers import (
    detect_aruco,
    detect_index_tips_bgr,
    quad_to_rect_homography,
)

mp_hands = mp.solutions.hands

router = APIRouter()   # 🔥 FastAPI 앱이 아니라 Router 이어야 함

ROI_SCALE = 1.5


class ROIRequest(BaseModel):
    image: str  # base64 이미지


class TipRequest(BaseModel):
    image: str       # base64
    roiSize: list    # [W, H]
    H_mat: list      # 3x3 matrix


def decode_frame(b64):
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ===============================
# 1) ROI Calibration
# ===============================
@router.post("/calibrate-roi")
def calibrate_roi(req: ROIRequest):
    frame = decode_frame(req.image)
    display = frame.copy()

    ok, centers, ordered_quad = detect_aruco(display, "DICT_4X4_50", draw=True)
    if not ordered_quad is not None:
        return {
            "ok": False,
            "reason": "Aruco 4개가 모두 감지되지 않았습니다."
        }

    _, (W0, H0), _ = quad_to_rect_homography(ordered_quad, None)
    new_wh = (int(W0 * ROI_SCALE), int(H0 * ROI_SCALE))

    H_mat, (W, H), _ = quad_to_rect_homography(ordered_quad, new_wh)

    # ROI 이미지 생성
    roi_img = cv2.warpPerspective(frame, H_mat, (W, H), flags=cv2.INTER_CUBIC)
    ok2, buf = cv2.imencode(".jpg", roi_img)
    roi_b64 = base64.b64encode(buf).decode("utf-8") if ok2 else None

    # Debug
    cv2.putText(display, f"ROI calibrated: {W}x{H}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    ok3, buf2 = cv2.imencode(".jpg", display)
    dbg_b64 = base64.b64encode(buf2).decode("utf-8") if ok3 else None

    return {
        "ok": True,
        "roi_w": W,
        "roi_h": H,
        "roi_size_str": f"{W}x{H}",
        "roi_image": roi_b64,
        "debug_image": dbg_b64,
        "H_mat": H_mat.tolist(),
    }


# ===============================
# 2) Finger Tip Detection
# ===============================
@router.post("/finger-tip")
def finger_tip(req: TipRequest):
    frame = decode_frame(req.image)
    display = frame.copy()

    W, H = req.roiSize
    H_mat = np.array(req.H_mat)

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    ok_tips, tips_px = detect_index_tips_bgr(display, hands, draw=False, MP_AVAILABLE=True)
    if (not ok_tips) or (not tips_px):
        return {
            "ok": False,
            "reason": "검지 손가락을 찾지 못했습니다."
        }

    tip = tips_px[0]
    tip_arr = np.array([[tip]], dtype=np.float32)

    # Frame → ROI 좌표 변환
    pts_roi = cv2.perspectiveTransform(tip_arr, H_mat).reshape(-1, 2)
    x_roi, y_roi = float(pts_roi[0][0]), float(pts_roi[0][1])

    x_norm = x_roi / W
    y_norm = y_roi / H

    GRID_W, GRID_H = 60, 40
    gx = max(0, min(GRID_W - 1, int(round(x_norm * (GRID_W - 1)))))
    gy = max(0, min(GRID_H - 1, int(round(y_norm * (GRID_H - 1)))))

    cv2.circle(display, (int(tip[0]), int(tip[1])), 6, (0, 0, 255), -1)
    cv2.putText(display, f"ROI tip=({x_roi:.1f},{y_roi:.1f}) grid=({gx},{gy})",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    ok2, buf = cv2.imencode(".jpg", display)
    dbg_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "ok": True,
        "tip_frame": [float(tip[0]), float(tip[1])],
        "tip_roi": [x_roi, y_roi],
        "roi_w": W,
        "roi_h": H,
        "tip_norm": [x_norm, y_norm],
        "grid": [gx, gy],
        "debug_image": dbg_b64,
    }
