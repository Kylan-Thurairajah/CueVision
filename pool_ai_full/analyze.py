import cv2 as cv
from .vision import find_table_quad, warp_topdown, draw_overlay
from .detectors import HoughBallDetector
from .planning import pick_best_shot, advice_from_best

def analyze_pool_image(bgr, felt_mode='auto', detector='hough'):
    # 1) Rectify via felt
    quad, mask = find_table_quad(bgr, felt_mode=felt_mode)
    if quad is not None:
        warp, H = warp_topdown(bgr, quad)
        rectified = True
    else:
        warp, H = bgr, None
        rectified = False

    # 2) Detect balls (Hough by default)
    if detector == 'hough':
        det = HoughBallDetector(felt_mode=felt_mode)
        balls, cue_idx = det.detect(warp)
    else:
        raise ValueError("Unsupported detector; use 'hough'.")

    # 3) Plan shot
    best = pick_best_shot(warp, balls, cue_idx) if balls else None
    advice = advice_from_best(best, balls, cue_idx) if balls else "No balls detected."
    overlay = draw_overlay(warp, balls, cue_idx, best) if balls else warp.copy()

    return {
        "rectified": rectified,
        "num_balls": len(balls),
        "cue_idx": cue_idx,
        "best": best,
        "advice": advice,
        "overlay": overlay,
        "warp": warp
    }
