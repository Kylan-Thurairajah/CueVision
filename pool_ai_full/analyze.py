import cv2 as cv
from .vision import find_table_quad, warp_topdown, draw_overlay
from .detectors import HoughBallDetector
from .planning import pick_best_shot, advice_from_best

def analyze_pool_image(bgr, felt_mode='auto', detector='hough'):
    # 1) Rectify via felt
    quad, _ = find_table_quad(bgr, felt_mode=felt_mode)
    rectified = quad is not None
    warp = warp_topdown(bgr, quad)[0] if rectified else bgr

    # 2) Detect balls only on the calibrated table.
    if detector != 'hough':
        raise ValueError("Unsupported detector; use 'hough'.")
    balls, cue_idx = HoughBallDetector(felt_mode=felt_mode).detect(warp) if rectified else ([], None)

    # 3) Plan shot
    best = pick_best_shot(warp, balls, cue_idx) if cue_idx is not None else None
    if not rectified:
        outcome = "no_usable_table"
        advice = "Could not find the table. Retake the photo with the whole table in view."
    elif cue_idx is None:
        outcome = "no_cue_ball"
        advice = "No cue ball detected. Retake the photo with the cue ball visible."
    else:
        outcome = "recommendation" if best is not None else "no_clear_pot"
        advice = advice_from_best(best, balls, cue_idx)
    overlay = draw_overlay(warp, balls, cue_idx, best) if balls else warp.copy()

    return {
        "rectified": rectified,
        "outcome": outcome,
        "num_balls": len(balls),
        "cue_idx": cue_idx,
        "best": best,
        "advice": advice,
        "overlay": overlay,
        "warp": warp
    }
