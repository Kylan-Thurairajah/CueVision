import cv2 as cv
import numpy as np

# Manual override thresholds
GREEN_LOW  = (30,  30,  30)
GREEN_HIGH = (90, 255, 255)
BLUE_LOW   = (90,  30,  30)
BLUE_HIGH  = (130,255,255)

TOPDOWN_W, TOPDOWN_H = 1024, 512  # ~2:1 aspect

def _morph(mask):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    return mask

def mask_felt(bgr, mode="auto"):
    """Return binary felt mask. mode: 'auto', 'green', 'blue'."""
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    if mode == "green":
        return _morph(cv.inRange(hsv, GREEN_LOW, GREEN_HIGH))
    if mode == "blue":
        return _morph(cv.inRange(hsv, BLUE_LOW, BLUE_HIGH))

    # AUTO: pick dominant hue among saturated/bright pixels
    Smin, Vmin = 40, 40
    satbright = cv.inRange(hsv, (0, Smin, Vmin), (179, 255, 255))
    hue = hsv[...,0]
    hist = cv.calcHist([hue],[0], satbright, [180], [0,180])
    peak = int(hist.argmax())
    band = 12  # +/- around peak
    low = (peak - band) % 180
    high = (peak + band) % 180

    if low <= high:
        mask_h = cv.inRange(hue, low, high)
    else:
        mask_h = cv.bitwise_or(cv.inRange(hue, 0, high), cv.inRange(hue, low, 179))

    mask = cv.bitwise_and(satbright, mask_h)
    return _morph(mask)

def _order_corners_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    order = np.argsort(ang)
    pts = pts[order]
    idx = np.argmin(pts.sum(axis=1))  # TL first
    pts = np.vstack([pts[idx:], pts[:idx]])
    return pts

def find_table_quad(bgr, felt_mode="auto"):
    """Find felt via color mask; return 4-point quad (TL,TR,BR,BL) and mask."""
    mask = mask_felt(bgr, mode=felt_mode)
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    cnt = max(cnts, key=cv.contourArea)
    if cv.contourArea(cnt) < 0.05 * (bgr.shape[0]*bgr.shape[1]):
        return None, mask

    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx) >= 4:
        hull = cv.convexHull(approx)
        if len(hull) >= 4:
            if len(hull) > 4:
                rect = cv.minAreaRect(hull)
                box = cv.boxPoints(rect)
                quad = _order_corners_clockwise(box)
            else:
                quad = _order_corners_clockwise(hull.reshape(-1,2))
        else:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            quad = _order_corners_clockwise(box)
    else:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        quad = _order_corners_clockwise(box)

    return quad, mask

def warp_topdown(bgr, quad):
    dst = np.float32([[0,0],[TOPDOWN_W-1,0],[TOPDOWN_W-1,TOPDOWN_H-1],[0,TOPDOWN_H-1]])
    H = cv.getPerspectiveTransform(np.float32(quad), dst)
    warp = cv.warpPerspective(bgr, H, (TOPDOWN_W, TOPDOWN_H))
    return warp, H

def table_pockets(ball_radius=None):
    """Return 6 pocket centers inset from rails to avoid rail-grazing paths."""
    w, h = TOPDOWN_W, TOPDOWN_H
    inset = int((ball_radius or max(h//28,6)) * 1.4)
    return np.array([
        [inset, inset], [w//2, inset], [w-1-inset, inset],
        [inset, h-1-inset], [w//2, h-1-inset], [w-1-inset, h-1-inset]
    ], dtype=np.float32)

def draw_overlay(warp_bgr, balls, cue_idx, best):
    vis = warp_bgr.copy()
    for idx,(x,y,r) in enumerate(balls):
        color = (0,255,0) if idx==cue_idx else (0,165,255)
        cv.circle(vis, (int(x),int(y)), int(r), color, 2)
        cv.putText(vis, str(idx), (int(x)-6,int(y)-8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
    for (px,py) in table_pockets().astype(int):
        cv.circle(vis, (px,py), 10, (255,0,0), 2)
    if best:
        i = best["object_idx"]
        C = (int(balls[cue_idx][0]), int(balls[cue_idx][1]))
        G = (int(best["ghost"][0]), int(best["ghost"][1]))
        O = (int(balls[i][0]), int(balls[i][1]))
        P = (int(best["pocket"][0]), int(best["pocket"][1]))
        cv.line(vis, C, G, (0,255,255), 2)
        cv.line(vis, O, P, (255,0,255), 2)
        cv.circle(vis, G, 6, (0,255,255), -1)
        cv.circle(vis, P, 8, (255,0,255), 2)
    return vis
