import cv2 as cv
import numpy as np

from .vision import mask_felt

class HoughBallDetector:
    """Classical Hough-based detector (no ML)."""
    def __init__(self, felt_mode="auto"):
        self.felt_mode = felt_mode
        self.HOUGH_DP = 1.2
        self.PARAM1 = 100
        self.PARAM2 = 14   # lower → more circles (and more false positives)
        self.BLUR = (9,9)

    @staticmethod
    def estimate_ball_radius_px(h):
        r_est = max(6, int(h/28))
        return int(r_est*0.7), int(r_est*1.4)

    def detect(self, warp_bgr):
        hsv = cv.cvtColor(warp_bgr, cv.COLOR_BGR2HSV)
        felt = mask_felt(warp_bgr, mode=self.felt_mode)
        not_felt = cv.bitwise_not(felt)
        fg = cv.GaussianBlur(not_felt, self.BLUR, 1.5)

        minR, maxR = self.estimate_ball_radius_px(warp_bgr.shape[0])
        circles = cv.HoughCircles(
            fg, cv.HOUGH_GRADIENT, dp=self.HOUGH_DP,
            minDist=max(10, minR*2),
            param1=self.PARAM1, param2=self.PARAM2,
            minRadius=minR, maxRadius=maxR
        )
        balls = []
        if circles is not None:
            for (x,y,r) in np.round(circles[0,:]).astype(int):
                if 3 < x < warp_bgr.shape[1]-3 and 3 < y < warp_bgr.shape[0]-3:
                    balls.append((x,y,r))
        if not balls:
            return [], None

        # Cue ball heuristic: lowest S, highest V
        cuescore = []
        for i,(x,y,r) in enumerate(balls):
            y0,y1 = max(0,y-r), min(y+r, hsv.shape[0])
            x0,x1 = max(0,x-r), min(x+r, hsv.shape[1])
            patch = hsv[y0:y1, x0:x1]
            if patch.size == 0:
                cuescore.append((i, 1e9)); continue
            S = patch[...,1].astype(np.float32).mean()
            V = patch[...,2].astype(np.float32).mean()
            score = S - 0.5*V
            cuescore.append((i, score))
        cue_idx = sorted(cuescore, key=lambda t: t[1])[0][0]
        return balls, cue_idx

class TRTBallsDetector:
    """Placeholder: swap to TensorRT for learned detection later."""
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        raise NotImplementedError("TensorRT detector to be implemented later.")
