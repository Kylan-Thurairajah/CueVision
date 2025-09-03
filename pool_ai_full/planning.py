import numpy as np
from math import atan2, degrees
from .vision import table_pockets

CUT_ANGLE_MAX_DEG = 80
CLEARANCE_MARGIN  = 1.0

def dist_point_to_segment(p, a, b):
    ap = p - a; ab = b - a
    t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9), 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def path_clear(a, b, balls, ignore_indices, clearance):
    a = np.array(a, dtype=np.float32); b = np.array(b, dtype=np.float32)
    for i,(x,y,r) in enumerate(balls):
        if i in ignore_indices:
            continue
        if dist_point_to_segment(np.array([x,y], np.float32), a, b) < clearance:
            return False
    return True

def pick_best_shot(warp_bgr, balls, cue_idx):
    # Estimate typical ball radius for pocket inset and clearance
    r_mean = float(np.mean([r for _,_,r in balls])) if balls else 10.0
    pockets = table_pockets(ball_radius=r_mean)
    cue = np.array(balls[cue_idx][:2], dtype=np.float32)
    best = None

    # small lateral jitter around ghost point to tolerate tiny occlusions
    jitters = np.linspace(-0.8, 0.8, 5)  # in units of ball radii

    for i,(ox,oy,r) in enumerate(balls):
        if i == cue_idx:
            continue
        O = np.array([ox,oy], dtype=np.float32)
        for P in pockets:
            d = P - O; dn = d / (np.linalg.norm(d)+1e-9)
            # tangent unit vector for jitter
            t = np.array([-dn[1], dn[0]], dtype=np.float32)

            # try a few jittered aim points around ideal ghost
            for j in jitters:
                G = O - (2.0*r) * dn + (j * r) * t  # per-ball radius, slight lateral offset
                # clearance checks
                clr1 = path_clear(cue, G, balls, (cue_idx,i), r*CLEARANCE_MARGIN)
                clr2 = path_clear(O, P,   balls, (cue_idx,i), r*CLEARANCE_MARGIN)
                if not (clr1 and clr2):
                    continue

                # cut angle
                v1 = O - cue; v2 = P - O
                a1, a2 = atan2(v1[1], v1[0]), atan2(v2[1], v2[0])
                cut_deg = abs(degrees(a2 - a1))
                if cut_deg > 180: cut_deg = 360 - cut_deg
                if cut_deg > CUT_ANGLE_MAX_DEG:
                    continue

                pocket_dist = np.linalg.norm(P - O)
                score = 1000 - (cut_deg*5.5) - pocket_dist*0.22
                cand = dict(score=score, object_idx=i, pocket=P, ghost=G, cut_deg=cut_deg)
                if (best is None) or (score > best["score"]):
                    best = cand
    return best

def advice_from_best(best, balls, cue_idx):
    if not best:
        return "No clear direct pot; consider a safety or reposition."
    pockets = ["TL","TM","TR","BL","BM","BR"]
    labels_xy = table_pockets()
    label = pockets[int(np.argmin(np.linalg.norm(labels_xy - best["pocket"], axis=1)))]
    i = best["object_idx"]
    return f"Pot ball #{i} to {label}. Aim at the ghost point; cut ≈ {best['cut_deg']:.0f}°. Controlled speed."
