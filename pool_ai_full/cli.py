#!/usr/bin/env python3
import argparse, sys, cv2 as cv
from .analyze import analyze_pool_image

def main():
    p = argparse.ArgumentParser(description="Pool table analyzer (Jetson-ready, no ML).")
    p.add_argument("image", help="Input pool table image path")
    p.add_argument("--felt", choices=["auto","green","blue"], default="auto", help="Felt color mode")
    p.add_argument("--detector", choices=["hough"], default="hough", help="Ball detector backend")
    p.add_argument("--overlay", help="Save annotated overlay PNG path", default=None)
    args = p.parse_args()

    img = cv.imread(args.image)
    if img is None:
        print("Could not read input image.", file=sys.stderr)
        sys.exit(1)

    res = analyze_pool_image(img, felt_mode=args.felt, detector=args.detector)
    print(res["advice"])
    if args.overlay:
        cv.imwrite(args.overlay, res["overlay"])

if __name__ == "__main__":
    main()
