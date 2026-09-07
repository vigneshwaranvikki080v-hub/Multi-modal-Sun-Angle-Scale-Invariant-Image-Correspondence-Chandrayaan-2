"""
Generate a synthetic crater-like demo image pair for the registration pipeline.

This is ONLY a stand-in for the real Chandrayaan-2 TMC-2 pair — use it to
sanity-check register.py end-to-end and to record a quick demo video.
Swap in the real .IMG pair (via load_pds_image) before you submit if your
slide claims the real Tycho Crater pair.

Usage:
    python3 make_demo_pair.py [--out demo_data]

Produces:
    demo_data/source.png   (the "Oct" image)
    demo_data/target.png   (the "Dec" image — rotated + scaled + darker)
"""

import argparse
import os

import cv2
import numpy as np


def make_crater_image(size=600, n_craters=35, seed=42):
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 60, dtype=np.uint8)

    # base terrain texture (mild noise, smoothed)
    noise = rng.normal(0, 12, (size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=6)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # scattered craters: dark rim + bright/dark shading to fake a light source
    for _ in range(n_craters):
        cx = rng.integers(40, size - 40)
        cy = rng.integers(40, size - 40)
        r = rng.integers(8, 45)

        overlay = img.copy().astype(np.int32)
        cv2.circle(overlay, (cx, cy), r, -70, -1)               # crater floor (dark)
        cv2.circle(overlay, (cx, cy), r, 40, 3)                 # rim highlight
        cv2.circle(overlay, (cx - r // 3, cy - r // 3), max(r // 3, 2), 60, -1)  # sunlit inner wall
        img = np.clip(overlay, 0, 255).astype(np.uint8)

    return img


def make_target_from_source(src, angle=7.0, scale=0.93, brightness_shift=-30):
    h, w = src.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(src, M, (w, h), borderValue=60)

    # simulate a ~2-month sun-angle shift: darker + slightly different contrast
    shifted = rotated.astype(np.float32) * 0.85 + brightness_shift
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    return shifted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_data")
    ap.add_argument("--size", type=int, default=600)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    source = make_crater_image(size=args.size)
    target = make_target_from_source(source)

    cv2.imwrite(os.path.join(args.out, "source.png"), source)
    cv2.imwrite(os.path.join(args.out, "target.png"), target)

    print(f"Wrote {args.out}/source.png and {args.out}/target.png")


if __name__ == "__main__":
    main()
