"""
Lunar Image Registration - Baseline Pipeline (SIFT + RANSAC)
SIH26166 - Lunar Image Correspondence & Registration

Usage:
    python3 register.py <image1_path> <image2_path> [--out OUTDIR]

Works on any 2D grayscale-convertible image (PNG/JPG for demo,
or a GDAL-read .IMG converted to array — see load_pds_image() below
for the real Chandrayaan-2 TMC-2 path).
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. LOADING
# ---------------------------------------------------------------------------

def load_image_generic(path):
    """Load a normal image file (png/jpg) as grayscale uint8."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_pds_image(xml_path):
    """
    Load a real Chandrayaan-2 TMC-2/OHRC/IIRS PDS4 product via GDAL.
    Point this at the .xml label (GDAL follows it to the .img data file).

    Requires: pip install gdal
    """
    from osgeo import gdal
    ds = gdal.Open(xml_path)
    if ds is None:
        raise RuntimeError(f"GDAL could not open {xml_path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    # Normalize to 8-bit for SIFT (raw DN values are often 12/16-bit)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess(gray):
    """CLAHE for illumination normalization (important across different
    sun-angle passes, e.g. the Oct-26 vs Dec-22 pair)."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# 3. FEATURES + MATCHING
# ---------------------------------------------------------------------------

def detect_and_match(img1, img2, ratio=0.75):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Not enough keypoints detected — check image content/contrast.")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    return kp1, kp2, good


# ---------------------------------------------------------------------------
# 4. RANSAC HOMOGRAPHY
# ---------------------------------------------------------------------------

def estimate_homography(kp1, kp2, good_matches, reproj_thresh=4.0):
    if len(good_matches) < 4:
        raise RuntimeError(f"Only {len(good_matches)} good matches — need >=4 for homography.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    inlier_mask = mask.ravel().astype(bool)
    return H, inlier_mask, src_pts, dst_pts


# ---------------------------------------------------------------------------
# 5. METRICS
# ---------------------------------------------------------------------------

def compute_metrics(H, src_pts, dst_pts, inlier_mask, kp1, kp2, good_matches, t_elapsed):
    n_matches = len(good_matches)
    n_inliers = int(inlier_mask.sum())
    inlier_ratio = n_inliers / n_matches if n_matches else 0.0

    # RMSE of reprojection error (inliers only)
    src_in = src_pts[inlier_mask]
    dst_in = dst_pts[inlier_mask]
    projected = cv2.perspectiveTransform(src_in, H)
    err = np.linalg.norm(projected.reshape(-1, 2) - dst_in.reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")

    return {
        "keypoints_img1": len(kp1),
        "keypoints_img2": len(kp2),
        "good_matches": n_matches,
        "inliers": n_inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "rmse_px": round(rmse, 4),
        "time_sec": round(t_elapsed, 3),
    }


def save_metrics_csv(metrics, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(metrics.keys())
        w.writerow(metrics.values())


# ---------------------------------------------------------------------------
# 6. WARP + VISUALIZATION
# ---------------------------------------------------------------------------

def warp_image(img1, img2, H):
    h, w = img2.shape
    return cv2.warpPerspective(img1, H, (w, h))


def make_match_visual(img1, img2, kp1, kp2, good_matches, inlier_mask, out_path):
    inlier_matches = [m for m, keep in zip(good_matches, inlier_mask) if keep]
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches[:80], None,
        matchColor=(0, 255, 0), singlePointColor=None,
        matchesMask=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(out_path, vis)


def make_before_after(img1, img2, warped, out_path):
    """3-panel: original source | target | source warped onto target."""
    h, w = img2.shape
    panel1 = cv2.resize(img1, (w, h))
    overlay = cv2.addWeighted(img2, 0.5, warped, 0.5, 0)

    def label(im, text):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(im, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(im, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
        return im

    panels = [
        label(panel1, "Source (before)"),
        label(img2, "Target"),
        label(overlay, "Registered overlay (after)"),
    ]
    combined = np.hstack(panels)
    cv2.imwrite(out_path, combined)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_pipeline(path1, path2, out_dir, loader=load_image_generic):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    img1_raw = loader(path1)
    img2_raw = loader(path2)

    img1 = preprocess(img1_raw)
    img2 = preprocess(img2_raw)

    kp1, kp2, good = detect_and_match(img1, img2)
    H, inlier_mask, src_pts, dst_pts = estimate_homography(kp1, kp2, good)
    warped = warp_image(img1_raw, img2_raw, H)

    t_elapsed = time.time() - t0
    metrics = compute_metrics(H, src_pts, dst_pts, inlier_mask, kp1, kp2, good, t_elapsed)

    make_match_visual(img1_raw, img2_raw, kp1, kp2, good, inlier_mask,
                       os.path.join(out_dir, "matches.png"))
    make_before_after(img1_raw, img2_raw, warped,
                       os.path.join(out_dir, "before_after.png"))
    save_metrics_csv(metrics, os.path.join(out_dir, "metrics.csv"))

    print("=== Registration Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nHomography matrix H:\n{H}")

    return metrics, H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image1")
    ap.add_argument("image2")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    run_pipeline(args.image1, args.image2, args.out)
