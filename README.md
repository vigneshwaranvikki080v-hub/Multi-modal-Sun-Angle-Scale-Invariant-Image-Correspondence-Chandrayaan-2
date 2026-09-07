# Lunar Image Registration — SIH26166

Baseline pipeline for registering Chandrayaan-2 TMC-2/OHRC/IIRS lunar imagery using SIFT feature matching + RANSAC homography estimation.

## Overview

Given two lunar images of the same region (captured at different times, sun angles, or slight viewpoint offsets), this pipeline:

1. Loads the images (PDS4 `.IMG`/`.XML` via GDAL, or plain PNG/JPG for demos)
2. Preprocesses with CLAHE for illumination normalization
3. Detects SIFT keypoints and matches them (FLANN + Lowe's ratio test)
4. Estimates a homography via RANSAC
5. Warps and overlays the source onto the target
6. Outputs match visualization, before/after overlay, and quantitative metrics (RMSE, inlier ratio, keypoint counts, runtime)

## Files

| File | Purpose |
|---|---|
| `register.py` | Core registration pipeline (SIFT → RANSAC → homography → metrics) |
| `make_demo_pair.py` | Generates a synthetic crater-like image pair (rotated/scaled/illumination-shifted) for testing without real data |

## Setup

```bash
pip install opencv-python numpy
pip install gdal   # only needed for real Chandrayaan .IMG/.XML files
```

## Usage

### Quick demo (synthetic data, no download needed)

```bash
python3 make_demo_pair.py --out demo_data
python3 register.py demo_data/source.png demo_data/target.png --out demo_output
```

### Real Chandrayaan-2 data

```python
from register import load_pds_image, run_pipeline

run_pipeline(
    "ch2_tmc_ncn_20231026T0943001971_d_img_d18.xml",
    "ch2_tmc_ncn_20231222T0751399116_d_img_d18.xml",
    "output_real",
    loader=load_pds_image,
)
```

## Output

Each run produces, in the specified output folder:
- `matches.png` — visualized keypoint matches between the two images
- `before_after.png` — 3-panel: source | target | registered overlay
- `metrics.csv` — keypoint counts, good matches, inliers, inlier ratio, RMSE (px), runtime (s)

## Data source

Chandrayaan-2 TMC-2 calibrated products from [ISSDC Chandrayaan Map Browse](https://chmapbrowse.issdc.gov.in/MapBrowse/), Tycho Crater region.

## Status

Currently on synthetic demo data while real `.IMG` pair finishes downloading. Pipeline is identical for both — only the loader and input paths change.

## Roadmap

```
Fundamentals → Real Chandrayaan Data → SIFT Baseline → RANSAC/Geometric Registration
→ LoFTR → Illumination Robustness → Scale/Viewpoint Robustness
→ Uniform Correspondences → Sub-pixel Refinement → Cross-sensor Registration
→ Benchmarking → Final Application/Presentation
```
