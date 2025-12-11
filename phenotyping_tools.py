# phenotyping_tools.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class LeafMeasurement:
    leaf_id: int
    area_cm2: float
    height_cm: float
    area_per_height: float


# ============================================================
# GRID DETECTION – 1 cm SQUARE BOARD
# ============================================================

def _line_centers_from_binary(line_img: np.ndarray, axis: int) -> Optional[np.ndarray]:
    """
    Given a binary image with lines, collapse along axis and return
    the centres (in pixels) of each distinct line.
    """
    # projection along axis
    projection = line_img.sum(axis=axis)
    idx = np.where(projection > 0)[0]
    if len(idx) < 5:
        return None

    centers = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            centers.append((start + prev) / 2.0)
            start = prev = i
    centers.append((start + prev) / 2.0)

    centers = np.array(centers)
    centers.sort()
    return centers


def detect_grid_spacing_px_per_cm(img_bgr: np.ndarray) -> Optional[float]:
    """
    Detect grid spacing (pixels per centimetre) assuming a 1 cm square grid.

    Approach:
      * Convert to grayscale + equalize
      * Threshold to get dark grid lines as white
      * Use morphology to pull out horizontal & vertical line images
      * Find line centres and compute median spacing between neighbours
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Invert: grid lines become white on black background
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = bw.shape

    # Kernels sized relative to image: we want to pick up multi-cm long lines
    vert_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(15, h // 40))
    )
    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(15, w // 40), 1)
    )

    # Extract vertical lines
    vertical = cv2.erode(bw, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=1)

    # Extract horizontal lines
    horizontal = cv2.erode(bw, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

    # Find line centres
    centers_v = _line_centers_from_binary(vertical, axis=0)
    centers_h = _line_centers_from_binary(horizontal, axis=1)

    def spacing_from_centers(centers: np.ndarray) -> Optional[float]:
        if centers is None or len(centers) < 5:
            return None
        diffs = np.diff(centers)
        diffs = diffs[diffs > 1]  # ignore zero/1-pixel noise
        if len(diffs) == 0:
            return None
        median = np.median(diffs)
        # keep only diffs near the median (to ignore borders / big gaps)
        diffs = diffs[(diffs > 0.5 * median) & (diffs < 1.5 * median)]
        if len(diffs) == 0:
            return None
        return float(np.median(diffs))

    sp_v = spacing_from_centers(centers_v)
    sp_h = spacing_from_centers(centers_h)

    if sp_v and sp_h:
        return float((sp_v + sp_h) / 2.0)
    elif sp_v:
        return float(sp_v)
    elif sp_h:
        return float(sp_h)
    else:
        return None


def draw_grid_overlay(
    img_bgr: np.ndarray,
    px_per_cm: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw grid lines at 1 cm spacing over the original image so you
    can visually see if calibration looks right.
    """
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]

    step = int(round(px_per_cm))
    if step <= 0:
        return overlay

    # Start near the centre to avoid noisy outer region
    cx, cy = w // 2, h // 2

    # vertical lines
    x = cx
    while x < w:
        cv2.line(overlay, (x, 0), (x, h), color, thickness)
        x += step
    x = cx - step
    while x > 0:
        cv2.line(overlay, (x, 0), (x, h), color, thickness)
        x -= step

    # horizontal lines
    y = cy
    while y < h:
        cv2.line(overlay, (0, y), (w, y), color, thickness)
        y += step
    y = cy - step
    while y > 0:
        cv2.line(overlay, (0, y), (w, y), color, thickness)
        y -= step

    return overlay


# ============================================================
# LEAF SEGMENTATION
# ============================================================

def create_leaf_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for leafy material.

    Strategy:
      * HSV conversion
      * Threshold on saturation + value to remove white grid
      * Hue band for vegetative colours
      * Morphological clean-up + removal of tiny specks
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # basic thresholds to knock out white/grey board
    leaf_mask = (s > 35) & (v < 245)

    # Green-ish hue band (roughly 25–100 deg)
    green_band = (h > 25) & (h < 100)
    leaf_mask = (leaf_mask & green_band).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Remove very small blobs (threshold dropped a bit so small leaves survive)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        leaf_mask, connectivity=8
    )
    cleaned = np.zeros_like(leaf_mask)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area > 150:  # was 300 – lowered to keep small leaves
            cleaned[labels == lab] = 255

    return cleaned


def split_leaves_watershed(
    img_bgr: np.ndarray,
    leaf_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marker-based watershed to split overlapping leaves.

    Returns:
      markers        – int32 label image
      cleaned_mask   – uint8 binary mask
    """
    dist = cv2.distanceTransform(leaf_mask, cv2.DIST_L2, 5)
    # Slightly more aggressive split (0.35 instead of 0.4)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(leaf_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    ws_img = img_bgr.copy()
    cv2.watershed(ws_img, markers)

    cleaned_mask = np.zeros_like(leaf_mask)
    cleaned_mask[markers > 1] = 255

    return markers, cleaned_mask


def measure_leaves(markers: np.ndarray, px_per_cm: float) -> List[LeafMeasurement]:
    """
    Compute area (cm²), height (cm), and area:height for each leaf label.
    """
    measurements: List[LeafMeasurement] = []

    unique_labels = np.unique(markers)
    leaf_labels = [lab for lab in unique_labels if lab > 1]

    idx = 1
    for lab in sorted(leaf_labels):
        mask_leaf = (markers == lab).astype(np.uint8) * 255
        if cv2.countNonZero(mask_leaf) < 200:  # lower threshold so small leaves count
            continue

        contours, _ = cv2.findContours(
            mask_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        area_cm2 = area_px / (px_per_cm ** 2)
        height_cm = h / px_per_cm
        area_per_height = area_cm2 / height_cm if height_cm > 0 else 0.0

        measurements.append(
            LeafMeasurement(
                leaf_id=idx,
                area_cm2=float(area_cm2),
                height_cm=float(height_cm),
                area_per_height=float(area_per_height),
            )
        )
        idx += 1

    return measurements


def create_label_overlay(
    img_bgr: np.ndarray,
    markers: np.ndarray,
    measurements: List[LeafMeasurement],
) -> np.ndarray:
    """
    Draw bounding boxes and ID labels on a black background.
    """
    overlay = np.zeros_like(img_bgr)

    # Map order of measurement IDs onto marker labels by area
    unique_labels = np.unique(markers)
    leaf_labels = [lab for lab in unique_labels if lab > 1]

    label_id = 1
    for lab in sorted(leaf_labels):
        mask_leaf = (markers == lab).astype(np.uint8) * 255
        if cv2.countNonZero(mask_leaf) < 200:
            continue

        contours, _ = cv2.findContours(
            mask_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        cv2.drawContours(overlay, [contour], -1, (255, 255, 255), thickness=-1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"{label_id}",
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        label_id += 1

    return overlay


# ============================================================
# STREAMLIT UI
# ============================================================

class PhenotypingUI:
    """Rootweiler leaf phenotyping tool."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload a top-down photo with leaves on a **1 cm grid board**.  
            Rootweiler will:

            - Detect the grid spacing and convert pixels → **cm²**  
            - Segment individual leaves (even when overlapping)  
            - Report **leaf count, area, height, and area:height**  
            - Summarize **average leaf size** and **size variation**
            """
        )

        uploaded = st.file_uploader(
            "Upload phenotyping image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        pil_img = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # -------- Grid detection (NO manual adjustment) --------
        st.markdown("### Grid calibration (1 cm squares)")

        px_per_cm = detect_grid_spacing_px_per_cm(img_bgr)

        if px_per_cm is None or px_per_cm <= 0:
            st.error(
                "Could not automatically detect grid spacing. "
                "Make sure the board is visible and the grid lines are sharp."
            )
            # Show the image anyway for debugging
            st.image(img_rgb, caption="Uploaded image", use_column_width=True)
            return

        st.success(f"Detected grid spacing: ~{px_per_cm:.1f} pixels per 1 cm")

        # Overlay grid so you can visually confirm calibration
        overlay_bgr = draw_grid_overlay(img_bgr, px_per_cm)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        st.caption("Green overlay lines show the detected 1 cm spacing.")
        st.image(overlay_rgb, use_column_width=True)

        # -------- Segmentation + measurements --------
        leaf_mask = create_leaf_mask(img_bgr)
        markers, cleaned_mask = split_leaves_watershed(img_bgr, leaf_mask)
        measurements = measure_leaves(markers, px_per_cm)

        mask_display = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2RGB)
        label_overlay_bgr = create_label_overlay(img_bgr, markers, measurements)
        label_overlay_rgb = cv2.cvtColor(label_overlay_bgr, cv2.COLOR_BGR2RGB)

        st.markdown("### Segmentation overview")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Original image**")
            st.image(img_rgb, use_column_width=True)

        with c2:
            st.markdown("**Leaf labels**")
            st.image(label_overlay_rgb, use_column_width=True)

        with c3:
            st.markdown("**Binary mask**")
            st.image(mask_display, use_column_width=True)

        # -------- Measurements table --------
        st.markdown("### Leaf measurements")

        if not measurements:
            st.warning(
                "No leaves were confidently segmented. "
                "Try a clearer image or a different background."
            )
            return

        rows = []
        for m in measurements:
            rows.append(
                {
                    "Leaf ID": m.leaf_id,
                    "Area (cm²)": round(m.area_cm2, 2),
                    "Height (cm)": round(m.height_cm, 2),
                    "Area : height (cm)": round(m.area_per_height, 2),
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Summary stats (cm, cm² only – no pixels)
        areas = np.array([m.area_cm2 for m in measurements])
        heights = np.array([m.height_cm for m in measurements])

        st.markdown("#### Summary statistics")
        st.write(f"- **Leaf count:** {len(measurements)}")
        st.write(f"- **Average leaf area:** {areas.mean():.2f} cm²")
        st.write(f"- **Leaf area standard deviation:** {areas.std(ddof=1):.2f} cm²")
        st.write(f"- **Average leaf height:** {heights.mean():.2f} cm")
        st.write(f"- **Leaf height standard deviation:** {heights.std(ddof=1):.2f} cm")

        st.caption(
            "All metrics are derived from the segmented silhouette and the "
            "detected grid spacing. If the green overlay lines look aligned with "
            "the board squares, the cm² values should be in the right ballpark."
        )
