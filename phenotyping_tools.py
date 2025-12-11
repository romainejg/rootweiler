# phenotyping_tools.py

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class LeafMeasurement:
    leaf_id: int
    area_cm2: float
    height_cm: float
    area_per_height: float


# -----------------------------
# Grid detection (1 cm squares)
# -----------------------------

def detect_grid_spacing_px_per_cm(img_bgr: np.ndarray) -> Optional[float]:
    """
    Detect grid spacing (in pixels per centimetre) from a checkerboard-like
    background with 1 cm squares.

    Returns px_per_cm or None if detection fails.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Enhance grid lines
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 50, 150, apertureSize=3)

    # Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=150,
        minLineLength=80,
        maxLineGap=10,
    )
    if lines is None or len(lines) < 10:
        return None

    vertical_x = []
    horizontal_y = []

    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and abs(dy) > 10:
            # perfect vertical
            vertical_x.append(x1)
        elif dy == 0 and abs(dx) > 10:
            # perfect horizontal
            horizontal_y.append(y1)
        else:
            # classify by slope
            if abs(dx) < 5 and abs(dy) > 20:
                vertical_x.append(int((x1 + x2) / 2))
            elif abs(dy) < 5 and abs(dx) > 20:
                horizontal_y.append(int((y1 + y2) / 2))

    if len(vertical_x) < 5 and len(horizontal_y) < 5:
        return None

    def spacing_from_positions(pos_list: List[int]) -> Optional[float]:
        if len(pos_list) < 5:
            return None
        pos = np.array(sorted(pos_list))
        diffs = np.diff(pos)
        # Filter out very large gaps (borders) by robust statistics
        median = np.median(diffs)
        diffs = diffs[(diffs > 0.5 * median) & (diffs < 1.5 * median)]
        if len(diffs) == 0:
            return None
        return float(np.median(diffs))

    sp_v = spacing_from_positions(vertical_x)
    sp_h = spacing_from_positions(horizontal_y)

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
    Draw a coarse grid overlay every 1 cm on the image so the user can
    visually confirm the calibration.
    """
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]

    step = int(round(px_per_cm))
    if step <= 0:
        return overlay

    # Start near the center to avoid border inaccuracies
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


# -----------------------------
# Leaf segmentation
# -----------------------------

def create_leaf_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for leafy material.

    Strategy:
    - Convert to HSV
    - Threshold on saturation + value to remove white/gray grid
    - Optional hue band for vegetative colors
    - Morphological clean-up
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Basic thresholds – tuned to "greenish" but also works for other foliage
    # S > 40 to avoid white background, V < 245 to avoid specular glare.
    leaf_mask = (s > 40) & (v < 245)

    # Optional hue band (rough green range) – keeps tag mostly out
    # convert bool to uint8 for bitwise ops
    hue_mask1 = (h > 25) & (h < 100)  # green-ish
    leaf_mask = (leaf_mask & hue_mask1).astype(np.uint8) * 255

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Remove very small specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
    cleaned = np.zeros_like(leaf_mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 300:  # ignore tiny blobs
            cleaned[labels == label] = 255

    return cleaned


def split_leaves_watershed(img_bgr: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use marker-based watershed to split overlapping leaves.

    Returns:
      - markers (int32 array with label IDs)
      - cleaned_leaf_mask (uint8)
    """
    # distance transform to find leaf "cores"
    dist = cv2.distanceTransform(leaf_mask, cv2.DIST_L2, 5)
    # threshold at 0.4 of max distance – controls how aggressive splitting is
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # sure background by dilation
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(leaf_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # markers for watershed
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_for_ws = img_bgr.copy()
    cv2.watershed(img_for_ws, markers)

    # markers == -1 are boundaries; treat anything >1 as leaf
    cleaned_leaf_mask = np.zeros_like(leaf_mask)
    cleaned_leaf_mask[markers > 1] = 255

    return markers, cleaned_leaf_mask


def measure_leaves(
    markers: np.ndarray,
    px_per_cm: float,
) -> List[LeafMeasurement]:
    """
    Compute area (cm²), height (cm), and area:height for each leaf label.
    """
    measurements: List[LeafMeasurement] = []

    unique_labels = np.unique(markers)
    leaf_labels = [lab for lab in unique_labels if lab > 1]  # >1 => skip background/boundary

    for idx, lab in enumerate(sorted(leaf_labels), start=1):
        mask_leaf = (markers == lab).astype(np.uint8) * 255
        # skip tiny labels
        if cv2.countNonZero(mask_leaf) < 500:
            continue

        contours, _ = cv2.findContours(mask_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        height_px = h

        area_cm2 = area_px / (px_per_cm ** 2)
        height_cm = height_px / px_per_cm
        area_per_height = area_cm2 / height_cm if height_cm > 0 else 0.0

        measurements.append(
            LeafMeasurement(
                leaf_id=idx,
                area_cm2=float(area_cm2),
                height_cm=float(height_cm),
                area_per_height=float(area_per_height),
            )
        )

    return measurements


def create_label_overlay(
    img_bgr: np.ndarray,
    markers: np.ndarray,
    measurements: List[LeafMeasurement],
) -> np.ndarray:
    """
    Draw bounding boxes + IDs for each leaf on a black background.
    """
    overlay = np.zeros_like(img_bgr)
    marker_to_idx = {m.leaf_id: m for m in measurements}

    unique_labels = np.unique(markers)
    leaf_labels = [lab for lab in unique_labels if lab > 1]

    label_id = 1
    for lab in sorted(leaf_labels):
        mask_leaf = (markers == lab).astype(np.uint8) * 255
        if cv2.countNonZero(mask_leaf) < 500:
            continue

        contours, _ = cv2.findContours(mask_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


# -----------------------------
# Streamlit UI
# -----------------------------

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

        # --- Grid detection ---
        st.markdown("### Grid calibration (1 cm squares)")

        auto_px_per_cm = detect_grid_spacing_px_per_cm(img_bgr)

        if auto_px_per_cm is not None:
            st.success(f"Auto-detected grid spacing: ~{auto_px_per_cm:.1f} px per 1 cm")
            default_px = float(auto_px_per_cm)
        else:
            st.warning("Could not auto-detect grid spacing reliably. Please enter manually.")
            default_px = 50.0

        px_per_cm = st.number_input(
            "Pixels per centimetre (override if needed)",
            min_value=1.0,
            max_value=1000.0,
            value=default_px,
            step=1.0,
        )

        # Grid overlay for sanity check
        overlay_bgr = draw_grid_overlay(img_bgr, px_per_cm)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        st.caption("Overlay of 1 cm grid (green lines) to check calibration.")
        st.image(overlay_rgb, use_column_width=True)

        # --- Segmentation & measurements ---
        leaf_mask = create_leaf_mask(img_bgr)
        markers, cleaned_mask = split_leaves_watershed(img_bgr, leaf_mask)
        measurements = measure_leaves(markers, px_per_cm=px_per_cm)

        # Binary mask for display
        mask_display = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2RGB)

        # Label overlay
        label_overlay_bgr = create_label_overlay(img_bgr, markers, measurements)
        label_overlay_rgb = cv2.cvtColor(label_overlay_bgr, cv2.COLOR_BGR2RGB)

        # Original RGB for display
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

        # --- Measurements table ---
        st.markdown("### Leaf measurements")

        if not measurements:
            st.warning(
                "No leaves were confidently segmented. "
                "Try a clearer image, adjust lighting, or tweak the grid calibration."
            )
            return

        # Build table
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

        import pandas as pd

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Summary stats
        area_vals = np.array([m.area_cm2 for m in measurements])
        height_vals = np.array([m.height_cm for m in measurements])

        st.markdown("#### Summary statistics")
        st.write(f"- **Leaf count:** {len(measurements)}")
        st.write(f"- **Average leaf area:** {area_vals.mean():.2f} cm²")
        st.write(f"- **Leaf area standard deviation:** {area_vals.std(ddof=1):.2f} cm²")
        st.write(f"- **Average leaf height:** {height_vals.mean():.2f} cm")
        st.write(f"- **Leaf height standard deviation:** {height_vals.std(ddof=1):.2f} cm")

        st.caption(
            "Area is estimated from the segmented silhouette and the detected grid spacing. "
            "If calibration looks off, adjust the pixels-per-cm value and rerun."
        )
