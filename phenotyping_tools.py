# phenotyping_tools.py

from __future__ import annotations

from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Type alias
# -----------------------------

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# -----------------------------
# Small helpers
# -----------------------------

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """OpenCV loads as BGR, Streamlit expects RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------
# Grid detection
# -----------------------------

def detect_grid_spacing(
    image_bgr: np.ndarray,
) -> Tuple[Optional[float], Optional[float], np.ndarray]:
    """
    Detect 1 cm grid spacing using Hough line detection.

    Returns:
        pixels_per_cm2 (float or None),
        pixels_per_cm (float or None),
        overlay_rgb (np.ndarray) – original image with a green grid drawn.
    """
    h, w = image_bgr.shape[:2]

    # Focus on central board region to avoid outer rulers & leaves
    roi_x1 = int(0.10 * w)
    roi_x2 = int(0.90 * w)
    roi_y1 = int(0.05 * h)
    roi_y2 = int(0.75 * h)
    roi = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 120, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=130,
        minLineLength=60,
        maxLineGap=8,
    )

    if lines is None or len(lines) < 10:
        # Could not detect reliably
        overlay = bgr_to_rgb(image_bgr.copy())
        return None, None, overlay

    vertical_positions = []
    horizontal_positions = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 5:  # vertical line
            cx = (x1 + x2) // 2
            vertical_positions.append(cx)
        elif abs(y1 - y2) < 5:  # horizontal line
            cy = (y1 + y2) // 2
            horizontal_positions.append(cy)

    spacings: List[float] = []

    if len(vertical_positions) > 5:
        xs = np.sort(np.array(vertical_positions))
        dx = np.diff(xs)
        spacings.extend(dx[(dx > 5) & (dx < 200)])

    if len(horizontal_positions) > 5:
        ys = np.sort(np.array(horizontal_positions))
        dy = np.diff(ys)
        spacings.extend(dy[(dy > 5) & (dy < 200)])

    if len(spacings) < 5:
        overlay = bgr_to_rgb(image_bgr.copy())
        return None, None, overlay

    spacing_px = float(np.median(spacings))
    pixels_per_cm = spacing_px              # 1 grid cell ≈ 1 cm
    pixels_per_cm2 = pixels_per_cm ** 2

    # ---- Draw a grid overlay for user sanity check ----
    overlay = image_bgr.copy()
    step = int(round(spacing_px))

    # Start grid roughly at same ROI offset
    start_x = roi_x1
    start_y = roi_y1
    end_x = roi_x2
    end_y = roi_y2

    # Vertical lines
    for x in range(start_x, end_x, step):
        cv2.line(overlay, (x, start_y), (x, end_y), (0, 255, 0), 1)

    # Horizontal lines
    for y in range(start_y, end_y, step):
        cv2.line(overlay, (start_x, y), (end_x, y), (0, 255, 0), 1)

    return pixels_per_cm2, pixels_per_cm, bgr_to_rgb(overlay)


# -----------------------------
# Leaf segmentation
# -----------------------------

def segment_leaves(
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[BBox]]:
    """
    Segment individual leaves using HSV threshold + morphology + watershed.

    Returns
    -------
    leaf_mask : np.ndarray
        Binary mask (255 = leaf, 0 = background)
    labels : np.ndarray
        Connected component labels (0 = background)
    label_ids : List[int]
        List of label IDs that we keep as "leaf objects"
    bboxes : List[BBox]
        Bounding boxes (x, y, w, h) for each label_id
    """
    # HSV threshold for green-ish lettuce
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # These ranges are tuned for your board + lettuce, adjust if needed
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up grid lines & noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Distance transform to split overlapping leaves
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Higher threshold -> fewer, more distinct seeds
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # ensure background is 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_bgr, markers)

    leaf_mask = np.zeros_like(mask)
    leaf_mask[markers > 1] = 255

    # Connected components on final mask
    num_labels, labels = cv2.connectedComponents(leaf_mask)

    label_ids: List[int] = []
    bboxes: List[BBox] = []

    for lab in range(1, num_labels):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        area_px = w * h

        # Filter tiny bits (change threshold if you want even smaller leaves)
        if area_px < 2000:
            continue

        label_ids.append(lab)
        bboxes.append((x_min, y_min, w, h))

    return leaf_mask, labels, label_ids, bboxes


def make_label_overlay(
    labels: np.ndarray,
    label_ids: List[int],
    bboxes: List[BBox],
) -> np.ndarray:
    """Create a black background + white leaves + green bounding boxes."""
    h, w = labels.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[labels > 0] = (255, 255, 255)

    for lab, (x, y, w_box, h_box) in zip(label_ids, bboxes):
        cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

    return overlay


def measure_leaves(
    labels: np.ndarray,
    label_ids: List[int],
    bboxes: List[BBox],
    pixels_per_cm2: Optional[float],
    pixels_per_cm: Optional[float],
) -> pd.DataFrame:
    """Compute per-leaf area, height, and area:height ratio in cm units."""
    rows = []

    # If grid detection failed, we still compute pixel areas, but cm will be NA
    scale_area = pixels_per_cm2 is not None and pixels_per_cm2 > 0
    scale_len = pixels_per_cm is not None and pixels_per_cm > 0

    for idx, (lab, (x, y, w, h)) in enumerate(zip(label_ids, bboxes), start=1):
        mask_leaf = (labels == lab)
        area_px = int(mask_leaf.sum())

        if scale_area:
            area_cm2 = area_px / pixels_per_cm2  # type: ignore
        else:
            area_cm2 = float("nan")

        if scale_len:
            height_cm = h / pixels_per_cm  # type: ignore
        else:
            height_cm = float("nan")

        if scale_area and scale_len and height_cm > 0:
            ratio = area_cm2 / height_cm
        else:
            ratio = float("nan")

        rows.append(
            {
                "Leaf ID": idx,
                "Area (cm²)": area_cm2,
                "Height (cm)": height_cm,
                "Area : height": ratio,
            }
        )

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Streamlit UI wrapper
# -----------------------------

class PhenotypingUI:
    """Rootweiler phenotyping tools (single-image leaf phenotyping)."""

    @classmethod
    def render(cls):
        st.subheader("Phenotyping – lettuce leaves on grid")

        st.markdown(
            """
            Upload an overhead image of leaves on the **1 cm grid board**.

            Rootweiler will try to:

            - Detect the grid and convert pixel area to **cm²**
            - Segment individual leaves (even when overlapping)
            - Measure per-leaf **area**, **height**, and **area:height**  
            - Summarise average leaf size and variability
            """
        )

        uploaded = st.file_uploader(
            "Upload image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            key="phenotype_image",
        )

        if uploaded is None:
            st.info("Upload an image to get started.")
            return

        # Read image into OpenCV BGR format
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Could not read this image.")
            return

        # --- Grid detection ---
        pixels_per_cm2, pixels_per_cm, grid_overlay_rgb = detect_grid_spacing(image_bgr)

        with st.expander("Grid calibration (1 cm squares)", expanded=True):
            if pixels_per_cm2 is None:
                st.error(
                    "Could not automatically detect grid spacing. "
                    "Make sure the board is visible and the grid lines are sharp."
                )
            else:
                px_cm = pixels_per_cm if pixels_per_cm is not None else float("nan")
                st.success(
                    f"Estimated grid spacing: ~**{px_cm:.1f} px per 1 cm** "
                    f"(1 cm² ≈ {pixels_per_cm2:.0f} px²)."
                )
            st.image(
                grid_overlay_rgb,
                caption="Original image with detected grid overlay",
                use_column_width=True,
            )

        # --- Leaf segmentation ---
        leaf_mask, labels, label_ids, bboxes = segment_leaves(image_bgr)

        if not label_ids:
            st.error(
                "No leaves detected. You may need stronger contrast between leaves and background."
            )
            return

        label_overlay = make_label_overlay(labels, label_ids, bboxes)

        # --- Visual overview ---
        st.markdown("### Segmentation overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original image**")
            st.image(bgr_to_rgb(image_bgr), use_column_width=True)
        with col2:
            st.markdown("**Leaf labels (bounding boxes)**")
            st.image(bgr_to_rgb(label_overlay), use_column_width=True)
        with col3:
            st.markdown("**Binary leaf mask**")
            st.image(leaf_mask, clamp=True, use_column_width=True)

        # --- Measurements ---
        df = measure_leaves(labels, label_ids, bboxes, pixels_per_cm2, pixels_per_cm)

        st.markdown("### Leaf measurements")
        st.dataframe(df.style.format({"Area (cm²)": "{:.2f}", "Height (cm)": "{:.2f}", "Area : height": "{:.2f}"}),
                     use_container_width=True)

        # Summary stats if we have cm²
        if pixels_per_cm2 is not None and not df["Area (cm²)"].isna().all():
            mean_area = df["Area (cm²)"].mean()
            std_area = df["Area (cm²)"].std()
            mean_ratio = df["Area : height"].mean()

            st.markdown("#### Summary")
            st.write(f"- Leaf count: **{len(df)}**")
            st.write(f"- Average area: **{mean_area:.2f} cm²**")
            st.write(f"- Area standard deviation: **{std_area:.2f} cm²**")
            if not np.isnan(mean_ratio):
                st.write(f"- Average area:height ratio: **{mean_ratio:.2f}**")
        else:
            st.caption(
                "Grid spacing was not detected, so areas are in pixels only. "
                "Try another image or adjust the board visibility."
            )
