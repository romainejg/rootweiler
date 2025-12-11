import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional

from PIL import Image
import streamlit as st


@dataclass
class LeafMeasurement:
    leaf_id: int
    area_cm2: float
    height_cm: float
    area_height_ratio: float


def _auto_grid_spacing_projection(gray: np.ndarray) -> Optional[float]:
    """Estimate grid spacing (in pixels) using edge projections."""
    # Focus on central region to avoid rulers/borders
    h, w = gray.shape
    y0, y1 = int(0.1 * h), int(0.9 * h)
    x0, x1 = int(0.1 * w), int(0.9 * w)
    roi = gray[y0:y1, x0:x1]

    # Edge detection
    edges = cv2.Canny(roi, 50, 150)

    # Project edges horizontally and vertically
    col_sum = edges.sum(axis=0)
    row_sum = edges.sum(axis=1)

    def _estimate_from_projection(signal: np.ndarray) -> Optional[float]:
        if signal.max() == 0:
            return None
        # Normalize
        s = (signal - signal.min()) / (signal.max() - signal.min() + 1e-6)
        # Threshold to keep strong lines
        mask = s > 0.4
        idx = np.where(mask)[0]
        if len(idx) < 10:
            return None
        # Find run boundaries -> approximate grid line centers
        centers: List[int] = []
        run_start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
                continue
            # close run
            centers.append((run_start + prev) // 2)
            run_start = i
            prev = i
        centers.append((run_start + prev) // 2)
        if len(centers) < 5:
            return None
        diffs = np.diff(centers)
        # Filter outliers
        m = np.median(diffs)
        diffs = diffs[(diffs > 0.5 * m) & (diffs < 1.5 * m)]
        if len(diffs) == 0:
            return None
        return float(np.median(diffs))

    dx = _estimate_from_projection(col_sum)
    dy = _estimate_from_projection(row_sum)

    candidates = [v for v in (dx, dy) if v is not None]
    if not candidates:
        return None
    return float(np.mean(candidates))


def _segment_leaves(image_bgr: np.ndarray, vcrop_start: float = 0.4) -> np.ndarray:
    """
    Segment leaves using HSV thresholding, focusing on lower part of the image
    where leaves usually are. Returns binary mask (uint8 0/255).
    """
    h, w, _ = image_bgr.shape
    y_start = int(vcrop_start * h)
    roi = image_bgr[y_start:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Green-ish range (tuneable later if needed)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Put ROI mask back into full-size mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y_start:, :] = mask_clean
    return full_mask


def _measure_leaves(
    mask: np.ndarray,
    px_per_cm: float,
    min_area_cm2: float = 1.0,
) -> Tuple[List[LeafMeasurement], np.ndarray]:
    """
    From a binary mask, find connected leaves and measure area/height.
    Returns list of LeafMeasurement and an overlay image with contours drawn.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    h, w = mask.shape
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    leaves: List[LeafMeasurement] = []
    leaf_id = 1
    px_per_cm2 = px_per_cm**2

    for label in range(1, num_labels):
        area_px = stats[label, cv2.CC_STAT_AREA]
        if area_px < min_area_cm2 * px_per_cm2:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        area_cm2 = area_px / px_per_cm2
        height_cm = bh / px_per_cm
        ratio = area_cm2 / height_cm if height_cm > 0 else 0.0

        leaves.append(LeafMeasurement(leaf_id, area_cm2, height_cm, ratio))

        # Draw bounding box + ID
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            str(leaf_id),
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        leaf_id += 1

    return leaves, overlay


class PhenotypingUI:
    """Streamlit UI for leaf phenotyping on a 1 cm grid background."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload an image of leaves on a **1 cm grid board**.  
            Rootweiler will:
            - estimate the grid spacing to convert pixels → cm  
            - segment individual leaves (even with some overlap)  
            - measure area, height, and area/height ratio per leaf
            """
        )

        uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
        if uploaded is None:
            st.info("Upload a photo of the grid board with leaves to begin.")
            return

        pil_img = Image.open(uploaded).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # ---------- Grid calibration ----------
        with st.expander("Grid calibration (1 cm squares)", expanded=True):
            auto_spacing = _auto_grid_spacing_projection(gray)

            if auto_spacing is not None and auto_spacing > 3:
                st.success(
                    f"Auto-detected grid spacing: ~{auto_spacing:0.1f} px per 1 cm"
                )
                default_px_cm = float(auto_spacing)
            else:
                st.warning(
                    "Could not confidently detect grid spacing automatically. "
                    "Use the manual control below."
                )
                h, w = gray.shape
                # Assume ~30 squares across as a fallback
                default_px_cm = w / 30.0

            h, w = gray.shape
            st.markdown(
                f"<small>Image size: {w} × {h} px. Adjust the slider if the auto value looks off.</small>",
                unsafe_allow_html=True,
            )

            px_per_cm = st.slider(
                "Pixels per centimetre",
                min_value=10.0,
                max_value=200.0,
                value=float(default_px_cm),
                step=0.5,
            )

            st.caption(
                "Tip: one grid square is 1 cm. "
                "If you know the number of squares across the image, "
                "px/cm = image_width_px ÷ squares_across."
            )

        # ---------- Segmentation ----------
        st.markdown("### Segmentation preview")

        vcrop = st.slider(
            "Vertical crop start (for leaf region)",
            min_value=0.0,
            max_value=0.8,
            value=0.4,
            step=0.05,
            help=(
                "Everything above this fraction of the image height is ignored "
                "for leaf segmentation."
            ),
        )

        mask = _segment_leaves(image_bgr, vcrop_start=vcrop)
        leaves, overlay = _measure_leaves(mask, px_per_cm=px_per_cm, min_area_cm2=1.0)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Segmented leaves**")
            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                use_column_width=True,
            )

        with col2:
            st.markdown("**Binary mask**")
            st.image(mask, use_column_width=True, clamp=True)

        # ---------- Measurements ----------
        st.markdown("### Leaf measurements")

        if not leaves:
            st.warning(
                "No leaves detected with current settings. You can try:\n"
                "- lowering the vertical crop start\n"
                "- retaking the photo with stronger contrast\n"
                "- or, if you're comfortable editing code, adjusting the green HSV thresholds."
            )
            return

        rows = []
        for lm in leaves:
            rows.append(
                {
                    "Leaf ID": lm.leaf_id,
                    "Area (cm²)": round(lm.area_cm2, 2),
                    "Height (cm)": round(lm.height_cm, 2),
                    "Area : height (cm)": round(lm.area_height_ratio, 2),
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Summary stats
        st.markdown("#### Summary statistics")
        st.write(f"- Leaf count: **{len(leaves)}**")
        st.write(f"- Mean area: **{df['Area (cm²)'].mean():.2f} cm²**")
        st.write(f"- Area standard deviation: **{df['Area (cm²)'].std():.2f} cm²**")
        st.write(f"- Mean height: **{df['Height (cm)'].mean():.2f} cm**")
