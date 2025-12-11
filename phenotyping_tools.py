# phenotyping.py

import os
import io
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Try importing Roboflow SDK
try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None  # type: ignore

# ---------- Types ----------
BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# ============================================================
# 1. Calibration: estimate pixels per cm² from the grid
# ============================================================

def estimate_pixels_per_cm2(
    image_bgr: np.ndarray,
    min_area: int = 800,
    max_aspect_diff: int = 10,
    min_squares: int = 20,
) -> Tuple[Optional[float], Optional[BBox]]:
    """
    Estimate pixels per cm² by detecting grid squares in the background.

    Returns:
        (pixels_per_cm2, calibration_bbox)
        - pixels_per_cm2: average area in pixels of the best-matching squares
        - calibration_bbox: (x, y, w, h) of one representative square (for overlay)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Square-ish, big enough to be a grid cell, not a tiny speck
        if area > min_area and abs(w - h) < max_aspect_diff:
            squares.append((x, y, w, h))

    if len(squares) < min_squares:
        return None, None

    areas = [w * h for (_, _, w, h) in squares]
    median_area = float(np.median(areas))

    # Sort by closeness to the median area
    squares_sorted = sorted(
        squares, key=lambda s: abs((s[2] * s[3]) - median_area)
    )
    best_squares = squares_sorted[:min_squares]

    # Average their areas
    avg_area = float(np.mean([w * h for (_, _, w, h) in best_squares]))

    # Pick the single square closest to the median for overlay
    calib_square = min(
        squares,
        key=lambda s: abs((s[2] * s[3]) - median_area),
    )

    return avg_area, calib_square


# ============================================================
# 2. Roboflow integration (with full debugging)
# ============================================================

def _call_roboflow_workflow(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Call Roboflow workflow using inference-sdk.

    Returns:
        result dict (workflow output) or None if anything fails.

    On failure, prints full traceback in Streamlit so we can debug.
    """
    api_key = st.secrets.get("ROBOFLOW_API_KEY", None)
    if not api_key:
        st.info(
            "No ROBOFLOW_API_KEY found in secrets. "
            "Using color-based segmentation instead."
        )
        return None

    if InferenceHTTPClient is None:
        st.info(
            "inference-sdk is not installed. "
            "Add 'inference-sdk' to requirements.txt to enable Roboflow."
        )
        return None

    # Save bytes to a temporary file; the SDK likes file paths
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    try:
        result = client.run_workflow(
            workspace_name="rootweiler",
            workflow_id="find-leaves-3",
            images={"image": tmp_path},  # IMPORTANT: dict, not list, not bytes
        )

        # You can uncomment this once to see raw structure
        # st.write("Roboflow raw result:", result)

        return result

    except Exception:
        # FULL traceback to help debug the exact problem
        st.error("Roboflow workflow failed. Showing traceback for debugging:")
        st.code(traceback.format_exc())
        return None

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _find_predictions(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Recursively search for a 'predictions' list in a nested dict/list
    Roboflow workflows often nest outputs.
    """
    if isinstance(obj, dict):
        if "predictions" in obj and isinstance(obj["predictions"], list):
            return obj["predictions"]
        for v in obj.values():
            res = _find_predictions(v)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = _find_predictions(item)
            if res is not None:
                return res
    return None


def _mask_from_roboflow(
    result: Dict[str, Any],
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Build a binary leaf mask and metadata from Roboflow workflow output.

    Expects predictions to be either:
      - Polygons: each prediction has "points": [{"x":..., "y":...}, ...]
      - Boxes:    "x","y","width","height"

    Returns:
        mask: uint8 mask (0/255) of leaves
        leaves: list of dicts with "bbox" (x,y,w,h) and "area_px"
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    leaves: List[Dict[str, Any]] = []

    predictions = _find_predictions(result)
    if not predictions:
        return mask, leaves

    leaf_id = 1
    for pred in predictions:
        # Polygon case
        if "points" in pred and isinstance(pred["points"], list):
            pts = []
            for p in pred["points"]:
                if "x" in p and "y" in p:
                    pts.append([p["x"], p["y"]])
            if len(pts) < 3:
                continue
            poly = np.array(pts, dtype=np.int32)
            temp_mask = np.zeros_like(mask)
            cv2.fillPoly(temp_mask, [poly], 255)

            area_px = float(cv2.countNonZero(temp_mask))
            if area_px < 100:  # filter tiny specks
                continue

            # Add to global mask
            mask = cv2.bitwise_or(mask, temp_mask)

            x, y, bw, bh = cv2.boundingRect(poly)
            leaves.append(
                {
                    "id": leaf_id,
                    "bbox": (x, y, bw, bh),
                    "area_px": area_px,
                }
            )
            leaf_id += 1

        # Bounding box case
        elif all(k in pred for k in ["x", "y", "width", "height"]):
            cx = float(pred["x"])
            cy = float(pred["y"])
            bw = float(pred["width"])
            bh = float(pred["height"])

            x = int(cx - bw / 2)
            y = int(cy - bh / 2)
            w_box = int(bw)
            h_box = int(bh)

            # Clamp
            x = max(0, x)
            y = max(0, y)
            if x + w_box > w:
                w_box = w - x
            if y + h_box > h:
                h_box = h - y

            if w_box <= 0 or h_box <= 0:
                continue

            temp_mask = np.zeros_like(mask)
            cv2.rectangle(
                temp_mask,
                (x, y),
                (x + w_box, y + h_box),
                255,
                thickness=-1,
            )
            area_px = float(cv2.countNonZero(temp_mask))
            if area_px < 100:
                continue

            mask = cv2.bitwise_or(mask, temp_mask)

            leaves.append(
                {
                    "id": leaf_id,
                    "bbox": (x, y, w_box, h_box),
                    "area_px": area_px,
                }
            )
            leaf_id += 1

    return mask, leaves


# ============================================================
# 3. Fallback: simple HSV + watershed segmentation
# ============================================================

def _segment_leaves_hsv(image_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Simple green-leaf segmentation using HSV + watershed, used if Roboflow fails.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # A rough "green-ish" range
    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=2)

    # Distance transform for watershed
    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    sure_bg = cv2.dilate(clean, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_for_ws = image_bgr.copy()
    markers = cv2.watershed(img_for_ws, markers)

    # Build leaf mask and leaf list
    h, w = clean.shape
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    leaves: List[Dict[str, Any]] = []

    unique_labels = np.unique(markers)
    leaf_id = 1
    for label in unique_labels:
        if label <= 1:
            continue
        comp_mask = (markers == label).astype(np.uint8) * 255
        area_px = float(cv2.countNonZero(comp_mask))
        if area_px < 500:  # filter tiny bits
            continue

        leaf_mask = cv2.bitwise_or(leaf_mask, comp_mask)

        ys, xs = np.where(comp_mask > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1

        leaves.append(
            {
                "id": leaf_id,
                "bbox": (int(x_min), int(y_min), int(bw), int(bh)),
                "area_px": area_px,
            }
        )
        leaf_id += 1

    return leaf_mask, leaves


# ============================================================
# 4. Metrics computation & visuals
# ============================================================

def compute_leaf_metrics(
    leaves: List[Dict[str, Any]],
    pixels_per_cm2: Optional[float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Given leaf list and pixel->cm² scale, compute per-leaf and summary metrics.
    """
    if not leaves:
        return pd.DataFrame(), {}

    # Plant height (in pixels): vertical span across all leaf bounding boxes
    y_min = min(y for _, (x, y, w, h) in enumerate(l["bbox"] for l in leaves))
    y_max = max(y + h for _, (x, y, w, h) in enumerate(l["bbox"] for l in leaves))
    plant_height_px = max(1, y_max - y_min)

    rows = []
    for leaf in leaves:
        leaf_id = leaf["id"]
        x, y, w, h = leaf["bbox"]
        area_px = leaf["area_px"]

        if pixels_per_cm2 and pixels_per_cm2 > 0:
            area_cm2 = area_px / pixels_per_cm2
            side_cm = np.sqrt(pixels_per_cm2) if pixels_per_cm2 > 0 else 1.0
            width_cm = w / side_cm
            height_cm = h / side_cm
        else:
            area_cm2 = float("nan")
            width_cm = float("nan")
            height_cm = float("nan")

        # ratio in %, using pixel heights (scale cancels)
        leaf_ratio = 100.0 * h / plant_height_px

        rows.append(
            {
                "Leaf ID": leaf_id,
                "Area (cm²)": area_cm2,
                "Width (cm)": width_cm,
                "Height (cm)": height_cm,
                "Leaf / plant height (%)": leaf_ratio,
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "leaf_count": int(len(leaves)),
        "mean_area_cm2": float(df["Area (cm²)"].mean())
        if pixels_per_cm2
        else float("nan"),
        "std_area_cm2": float(df["Area (cm²)"].std())
        if pixels_per_cm2
        else float("nan"),
    }

    return df, summary


def make_overlay_image(
    image_bgr: np.ndarray,
    leaves: List[Dict[str, Any]],
    calib_square: Optional[BBox],
) -> np.ndarray:
    """
    Draw leaf bounding boxes and calibration square on top of the original image.
    """
    overlay = image_bgr.copy()

    # Draw calibration square in magenta
    if calib_square is not None:
        x, y, w, h = calib_square
        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (255, 0, 255),
            2,
        )
        cv2.putText(
            overlay,
            "1cm² ref",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw leaf bounding boxes in green
    for leaf in leaves:
        x, y, w, h = leaf["bbox"]
        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            overlay,
            str(leaf["id"]),
            (x, max(0, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Convert BGR → RGB for Streamlit
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


# ============================================================
# 5. Streamlit UI wrapper
# ============================================================

class PhenotypingUI:
    """Rootweiler phenotyping / leaf metrics tool."""

    @classmethod
    def render(cls):
        st.subheader("Phenotyping")

        st.markdown(
            """
            Upload a single **top-view image** of a lettuce plant on the grid background.

            Rootweiler will:
            - Use the grid to estimate **cm² per pixel**
            - Try **Roboflow** leaf segmentation (with fallback to color-based segmentation)
            - Count leaves and estimate:
              - Leaf area (cm²)
              - Leaf bounding box size (cm)
              - Leaf / plant height ratio
              - Average leaf size and size variation
            """
        )

        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
        )

        if not uploaded:
            return

        # Read bytes once
        file_bytes = uploaded.read()

        # Decode image with OpenCV
        file_array = np.frombuffer(file_bytes, np.uint8)
        image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Could not decode image. Please try a different file.")
            return

        # Show original
        st.markdown("#### Original image")
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

        # 1) Calibration from grid
        pixels_per_cm2, calib_square = estimate_pixels_per_cm2(image_bgr)
        if pixels_per_cm2 is None:
            st.warning(
                "Could not reliably detect grid squares. "
                "Scale-dependent metrics (cm²) may be unavailable."
            )
        else:
            st.success(
                f"Calibration: estimated **{pixels_per_cm2:.0f} pixels per cm²** "
                "(using grid squares)."
            )

        # 2) Segmentation: Roboflow first, then fallback
        st.markdown("#### Leaf segmentation")

        with st.spinner("Running segmentation…"):
            rf_result = _call_roboflow_workflow(file_bytes)

            if rf_result is not None:
                mask, leaves = _mask_from_roboflow(rf_result, image_bgr.shape[:2])
                if not leaves:
                    st.warning(
                        "Roboflow returned no usable leaf predictions. "
                        "Falling back to color-based segmentation."
                    )
                    mask, leaves = _segment_leaves_hsv(image_bgr)
            else:
                mask, leaves = _segment_leaves_hsv(image_bgr)

        if not leaves:
            st.error("No leaves detected – try a clearer image or retrain the model.")
            return

        # 3) Metrics
        df_leaves, summary = compute_leaf_metrics(leaves, pixels_per_cm2)

        # 4) Overlay image
        overlay_rgb = make_overlay_image(image_bgr, leaves, calib_square)

        col_img, col_mask = st.columns(2)
        with col_img:
            st.markdown("**Detected leaves (overlay)**")
            st.image(overlay_rgb, use_container_width=True)

        with col_mask:
            st.markdown("**Binary leaf mask**")
            st.image(mask, clamp=True, use_container_width=True)

        # 5) Summary
        st.markdown("### Leaf metrics")

        st.write(f"- **Leaf count**: {summary.get('leaf_count', 0)}")

        if pixels_per_cm2:
            mean_area = summary.get("mean_area_cm2", float("nan"))
            std_area = summary.get("std_area_cm2", float("nan"))
            st.write(
                f"- **Average leaf area**: {mean_area:.2f} cm² "
                f"(± {std_area:.2f} cm²)"
            )
        else:
            st.write(
                "- Average leaf area: scale unknown (grid calibration failed)."
            )

        st.markdown("#### Per-leaf table")
        st.dataframe(df_leaves.style.format(precision=2), use_container_width=True)

        # 6) Leaf area distribution
        if pixels_per_cm2:
            st.markdown("#### Leaf area distribution")
            st.bar_chart(df_leaves.set_index("Leaf ID")["Area (cm²)"])

        st.caption(
            "Note: Roboflow segmentation quality will strongly influence these metrics. "
            "For best results, train on images similar to your production setup."
        )