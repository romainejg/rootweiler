# phenotyping_tools.py

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Optional OpenCV (we'll show a nice message if it's missing)
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    HAS_CV2 = False

# Roboflow inference SDK
try:
    from inference_sdk import InferenceHTTPClient

    HAS_ROBOFLOW = True
except ImportError:
    InferenceHTTPClient = None  # type: ignore
    HAS_ROBOFLOW = False


BBox = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class LeafMeasurement:
    leaf_id: int
    bbox: BBox
    area_cm2: Optional[float]
    height_cm: Optional[float]
    leaf_to_plant_height_ratio: Optional[float]


# -------------------------------------------------
# Utility: load image (Streamlit upload -> PIL & cv2)
# -------------------------------------------------

def _load_image_from_upload(uploaded_file) -> Tuple[np.ndarray, Image.Image, bytes]:
    """
    Convert a Streamlit uploaded_file into:
      - cv2 BGR ndarray
      - PIL Image (RGB)
      - raw bytes
    """
    image_bytes: bytes = uploaded_file.getvalue()

    # PIL
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # OpenCV (BGR)
    img_array = np.frombuffer(image_bytes, np.uint8)
    cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return cv_img, pil_img, image_bytes


# -------------------------------------------------
# Grid detection: estimate pixels per cm²
# -------------------------------------------------

def estimate_pixels_per_cm2(
    image_bgr: np.ndarray,
    min_square_area_px: int = 500,
    min_squares: int = 30,
) -> Tuple[Optional[float], Optional[BBox]]:
    """
    Estimate pixels per cm² by detecting square-like regions (the 1 cm² grid).

    Returns:
      (pixels_per_cm2, example_square_bbox)
    or
      (None, None) if detection fails.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_square_area_px:
            continue
        # roughly square
        if abs(w - h) < 0.15 * max(w, h):
            squares.append((x, y, w, h))

    if len(squares) < min_squares:
        return None, None

    areas = [w * h for (_, _, w, h) in squares]
    median_area = np.median(areas)

    # Pick squares closest to the median area
    squares_sorted = sorted(squares, key=lambda s: abs((s[2] * s[3]) - median_area))
    top_squares = squares_sorted[: min(50, len(squares_sorted))]

    avg_area = float(np.mean([w * h for (_, _, w, h) in top_squares]))
    example_square = top_squares[0]
    return avg_area, example_square


# -------------------------------------------------
# Fallback HSV segmentation (no Roboflow)
# -------------------------------------------------

def hsv_segment_leaves(image_bgr: np.ndarray) -> List[BBox]:
    """
    Very simple HSV + watershed pipeline to approximate leaf regions.
    Used as a fallback if Roboflow isn't available/working.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Green-ish leaves – you may tweak these if needed
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Distance transform + watershed to separate touching regions
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image_bgr.copy(), markers)

    # Create binary mask of leaf regions
    leaf_mask = np.zeros_like(mask)
    leaf_mask[markers > 1] = 255

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[BBox] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # Filter out tiny specks
        if area < 200:
            continue
        boxes.append((x, y, w, h))

    return boxes


# -------------------------------------------------
# Roboflow integration
# -------------------------------------------------

def run_leaf_segmentation_roboflow(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Run your Roboflow workflow on the given image bytes.
    Returns the JSON result or None if it fails.
    """
    if not HAS_ROBOFLOW or InferenceHTTPClient is None:
        return None

    # API key from Streamlit secrets
    try:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        st.warning(
            "Roboflow API key not found in Streamlit secrets (`ROBOFLOW_API_KEY`). "
            "Using fallback color-based segmentation."
        )
        return None

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    try:
        # NOTE: no `use_cache` arg here – your SDK version doesn’t support it
        result = client.run_workflow(
            workspace_name="rootweiler",
            workflow_id="find-leaves-3",
            images={"image": image_bytes},
        )
        return result

    except Exception as e:
        st.warning(
            "Roboflow workflow failed or is not configured correctly. "
            "Falling back to simple HSV segmentation.\n\n"
            f"Details: {e}"
        )
        return None


def parse_roboflow_boxes(
    result: Dict[str, Any],
    img_w: int,
    img_h: int,
) -> List[BBox]:
    """
    Try to extract bounding boxes from a Roboflow workflow result.
    This is written to cover a common pattern:
        { "predictions": [ { "x": ..., "y": ..., "width": ..., "height": ... }, ... ] }

    If your workflow output differs, you may need to adjust this function.
    """
    preds = None

    if isinstance(result, dict):
        if "predictions" in result and isinstance(result["predictions"], list):
            preds = result["predictions"]
        elif "output" in result and isinstance(result["output"], dict):
            if "predictions" in result["output"]:
                preds = result["output"]["predictions"]
    elif isinstance(result, list):
        # Some workflows return a list of outputs
        for item in result:
            if isinstance(item, dict) and "predictions" in item:
                preds = item["predictions"]
                break

    if preds is None:
        return []

    boxes: List[BBox] = []
    for p in preds:
        # Center-based bbox is typical: x,y,width,height in pixels
        try:
            if "x" in p and "y" in p and "width" in p and "height" in p:
                cx = float(p["x"])
                cy = float(p["y"])
                w = float(p["width"])
                h = float(p["height"])
            elif "bbox" in p and isinstance(p["bbox"], dict):
                bb = p["bbox"]
                cx = float(bb["x"])
                cy = float(bb["y"])
                w = float(bb["width"])
                h = float(bb["height"])
            else:
                continue

            x_min = int(max(0, cx - w / 2))
            y_min = int(max(0, cy - h / 2))
            x_max = int(min(img_w, cx + w / 2))
            y_max = int(min(img_h, cy + h / 2))

            w_px = max(1, x_max - x_min)
            h_px = max(1, y_max - y_min)
            boxes.append((x_min, y_min, w_px, h_px))
        except Exception:
            continue

    return boxes


# -------------------------------------------------
# Metrics computations
# -------------------------------------------------

def compute_leaf_metrics(
    boxes: List[BBox],
    pixels_per_cm2: Optional[float],
) -> Tuple[List[LeafMeasurement], Dict[str, Optional[float]]]:
    """
    Compute per-leaf cm², leaf height, and leaf-to-plant-height ratio.
    If pixels_per_cm2 is None, areas and heights are returned as None.
    """
    if not boxes:
        return [], {
            "total_leaves": 0,
            "mean_area_cm2": None,
            "std_area_cm2": None,
        }

    # Determine plant height from bounding boxes (top to bottom of all leaves)
    y_min_all = min(y for (x, y, w, h) in boxes)
    y_max_all = max(y + h for (x, y, w, h) in boxes)
    plant_height_px = max(1, y_max_all - y_min_all)

    if pixels_per_cm2 is not None:
        px_per_cm = np.sqrt(pixels_per_cm2)
        plant_height_cm = plant_height_px / px_per_cm
    else:
        plant_height_cm = None

    leaf_measurements: List[LeafMeasurement] = []
    areas_cm2: List[float] = []

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        if pixels_per_cm2 is not None:
            area_cm2 = (w * h) / pixels_per_cm2
            leaf_height_cm = h / np.sqrt(pixels_per_cm2)
            if plant_height_cm and plant_height_cm > 0:
                ratio = leaf_height_cm / plant_height_cm
            else:
                ratio = None

            areas_cm2.append(area_cm2)
        else:
            area_cm2 = None
            leaf_height_cm = None
            ratio = None

        lm = LeafMeasurement(
            leaf_id=i,
            bbox=(x, y, w, h),
            area_cm2=area_cm2,
            height_cm=leaf_height_cm,
            leaf_to_plant_height_ratio=ratio,
        )
        leaf_measurements.append(lm)

    if areas_cm2:
        mean_area = float(np.mean(areas_cm2))
        std_area = float(np.std(areas_cm2))
    else:
        mean_area = None
        std_area = None

    summary = {
        "total_leaves": len(leaf_measurements),
        "mean_area_cm2": mean_area,
        "std_area_cm2": std_area,
    }

    return leaf_measurements, summary


# -------------------------------------------------
# Visualization helpers
# -------------------------------------------------

def draw_calibration_overlay(
    pil_img: Image.Image,
    square_bbox: Optional[BBox],
) -> Image.Image:
    """
    Draw the calibration square on the original image (if detected).
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)

    if square_bbox is not None:
        x, y, w, h = square_bbox
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=3)
        draw.text((x, max(0, y - 15)), "Calibration 1 cm²", fill="red")

    return img


def draw_leaf_overlays(
    pil_img: Image.Image,
    leaves: List[LeafMeasurement],
) -> Image.Image:
    """
    Draw bounding boxes and IDs + area (cm²) on a copy of the image.
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)

    # Try to use a reasonably readable default font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for lm in leaves:
        x, y, w, h = lm.bbox
        draw.rectangle([(x, y), (x + w, y + h)], outline="lime", width=2)

        if lm.area_cm2 is not None:
            label = f"{lm.leaf_id}: {lm.area_cm2:.1f} cm²"
        else:
            label = f"{lm.leaf_id}"

        text_y = max(0, y - 16)
        draw.rectangle(
            [(x, text_y), (x + 120, text_y + 16)],
            fill="black",
        )
        draw.text((x + 2, text_y), label, fill="white", font=font)

    return img


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

class PhenotypingUI:
    """Rootweiler phenotyping: leaf count, leaf sizes, and ratios."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        if not HAS_CV2:
            st.error(
                "Leaf phenotyping requires OpenCV (`opencv-python-headless`).\n\n"
                "Your environment does not have it installed, so this feature "
                "cannot run. Other Rootweiler tools will still work."
            )
            return

        st.markdown(
            """
            Upload a **top-down image** of lettuce plants on the 1 cm² grid.

            Rootweiler will:
            - Count leaves
            - Estimate individual leaf areas (cm²)
            - Estimate a simple leaf-to-plant-height ratio
            - Show which calibration square was used for the grid
            """
        )

        uploaded_file = st.file_uploader(
            "Upload phenotyping image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is None:
            st.info("Upload an image to begin.")
            return

        # Segmentation mode
        mode = st.radio(
            "Segmentation mode",
            ["Roboflow workflow (recommended)", "Color-based fallback only"],
            index=0,
        )
        use_roboflow = (mode == "Roboflow workflow (recommended)")

        if st.button("Run leaf analysis", type="primary"):
            cls._run_analysis(uploaded_file, use_roboflow)

    @classmethod
    def _run_analysis(cls, uploaded_file, use_roboflow: bool):
        # Load image
        cv_img, pil_img, image_bytes = _load_image_from_upload(uploaded_file)
        h, w = cv_img.shape[:2]

        # 1. Calibration: pixels per cm²
        pixels_per_cm2, square_bbox = estimate_pixels_per_cm2(cv_img)

        if pixels_per_cm2 is None:
            st.warning(
                "Could not reliably detect the 1 cm² grid. "
                "Leaf areas and heights will be shown without cm units."
            )
        else:
            st.success(
                f"Estimated grid scale: **{pixels_per_cm2:.0f} px / cm²** "
                f"(using detected squares)."
            )

        # 2. Segmentation: Roboflow or fallback
        boxes: List[BBox] = []

        if use_roboflow:
            rf_result = run_leaf_segmentation_roboflow(image_bytes)
            if rf_result is not None:
                boxes = parse_roboflow_boxes(rf_result, img_w=w, img_h=h)

                if not boxes:
                    st.warning(
                        "Roboflow returned no leaf boxes or an unexpected format. "
                        "Switching to color-based fallback."
                    )

        if not boxes:
            boxes = hsv_segment_leaves(cv_img)

        if not boxes:
            st.error(
                "No leaves detected with either Roboflow or the fallback method. "
                "You may need a clearer top-down image on the grid."
            )
            return

        # 3. Metrics
        leaf_measurements, summary = compute_leaf_metrics(boxes, pixels_per_cm2)

        # 4. Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original image + calibration square")
            img_cal = draw_calibration_overlay(pil_img, square_bbox)
            st.image(img_cal, use_container_width=True)

        with col2:
            st.markdown("#### Leaves with IDs and area (cm²)")
            img_leaves = draw_leaf_overlays(pil_img, leaf_measurements)
            st.image(img_leaves, use_container_width=True)

        # 5. Summary numbers
        st.markdown("### Summary")

        total_leaves = summary["total_leaves"]
        mean_area = summary["mean_area_cm2"]
        std_area = summary["std_area_cm2"]

        cols = st.columns(3)
        cols[0].metric("Leaves detected", f"{total_leaves}")
        if mean_area is not None:
            cols[1].metric("Mean leaf area", f"{mean_area:.1f} cm²")
        else:
            cols[1].metric("Mean leaf area", "N/A")

        if std_area is not None:
            cols[2].metric("Leaf area SD", f"{std_area:.1f} cm²")
        else:
            cols[2].metric("Leaf area SD", "N/A")

        # 6. Detailed table
        st.markdown("### Per-leaf details")

        table_rows = []
        for lm in leaf_measurements:
            x, y, w_box, h_box = lm.bbox
            table_rows.append(
                {
                    "Leaf ID": lm.leaf_id,
                    "Area (cm²)": None if lm.area_cm2 is None else round(lm.area_cm2, 2),
                    "Leaf height (cm)": None if lm.height_cm is None else round(lm.height_cm, 2),
                    "Leaf:plant height ratio": None
                    if lm.leaf_to_plant_height_ratio is None
                    else round(lm.leaf_to_plant_height_ratio, 3),
                    "x": x,
                    "y": y,
                    "w (px)": w_box,
                    "h (px)": h_box,
                }
            )

        st.dataframe(table_rows, use_container_width=True)

        st.caption(
            "Areas and heights depend on the detected 1 cm² grid. "
            "If grid detection fails, values are shown without physical units."
        )
