# phenotyping.py

import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient


# -----------------------------
# Data structures / type aliases
# -----------------------------

@dataclass
class LeafMeasurement:
    leaf_id: int
    area_cm2: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    ratio_area_to_height: Optional[float]


BBox = Tuple[int, int, int, int]


# -----------------------------
# Calibration: pixels per cm²
# -----------------------------

def calculate_pixels_per_cm2_with_example(
    image_bgr: np.ndarray,
) -> Tuple[Optional[float], Optional[BBox]]:
    """
    Estimate pixels per cm² using grid squares in the background.

    Returns:
        pixels_per_cm2 (float or None),
        example_square_bbox (x, y, w, h) or None
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[BBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Roughly square and not tiny
        if abs(w - h) < 10 and w * h > 1000:
            squares.append((x, y, w, h))

    if len(squares) < 20:
        return None, None

    areas = [w * h for (_, _, w, h) in squares]
    median_area = float(np.median(areas))

    # pick the squares closest to the median area
    squares_sorted = sorted(
        squares,
        key=lambda s: abs((s[2] * s[3]) - median_area)
    )
    chosen = squares_sorted[:20]
    avg_area = float(np.mean([w * h for (_, _, w, h) in chosen]))

    # Use the first of the chosen ones as example, for overlay
    example_square = chosen[0]

    pixels_per_cm2 = avg_area  # each grid square is 1 cm²
    return pixels_per_cm2, example_square


# -----------------------------
# Simple HSV fallback mask
# -----------------------------

def create_mask_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """
    Simple HSV + watershed leaf mask as a fallback if Roboflow fails.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Quite broad green range; you can refine these if needed
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_bgr, markers)
    final_mask = np.zeros_like(mask)
    final_mask[markers > 1] = 255

    return final_mask


# -----------------------------
# Roboflow workflow integration
# -----------------------------

def run_roboflow_segmentation(
    image_bgr: np.ndarray,
    workspace_name: str = "rootweiler",
    workflow_id: str = "find-leaves-3",
) -> Optional[np.ndarray]:
    """
    Call a Roboflow workflow to segment leaves.

    - Uses a temporary file path for the image (Roboflow example style).
    - Expects the workflow to output detections with polygons or bounding boxes.
    - Returns a binary mask (255 where leaf, 0 elsewhere) on success, or None if
      anything doesn't line up.
    """

    api_key = st.secrets.get("ROBOFLOW_API_KEY", None)
    if not api_key:
        st.info("No Roboflow API key found in secrets. Using color-based segmentation.")
        return None

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    # Write temp image file (JPEG) and pass its path string
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, image_bgr)

    try:
        # ⚠️ IMPORTANT: keys in `images={...}` must match the name of the image input
        # node in your Roboflow workflow. Docs example uses "image".
        result = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={"image": temp_path},
        )
    except Exception as e:
        st.warning(
            "Roboflow workflow failed or is not configured correctly. "
            "Falling back to color-based segmentation."
        )
        # Show full traceback in the app so we can see what's wrong
        st.exception(e)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Show raw result (once) to understand structure
    with st.expander("Debug: raw Roboflow result", expanded=False):
        st.json(result)

    # Try to find predictions
    predictions = None

    if isinstance(result, dict):
        if "predictions" in result:
            predictions = result["predictions"]
        elif "image" in result and isinstance(result["image"], dict) and "predictions" in result["image"]:
            predictions = result["image"]["predictions"]
        elif "steps" in result and isinstance(result["steps"], dict):
            # Try to find first step with "predictions"
            for step_name, step in result["steps"].items():
                if isinstance(step, dict) and "predictions" in step:
                    st.caption(f"Using predictions from workflow step: **{step_name}**")
                    predictions = step["predictions"]
                    break

    if not predictions:
        st.warning(
            "Roboflow workflow ran but no 'predictions' field was found in the response. "
            "Showing raw result above – we may need to adjust how we parse it."
        )
        return None

    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # We will handle both bounding boxes and polygons if available
    for det in predictions:
        # Example keys might be: {'x','y','width','height'} or 'points'
        if isinstance(det, dict) and "points" in det:  # polygon
            pts = np.array(det["points"], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        elif isinstance(det, dict) and {"x", "y", "width", "height"} <= det.keys():
            cx = det["x"]
            cy = det["y"]
            bw = det["width"]
            bh = det["height"]
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Final clean-up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


# -----------------------------
# Leaf measurements
# -----------------------------

def measure_leaves_from_mask(
    mask: np.ndarray,
    pixels_per_cm2: float,
) -> Tuple[List[LeafMeasurement], Optional[float]]:
    """
    Given a binary leaf mask and pixels/cm², compute per-leaf and plant metrics.

    Returns:
        - list of LeafMeasurement
        - plant_height_cm (float)
    """
    # Plant height: vertical extent of all leaf pixels
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return [], None

    plant_top = ys.min()
    plant_bottom = ys.max()
    plant_height_px = plant_bottom - plant_top
    plant_height_cm = plant_height_px / np.sqrt(pixels_per_cm2)

    # Contours → each leaf
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    measurements: List[LeafMeasurement] = []
    leaf_id = 1
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < 50:  # tiny noise
            continue

        area_cm2 = area_px / pixels_per_cm2
        x, y, w, h = cv2.boundingRect(cnt)

        if plant_height_cm and plant_height_cm > 0:
            ratio = area_cm2 / plant_height_cm
        else:
            ratio = None

        measurements.append(
            LeafMeasurement(
                leaf_id=leaf_id,
                area_cm2=area_cm2,
                bbox=(x, y, w, h),
                ratio_area_to_height=ratio,
            )
        )
        leaf_id += 1

    return measurements, plant_height_cm


def draw_overlays(
    original_bgr: np.ndarray,
    mask: np.ndarray,
    leaves: List[LeafMeasurement],
    calib_square: Optional[BBox],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return two RGB images:
      - original image with leaf boxes + IDs + calibration square
      - mask visualization
    """
    vis_orig = original_bgr.copy()
    vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Draw calibration square
    if calib_square is not None:
        x, y, w, h = calib_square
        cv2.rectangle(vis_orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            vis_orig,
            "1 cm² ref",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw leaf boxes + IDs
    for leaf in leaves:
        x, y, w, h = leaf.bbox
        cv2.rectangle(vis_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis_orig,
            str(leaf.leaf_id),
            (x, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Convert BGR to RGB for Streamlit display
    vis_orig_rgb = cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB)
    vis_mask_rgb = cv2.cvtColor(vis_mask, cv2.COLOR_BGR2RGB)

    return vis_orig_rgb, vis_mask_rgb


# -----------------------------
# Streamlit UI wrapper
# -----------------------------

class PhenotypingUI:
    """Rootweiler phenotyping: leaf segmentation & measurements."""

    @classmethod
    def render(cls):
        st.subheader("Phenotyping – Leaf segmentation")

        st.markdown(
            """
            Upload a plant image on a calibration grid.  
            Rootweiler will try to:
            - Detect and separate individual leaves  
            - Use the grid squares (1 cm² each) as a **scale**  
            - Report **leaf count, leaf area (cm²)**, and **leaf-area-to-height ratio**  
            """
        )

        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            help="Use a top-down photo on the square grid background when possible.",
        )

        if uploaded is None:
            st.info("Upload a plant image to begin.")
            return

        # Read image into OpenCV (BGR)
        pil_img = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Show original
        st.markdown("#### Original image")
        st.image(img_rgb, use_container_width=True)

        if st.button("Run phenotyping", type="primary"):
            cls._run_analysis(img_bgr)

    @classmethod
    def _run_analysis(cls, image_bgr: np.ndarray):
        # 1) Calibration
        pixels_per_cm2, calib_square = calculate_pixels_per_cm2_with_example(image_bgr)
        if pixels_per_cm2 is None:
            st.error(
                "Could not reliably detect grid squares to calibrate scale. "
                "Try a clearer image with visible 1 cm² grid."
            )
            return

        st.markdown(f"- Estimated scale: **{pixels_per_cm2:.1f} pixels per cm²**")

        # 2) Try Roboflow segmentation
        mask = run_roboflow_segmentation(image_bgr)
        if mask is None:
            mask = create_mask_hsv(image_bgr)

        # 3) Leaf measurements
        leaves, plant_height_cm = measure_leaves_from_mask(mask, pixels_per_cm2)
        if not leaves:
            st.warning("No leaf regions detected. Try another image or adjust your workflow.")
            return

        # 4) Overlays
        vis_orig_rgb, vis_mask_rgb = draw_overlays(
            original_bgr=image_bgr,
            mask=mask,
            leaves=leaves,
            calib_square=calib_square,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Segmented leaves (with IDs)")
            st.image(vis_orig_rgb, use_container_width=True)
        with col2:
            st.markdown("#### Leaf mask")
            st.image(vis_mask_rgb, use_container_width=True)

        # 5) Numeric results (all in cm / cm², no pixels)
        areas = np.array([leaf.area_cm2 for leaf in leaves], dtype=float)
        ratios = np.array(
            [leaf.ratio_area_to_height for leaf in leaves if leaf.ratio_area_to_height is not None],
            dtype=float,
        )

        st.markdown("### Leaf metrics")

        st.write(f"- **Leaf count:** {len(leaves)}")
        if plant_height_cm is not None:
            st.write(f"- **Plant height (vertical leaf extent):** {plant_height_cm:.1f} cm")

        st.write(f"- **Average leaf area:** {areas.mean():.2f} cm²")
        st.write(f"- **Leaf area standard deviation:** {areas.std(ddof=1):.2f} cm²")

        if ratios.size > 0:
            st.write(f"- **Mean leaf area : plant height ratio:** {ratios.mean():.3f} (cm²/cm)")
        else:
            st.write("- Leaf-to-height ratio: not available (height could not be determined).")

        # Table with all leaves
        rows = []
        for leaf in sorted(leaves, key=lambda l: l.leaf_id):
            row = {
                "Leaf ID": leaf.leaf_id,
                "Area (cm²)": round(leaf.area_cm2, 2),
            }
            if leaf.ratio_area_to_height is not None:
                row["Area / Height (cm²/cm)"] = round(leaf.ratio_area_to_height, 3)
            rows.append(row)

        st.markdown("#### Per-leaf details")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)