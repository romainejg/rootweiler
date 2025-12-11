# phenotyping_tools.py

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
@dataclass
class LeafMeasurement:
    id: int
    area_cm2: float
    height_cm: float
    area_height_ratio: float


# ---------------------------------------------------------------------
# Grid calibration – detect 1 cm² squares
# ---------------------------------------------------------------------
def _detect_grid_squares(image_bgr: np.ndarray) -> Tuple[Optional[float], List[Tuple[int, int, int, int]]]:
    """
    Detect square-ish contours and estimate pixel area for a 1 cm² grid square.
    Returns: pixels_per_cm2 (float | None), list of chosen square bounding boxes.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    squares: List[Tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if abs(w - h) < 10 and w * h > 1000:  # roughly square and not tiny
            squares.append((x, y, w, h))

    if len(squares) < 20:
        return None, []

    areas = [w * h for (_, _, w, h) in squares]
    median_area = np.median(areas)

    squares_sorted = sorted(squares, key=lambda s: abs((s[2] * s[3]) - median_area))
    chosen = squares_sorted[:20]

    avg_area = float(np.mean([w * h for (_, _, w, h) in chosen]))
    pixels_per_cm2 = avg_area

    return pixels_per_cm2, chosen


def _overlay_grid_boxes(image_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Return a copy of the image with green rectangles drawn on detected grid squares."""
    out = image_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


# ---------------------------------------------------------------------
# Roboflow workflow wrapper
# ---------------------------------------------------------------------
def _run_roboflow_workflow(image_bytes: bytes) -> Optional[Dict[str, object]]:
    """
    Call the Roboflow workflow 'leafy' and return predictions.
    Returns None if anything fails.
    """
    api_key = None
    if "ROBOFLOW_API_KEY" in st.secrets:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
        api_key = st.secrets["roboflow"]["api_key"]

    if not api_key:
        st.info("Roboflow API key not found in secrets – using color-based segmentation instead.")
        return None

    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
    except Exception as e:
        st.info(f"Could not initialize Roboflow client ({type(e).__name__}). Falling back to color segmentation.")
        return None

    tmp_path = "tmp_phenotype_image.jpg"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)

    try:
        result = client.run_workflow(
            workspace_name="rootweiler",
            workflow_id="leafy",
            images={"image": tmp_path},
        )
    except Exception as e:
        st.info(f"Roboflow workflow call failed ({type(e).__name__}). Falling back to color segmentation.")
        return None
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if isinstance(result, list) and len(result) > 0:
        obj = result[0]
    elif isinstance(result, dict):
        obj = result
    else:
        return None

    output2 = obj.get("output2")
    if not isinstance(output2, dict):
        return None

    preds = output2.get("predictions")
    if not isinstance(preds, list) or len(preds) == 0:
        return None

    return {"predictions": preds}


def _mask_from_roboflow_predictions(image_shape: Tuple[int, int, int], predictions: List[dict]) -> np.ndarray:
    """Build a binary mask from Roboflow polygons."""
    h, w, _ = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for pred in predictions:
        pts = pred.get("points")
        if not pts or not isinstance(pts, list):
            continue
        poly = np.array([[p["x"], p["y"]] for p in pts if "x" in p and "y" in p], dtype=np.int32)
        if poly.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [poly], 255)

    return mask


# ---------------------------------------------------------------------
# Fallback color-based segmentation
# ---------------------------------------------------------------------
def _color_based_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Simple green-ish segmentation as fallback."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


# ---------------------------------------------------------------------
# Leaf measurement helpers
# ---------------------------------------------------------------------
def _measure_leaves(mask: np.ndarray, pixels_per_cm2: float) -> List[LeafMeasurement]:
    """Use connected components on the binary mask to measure each leaf."""
    if pixels_per_cm2 <= 0:
        raise ValueError("pixels_per_cm2 must be positive")

    pixels_per_cm = float(np.sqrt(pixels_per_cm2))
    num_labels, labels = cv2.connectedComponents(mask)
    measurements: List[LeafMeasurement] = []

    for label_id in range(1, num_labels):
        component = (labels == label_id)
        area_px = int(np.count_nonzero(component))
        if area_px < 500:
            continue

        ys, xs = np.where(component)
        if len(ys) == 0:
            continue

        height_px = ys.max() - ys.min() + 1
        area_cm2 = area_px / pixels_per_cm2
        height_cm = height_px / pixels_per_cm
        ratio = area_cm2 / height_cm if height_cm > 0 else 0.0

        measurements.append(
            LeafMeasurement(
                id=len(measurements) + 1,
                area_cm2=area_cm2,
                height_cm=height_cm,
                area_height_ratio=ratio,
            )
        )

    return measurements


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
class PhenotypingUI:
    """Leaf phenotyping tool using Roboflow (preferred) + grid calibration."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")
        st.markdown(
            """
            Upload a **phenotyping photo** taken on the 1 cm grid board.  
            Rootweiler will attempt to detect the grid squares, calibrate pixel size, and then
            segment leaves using Roboflow (if API key available) or fallback color segmentation.
            """
        )

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is None:
            st.info("Upload a photo to start.")
            return

        file_bytes = uploaded_file.read()
        file_array = np.frombuffer(file_bytes, np.uint8)
        image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Could not read the uploaded image.")
            return

        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)

        pixels_per_cm2, boxes = _detect_grid_squares(image_bgr)
        if not pixels_per_cm2:
            st.warning("Could not detect enough grid squares for calibration.")
            return

        st.image(cv2.cvtColor(_overlay_grid_boxes(image_bgr, boxes), cv2.COLOR_BGR2RGB),
                 caption="Detected grid squares", use_column_width=True)

        roboflow_result = _run_roboflow_workflow(file_bytes)
