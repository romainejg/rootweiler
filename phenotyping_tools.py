# phenotyping_tools.py - Final Attempt with Simplified PIL Handling

import io
import os
import streamlit as st
import cv2
import numpy as np
import base64
import requests
from typing import List, Optional, Tuple, Dict

# ---------------------------------------------------------------------
# Roboflow Workflow Integration (Using Direct HTTP POST) - UNCHANGED
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Running Roboflow segmentation...")
def _run_roboflow_workflow(image_bytes: bytes) -> Tuple[Optional[Dict[str, object]], bool]:
    # 1. Key Retrieval
    api_key = None
    if "ROBOFLOW_API_KEY" in st.secrets:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
        api_key = st.secrets["roboflow"]["api_key"]
    if not api_key: return None, False
    api_key = api_key.strip()
    
    # 2. Prepare the request
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
    except Exception:
        return None, True

    url = "https://detect.roboflow.com/infer/workflows/rootweiler/leafy"
    params = {"api_key": api_key}
    payload = {"inputs": {"image": {"type": "base64", "value": base64_image}}}

    # 3. Send the Request
    result = None
    try:
        response = requests.post(url, params=params, json=payload)
        response.raise_for_status()
        result = response.json()
    except Exception:
        return None, True

    # 4. Process Results
    obj = result
    if not isinstance(obj, dict): return None, True
    preds = obj.get("output2")
    if not isinstance(preds, list) or len(preds) == 0: return None, True

    return {"predictions": preds}, True


def _mask_from_roboflow_predictions(image_shape: Tuple[int, int, int], predictions: List[dict]) -> np.ndarray:
    """Build a binary mask from Roboflow instance segmentation polygons."""
    h, w, _ = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for pred in predictions:
        pts = pred.get("points")
        if not pts or not isinstance(pts, list): continue
        poly = np.array([[int(p["x"]), int(p["y"])] for p in pts if "x" in p and "y" in p], dtype=np.int32)
        if poly.shape[0] < 3: continue
        cv2.fillPoly(mask, [poly], 255)
    return mask


# ---------------------------------------------------------------------
# Streamlit UI (Bare Bones) - UPDATED FILE HANDLING
# ---------------------------------------------------------------------
class PhenotypingUI:
    """Bare-bones UI to upload image and display segmentation mask."""

    @classmethod
    def render(cls):
        st.subheader("Leaf Segmentation Mask Generator")
        st.markdown("Upload an image to generate the instance segmentation mask.")

        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        image_bytes = uploaded.read()
        
        # --- CRITICAL SAFETY CHECK (Enhanced) ---
        if not image_bytes:
            st.error("❌ The uploaded file is empty or corrupted (zero bytes). Please check the file and try again.")
            return
        # -----------------------------

        # --- SIMPLIFIED PIL OPENING (Crucial change) ---
        try:
            # Open the image without immediate conversion to eliminate a potential source of failure
            pil_img = Image.open(io.BytesIO(image_bytes))
            
            # Now, safely convert it to RGB
            img_rgb = np.array(pil_img.convert("RGB")) 
            
        except Exception as e:
            # Report the exact exception to help identify library/format issues
            st.error(f"❌ Could not open the uploaded file. Ensure it is a valid JPG or PNG image. (Error: {type(e).__name__})")
            return
        # -----------------------------------------------

        # --- Segmentation ---
        with st.spinner("Generating mask..."):
            rf_result, attempted_rf = _run_roboflow_workflow(image_bytes)

        if rf_result is not None:
            mask_bin = _mask_from_roboflow_predictions(img_rgb.shape, rf_result["predictions"])
            st.success(f"✅ Segmentation Success: Found {len(rf_result['predictions'])} leaf instances.")
            
            # Display
            c1, c2 = st.columns(2)
            with c1: 
                st.image(img_rgb, caption="Original Image", use_container_width=True)
            with c2: 
                st.image(mask_bin, caption="Generated Binary Mask", use_container_width=True)

        else:
            if attempted_rf:
                st.error("❌ Roboflow failed to return a mask. This may mean zero leaves were detected, or a network/API issue occurred.")
            else:
                st.error("❌ API Key not found in `secrets.toml`. Cannot generate mask.")
