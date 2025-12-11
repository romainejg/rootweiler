# phenotyping_tools.py - Bare Bones Mask Output with Direct HTTP Call

import io
import os
import streamlit as st
import cv2
import numpy as np
import base64
import requests # <-- We rely on this standard library now
from typing import List, Optional, Tuple, Dict

# No longer need to import inference_sdk if we use requests

# ---------------------------------------------------------------------
# Roboflow Workflow Integration (Using Direct HTTP POST)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Running Roboflow segmentation...")
def _run_roboflow_workflow(image_bytes: bytes) -> Tuple[Optional[Dict[str, object]], bool]:
    
    # 1. Key Retrieval (Retained for validation)
    api_key = None
    if "ROBOFLOW_API_KEY" in st.secrets:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
        api_key = st.secrets["roboflow"]["api_key"]

    if not api_key:
        return None, False

    api_key = api_key.strip()
    
    # 2. Prepare the request (Image to Base64)
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
    except Exception:
        return None, True

    url = "https://detect.roboflow.com/infer/workflows/rootweiler/leafy"
    
    params = {
        "api_key": api_key
    }
    
    # Payload matches Roboflow Workflow input expectations for base64
    payload = {
        "inputs": {
            "image": {
                "type": "base64",
                "value": base64_image
            }
        }
    }

    # 3. Send the Request
    result = None
    try:
        response = requests.post(url, params=params, json=payload)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        result = response.json()
        
    except requests.exceptions.RequestException as e:
        # Check if it's an API key issue (though less likely now)
        if hasattr(e, 'response') and (e.response.status_code == 401 or e.response.status_code == 403):
            st.error("❌ Roboflow Auth Error via HTTP. Key permissions may be incorrect.")
        else:
            st.error(f"❌ Network/HTTP Error: {e}")
        return None, True
    except Exception as e:
        st.error(f"❌ General Error during HTTP request: {e}")
        return None, True


    # 4. Process Results
    
    # A successful HTTP workflow call returns a dictionary, not a list
    obj = result
    
    if not isinstance(obj, dict):
        st.error("❌ API Response was not a valid dictionary object.")
        return None, True
    
    # Extract predictions from "output2" (Assuming this is still the workflow output name)
    preds = obj.get("output2")
    
    if not isinstance(preds, list) or len(preds) == 0:
        st.warning("⚠️ Prediction list ('output2') was empty. Model saw nothing.")
        return None, True

    return {"predictions": preds}, True


def _mask_from_roboflow_predictions(image_shape: Tuple[int, int, int], predictions: List[dict]) -> np.ndarray:
    """
    Build a binary mask from Roboflow instance segmentation polygons.
    """
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
# Streamlit UI (Bare Bones)
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
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)

        # --- Segmentation ---
        with st.spinner("Generating mask..."):
            rf_result, attempted_rf = _run_roboflow_workflow(image_bytes)

        if rf_result is not None:
            mask_bin = _mask_from_roboflow_predictions(img_rgb.shape, rf_result["predictions"])
            st.success(f"✅ Segmentation Success: Found {len(rf_result['predictions'])} leaf instances.")
            
            # Use original and mask side-by-side
            c1, c2 = st.columns(2)
            with c1: 
                st.image(img_rgb, caption="Original Image", use_container_width=True)
            with c2: 
                st.image(mask_bin, caption="Generated Binary Mask", use_container_width=True)

        else:
            if attempted_rf:
                st.error("❌ Roboflow failed to return a mask. The API call completed, but no leaves were detected, or a non-SDK error occurred.")
            else:
                st.error("❌ API Key not found in `secrets.toml`. Cannot generate mask.")
