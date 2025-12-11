# phenotyping_tools.py - Bare Bones Mask Output with Key Debug

import io
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict

# Roboflow SDK Import
from inference_sdk import InferenceHTTPClient

# ---------------------------------------------------------------------
# Roboflow SDK Integration (Minimal Version)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Running Roboflow segmentation...")
def _run_roboflow_workflow(image_bytes: bytes) -> Tuple[Optional[Dict[str, object]], bool]:
    
    # ---------------------------------------------------------------
    # 1. KEY RETRIEVAL AND DEBUG BLOCK
    # ---------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔑 API Key Access Debug")
    
    api_key = None
    key_source = "None Found"

    if "ROBOFLOW_API_KEY" in st.secrets:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
        key_source = "ROBOFLOW_API_KEY (Top Level)"
    elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
        api_key = st.secrets["roboflow"]["api_key"]
        key_source = "roboflow/api_key (Section Level)"

    # Check if the key was found
    if api_key is None:
        st.error(f"❌ Key Retrieval Failed. Checked sources: {key_source}")
        st.markdown("---")
        return None, False

    # Check the state of the key after retrieval
    original_key_length = len(api_key)
    api_key = api_key.strip()
    stripped_key_length = len(api_key)

    if stripped_key_length == 0:
        st.error(f"❌ Key Found but is Empty/Whitespace only (Length: {original_key_length}).")
        st.markdown("---")
        return None, False

    if original_key_length != stripped_key_length:
        st.warning(f"⚠️ Key successfully stripped! Length was {original_key_length}, is now {stripped_key_length}.")
    
    st.success(f"✅ Key Found and Ready. Source: {key_source}, Final Length: {stripped_key_length}.")
    st.markdown("---")
    
    # End of Debug Block
    # ---------------------------------------------------------------

    # 2. Setup Client
    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            # We use the stripped api_key here
            api_key=api_key 
        )
    except Exception as e:
        st.error(f"❌ Client initialization failed (Pre-API call). Error: {e}")
        st.markdown("---")
        return None, True

    # 3. Save Temp File
    tmp_path = "tmp_phenotype_image.jpg"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)

    # 4. Run Workflow
    result = None
    try:
        result = client.run_workflow(
            workspace_name="rootweiler",
            workflow_id="leafy",
            images={
                "image": tmp_path
            },
            parameters={
                "output_message": "Segmentation started."
            }
        )
    except Exception as e:
        error_str = str(e)
        st.markdown("#### Debug API Call Error")
        st.error(f"❌ Roboflow Workflow Error: {e}")
        if "401" in error_str or "403" in error_str or "Unauthorized" in error_str:
             st.error("Authentication likely failed despite key being read.")
        st.markdown("---")
        return None, True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 5. Process Results
    st.markdown("#### Debug API Response (Raw)")
    st.json(result) # <-- Raw response for inspection

    if not isinstance(result, list) or len(result) == 0:
        st.error("❌ Raw result was empty/invalid list from API.")
        return None, True

    # Get the first result object
    obj = result[0]
    
    # Extract predictions from "output2"
    preds = obj.get("output2")
    
    if not isinstance(preds, list) or len(preds) == 0:
        st.warning("⚠️ Prediction list ('output2') was empty. Model saw nothing or key is still failing.")
        return None, True
    
    st.markdown("---")

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
                # Ensure mask is 3-channel for colored visualization if needed, but binary is fine
                st.image(mask_bin, caption="Generated Binary Mask", use_container_width=True)

        else:
            if attempted_rf:
                st.error("❌ Roboflow failed to return a mask. Check API Key validity/permissions and Raw Response above.")
            else:
                st.error("❌ API Key not found in `secrets.toml`. Cannot generate mask.")
