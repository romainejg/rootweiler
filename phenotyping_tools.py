import os
import base64
import tempfile
from typing import Any, Dict, List, Union

import streamlit as st
from inference_sdk import InferenceHTTPClient


# -------------------------
# Helpers
# -------------------------

def get_client(api_key: str, api_url: str) -> InferenceHTTPClient:
    """
    Create an InferenceHTTPClient.

    For hosted workflows use:
        api_url="https://detect.roboflow.com"
    or (if your deploy snippet says so) "https://serverless.roboflow.com".
    """
    return InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key,
    )


def decode_base64_image(b64_val: str) -> bytes:
    """
    Handles both raw base64 and data URLs (e.g. 'data:image/png;base64,...').
    Returns raw image bytes.
    """
    if b64_val.startswith("data:image"):
        b64_val = b64_val.split(",", 1)[-1]
    return base64.b64decode(b64_val)


def extract_first_item(result: Any) -> Dict:
    """
    Workflows often return a list containing one dict.
    This normalizes list/dict into a single dict.
    """
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    if isinstance(result, dict):
        return result
    return {}


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Roboflow Workflow Test", layout="centered")
st.title("🔬 Roboflow Workflow Sanity Check")

st.markdown(
    """
This app **only** tests your Roboflow workflow call and output, 
before we plug it into your phenotyping logic.

1. Enter your **API key**, **workspace name**, and **workflow id**.  
2. Upload an image.  
3. Click **Run workflow**.  

If everything is wired correctly, you'll see:
- Raw JSON from Roboflow
- Any **base64 image outputs** decoded and displayed
"""
)

# --- Config inputs ---
st.sidebar.header("Roboflow configuration")

default_api_key = os.getenv("ROBOFLOW_API_KEY", "")
api_key = st.sidebar.text_input("API key", type="password", value=default_api_key)
workspace_name = st.sidebar.text_input("Workspace name", value="rootweiler")
workflow_id = st.sidebar.text_input("Workflow ID", value="leafy")

api_url = st.sidebar.selectbox(
    "API URL",
    options=[
        "https://detect.roboflow.com",      # hosted workflows (per docs)
        "https://serverless.roboflow.com",  # used in some deploy snippets
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Python must have `inference-sdk` installed.")


uploaded_file = st.file_uploader(
    "Upload a test phenotyping image",
    type=["jpg", "jpeg", "png"],
)

run_button = st.button("Run workflow")

if run_button:
    if not api_key:
        st.error("Please provide your Roboflow API key.")
    elif not workspace_name or not workflow_id:
        st.error("Please provide workspace name and workflow id.")
    elif uploaded_file is None:
        st.error("Please upload an image.")
    else:
        # ---------------------------------
        # 1. Initialize client
        # ---------------------------------
        try:
            client = get_client(api_key=api_key, api_url=api_url)
        except Exception as e:
            st.error(f"Failed to create InferenceHTTPClient: {type(e).__name__}: {e}")
            st.stop()

        # ---------------------------------
        # 2. Save uploaded image to temp file
        # ---------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.info("Calling Roboflow workflow…")

        try:
            # IMPORTANT: do NOT pass `use_cache` here to avoid your TypeError
            result = client.run_workflow(
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                images={"image": tmp_path},
                # parameters={}  # add if your workflow has parameters
            )
        except Exception as e:
            st.error(f"Workflow call failed: {type(e).__name__}: {e}")
            # Clean up temp file
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            st.stop()

        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        st.success("Workflow call succeeded ✅")

        # ---------------------------------
        # 3. Show raw JSON
        # ---------------------------------
        st.subheader("Raw workflow result")
        st.json(result)

        # Normalize to first item
        item = extract_first_item(result)

        # ---------------------------------
        # 4. Try to display any base64 images
        # ---------------------------------
        st.subheader("Decoded image outputs (if present)")

        # Common output names for your case:
        # - "output"   -> mask_visualization.image (base64)
        # - "output2"  -> predictions (list, not an image)
        image_fields: List[str] = []

        for key, value in item.items():
            # Heuristic: try fields that look like base64 images
            if isinstance(value, str) and ("data:image" in value or len(value) > 100):
                image_fields.append(key)

        if not image_fields:
            st.info(
                "No obvious base64 image fields found in the result. "
                "Check your workflow Outputs configuration (e.g. JsonField vs. Image Field)."
            )
        else:
            for field_name in image_fields:
                try:
                    img_bytes = decode_base64_image(item[field_name])
                    st.markdown(f"**Output field:** `{field_name}`")
                    st.image(img_bytes, use_container_width=True)
                except Exception as e:
                    st.warning(
                        f"Field `{field_name}` looked like base64, "
                        f"but could not be decoded: {type(e).__name__}: {e}"
                    )

        # ---------------------------------
        # 5. Show a quick peek at predictions if present
        # ---------------------------------
        st.subheader("Prediction list (if present)")

        preds = None
        # Your JSON schema:
        #   output2 -> $steps.model.predictions
        if "output2" in item:
            preds = item["output2"]
        elif "predictions" in item:
            preds = item["predictions"]

        if isinstance(preds, list) and preds:
            st.write(f"Found {len(preds)} predictions. Showing first 5:")
            st.json(preds[:5])
        else:
            st.info(
                "No `output2` or `predictions` list found. "
                "If your workflow is instance segmentation, make sure the Output "
                "is configured to expose `$steps.<your_model_step>.predictions`."
            )
