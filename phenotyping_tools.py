# phenotyping_tools.py
#
# Minimal phenotyping test:
# - Upload an image
# - Call Roboflow "leafy" workflow via HTTP
# - Show the raw JSON response
#
# Keeps the same PhenotypingUI class and render() signature
# so app.py does not need to change.

import base64
import io
import os
from typing import Optional, Dict, Any

import requests
import streamlit as st
from PIL import Image


class PhenotypingUI:
    """
    Minimal phenotyping UI to prove that the Roboflow workflow call works
    inside your Streamlit app.

    Flow:
    - User uploads an image
    - We display the image
    - We call the Roboflow Workflow API for 'rootweiler/leafy'
    - We print the JSON result

    Once this is solid, you can wire the JSON into your measurement logic.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @classmethod
    def _get_api_key(cls) -> Optional[str]:
        """
        Get Roboflow API key from Streamlit secrets.

        Supports either:
        - ROBOFLOW_API_KEY = "..."
        or
        - [roboflow]
          api_key = "..."
        in .streamlit/secrets.toml
        """
        api_key = None

        if "ROBOFLOW_API_KEY" in st.secrets:
            api_key = st.secrets["ROBOFLOW_API_KEY"]
        elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
            api_key = st.secrets["roboflow"]["api_key"]

        if not api_key:
            st.error(
                "Roboflow API key not found.\n\n"
                "Please add one of the following to `.streamlit/secrets.toml`:\n\n"
                'ROBOFLOW_API_KEY = "your_key_here"\n\n'
                "or\n\n"
                "[roboflow]\n"
                'api_key = "your_key_here"\n'
            )
            return None

        return api_key

    @classmethod
    def _call_workflow_http(
        cls, api_key: str, workspace_name: str, workflow_id: str, image_bytes: bytes
    ) -> Dict[str, Any]:
        """
        Call Roboflow workflow via the HTTP REST API directly.

        Endpoint (from docs):
        POST https://detect.roboflow.com/infer/workflows/<workspace>/<workflow-id>

        Body:
        {
          "api_key": "...",
          "inputs": {
            "image": {
              "type": "base64",
              "value": "<base64-encoded-image>"
            }
          }
        }
        """
        # Encode image as base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        url = f"https://detect.roboflow.com/infer/workflows/{workspace_name}/{workflow_id}"

        payload = {
            "api_key": api_key,
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": image_b64,
                }
            },
        }

        # You can add more inputs here later if your workflow expects them

        resp = requests.post(url, json=payload, timeout=60)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Try to show any error message from Roboflow in addition to status code
            msg = f"HTTP {resp.status_code}"
            try:
                j = resp.json()
                # Roboflow errors often come with 'error' or 'message' fields
                err_text = j.get("error") or j.get("message")
                if err_text:
                    msg += f" – {err_text}"
            except Exception:
                pass
            raise RuntimeError(f"Roboflow HTTP error: {msg}") from e

        try:
            result = resp.json()
        except Exception as e:
            raise RuntimeError(f"Could not decode JSON from Roboflow: {e}") from e

        return result

    # ------------------------------------------------------------------
    # Public UI
    # ------------------------------------------------------------------
    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping (Roboflow workflow test)")

        st.markdown(
            """
            This is a **minimal test** of your Roboflow workflow integration.

            It will:

            1. Let you upload a phenotyping image  
            2. Send it to the Roboflow **leafy** workflow via HTTP  
            3. Show the raw JSON response from Roboflow  

            Once this works, you can plug the outputs into your
            grid calibration, masks, and leaf measurements.
            """
        )

        uploaded = st.file_uploader(
            "Upload a leaf / phenotyping image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        # Read the uploaded bytes
        image_bytes = uploaded.read()

        # Show the uploaded image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}")
            return

        st.image(pil_img, caption="Uploaded image", use_column_width=True)

        # Get API key
        api_key = cls._get_api_key()
        if api_key is None:
            return

        # Workspace and workflow ID: adjust if your names differ
        workspace_name = "rootweiler"  # from your description
        workflow_id = "leafy"          # from your description

        st.caption(f"Workflow: `{workspace_name}/{workflow_id}`")

        if st.button("Run Roboflow 'leafy' workflow"):
            with st.spinner("Calling Roboflow workflow..."):
                try:
                    result = cls._call_workflow_http(
                        api_key=api_key,
                        workspace_name=workspace_name,
                        workflow_id=workflow_id,
                        image_bytes=image_bytes,
                    )
                except Exception as e:
                    st.error(f"Roboflow workflow call failed: {e}")
                    return

            st.success("Successfully received a response from Roboflow.")
            st.markdown("### Raw JSON result")
            st.json(result)
