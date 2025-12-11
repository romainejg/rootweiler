# phenotyping_tools.py (bare-bones test version)

import io
import os

import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient


class PhenotypingUI:
    """
    Minimal phenotyping UI just to prove:
    - We can upload an image
    - We can call the Roboflow 'leafy' workflow
    - We see the raw JSON response in the app

    This keeps the same class name and render() method
    so app.py does not need to change.
    """

    @classmethod
    def _get_client(cls) -> InferenceHTTPClient | None:
        """Create a Roboflow InferenceHTTPClient from st.secrets."""
        api_key = None

        # Option 1: flat key in secrets.toml
        if "ROBOFLOW_API_KEY" in st.secrets:
            api_key = st.secrets["ROBOFLOW_API_KEY"]

        # Option 2: nested key like:
        # [roboflow]
        # api_key = "..."
        elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
            api_key = st.secrets["roboflow"]["api_key"]

        if not api_key:
            st.error(
                "Roboflow API key not found.\n\n"
                "Add one of the following to `.streamlit/secrets.toml`:\n\n"
                'ROBOFLOW_API_KEY = "your_key_here"\n\n'
                "or\n\n"
                '[roboflow]\napi_key = "your_key_here"'
            )
            return None

        try:
            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key,
            )
            return client
        except Exception as e:
            st.error(f"Could not initialize Roboflow client: {e}")
            return None

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping (Roboflow test)")

        st.markdown(
            """
            This is a **minimal test** to check that:

            1. You can upload an image  
            2. The image is sent to the Roboflow **leafy** workflow  
            3. The app receives and displays the raw JSON response  

            Once this works, we can plug the results into your
            more advanced grid calibration + measurement logic.
            """
        )

        uploaded = st.file_uploader(
            "Upload a leaf / phenotyping image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        # Read bytes once
        image_bytes = uploaded.read()

        # Show the uploaded image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(pil_img, caption="Uploaded image", use_column_width=True)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            return

        client = cls._get_client()
        if client is None:
            # Error already shown
            return

        # Small UX: only call Roboflow when the user clicks the button
        if st.button("Run Roboflow workflow"):
            # Roboflow's example uses a file path, so we write a temporary file
            tmp_path = "tmp_phenotype_image.jpg"
            try:
                with open(tmp_path, "wb") as f:
                    f.write(image_bytes)

                try:
                    result = client.run_workflow(
                        workspace_name="rootweiler",
                        workflow_id="leafy",
                        images={"image": tmp_path},  # must match your Roboflow workflow
                        parameters={
                            "output_message": (
                                "Your model is being initialized, "
                                "try again in a few seconds."
                            )
                        },
                        use_cache=True,
                    )
                except Exception as e:
                    st.error(f"Roboflow workflow call failed: {e}")
                    result = None

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            if result is not None:
                st.success("Successfully received a response from Roboflow:")
                st.json(result)
            else:
                st.warning("No result returned from Roboflow.")
