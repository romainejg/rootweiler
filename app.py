import streamlit as st
from PIL import Image, ImageOps
import os

# Import your existing modules (logic will need web-UI wrappers later)
import leafAnalysis
import jpgExtract
import debugger
import boxing


# -----------------------
# Helper functions
# -----------------------

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def crop_and_fit_image(image_path, height):
    """
    Similar idea as your Tkinter version:
    open the image, maintain aspect ratio, and crop/fit to a specific height.
    """
    image = Image.open(image_path)
    aspect_ratio = image.width / image.height
    width = int(height * aspect_ratio)
    image = ImageOps.fit(image, (width, height), Image.LANCZOS)
    return image


def go_to_page(page_name: str):
    """Set the current page in session state."""
    st.session_state["page"] = page_name
    # Force Streamlit to re-render with the new page
    st.experimental_rerun()


# -----------------------
# Page renderers
# -----------------------

def show_main_interface():
    script_dir = get_script_dir()
    logo_path = os.path.join(script_dir, "logo.png")
    enza_path = os.path.join(script_dir, "enza.png")
    cea_path = os.path.join(script_dir, "cea.png")

    with st.container():
        st.markdown('<div class="rootrott-main">', unsafe_allow_html=True)

        # --- Hero section ---
        col_logo, col_text = st.columns([1, 2])

        with col_logo:
            if os.path.exists(logo_path):
                logo_image = Image.open(logo_path).resize((260, 260), Image.LANCZOS)
                st.image(logo_image, use_column_width=False)

        with col_text:
            st.markdown('<div class="rootrott-hero-title">RootRott.io</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="rootrott-hero-subtitle">'
                "Leaf imaging tools for controlled environment agriculture.<br>"
                "Upload your files, analyze your plants, and export results in seconds."
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")  # small spacer

        # --- Tool cards grid ---
        st.markdown("#### Tools")

        row1 = st.columns(2)
        row2 = st.columns(2)

        # Extractor
        with row1[0]:
            st.markdown(
                """
                <div class="rootrott-card">
                    <h4>🧾 Extractor</h4>
                    <p>Pull images directly from PDF, Word, PowerPoint, and Excel files for further analysis or archiving.</p>
                """,
                unsafe_allow_html=True,
            )
            with st.container():
                # Put the button in a wrapper so CSS can target it
                st.markdown('<div class="rootrott-card-button">', unsafe_allow_html=True)
                if st.button("Open Extractor", key="home_extractor"):
                    go_to_page("extractor")
                st.markdown("</div></div>", unsafe_allow_html=True)

        # Leaf Analysis
        with row1[1]:
            st.markdown(
                """
                <div class="rootrott-card">
                    <h4>🌿 Leaf Analysis</h4>
                    <p>Batch-measure leaf sizes from images and export structured Excel reports for your trials.</p>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="rootrott-card-button">', unsafe_allow_html=True)
            if st.button("Open Leaf Analysis", key="home_leaf"):
                go_to_page("leaf_analysis")
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Box Plot
        with row2[0]:
            st.markdown(
                """
                <div class="rootrott-card">
                    <h4>📊 Box Plot</h4>
                    <p>Create publication-ready box plots from your Excel datasets with custom ordering and styling.</p>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="rootrott-card-button">', unsafe_allow_html=True)
            if st.button("Open Box Plot", key="home_box"):
                go_to_page("boxing")
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Debugger
        with row2[1]:
            st.markdown(
                """
                <div class="rootrott-card">
                    <h4>🧪 Debugger</h4>
                    <p>Visualize segmentation steps and tune HSV & watershed parameters for new cultivars and lighting setups.</p>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="rootrott-card-button">', unsafe_allow_html=True)
            if st.button("Open Debugger", key="home_debug"):
                go_to_page("debugger")
            st.markdown("</div></div>", unsafe_allow_html=True)

        # --- Footer logos ---
        st.markdown('<div class="rootrott-footer">', unsafe_allow_html=True)
        fcol1, fcol2, fcol3 = st.columns([1, 1, 1])

        with fcol1:
            if os.path.exists(enza_path):
                try:
                    enza_image = crop_and_fit_image(enza_path, 60)
                    st.image(enza_image, caption="Enza Zaden")
                except Exception as e:
                    st.write(f"Error loading enza.png: {e}")

        with fcol3:
            if os.path.exists(cea_path):
                try:
                    cea_image = crop_and_fit_image(cea_path, 40)
                    st.image(cea_image, caption="CEA Seed")
                except Exception as e:
                    st.write(f"Error loading cea.png: {e}")

        with fcol2:
            st.write("")  # spacer / balance

        st.markdown("</div>", unsafe_allow_html=True)  # end footer
        st.markdown("</div>", unsafe_allow_html=True)  # end rootrott-main


def show_jpg_extract_interface():
    st.title("RootRott.io - Extractor")
    st.markdown("_JPG Extract interface_")

    # Simple file uploader as a starting point
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        st.write("Here you can call your jpgExtract logic.")
        st.code(
            "result = jpgExtract.process_image(image)\n"
            "st.write(result)",
            language="python",
        )
        # TODO: Replace with actual function(s) from jpgExtract that do not rely on Tkinter

    if st.button("Back to main menu"):
        go_to_page("main")


def show_leaf_analysis_interface():
    st.title("RootRott.io - Leaf Analysis")
    st.markdown("_Leaf Analysis interface_")

    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Leaf image", use_column_width=True)

        st.write("Here you can call your leafAnalysis logic.")
        st.code(
            "result = leafAnalysis.analyze_leaf_image(image)\n"
            "st.write(result)",
            language="python",
        )
        # TODO: Replace with actual function(s) from leafAnalysis that do not rely on Tkinter

    if st.button("Back to main menu"):
        go_to_page("main")


def show_boxing_interface():
    st.title("RootRott.io - Box Plot")
    st.markdown("_Boxing / Box Plot interface_")

    st.write("Add your box plot / boxing-related inputs here.")
    st.code(
        "# Example skeleton:\n"
        "# data = load_some_data()\n"
        "# fig = boxing.create_box_plot(data)\n"
        "# st.pyplot(fig)",
        language="python",
    )
    # TODO: Replace with real boxing module calls

    if st.button("Back to main menu"):
        go_to_page("main")


def show_debugger_interface():
    st.title("RootRott.io - Debug")
    st.markdown("_Debugger interface_")

    st.write("Add controls to run your debugger module here.")
    st.code(
        "# Example skeleton:\n"
        "# debug_info = debugger.run_debug_checks()\n"
        "# st.json(debug_info)",
        language="python",
    )
    # TODO: Replace with real debugger module calls

    if st.button("Back to main menu"):
        go_to_page("main")


# -----------------------
# Main entry point
# -----------------------

def main():
    st.set_page_config(page_title="RootRott.io", layout="wide")

    # ---- Global CSS ----
    st.markdown(
        """
        <style>
        /* Global background & font tweaks */
        .stApp {
            background: radial-gradient(circle at top left, #f5fff7, #eaf4ff 40%, #f7f7ff 80%);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Center main content a bit more and give max-width */
        .rootrott-main {
            max-width: 1100px;
            margin: 0 auto;
            padding-bottom: 3rem;
        }

        /* Hero title */
        .rootrott-hero-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            letter-spacing: 0.02em;
        }

        .rootrott-hero-subtitle {
            font-size: 1.1rem;
            color: #4b5563;
            margin-bottom: 2rem;
        }

        /* Tool cards */
        .rootrott-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.2);
            height: 100%;
        }

        .rootrott-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .rootrott-card p {
            font-size: 0.95rem;
            color: #4b5563;
            margin: 0 0 0.8rem 0;
        }

        .rootrott-card-button button {
            width: 100% !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
        }

        /* Footer logos row */
        .rootrott-footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px dashed rgba(148, 163, 184, 0.6);
            font-size: 0.8rem;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state["page"] = "main"

    current_page = st.session_state["page"]

    if current_page == "main":
        show_main_interface()
    elif current_page == "extractor":
        show_jpg_extract_interface()
    elif current_page == "leaf_analysis":
        show_leaf_analysis_interface()
    elif current_page == "boxing":
        show_boxing_interface()
    elif current_page == "debugger":
        show_debugger_interface()
    else:
        show_main_interface()


if __name__ == "__main__":
    main()


