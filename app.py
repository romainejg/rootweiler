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
    st.title("RootRott.io")
    st.markdown("### Select an option below to start")

    script_dir = get_script_dir()
    logo_path = os.path.join(script_dir, "logo.png")

    # Center the logo
    if os.path.exists(logo_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((300, 300), Image.LANCZOS)
            st.image(logo_image)

    st.write("")  # spacing

    # Buttons mirroring your Tkinter main menu
    if st.button("Extractor", use_container_width=True):
        go_to_page("extractor")

    if st.button("Leaf Analysis", use_container_width=True):
        go_to_page("leaf_analysis")

    if st.button("Box Plot", use_container_width=True):
        go_to_page("boxing")

    if st.button("Debug", use_container_width=True):
        go_to_page("debugger")

    st.write("")  # spacing

    # Footer images (Enza and CEA) similar to your Tkinter footer_frame
    script_dir = get_script_dir()
    enza_path = os.path.join(script_dir, "enza.png")
    cea_path = os.path.join(script_dir, "cea.png")

    footer_cols = st.columns(2)

    with footer_cols[0]:
        if os.path.exists(enza_path):
            try:
                enza_image = crop_and_fit_image(enza_path, 150)
                st.image(enza_image)
            except Exception as e:
                st.write(f"Error loading enza.png: {e}")

    with footer_cols[1]:
        if os.path.exists(cea_path):
            try:
                cea_image = crop_and_fit_image(cea_path, 40)
                st.image(cea_image)
            except Exception as e:
                st.write(f"Error loading cea.png: {e}")


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

    # Initialize page state
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
        # Fallback
        show_main_interface()


if __name__ == "__main__":
    main()
