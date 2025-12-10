import os
import io
import json

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2

import leafAnalysis
import jpgExtract
import boxing
import debugger


# -----------------------
# Helpers
# -----------------------

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def crop_and_fit_image(image_path, height):
    image = Image.open(image_path)
    aspect_ratio = image.width / image.height
    width = int(height * aspect_ratio)
    return image.resize((width, height), Image.LANCZOS)


# -----------------------
# Global styling
# -----------------------

def inject_css():
    st.markdown(
        """
        <style>
        /* App background + typography */
        .stApp {
            background: radial-gradient(circle at top left, #f4fff7, #eaf4ff 45%, #f9f9ff 90%);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Main content width */
        .block-container {
            max-width: 1150px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: #111827;
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #f9fafb;
        }

        .rw-sidebar-title {
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #f9fafb;
            margin-bottom: 0.25rem;
        }
        .rw-sidebar-subtitle {
            font-size: 0.8rem;
            color: #9ca3af;
            margin-bottom: 1rem;
        }

        /* Card-style boxes */
        .rw-card {
            background: rgba(255, 255, 255, 0.98);
            border-radius: 18px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.2);
            margin-bottom: 1.3rem;
        }

        .rw-card h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }

        .rw-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background: rgba(16, 185, 129, 0.1);
            color: #047857;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .rw-stat-label {
            font-size: 0.8rem;
            color: #6b7280;
        }
        .rw-stat-value {
            font-size: 1.4rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------
# Sidebar navigation
# -----------------------

def sidebar_nav():
    script_dir = get_script_dir()
    logo_path = os.path.join(script_dir, "logo.png")

    with st.sidebar:
        # Logo + title
        cols = st.columns([1, 3])
        with cols[0]:
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path).resize((60, 60), Image.LANCZOS)
                st.image(logo_img, use_column_width=True)
        with cols[1]:
            st.markdown('<div class="rw-sidebar-title">ROOTWEILER</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="rw-sidebar-subtitle">Digital sidekick for greenhouse teams.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        section = st.radio(
            "Sections",
            [
                "Home",
                "Calculators",
                "Climate",
                "Phenotyping",
                "Data & Graphs",
                "Imaging",
            ],
            index=0,
        )

        st.markdown("---")
        st.caption("Tip: All processing happens in the cloud. No local installs needed.")

    return section


# -----------------------
# Section: Home
# -----------------------

def render_home():
    script_dir = get_script_dir()
    enza_path = os.path.join(script_dir, "enza.png")
    cea_path = os.path.join(script_dir, "cea.png")

    # Hero
    col1, col2 = st.columns([1.1, 1.6])
    with col1:
        st.markdown("### Rootweiler")
        st.markdown(
            "_A support tool for greenhouse growers, plant breeders, and agronomists._"
        )
        st.markdown(
            """
            Rootweiler helps you turn messy imaging and trial files into **clean, usable data**.

            - 🧾 Extract images straight from reports & slide decks  
            - 🌿 Measure leaf size and structure at scale  
            - 📊 Generate clean plots for updates & presentations  
            - 🧪 Debug segmentation settings for new cultivars and lighting setups  
            """
        )

    with col2:
        st.markdown('<div class="rw-card">', unsafe_allow_html=True)
        st.markdown('<span class="rw-pill">Greenhouse support</span>', unsafe_allow_html=True)
        st.markdown("#### Why Rootweiler?")
        st.markdown(
            """
            Greenhouse teams juggle **crop care, imaging, and reporting** under pressure.
            Rootweiler takes care of the digital grunt work so you can spend more time in the crop,
            less time in Excel.

            - Central place to try tools  
            - Simple enough for technicians  
            - Flexible enough for data scientists  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Quick stats / use cases
    st.markdown("#### Designed for your team")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="rw-card">', unsafe_allow_html=True)
        st.markdown("**Growers**", unsafe_allow_html=False)
        st.markdown(
            "- Check leaf size distributions\n- Compare blocks or recipes\n- Document issues with images"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="rw-card">', unsafe_allow_html=True)
        st.markdown("**Breeders**")
        st.markdown(
            "- Measure traits across trials\n- Export clean phenotype tables\n- Standardize imaging workflows"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="rw-card">', unsafe_allow_html=True)
        st.markdown("**Agronomists & R&D**")
        st.markdown(
            "- Explore responses vs climate\n- Build internal protocols\n- Share reproducible visuals"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Partners / footer
    st.markdown("#### Partner logos")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        if os.path.exists(enza_path):
            try:
                st.image(crop_and_fit_image(enza_path, 55), caption="Enza Zaden")
            except Exception as e:
                st.write(f"Error loading enza.png: {e}")
    with fc2:
        st.write("")
    with fc3:
        if os.path.exists(cea_path):
            try:
                st.image(crop_and_fit_image(cea_path, 40), caption="CEA Seed")
            except Exception as e:
                st.write(f"Error loading cea.png: {e}")

    st.caption("This is an early prototype. Tools and UX will evolve with real grower feedback.")


# -----------------------
# Section: Calculators (placeholder)
# -----------------------

def render_calculators():
    st.markdown("### Calculators")
    st.markdown(
        """
        This section will host quick calculators relevant to greenhouse operations, for example:

        - 🌞 Daily Light Integral (DLI)  
        - 🌡 Growing Degree Days (GDD)  
        - 📦 Yield per m² projections  

        For now, this area is a **placeholder** while we focus on imaging and phenotyping.
        """,
    )
    st.info("If you have specific calculators in mind, jot them down and we can design around that.")


# -----------------------
# Section: Climate (placeholder)
# -----------------------

def render_climate():
    st.markdown("### Climate")
    st.markdown(
        """
        In a future iteration, the Climate area could:

        - Pull climate exports (e.g. from Priva, Hoogendoorn, Argus)  
        - Aggregate them per zone / recipe / experiment  
        - Link climate patterns to phenotyping outputs  

        Right now, no tools are wired here yet — think of this as a **reserved space** for future growth.
        """
    )


# -----------------------
# Section: Imaging (Extractor)
# -----------------------

def render_imaging():
    st.markdown("### Imaging")
    st.markdown(
        """
        Use Imaging tools when you want to **pull images out of existing files**  
        like trial reports, slide decks, or Excel files.
        """
    )

    tab_extractor, = st.tabs(["Image Extractor"])

    with tab_extractor:
        render_extractor_tool()


def render_extractor_tool():
    st.subheader("🧾 Image Extractor")
    st.write("Upload a document and extract embedded images for reuse or analysis.")

    uploaded_file = st.file_uploader(
        "Upload PDF, Word, PowerPoint, or Excel file",
        type=["pdf", "docx", "pptx", "xlsx"],
    )

    if uploaded_file is None:
        st.info("Upload a file to begin.")
        return

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    data = uploaded_file.read()

    try:
        images = jpgExtract.extract_images_from_bytes(data, ext)
    except Exception as e:
        st.error(f"Error extracting images: {e}")
        return

    if not images:
        st.warning("No images big enough were found in this file.")
        return

    st.success(f"Found {len(images)} image(s).")

    for idx, (img, suggested_name) in enumerate(images):
        st.image(img, caption=suggested_name, use_column_width=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button(
            label=f"Download {suggested_name}",
            data=buf.getvalue(),
            file_name=suggested_name,
            mime="image/png",
            key=f"download_{idx}",
        )


# -----------------------
# Section: Phenotyping (Leaf Analysis + Debugger)
# -----------------------

def render_phenotyping():
    st.markdown("### Phenotyping")
    st.markdown(
        """
        Phenotyping tools help you **quantify traits from images** and  
        tune the segmentation pipeline for new crops, recipes, or cameras.
        """
    )

    tab_leaf, tab_debug = st.tabs(["Leaf Analysis", "Segmentation Debugger"])

    with tab_leaf:
        render_leaf_analysis_tool()

    with tab_debug:
        render_debugger_tool()


def render_leaf_analysis_tool():
    st.subheader("🌿 Leaf Analysis")
    st.write("Upload a leaf image to segment and measure leaf objects.")

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_file",
    )

    if uploaded_file is None:
        st.info("Upload an image to start analysis.")
        return

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not read the uploaded image.")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Original image", use_column_width=True)

    try:
        mask, measurements, px_per_cm2 = leafAnalysis.analyze_image(image_bgr)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return

    st.image(mask, caption="Segmentation mask", use_column_width=True, clamp=True)

    rows = []
    for i, (x, y, w, h) in enumerate(measurements, start=1):
        width_cm = round(w / np.sqrt(px_per_cm2), 1)
        height_cm = round(h / np.sqrt(px_per_cm2), 1)
        rows.append(
            {"Object": i, "Width (cm)": width_cm, "Height (cm)": height_cm, "x": x, "y": y}
        )

    if not rows:
        st.warning("No objects above the minimum size threshold were found.")
        return

    df = pd.DataFrame(rows)
    st.markdown("#### Measurements")
    st.dataframe(df, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Measurements")
    st.download_button(
        label="Download measurements as Excel",
        data=buf.getvalue(),
        file_name="leaf_measurements.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def render_debugger_tool():
    st.subheader("🧪 Segmentation Debugger")
    st.write("Tune HSV & watershed parameters and see each stage of the pipeline.")

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        key="debug_file",
    )

    params = debugger.SegmentationParams()

    st.markdown("##### Parameters")
    c1, c2 = st.columns(2)

    with c1:
        params.lower_hue = st.slider("Lower Hue", 0, 179, params.lower_hue)
        params.upper_hue = st.slider("Upper Hue", 0, 179, params.upper_hue)
        params.lower_saturation = st.slider(
            "Lower Saturation", 0, 255, params.lower_saturation
        )
        params.upper_saturation = st.slider(
            "Upper Saturation", 0, 255, params.upper_saturation
        )
        params.lower_value = st.slider("Lower Value", 0, 255, params.lower_value)
        params.upper_value = st.slider("Upper Value", 0, 255, params.upper_value)

    with c2:
        params.kernel_size = st.slider("Kernel Size (odd numbers)", 1, 15, params.kernel_size, step=2)
        params.morph_iterations = st.slider("Morph Iterations", 0, 10, params.morph_iterations)
        params.dilate_iterations = st.slider("Dilate Iterations", 0, 10, params.dilate_iterations)
        params.dist_transform_threshold = st.slider(
            "Dist Transform Threshold", 0.0, 1.0, params.dist_transform_threshold, step=0.05
        )

    if uploaded_file is None:
        st.info("Upload an image to view the debug plots.")
        return

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not read the uploaded image.")
        return

    fig = debugger.create_debug_figure(image_bgr, uploaded_file.name, params)
    st.pyplot(fig)

    # Save parameters as JSON
    config_str = json.dumps(params.__dict__, indent=2)
    st.download_button(
        label="Download parameter config (JSON)",
        data=config_str,
        file_name="segmentation_params.json",
        mime="application/json",
    )


# -----------------------
# Section: Data & Graphs (Box Plot)
# -----------------------

def render_data_graphs():
    st.markdown("### Data & Graphs")
    st.markdown(
        """
        Use this area when you already have **numeric data in tables** and want  
        quick visual summaries for team updates or reports.
        """
    )

    tab_box, = st.tabs(["Box Plot"])

    with tab_box:
        render_box_plot_tool()


def render_box_plot_tool():
    st.subheader("📊 Box Plot")
    st.write("Upload an Excel file and create a styled box plot.")

    uploaded_file = st.file_uploader(
        "Upload Excel file",
        type=["xlsx", "xls"],
        key="box_file",
    )

    if uploaded_file is None:
        st.info("Upload an Excel file to continue.")
        return

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        return

    if df.empty:
        st.warning("The uploaded Excel file appears to be empty.")
        return

    st.write("Preview of data:")
    st.dataframe(df.head(), use_container_width=True)

    columns = df.columns.tolist()
    x_column = st.selectbox("X-axis column", options=columns)
    y_column = st.selectbox("Y-axis column", options=columns)

    default_title = f"Box Plot of {y_column} by {x_column}"
    plot_title = st.text_input("Plot title", value=default_title)
    xlabel = st.text_input("X-axis label", value=x_column)
    ylabel = st.text_input("Y-axis label", value=y_column)

    if st.button("Generate box plot"):
        try:
            fig = boxing.generate_box_plot_figure(
                data=df,
                x_column=x_column,
                y_column=y_column,
                plot_title=plot_title,
                x_label=xlabel,
                y_label=ylabel,
            )
        except Exception as e:
            st.error(f"Error generating plot: {e}")
            return

        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            label="Download plot as PNG",
            data=buf.getvalue(),
            file_name="box_plot.png",
            mime="image/png",
        )


# -----------------------
# Main app
# -----------------------

def main():
    st.set_page_config(page_title="Rootweiler", layout="wide")
    inject_css()

    section = sidebar_nav()

    if section == "Home":
        render_home()
    elif section == "Calculators":
        render_calculators()
    elif section == "Climate":
        render_climate()
    elif section == "Phenotyping":
        render_phenotyping()
    elif section == "Data & Graphs":
        render_data_graphs()
    elif section == "Imaging":
        render_imaging()
    else:
        render_home()


if __name__ == "__main__":
    main()
