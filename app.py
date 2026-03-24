
import os
import io
import json

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import base64

import cv2
import climate_analyzer
import calculators
from imaging_tools import ImagingToolsUI
from nutrient_tools import NutrientToolsUI


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
        @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700;800&display=swap');

        :root {
            --rw-green: #45C96B;
            --rw-purple: #8C8BFF;
            --rw-grey: #E5E5E5;
            --rw-yellow: #FFD750;
            --rw-red: #ED695D;
            --rw-dark: #111111;
            --rw-light-bg: #F5FAFF;
        }

        /* App background + Typography */
        .stApp {
            background: #ffffff;
            font-family: "Rubik", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: "Urbane Rounded", "Rubik", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #111111;
        }

        .block-container {
            max-width: 1150px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* Sidebar styling – light, readable */
        [data-testid="stSidebar"] {
            background: #F5F7FB;
            color: #111827;
            border-right: 1px solid #E5E7EB;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #111827;
        }

        .rw-sidebar-title {
            font-size: 1.1rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #111827;
            margin-bottom: 0.15rem;
        }
        .rw-sidebar-subtitle {
            font-size: 0.8rem;
            color: #4B5563;
            margin-bottom: 0.9rem;
        }

        /* Home hero */
        .rw-hero-hello {
            font-size: 2.6rem;
            color: #4B5563;
            margin-top: 7rem;
            margin-bottom: 0.2rem;
        }
        .rw-hero-name {
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }
        .rw-hero-intro {
            font-size: 1.2rem;
            color: #4B5563;
            line-height: 1.2;
        }

        .rw-hero-icon-row {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-top: 3rem;
            margin-bottom: 1rem;
        }
        .rw-hero-icon-circle {
            width: 40px;
            height: 40px;
            border-radius: 999px;
            background: #111111;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #F9FAFB;
            font-size: 0.9rem;
            font-weight: 700;
        }
        .rw-hero-icon-text {
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6B7280;
        }

        /* Thin divider */
        .rw-divider {
            border: none;
            border-top: 1px solid #E5E7EB;
            margin: 2.5rem 0 1.5rem 0;
        }

        /* Bottom contact row */
        .rw-contact-label {
            font-size: 0.9rem;
            color: #6B7280;
            margin-bottom: 0.25rem;
        }
        .rw-contact-value {
            font-size: 1.0rem;
            font-weight: 500;
            color: #111827;
        }

        /* ---------------------------------------- */
        /* Force light theme + readable text       */
        /* ---------------------------------------- */

        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="stMarkdownContainer"] {
            color: #111111 !important;
            background-color: #ffffff !important;
        }

        /* Main text elements */
        p, span, label, li, div {
            color: #111111;
        }

        /* Sidebar text (including items in mobile menu) */
        [data-testid="stSidebar"] * {
            color: #111827 !important;
        }

        /* Links */
        a {
            color: #2563EB;
        }
        a:hover {
            color: #1D4ED8;
        }

        /* Mobile tweaks */
        @media (max-width: 768px) {
            .rw-hero-hello {
                font-size: 1.6rem;
                margin-top: 4rem;
            }
            .rw-hero-name {
                font-size: 2.2rem;
            }
            .rw-hero-intro {
                font-size: 1.0rem;
                line-height: 1.4;
            }
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
    logo_path = os.path.join(script_dir, "assets", "logo.png")

    with st.sidebar:

        # --- Large centered logo on top ---
        if os.path.exists(logo_path):
            st.markdown(
                """
                <div style="display:flex; justify-content:center; margin-top:10px; margin-bottom:0px;">
                    <img src="data:image/png;base64,{}" style="width:180px; height:auto;" />
                </div>
                """.format(
                    base64.b64encode(open(logo_path, "rb").read()).decode()
                ),
                unsafe_allow_html=True,
            )

        # --- Title + Subtitle ---
        st.markdown(
            '<div class="rw-sidebar-title" style="text-align:center; margin-top:10px;">ROOTWEILER</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="rw-sidebar-subtitle" style="text-align:center;">Digital support for greenhouse teams.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # --- Navigation radio buttons ---
        section = st.radio(
            "Sections",
            [
                "Home",
                "DLI Calculator",
                "VPD/HD Calculator",
                "MGS tools",
                "DWC tools",
                "Nutrient tools",
                "Climate Analyzer",
                "Unit Converter",
                "PDF and Imaging",
            ],
            index=0,
        )

        st.markdown("---")
        st.caption("Tip: All processing happens in the cloud. No local installs needed.")

    return section

# -----------------------
# Section: Home (minimal hero layout)
# -----------------------

def render_home():
    script_dir = get_script_dir()
    lettuce1 = os.path.join(script_dir, "assets", "lettuce1.jpg")
    logo_path = os.path.join(script_dir, "assets", "logo.png")

    # Small icon row at very top of page
    st.markdown(
        '<div class="rw-hero-icon-row">'
        '<div class="rw-hero-icon-circle">R</div>'
        '<div class="rw-hero-icon-text">Rootweiler • greenhouse support</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Hero: image left, text right
    col_img, col_text = st.columns([1, 1.4])

    with col_img:
        if os.path.exists(lettuce1):
            hero_img = Image.open(lettuce1)
        elif os.path.exists(logo_path):
            hero_img = Image.open(logo_path)
        else:
            hero_img = None

        if hero_img is not None:
            st.image(hero_img, use_column_width=True)
        else:
            st.write("")

    with col_text:
        st.markdown('<div class="rw-hero-hello">Hoi!</div>', unsafe_allow_html=True)
        st.markdown('<div class="rw-hero-name">Welcome to Rootweiler.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="rw-hero-intro">'
            "A non-profit app for greenhouse leafy leaders who live their days between crops, climate, and endless data. "
            "Rootweiler is a hub of digital tools to assist with many aspects of the controlled environment agriculture industry."
            "</div>",
            unsafe_allow_html=True,
        )

    # Thin divider
    st.markdown('<hr class="rw-divider">', unsafe_allow_html=True)

    # Bottom row: logo – contact – logo, all centered with similar heights
    col_center,_,_ = st.columns([1,1,1])

    with col_center:
        st.markdown('<div class="rw-contact-label">Contact</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="rw-contact-value">j.gray@enzazaden.com</div>',
            unsafe_allow_html=True,
        )



# -----------------------
# Section: DLI Calculator
# -----------------------

def render_dli_calculator():
    st.markdown("## DLI Calculator")
    calculators.DLICalculator.render()


# -----------------------
# Section: VPD/HD Calculator
# -----------------------

def render_vpd_hd_calculator():
    st.markdown("## VPD/HD Calculator")

    tabs = st.tabs(["Vapor Pressure Deficit (VPD)", "Humidity Deficit (HD)"])

    with tabs[0]:
        calculators.VPDCalculator.render()

    with tabs[1]:
        calculators.HumidityDeficitCalculator.render()


# -----------------------
# Section: MGS tools
# -----------------------

def render_mgs_tools():
    st.markdown("## MGS tools")
    calculators.MGSLettuceCalculator.render()


# -----------------------
# Section: DWC tools
# -----------------------

def render_dwc_tools():
    st.markdown("## DWC tools")
    st.info("DWC tools are coming soon. Stay tuned!")


# -----------------------
# Section: Nutrient tools
# -----------------------

def render_nutrient_tools():
    NutrientToolsUI.render()


# -----------------------
# Section: Climate Analyzer
# -----------------------

def render_climate_analyzer():
    st.markdown("## Climate Analyzer")
    climate_analyzer.ClimateAnalyzerUI.render()


# -----------------------
# Section: Unit Converter
# -----------------------

def render_unit_converter():
    st.markdown("## Unit Converter")
    calculators.UnitConverterCalculator.render()


# -----------------------
# Section: PDF and Imaging
# -----------------------

def render_pdf_imaging():
    ImagingToolsUI.render()


# -----------------------
# Main app
# -----------------------

def main():
    st.set_page_config(page_title="Rootweiler", layout="wide")
    inject_css()

    section = sidebar_nav()

    if section == "Home":
        render_home()
    elif section == "DLI Calculator":
        render_dli_calculator()
    elif section == "VPD/HD Calculator":
        render_vpd_hd_calculator()
    elif section == "MGS tools":
        render_mgs_tools()
    elif section == "DWC tools":
        render_dwc_tools()
    elif section == "Nutrient tools":
        render_nutrient_tools()
    elif section == "Climate Analyzer":
        render_climate_analyzer()
    elif section == "Unit Converter":
        render_unit_converter()
    elif section == "PDF and Imaging":
        render_pdf_imaging()
    else:
        render_home()


if __name__ == "__main__":
    main()

