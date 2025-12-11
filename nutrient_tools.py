# nutrient_tools.py

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Helper dataclasses / constants
# -----------------------------

@dataclass
class Fertilizer:
    name: str
    n: float   # % N
    p2o5: float  # % P₂O₅
    k2o: float  # % K₂O


# A simple 3-salt lettuce recipe basis
CALCIUM_NITRATE = Fertilizer("Calcium nitrate (15.5-0-0)", n=15.5, p2o5=0.0, k2o=0.0)
POTASSIUM_NITRATE = Fertilizer("Potassium nitrate (13-0-46)", n=13.0, p2o5=0.0, k2o=46.0)
MONO_POTASSIUM_PHOSPHATE = Fertilizer("Mono potassium phosphate (0-52-34)", n=0.0, p2o5=52.0, k2o=34.0)

LETTUCE_NPK_RATIO = (2, 1, 3)  # N:P₂O₅:K₂O target pattern


# Lettuce tissue test “typical” ranges (approximate; dry matter basis)
LETTUCE_TISSUE_RANGES = {
    "N (%)": (3.0, 4.5),
    "P (%)": (0.4, 0.7),
    "K (%)": (3.0, 6.0),
    "Ca (%)": (0.8, 1.4),
    "Mg (%)": (0.2, 0.5),
    "S (%)": (0.3, 0.6),
    "Fe (ppm)": (80, 150),
    "Mn (ppm)": (40, 200),
    "Zn (ppm)": (20, 70),
    "Cu (ppm)": (5, 15),
    "B (ppm)": (25, 60),
    "Mo (ppm)": (1, 5),
}


# ---------------------------------
# Core calculation helper functions
# ---------------------------------

def scale_stock_concentration(
    old_ratio: float,
    new_ratio: float,
    grams_per_l_old: float,
) -> float:
    """
    Scale stock solution concentration to keep the *working solution* the same
    when you change injector ratio.

    Ratios are X in "1:X".

    We want:
        Cs_old / R_old  =  Cs_new / R_new

    → Cs_new = Cs_old * R_new / R_old
    """
    if old_ratio <= 0 or new_ratio <= 0:
        return grams_per_l_old

    return grams_per_l_old * (new_ratio / old_ratio)


def grams_for_target_ppm(
    target_ppm: float,
    volume_l: float,
    nutrient_percent: float,
) -> float:
    """
    Given:
      - target_ppm: desired ppm of a nutrient
      - volume_l: solution volume in liters
      - nutrient_percent: % nutrient (e.g. 15.5 for 15.5% N)

    Return grams of fertilizer to add.
    """
    if nutrient_percent <= 0 or volume_l <= 0 or target_ppm <= 0:
        return 0.0

    # ppm = mg/L of nutrient
    total_mg_nutrient = target_ppm * volume_l
    total_g_nutrient = total_mg_nutrient / 1000.0
    fraction = nutrient_percent / 100.0

    grams_fertilizer = total_g_nutrient / fraction
    return grams_fertilizer


def approximate_ppm_from_ec(ec_ms_cm: float, factor: float = 640.0) -> float:
    """
    Rough conversion: ppm ≈ EC (mS/cm) * factor.
    factor 500–700 is common; 640 is a decent middle for mixed salts.
    """
    return ec_ms_cm * factor


def calculate_lettuce_npk_recipe(
    target_n_ppm: float,
    ratio: Tuple[float, float, float] = LETTUCE_NPK_RATIO,
) -> Optional[Dict[str, float]]:
    """
    Very simplified 3-fertilizer mixer for lettuce-style N:P₂O₅:K₂O ratios.

    We assume mixing in 1000 L of solution and solve for grams per 1000 L:
        x = Ca(NO3)2 (15.5-0-0)
        y = KNO3 (13-0-46)
        z = KH2PO4 (0-52-34)

    For 1000 L, ppm = grams * % / 100.
    """
    Rn, Rp, Rk = ratio

    if target_n_ppm <= 0 or Rn <= 0:
        return None

    # Scale P₂O₅ and K₂O targets based on N
    scale = target_n_ppm / Rn
    target_p2o5 = Rp * scale
    target_k2o = Rk * scale

    # 0.52 * z = P₂O₅_target       → z
    # 0.46 * y + 0.34 * z = K₂O    → y
    # 0.155 * x + 0.13 * y = N     → x

    z = target_p2o5 / (MONO_POTASSIUM_PHOSPHATE.p2o5 / 100.0)  # 0.52
    y = (target_k2o - (MONO_POTASSIUM_PHOSPHATE.k2o / 100.0) * z) / (POTASSIUM_NITRATE.k2o / 100.0)  # 0.46
    x = (target_n_ppm - (POTASSIUM_NITRATE.n / 100.0) * y) / (CALCIUM_NITRATE.n / 100.0)  # 0.155

    if x < 0 or y < 0 or z < 0 or not np.isfinite(x + y + z):
        return None

    return {
        CALCIUM_NITRATE.name: x,  # g/1000 L
        POTASSIUM_NITRATE.name: y,
        MONO_POTASSIUM_PHOSPHATE.name: z,
        "__targets__": {
            "N (ppm)": target_n_ppm,
            "P₂O₅ (ppm)": target_p2o5,
            "K₂O (ppm)": target_k2o,
        },
    }


def acid_required_for_alkalinity_drop(
    initial_alk_ppm_caco3: float,
    target_alk_ppm_caco3: float,
    acid_type: str = "Nitric acid 67%",
) -> Dict[str, float]:
    """
    Super-simplified water alkalinity correction.

    Assumes alkalinity is given as mg/L CaCO3.
    We estimate mL of strong acid per m³ (1000 L) to drop from initial to target.

    For nitric 67%:
        density ≈ 1.41 g/mL
        w/w ≈ 0.67
        → ~15 normal (N ≈ 15)

    For phosphoric 75%:
        density ≈ 1.58 g/mL
        w/w ≈ 0.75
        effective normality ~ 14 N (very rough, as multiprotic).

    This is intentionally approximate – just a starting point for growers.
    """
    if initial_alk_ppm_caco3 <= target_alk_ppm_caco3:
        return {
            "delta_alk": 0.0,
            "acid_ml_per_l": 0.0,
            "acid_ml_per_m3": 0.0,
        }

    delta_alk = initial_alk_ppm_caco3 - target_alk_ppm_caco3

    # mg/L CaCO3 → meq/L: divide by 50
    meq_l = delta_alk / 50.0

    if acid_type == "Nitric acid 67%":
        normality = 15.0
    else:
        normality = 14.0

    # meq/L = N * (mL/L) / 1000
    # → mL/L = meq/L * 1000 / N
    acid_ml_per_l = meq_l * 1000.0 / normality
    acid_ml_per_m3 = acid_ml_per_l * 1000.0

    return {
        "delta_alk": delta_alk,
        "acid_ml_per_l": acid_ml_per_l,
        "acid_ml_per_m3": acid_ml_per_m3,
    }


def interpret_leaf_value(value: float, low: float, high: float) -> str:
    if np.isnan(value):
        return "No value"
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Within range"


# -----------------------------
# Streamlit UI for Nutrients
# -----------------------------

class NutrientToolsUI:
    """Rootweiler Nutrient tools suite."""

    @classmethod
    def render(cls):
        st.subheader("Nutrient tools")

        tabs = st.tabs(
            [
                "EC / ppm mixer",
                "Lettuce N–P–K ratio",
                "Water quality",
                "Leaf tissue (lettuce)",
                "Stock recipe converter",  # ⬅ NEW 5th tab
            ]
        )

        with tabs[0]:
            cls._tab_ec_ppm_mixer()

        with tabs[1]:
            cls._tab_lettuce_npk()

        with tabs[2]:
            cls._tab_water_quality()

        with tabs[3]:
            cls._tab_leaf_tissue()

        with tabs[4]:
            cls._tab_stock_converter()  # ⬅ this now matches the 5th tab

    # ------------------ Tab 1: EC / ppm mixer ------------------ #

    @staticmethod
    def _tab_ec_ppm_mixer():
        st.markdown(
            """
            ### EC / ppm mixing helper

            Quick helper for **single fertilizer products** – works best when you know
            the % nutrient (e.g. %N, %K₂O) and the tank volume.

            This doesn’t try to be a full fertigation designer – just a fast way
            to answer “how many grams do I weigh out?”.
            """
        )
        # ... (rest of your existing Tab 1 code unchanged)

    # ------------------ Tab 2: Lettuce N–P–K ratio ------------------ #

    @staticmethod
    def _tab_lettuce_npk():
        # ... (unchanged)

    # ------------------ Tab 3: Water quality ------------------ #

    @staticmethod
    def _tab_water_quality():
        # ... (unchanged)

    # ------------------ Tab 4: Lettuce leaf tissue ------------------ #

    @staticmethod
    def _tab_leaf_tissue():
        # ... (unchanged)

    # ------------------ Tab 5: Stock recipe converter ------------------ #

    @staticmethod
    def _tab_stock_converter():
        st.markdown(
            """
            ### Stock recipe converter

            Use this when you **change injector ratio** (for example from 1:100 to 1:150)
            and want to keep the **same working solution** in the greenhouse.

            This tool scales the **stock solution grams per liter** for each product.
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            old_ratio = st.number_input(
                "Old injection ratio (X in 1:X)",
                min_value=1.0,
                value=100.0,
                step=1.0,
                help="For a 1:100 injector, enter 100. For 1:150, enter 150.",
            )
        with col2:
            new_ratio = st.number_input(
                "New injection ratio (X in 1:X)",
                min_value=1.0,
                value=150.0,
                step=1.0,
                help="The new injector setting you want to use.",
            )

        st.markdown("#### Stock components")

        n_products = st.number_input(
            "Number of fertilizer components in this stock tank",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

        products = []
        for i in range(n_products):
            c1, c2 = st.columns([2, 1])
            with c1:
                name = st.text_input(
                    f"Product {i+1} name",
                    value=f"Product {i+1}",
                    key=f"stock_name_{i}",
                )
            with c2:
                grams_per_l_old = st.number_input(
                    f"Old g/L (Product {i+1})",
                    min_value=0.0,
                    value=100.0,
                    step=1.0,
                    key=f"stock_gpl_{i}",
                )
            products.append((name, grams_per_l_old))

        if st.button("Convert recipe", type="primary"):
            rows = []
            factor = new_ratio / old_ratio if old_ratio > 0 else 1.0

            for name, gpl_old in products:
                gpl_new = scale_stock_concentration(old_ratio, new_ratio, gpl_old)
                rows.append(
                    {
                        "Fertilizer": name,
                        "Old g/L (stock)": round(gpl_old, 2),
                        "New g/L (stock)": round(gpl_new, 2),
                    }
                )

            st.markdown("#### Converted stock recipe")
            st.table(pd.DataFrame(rows))

            st.markdown("#### Scale factor")
            st.write(
                f"- Scale factor applied: **×{factor:.3f}**\n"
                f"- Example: 100 g/L → {100 * factor:.1f} g/L"
            )

            st.caption(
                "Logic: working concentration ≈ stock concentration ÷ injector ratio.\n"
                "We keep (stock / ratio) the same, so when you change from 1:100 to 1:150, "
                "stock concentration is multiplied by 150/100."
            )

