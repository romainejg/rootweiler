# calculators.py

import streamlit as st
import numpy as np


class DLICalculator:
    """Daily Light Integral calculator."""

    @staticmethod
    def compute_dli(ppfd: float, hours: float) -> float:
        """
        Compute DLI (mol·m⁻²·day⁻¹) from PPFD (µmol·m⁻²·s⁻¹)
        and photoperiod in hours.
        """
        if ppfd <= 0 or hours <= 0:
            return 0.0
        return ppfd * hours * 3600.0 / 1_000_000.0

    @classmethod
    def render(cls):
        st.subheader("Daily Light Integral (DLI)")

        st.markdown(
            """
            DLI describes the total amount of photosynthetically active light a crop
            receives over a day, in **mol·m⁻²·day⁻¹**.

            This calculator uses:

            - **PPFD** in µmol·m⁻²·s⁻¹  
            - **Photoperiod** in hours per day  
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            ppfd = st.number_input(
                "Average PPFD (µmol·m⁻²·s⁻¹)",
                min_value=0.0,
                max_value=5000.0,
                value=200.0,
                step=10.0,
            )

        with col2:
            hours = st.number_input(
                "Photoperiod (hours per day)",
                min_value=0.0,
                max_value=24.0,
                value=16.0,
                step=0.5,
            )

        if ppfd > 0 and hours > 0:
            dli = cls.compute_dli(ppfd, hours)
            st.markdown("### Result")
            st.write(f"**DLI: {dli:.2f} mol·m⁻²·day⁻¹**")
        else:
            st.info("Enter PPFD and photoperiod above zero to see the DLI.")


class VPDCalculator:
    """Vapor Pressure Deficit calculator."""

    @staticmethod
    def compute_vpd_kpa(temp_c: float, rh: float) -> float:
        """
        Compute VPD in kPa from temperature in °C and RH in %.
        Uses standard Tetens formula for saturation vapor pressure.
        """
        if rh <= 0 or rh > 100:
            return 0.0

        # Saturation vapor pressure (kPa) at temp_c
        es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
        # Actual vapor pressure (kPa)
        ea = es * (rh / 100.0)
        vpd = es - ea
        return max(vpd, 0.0)

    @classmethod
    def render(cls):
        st.subheader("Vapor Pressure Deficit (VPD)")

        st.markdown(
            """
            VPD describes the drying power of the air and is closely tied to crop transpiration.

            This calculator uses:

            - **Air temperature** (°C or °F)  
            - **Relative humidity** (%)  

            and returns VPD in **kPa**.
            """
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            temp_unit = st.radio(
                "Temperature unit",
                ["°C", "°F"],
                index=0,
                horizontal=True,
            )

        with col2:
            temp_input = st.number_input(
                f"Air temperature ({temp_unit})",
                value=25.0,
                step=0.5,
            )

        with col3:
            rh = st.number_input(
                "Relative Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0,
            )

        # Convert to °C if user entered °F
        if temp_unit == "°F":
            temp_c = (temp_input - 32.0) * 5.0 / 9.0
        else:
            temp_c = temp_input

        if 0 < rh <= 100:
            vpd = cls.compute_vpd_kpa(temp_c, rh)
            st.markdown("### Result")
            st.write(f"**VPD: {vpd:.2f} kPa**")
        else:
            st.info("Set relative humidity between 0 and 100% to calculate VPD.")


class GutterPlantDensityCalculator:
    """
    Calculator for plants per square meter in a gutter system.

    Inputs:
    - Gutter width
    - Gutter length
    - Spacing between gutters
    - Plants per gutter

    All length inputs can be in m, cm, ft, or in.
    """

    @staticmethod
    def to_meters(value: float, unit: str) -> float:
        """Convert a length to meters based on unit."""
        if value is None:
            return 0.0

        unit = unit.lower()
        if unit == "m":
            return value
        elif unit == "cm":
            return value / 100.0
        elif unit == "ft":
            return value * 0.3048
        elif unit == "in":
            return value * 0.0254
        else:
            return value  # fallback (treat as meters)

    @staticmethod
    def compute_plants_per_m2(
        gutter_length_m: float,
        gutter_width_m: float,
        gutter_spacing_m: float,
        plants_per_gutter: float,
    ) -> float:
        """
        Compute plants per m².

        Assumes:
        - Each gutter occupies width = gutter_width + spacing_between_gutters
        - Area per gutter = gutter_length * (gutter_width + spacing)
        """
        if gutter_length_m <= 0 or plants_per_gutter <= 0:
            return 0.0

        pitch_m = gutter_width_m + gutter_spacing_m  # total width per gutter
        if pitch_m <= 0:
            return 0.0

        area_per_gutter_m2 = gutter_length_m * pitch_m
        if area_per_gutter_m2 <= 0:
            return 0.0

        return plants_per_gutter / area_per_gutter_m2

    @classmethod
    def render(cls):
        st.subheader("Plants per m² (Gutter Layout)")

        st.markdown(
            """
            This calculator estimates **plants per square meter** for a gutter-based system.

            It uses:

            - **Gutter width**  
            - **Gutter length**  
            - **Spacing between gutters** (gap between gutters)  
            - **Plants per gutter**  

            All length inputs can be in **m, cm, ft, or in**.
            """
        )

        # Layout inputs
        col1, col2 = st.columns(2)

        with col1:
            gutter_length_val = st.number_input(
                "Gutter length",
                min_value=0.0,
                value=100.0,
                step=10.0,
                key="gutter_length_val",
            )
            gutter_length_unit = st.selectbox(
                "Length unit",
                ["m", "cm", "ft", "in"],
                index=0,
                key="gutter_length_unit",
            )

            gutter_width_val = st.number_input(
                "Gutter width",
                min_value=0.0,
                value=0.2,
                step=0.05,
                key="gutter_width_val",
            )
            gutter_width_unit = st.selectbox(
                "Width unit",
                ["m", "cm", "ft", "in"],
                index=0,
                key="gutter_width_unit",
            )

        with col2:
            spacing_val = st.number_input(
                "Spacing between gutters",
                min_value=0.0,
                value=0.5,
                step=0.05,
                key="gutter_spacing_val",
            )
            spacing_unit = st.selectbox(
                "Spacing unit",
                ["m", "cm", "ft", "in"],
                index=0,
                key="gutter_spacing_unit",
            )

            plants_per_gutter = st.number_input(
                "Plants per gutter",
                min_value=0.0,
                value=30.0,
                step=1.0,
                key="plants_per_gutter",
            )

        # Convert to meters
        length_m = cls.to_meters(gutter_length_val, gutter_length_unit)
        width_m = cls.to_meters(gutter_width_val, gutter_width_unit)
        spacing_m = cls.to_meters(spacing_val, spacing_unit)

        if plants_per_gutter > 0 and length_m > 0 and (width_m + spacing_m) > 0:
            plants_per_m2 = cls.compute_plants_per_m2(
                length_m,
                width_m,
                spacing_m,
                plants_per_gutter,
            )

            st.markdown("### Result")
            st.write(f"**Plants per m²: {plants_per_m2:.2f}**")

            # Optional: show intermediate values
            with st.expander("Show details", expanded=False):
                st.write(f"Gutter length (m): `{length_m:.3f}`")
                st.write(f"Gutter width (m): `{width_m:.3f}`")
                st.write(f"Spacing between gutters (m): `{spacing_m:.3f}`")
                pitch_m = width_m + spacing_m
                st.write(f"Total pitch (width per gutter) (m): `{pitch_m:.3f}`")
                area_per_gutter = length_m * pitch_m
                st.write(f"Area per gutter (m²): `{area_per_gutter:.3f}`")
        else:
            st.info("Enter non-zero values for length, widths, spacing and plants per gutter to see the result.")
