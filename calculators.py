import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import canopy_closure as cc


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

            This calculator accepts light input as either **PPFD** (µmol·m⁻²·s⁻¹) or
            **PAR** (W·m⁻²). PAR in W·m⁻² is converted to PPFD using the standard
            approximation: *PPFD ≈ PAR × 4.57*.
            """
        )

        col_mode, _ = st.columns([1, 2])
        with col_mode:
            light_input_mode = st.radio(
                "Light input type",
                ["PPFD (µmol·m⁻²·s⁻¹)", "PAR (W·m⁻²)"],
                index=0,
                horizontal=True,
                key="dli_light_mode",
            )

        col1, col2 = st.columns(2)

        with col1:
            if light_input_mode == "PPFD (µmol·m⁻²·s⁻¹)":
                light_value = st.number_input(
                    "Average PPFD (µmol·m⁻²·s⁻¹)",
                    min_value=0.0,
                    max_value=5000.0,
                    value=200.0,
                    step=10.0,
                    key="dli_ppfd",
                )
                ppfd = light_value
            else:
                light_value = st.number_input(
                    "Average PAR (W·m⁻²)",
                    min_value=0.0,
                    max_value=1200.0,
                    value=44.0,
                    step=1.0,
                    key="dli_par",
                )
                ppfd = light_value * 4.57  # PAR (W/m²) → PPFD (µmol/m²/s)

        with col2:
            hours = st.number_input(
                "Photoperiod (hours per day)",
                min_value=0.0,
                max_value=24.0,
                value=16.0,
                step=0.5,
                key="dli_hours",
            )

        if ppfd > 0 and hours > 0:
            dli = cls.compute_dli(ppfd, hours)
            st.markdown("### Result")
            if light_input_mode == "PAR (W·m⁻²)":
                st.write(f"Converted PPFD: **{ppfd:.1f} µmol·m⁻²·s⁻¹**")
            st.write(f"**DLI: {dli:.2f} mol·m⁻²·day⁻¹**")
        else:
            st.info("Enter a light value and photoperiod above zero to see the DLI.")


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


class HumidityDeficitCalculator:
    """Humidity Deficit (HD) calculator for greenhouse management."""

    @staticmethod
    def compute_saturation_absolute_humidity(temp_c: float) -> float:
        """
        Compute Saturation Absolute Humidity (AHs) in g/m³ at the given temperature.
        Uses the Tetens equation for saturation vapor pressure, then converts via
        the ideal gas law: AHs = (es_Pa × Mw) / (R × T_K)
        where Mw = 18.016 g/mol and R = 8.314 J/(mol·K).
        Reference: At 23 °C, AHs ≈ 20.56 g/m³.
        """
        es_kpa = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
        es_pa = es_kpa * 1000.0
        t_k = temp_c + 273.15
        ahs = (es_pa * 18.016) / (8.314 * t_k)
        return max(ahs, 0.0)

    @classmethod
    def compute_hd(cls, temp_c: float, rh: float) -> float:
        """
        Compute Humidity Deficit in g/m³.
        HD = AHs × (1 − RH/100)
        """
        if rh < 0 or rh > 100:
            return 0.0
        ahs = cls.compute_saturation_absolute_humidity(temp_c)
        return ahs * (1.0 - rh / 100.0)

    @staticmethod
    def interpret_hd_lettuce(hd: float) -> str:
        """Return a brief interpretation of HD for lettuce cultivation."""
        if hd < 3:
            return "Too Humid – risk of tip burn and disease; increase ventilation."
        elif hd <= 7:
            return "Optimal – ideal range for lettuce growth."
        else:
            return "Too Dry – excessive transpiration stress; raise humidity."

    @classmethod
    def render(cls):
        st.subheader("Humidity Deficit (HD)")

        st.markdown(
            """
            Humidity Deficit describes the amount of water vapour the air can still
            absorb before reaching saturation. It is a key driver of plant transpiration
            in greenhouses.

            This calculator uses:

            - **Air temperature** (°C or °F)
            - **Relative humidity** (%)

            and returns HD in **g/m³**.

            **Formula:**
            1. Saturation Absolute Humidity *AHs* (g/m³) via the Tetens equation
            2. *HD* = *AHs* × (1 − RH / 100)
            """
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            temp_unit = st.radio(
                "Temperature unit",
                ["°C", "°F"],
                index=0,
                horizontal=True,
                key="hd_temp_unit",
            )

        with col2:
            temp_input = st.number_input(
                f"Air temperature ({temp_unit})",
                value=23.0,
                step=0.5,
                key="hd_temp_input",
            )

        with col3:
            rh = st.number_input(
                "Relative Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0,
                key="hd_rh",
            )

        if temp_unit == "°F":
            temp_c = (temp_input - 32.0) * 5.0 / 9.0
        else:
            temp_c = temp_input

        if 0 <= rh <= 100:
            ahs = cls.compute_saturation_absolute_humidity(temp_c)
            hd = cls.compute_hd(temp_c, rh)
            interpretation = cls.interpret_hd_lettuce(hd)

            st.markdown("### Result")
            st.write(f"**Saturation Absolute Humidity (AHs): {ahs:.2f} g/m³**")
            st.write(f"**Humidity Deficit (HD): {hd:.2f} g/m³**")
            st.markdown(f"**Lettuce interpretation:** {interpretation}")
        else:
            st.info("Set relative humidity between 0 and 100% to calculate HD.")


class VPDHDCalculator:
    """Combined Vapor Pressure Deficit + Humidity Deficit calculator."""

    @classmethod
    def render(cls):
        st.subheader("VPD & HD Calculator")

        st.markdown(
            """
            Both **Vapor Pressure Deficit (VPD)** and **Humidity Deficit (HD)** are
            calculated from the same two inputs: air temperature and relative humidity.

            - **VPD** (kPa) — describes the drying power of the air; key driver of
              crop transpiration.
            - **HD** (g/m³) — the amount of water vapour the air can still absorb
              before saturation; directly linked to lettuce transpiration rates.
            """
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            temp_unit = st.radio(
                "Temperature unit",
                ["°C", "°F"],
                index=0,
                horizontal=True,
                key="vpdhd_temp_unit",
            )

        with col2:
            temp_input = st.number_input(
                f"Air temperature ({temp_unit})",
                value=23.0,
                step=0.5,
                key="vpdhd_temp_input",
            )

        with col3:
            rh = st.number_input(
                "Relative Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0,
                key="vpdhd_rh",
            )

        if temp_unit == "°F":
            temp_c = (temp_input - 32.0) * 5.0 / 9.0
        else:
            temp_c = temp_input

        if 0 <= rh <= 100:
            vpd = VPDCalculator.compute_vpd_kpa(temp_c, rh)
            hd = HumidityDeficitCalculator.compute_hd(temp_c, rh)
            ahs = HumidityDeficitCalculator.compute_saturation_absolute_humidity(temp_c)
            hd_interp = HumidityDeficitCalculator.interpret_hd_lettuce(hd)

            st.markdown("### Results")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric("Vapor Pressure Deficit (VPD)", f"{vpd:.2f} kPa")

            with res_col2:
                st.metric("Humidity Deficit (HD)", f"{hd:.2f} g/m³")

            st.markdown(f"**Lettuce HD interpretation:** {hd_interp}")

            with st.expander("Show intermediate values", expanded=False):
                st.write(f"Temperature (°C): `{temp_c:.2f}`")
                st.write(f"Relative Humidity: `{rh:.1f}%`")
                st.write(f"Saturation Absolute Humidity (AHs): `{ahs:.2f} g/m³`")
                es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
                st.write(f"Saturation Vapour Pressure (es): `{es:.4f} kPa`")
        else:
            st.info("Set relative humidity between 0 and 100% to calculate VPD and HD.")


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


class CanopyClosureCalculator:
    """
    Canopy Closure Estimator based on calibrated density anchor + normalized PDI 
    with density buffering and optional demerit modifiers.
    
    Model features:
    - Normalized PDI from environmental averages (T, DLI, CO2, VPD)
    - Calibrated anchor points at standard environment
    - Density buffering (high density less sensitive to environment)
    - Optional variety/climate/stress modifiers
    - Multi-stage transplant support
    """
    
    # Variety speed multipliers (slower varieties take more days)
    VARIETY_MULTIPLIERS = {
        "Green (default)": 1.00,
        "Red variety": 0.80,
        "Romaine/Cos": 0.95
    }
    
    # Climate and stress multipliers
    CLIMATE_INCONSISTENT_MULT = 0.90
    STRESS_MULT = 0.85

    @classmethod
    def render(cls):
        st.subheader("Canopy Closure Estimator")
        
        st.markdown(
            """
            **⚠️ This tool is still under construction** but provides estimates based on:
            
            - **Plant density** (plants/m²)
            - **Environmental averages**: Temperature, DLI, CO2, VPD
            - **Target closure %** (default 90%)
            - **Optional modifiers**: variety, climate consistency, stress periods
            - **Optional transplant stages**: multiple density stages if needed
            
            The model uses a calibrated anchor approach with density buffering, meaning high 
            densities (≥600 plants/m²) are less sensitive to environmental changes, while low 
            densities show more environmental sensitivity.
            """
        )
        
        # Create tabs for single-stage vs multi-stage
        mode_tabs = st.tabs(["Single Stage", "Multi-Stage (Transplants)"])
        
        # ===== SINGLE STAGE MODE =====
        with mode_tabs[0]:
            st.markdown("### Environmental Inputs")
            
            col1, col2 = st.columns(2)
            
            with col1:
                density = st.number_input(
                    "Density (plants/m²)",
                    min_value=1.0,
                    max_value=2000.0,
                    value=600.0,
                    step=10.0,
                    key="single_density"
                )
                
                temp = st.number_input(
                    "Avg Temperature (°C)",
                    min_value=-10.0,
                    max_value=45.0,
                    value=23.0,
                    step=0.5,
                    key="single_temp"
                )
                
                dli = st.number_input(
                    "Avg DLI (mol m⁻² d⁻¹)",
                    min_value=0.1,
                    max_value=60.0,
                    value=18.0,
                    step=0.5,
                    key="single_dli"
                )
            
            with col2:
                co2 = st.number_input(
                    "Avg CO2 (ppm)",
                    min_value=300.0,
                    max_value=2000.0,
                    value=800.0,
                    step=50.0,
                    key="single_co2"
                )
                
                vpd = st.number_input(
                    "Avg VPD (kPa)",
                    min_value=0.1,
                    max_value=3.0,
                    value=0.8,
                    step=0.1,
                    key="single_vpd"
                )
                
                target_pct = st.number_input(
                    "Target Closure (%)",
                    min_value=10.0,
                    max_value=99.0,
                    value=90.0,
                    step=5.0,
                    key="single_target"
                )
            
            st.markdown("### Modifiers (Optional)")
            
            mod1, mod2, mod3 = st.columns(3)
            
            with mod1:
                variety = st.selectbox(
                    "Variety",
                    list(cls.VARIETY_MULTIPLIERS.keys()),
                    index=0,
                    key="single_variety"
                )
            
            with mod2:
                inconsistent_climate = st.checkbox(
                    "Inconsistent climate",
                    value=False,
                    help="Check if climate conditions are inconsistent (10% slower)",
                    key="single_climate"
                )
            
            with mod3:
                known_stress = st.checkbox(
                    "Known stress period",
                    value=False,
                    help="Check if there's a known stress period (15% slower)",
                    key="single_stress"
                )
            
            # Compute speed multiplier
            variety_mult = cls.VARIETY_MULTIPLIERS[variety]
            climate_mult = cls.CLIMATE_INCONSISTENT_MULT if inconsistent_climate else 1.00
            stress_mult = cls.STRESS_MULT if known_stress else 1.00
            speed_multiplier = variety_mult * climate_mult * stress_mult
            
            # Compute results
            result = cc.canopy_days_to_target(
                D=density,
                T=temp,
                DLI=dli,
                CO2=co2,
                VPD=vpd,
                target_pct=target_pct,
                speed_mult=speed_multiplier
            )
            
            # Display results
            st.markdown("### Results")
            
            res1, res2, res3 = st.columns(3)
            
            with res1:
                st.metric("PDI (raw)", f"{result['pdi_raw']:.3f}")
                st.metric("PDI (clipped)", f"{result['pdi']:.3f}")
            
            with res2:
                st.metric("t90 anchor (days)", f"{result['t90_anchor']:.1f}")
                st.metric("Alpha (buffering)", f"{result['alpha']:.3f}")
            
            with res3:
                st.metric("Speed multiplier", f"{result['speed_multiplier']:.2f}")
                st.metric("Effective t90 (days)", f"{result['effective_t90']:.1f}")
            
            st.markdown("---")
            
            st.markdown(f"### **Days to {target_pct:.0f}% Closure: {result['t_target']:.1f} days**")
            
            # Generate and plot curve
            curve_data = cc.closure_curve(
                t90=result['effective_t90'],
                target_pct=target_pct
            )
            
            fig = go.Figure()
            
            # Main curve
            fig.add_trace(go.Scatter(
                x=curve_data['day'],
                y=curve_data['closure_pct'],
                mode='lines',
                name='Closure %',
                line=dict(color='#45C96B', width=3)
            ))
            
            # Target line
            fig.add_hline(
                y=target_pct,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Target: {target_pct:.0f}%",
                annotation_position="right"
            )
            
            # Target point
            fig.add_trace(go.Scatter(
                x=[result['t_target']],
                y=[target_pct],
                mode='markers',
                name=f'{target_pct:.0f}% closure',
                marker=dict(size=10, color='#ED695D')
            ))
            
            # 90% reference if different from target
            if abs(target_pct - 90.0) > 1.0:
                fig.add_trace(go.Scatter(
                    x=[result['effective_t90']],
                    y=[90.0],
                    mode='markers',
                    name='90% closure (ref)',
                    marker=dict(size=8, color='#8C8BFF', symbol='diamond')
                ))
            
            fig.update_layout(
                title=f"Canopy Closure Curve (Density: {density:.0f} plants/m²)",
                xaxis_title="Days",
                yaxis_title="Canopy Closure (%)",
                hovermode='x unified',
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show intermediate calculations
            with st.expander("Show intermediate calculations", expanded=False):
                st.markdown(f"""
                **Step 1: Normalized PDI**
                - Standard env: T={cc.T_REF}°C, DLI={cc.DLI_REF}, CO2={cc.CO2_REF}ppm, VPD={cc.VPD_REF}kPa
                - Your env: T={temp}°C, DLI={dli}, CO2={co2}ppm, VPD={vpd}kPa
                - PDI raw: {result['pdi_raw']:.3f}
                - PDI clipped [0.70, 1.30]: {result['pdi']:.3f}
                
                **Step 2: Density anchor at PDI=1**
                - Density: {density:.0f} plants/m²
                - t90_anchor: {result['t90_anchor']:.1f} days (days to 90% at standard env)
                
                **Step 3: Density buffering**
                - Alpha: {result['alpha']:.3f} (sensitivity to environment)
                - Higher density → lower alpha → less sensitive
                
                **Step 4: Environment adjustment**
                - t90 = t90_anchor × PDI^(-alpha)
                - t90 = {result['t90_anchor']:.1f} × {result['pdi']:.3f}^(-{result['alpha']:.3f})
                - t90 = {result['t90']:.1f} days
                
                **Step 5: Modifiers**
                - Variety: {variety} → {variety_mult:.2f}
                - Climate: {"Inconsistent" if inconsistent_climate else "Consistent"} → {climate_mult:.2f}
                - Stress: {"Yes" if known_stress else "No"} → {stress_mult:.2f}
                - Combined: {speed_multiplier:.2f}
                - Effective t90 = {result['t90']:.1f} / {speed_multiplier:.2f} = {result['effective_t90']:.1f} days
                
                **Step 6: Target closure**
                - Growth constant k = ln(10) / t90 = {result['k']:.4f}
                - Days to {target_pct:.0f}%: {result['t_target']:.1f} days
                """)
        
        # ===== MULTI-STAGE MODE =====
        with mode_tabs[1]:
            st.markdown("### Multi-Stage Transplant Mode")
            st.markdown(
                """
                Use this mode when you have multiple transplant stages at different densities.
                For example: plug stage → first transplant → final spacing.
                """
            )
            
            # Environment inputs (shared for all stages)
            st.markdown("#### Environmental Conditions (applies to all stages)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                temp_multi = st.number_input(
                    "Avg Temperature (°C)",
                    min_value=-10.0,
                    max_value=45.0,
                    value=23.0,
                    step=0.5,
                    key="multi_temp"
                )
                
                dli_multi = st.number_input(
                    "Avg DLI (mol m⁻² d⁻¹)",
                    min_value=0.1,
                    max_value=60.0,
                    value=18.0,
                    step=0.5,
                    key="multi_dli"
                )
            
            with col2:
                co2_multi = st.number_input(
                    "Avg CO2 (ppm)",
                    min_value=300.0,
                    max_value=2000.0,
                    value=800.0,
                    step=50.0,
                    key="multi_co2"
                )
                
                vpd_multi = st.number_input(
                    "Avg VPD (kPa)",
                    min_value=0.1,
                    max_value=3.0,
                    value=0.8,
                    step=0.1,
                    key="multi_vpd"
                )
            
            target_pct_multi = st.number_input(
                "Target Closure (%)",
                min_value=10.0,
                max_value=99.0,
                value=90.0,
                step=5.0,
                key="multi_target"
            )
            
            # Modifiers
            st.markdown("#### Modifiers")
            
            mod1, mod2, mod3 = st.columns(3)
            
            with mod1:
                variety_multi = st.selectbox(
                    "Variety",
                    list(cls.VARIETY_MULTIPLIERS.keys()),
                    index=0,
                    key="multi_variety"
                )
            
            with mod2:
                inconsistent_climate_multi = st.checkbox(
                    "Inconsistent climate",
                    value=False,
                    key="multi_climate"
                )
            
            with mod3:
                known_stress_multi = st.checkbox(
                    "Known stress period",
                    value=False,
                    key="multi_stress"
                )
            
            # Compute speed multiplier
            variety_mult_multi = cls.VARIETY_MULTIPLIERS[variety_multi]
            climate_mult_multi = cls.CLIMATE_INCONSISTENT_MULT if inconsistent_climate_multi else 1.00
            stress_mult_multi = cls.STRESS_MULT if known_stress_multi else 1.00
            speed_multiplier_multi = variety_mult_multi * climate_mult_multi * stress_mult_multi
            
            # Stage inputs
            st.markdown("#### Transplant Stages")
            st.markdown("Define each stage with density and days. Last stage goes until target closure.")
            
            num_stages = st.number_input(
                "Number of stages",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                key="num_stages"
            )
            
            stages = []
            for i in range(num_stages):
                st.markdown(f"**Stage {i+1}**")
                c1, c2 = st.columns(2)
                
                with c1:
                    stage_density = st.number_input(
                        f"Density (plants/m²)",
                        min_value=1.0,
                        max_value=2000.0,
                        value=1200.0 if i == 0 else (600.0 if i == 1 else 350.0),
                        step=10.0,
                        key=f"stage_{i}_density"
                    )
                
                with c2:
                    if i < num_stages - 1:
                        stage_days = st.number_input(
                            f"Days at this stage",
                            min_value=0.0,
                            max_value=100.0,
                            value=7.0 if i == 0 else 14.0,
                            step=1.0,
                            key=f"stage_{i}_days"
                        )
                    else:
                        st.markdown("**(Final stage: until target)**")
                        stage_days = None
                
                stages.append((stage_density, stage_days))
            
            # Compute multi-stage results
            multi_result = cc.canopy_days_multistage(
                stages=stages,
                T=temp_multi,
                DLI=dli_multi,
                CO2=co2_multi,
                VPD=vpd_multi,
                target_pct=target_pct_multi,
                speed_mult=speed_multiplier_multi
            )
            
            # Display results
            st.markdown("### Results")
            
            st.markdown(f"### **Total Days to {target_pct_multi:.0f}% Closure: {multi_result['total_days']:.1f} days**")
            st.markdown(f"**Final Closure: {multi_result['final_closure_pct']:.1f}%**")
            
            st.markdown("#### Stage Breakdown")
            
            # Create table of stages
            stage_table = []
            for stage_info in multi_result['stages_info']:
                stage_table.append({
                    'Stage': stage_info['stage'],
                    'Density (plants/m²)': f"{stage_info['density']:.0f}",
                    'Days': f"{stage_info['days']:.1f}",
                    'Closure Start (%)': f"{stage_info['closure_start']:.1f}",
                    'Closure End (%)': f"{stage_info['closure_end']:.1f}",
                    'Cumulative Days': f"{stage_info['cumulative_days']:.1f}"
                })
            
            st.dataframe(pd.DataFrame(stage_table), use_container_width=True)
            
            # Plot multi-stage curve
            fig_multi = go.Figure()
            
            for stage_info in multi_result['stages_info']:
                days_start = stage_info['cumulative_days'] - stage_info['days']
                days_end = stage_info['cumulative_days']
                
                # Generate curve for this stage
                k = stage_info['k']
                days_range = np.linspace(0, stage_info['days'], 50)
                closure_base = 100 * (1 - np.exp(-k * days_range))
                
                # Adjust for starting point
                closure_adjusted = stage_info['closure_start'] + \
                                 closure_base * (1 - stage_info['closure_start']/100.0)
                days_absolute = days_start + days_range
                
                fig_multi.add_trace(go.Scatter(
                    x=days_absolute,
                    y=closure_adjusted,
                    mode='lines',
                    name=f'Stage {stage_info["stage"]} (D={stage_info["density"]:.0f})',
                    line=dict(width=3)
                ))
            
            # Target line
            fig_multi.add_hline(
                y=target_pct_multi,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Target: {target_pct_multi:.0f}%",
                annotation_position="right"
            )
            
            fig_multi.update_layout(
                title="Multi-Stage Canopy Closure",
                xaxis_title="Days",
                yaxis_title="Canopy Closure (%)",
                hovermode='x unified',
                height=450
            )
            
            st.plotly_chart(fig_multi, use_container_width=True)


class UnitConverterCalculator:
    """
    Generic unit converter for common greenhouse / agronomy-relevant quantities:
    - Length
    - Area
    - Volume
    - Mass
    - Temperature
    """

    # Base-unit scaling factors for multiplicative conversions
    LENGTH_FACTORS = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "ft": 0.3048,
        "in": 0.0254,
    }

    AREA_FACTORS = {
        "m²": 1.0,
        "ft²": 0.09290304,
        "ha": 10_000.0,
        "acre": 4046.8564224,
    }

    VOLUME_FACTORS = {
        "L": 0.001,           # 1 L = 0.001 m³
        "m³": 1.0,
        "gal (US)": 0.00378541,
        "ft³": 0.0283168,
    }

    MASS_FACTORS = {
        "g": 0.001,    # 1 g = 0.001 kg
        "kg": 1.0,
        "lb": 0.45359237,
        "oz": 0.0283495231,
    }

    @staticmethod
    def convert_length(value: float, from_unit: str, to_unit: str) -> float:
        base = value * UnitConverterCalculator.LENGTH_FACTORS[from_unit]
        return base / UnitConverterCalculator.LENGTH_FACTORS[to_unit]

    @staticmethod
    def convert_area(value: float, from_unit: str, to_unit: str) -> float:
        base = value * UnitConverterCalculator.AREA_FACTORS[from_unit]
        return base / UnitConverterCalculator.AREA_FACTORS[to_unit]

    @staticmethod
    def convert_volume(value: float, from_unit: str, to_unit: str) -> float:
        base = value * UnitConverterCalculator.VOLUME_FACTORS[from_unit]
        return base / UnitConverterCalculator.VOLUME_FACTORS[to_unit]

    @staticmethod
    def convert_mass(value: float, from_unit: str, to_unit: str) -> float:
        base = value * UnitConverterCalculator.MASS_FACTORS[from_unit]
        return base / UnitConverterCalculator.MASS_FACTORS[to_unit]

    @staticmethod
    def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
        # Convert to °C first
        if from_unit == "°C":
            c = value
        elif from_unit == "°F":
            c = (value - 32.0) * 5.0 / 9.0
        elif from_unit == "K":
            c = value - 273.15
        else:
            c = value

        # Convert from °C to target
        if to_unit == "°C":
            return c
        elif to_unit == "°F":
            return c * 9.0 / 5.0 + 32.0
        elif to_unit == "K":
            return c + 273.15
        else:
            return c

    @classmethod
    def render(cls):
        st.subheader("Unit Converter")

        st.markdown(
            """
            A small helper for converting common greenhouse and agronomy units.

            Choose a quantity type, enter a value, and select the units to convert from and to.
            """
        )

        quantity_type = st.selectbox(
            "Quantity type",
            ["Length", "Area", "Volume", "Mass", "Temperature"],
            index=0,
        )

        if quantity_type == "Length":
            units = list(cls.LENGTH_FACTORS.keys())
        elif quantity_type == "Area":
            units = list(cls.AREA_FACTORS.keys())
        elif quantity_type == "Volume":
            units = list(cls.VOLUME_FACTORS.keys())
        elif quantity_type == "Mass":
            units = list(cls.MASS_FACTORS.keys())
        else:
            units = ["°C", "°F", "K"]

        col1, col2, col3 = st.columns([1, 1, 1.2])

        with col1:
            value = st.number_input("Value", value=1.0, step=0.1)

        with col2:
            from_unit = st.selectbox("From", units, index=0)

        with col3:
            to_unit = st.selectbox("To", units, index=min(1, len(units) - 1))

        result = None
        if quantity_type == "Length":
            result = cls.convert_length(value, from_unit, to_unit)
        elif quantity_type == "Area":
            result = cls.convert_area(value, from_unit, to_unit)
        elif quantity_type == "Volume":
            result = cls.convert_volume(value, from_unit, to_unit)
        elif quantity_type == "Mass":
            result = cls.convert_mass(value, from_unit, to_unit)
        elif quantity_type == "Temperature":
            result = cls.convert_temperature(value, from_unit, to_unit)

        if result is not None:
            st.markdown("### Result")
            st.write(f"**{value} {from_unit} = {result:.4g} {to_unit}**")


class MGSLettuceCalculator:
    """
    MGS (Mobile Gutter System) lettuce density calculator.

    Calculates per-zone and overall average seeds/plants per m² for MGS systems.

    Fixed system inputs (entered once):
    - Gutter length  (m, cm, ft, or in)
    - Gutter width   (m, cm, ft, or in)
    - Seeds per gutter

    Per-zone inputs (repeated for each zone):
    - Zone name / label
    - Zone length    (m, cm, ft, or in)
    - Zone spacing   (gap between gutters, m, cm, ft, or in)

    Formulae:
    - Footprint area per gutter = gutter_length × (gutter_width + zone_spacing)
    - Seeds per m² (per zone)   = seeds_per_gutter / footprint_area_per_gutter
    - Gutters per zone          = zone_length / (gutter_width + zone_spacing)
    - Total seeds per zone      = seeds_per_gutter × gutters_per_zone
    - Zone area                 = gutter_length × zone_length
    - Overall avg seeds/m²      = Σ(total_seeds) / Σ(zone_area)
    """

    _UNITS = ["m", "cm", "ft", "in"]
    _SQM_TO_SQFT = 10.7639  # 1 m² = 10.7639 ft²
    _CFG_KEY = "mgs_density_cfg"

    @classmethod
    def _cfg(cls) -> dict:
        """Return the persistent session-state config dict for this calculator."""
        if cls._CFG_KEY not in st.session_state:
            st.session_state[cls._CFG_KEY] = {}
        return st.session_state[cls._CFG_KEY]

    @staticmethod
    def to_meters(value: float, unit: str) -> float:
        """Convert a length value to metres."""
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
        return value  # fallback: treat as metres

    @staticmethod
    def compute_seeds_per_m2(
        gutter_length_m: float,
        gutter_width_m: float,
        spacing_m: float,
        seeds_per_gutter: float,
    ) -> float:
        """
        Compute seeds/m² from gutter geometry and seeding rate.

        footprint_area = gutter_length × (gutter_width + spacing)
        seeds_per_m²   = seeds_per_gutter / footprint_area
        """
        if gutter_length_m <= 0 or seeds_per_gutter <= 0:
            return 0.0
        pitch_m = gutter_width_m + spacing_m
        if pitch_m <= 0:
            return 0.0
        area_m2 = gutter_length_m * pitch_m
        if area_m2 <= 0:
            return 0.0
        return seeds_per_gutter / area_m2

    @classmethod
    def compute_zone_stats(
        cls,
        gutter_length_m: float,
        gutter_width_m: float,
        seeds_per_gutter: float,
        zone_length_m: float,
        zone_spacing_m: float,
    ) -> dict:
        """
        Compute statistics for a single zone.

        Returns a dict with keys:
        - gutters_in_zone  (float)
        - total_seeds      (float)
        - zone_area_m2     (float)
        - seeds_per_m2     (float)
        - seeds_per_sqft   (float)
        """
        pitch_m = gutter_width_m + zone_spacing_m
        if pitch_m <= 0 or gutter_length_m <= 0 or zone_length_m <= 0:
            return {
                "gutters_in_zone": 0.0,
                "total_seeds": 0.0,
                "zone_area_m2": 0.0,
                "seeds_per_m2": 0.0,
                "seeds_per_sqft": 0.0,
            }

        gutters_in_zone = zone_length_m / pitch_m
        total_seeds = seeds_per_gutter * gutters_in_zone
        zone_area_m2 = gutter_length_m * zone_length_m
        seeds_per_m2 = total_seeds / zone_area_m2 if zone_area_m2 > 0 else 0.0
        seeds_per_sqft = seeds_per_m2 / cls._SQM_TO_SQFT

        return {
            "gutters_in_zone": gutters_in_zone,
            "total_seeds": total_seeds,
            "zone_area_m2": zone_area_m2,
            "seeds_per_m2": seeds_per_m2,
            "seeds_per_sqft": seeds_per_sqft,
        }

    @classmethod
    def render(cls):
        st.subheader("MGS Lettuce Density")

        st.markdown(
            """
            This calculator estimates **plants per m²** for a
            **Mobile Gutter System (MGS)** lettuce setup.

            Enter the fixed system dimensions once, then configure each zone
            with the number of **days plants spend in that zone** and the
            gutter-to-gutter spacing.  
            Results show per-zone density and an overall **time-weighted average**.
            """
        )

        cfg = cls._cfg()

        st.markdown("#### System inputs")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            gutter_length_val = st.number_input(
                "Gutter length",
                min_value=0.0,
                value=cfg.get("gutter_length_val", 2.0),
                step=0.1,
                key="mgs_gutter_length_val",
            )
            cfg["gutter_length_val"] = gutter_length_val
        with col2:
            _gl_unit = cfg.get("gutter_length_unit", "m")
            gutter_length_unit = st.selectbox(
                "Unit",
                cls._UNITS,
                index=cls._UNITS.index(_gl_unit) if _gl_unit in cls._UNITS else 0,
                key="mgs_gutter_length_unit",
            )
            cfg["gutter_length_unit"] = gutter_length_unit
        with col3:
            seeds_per_gutter = st.number_input(
                "Seeds per gutter",
                min_value=0.0,
                value=cfg.get("seeds_per_gutter", 22.0),
                step=1.0,
                key="mgs_seeds_per_gutter",
            )
            cfg["seeds_per_gutter"] = seeds_per_gutter

        col4, col5 = st.columns([2, 1])
        with col4:
            gutter_width_val = st.number_input(
                "Gutter width",
                min_value=0.0,
                value=cfg.get("gutter_width_val", 16.0),
                step=1.0,
                key="mgs_gutter_width_val",
            )
            cfg["gutter_width_val"] = gutter_width_val
        with col5:
            _gw_unit = cfg.get("gutter_width_unit", "cm")
            gutter_width_unit = st.selectbox(
                "Unit",
                cls._UNITS,
                index=cls._UNITS.index(_gw_unit) if _gw_unit in cls._UNITS else 1,
                key="mgs_gutter_width_unit",
            )
            cfg["gutter_width_unit"] = gutter_width_unit

        st.markdown("---")
        st.markdown("#### Zone configuration")

        num_zones = st.number_input(
            "Number of zones",
            min_value=1,
            max_value=20,
            value=cfg.get("num_zones", 2),
            step=1,
            key="mgs_num_zones",
        )
        cfg["num_zones"] = int(num_zones)

        # Collect per-zone inputs
        zone_inputs = []
        zones_cfg = cfg.setdefault("zones", [])
        for i in range(int(num_zones)):
            # Grow the saved zones list as needed
            while len(zones_cfg) <= i:
                zones_cfg.append({})
            zc = zones_cfg[i]

            st.markdown(f"**Zone {i + 1}**")
            zc1, zc2, zc3, zc4 = st.columns([2, 2, 2, 1])
            with zc1:
                zone_name = st.text_input(
                    "Zone name",
                    value=zc.get("name", f"Zone {i + 1}"),
                    key=f"mgs_zone_name_{i}",
                )
                zc["name"] = zone_name
            with zc2:
                days_in_zone = st.number_input(
                    "Days in zone",
                    min_value=0.0,
                    value=zc.get("days_in_zone", 7.0),
                    step=1.0,
                    help="Number of days plants spend in this zone.",
                    key=f"mgs_days_in_zone_{i}",
                )
                zc["days_in_zone"] = days_in_zone
            with zc3:
                zone_spacing_val = st.number_input(
                    "Zone spacing",
                    min_value=0.0,
                    value=zc.get("spacing_val", 20.0),
                    step=1.0,
                    key=f"mgs_zone_spacing_val_{i}",
                )
                zc["spacing_val"] = zone_spacing_val
            with zc4:
                _sp_unit = zc.get("spacing_unit", "cm")
                zone_spacing_unit = st.selectbox(
                    "Unit",
                    cls._UNITS,
                    index=cls._UNITS.index(_sp_unit) if _sp_unit in cls._UNITS else 1,
                    key=f"mgs_zone_spacing_unit_{i}",
                )
                zc["spacing_unit"] = zone_spacing_unit
            zone_inputs.append(
                {
                    "name": zone_name,
                    "days_in_zone": days_in_zone,
                    "spacing_val": zone_spacing_val,
                    "spacing_unit": zone_spacing_unit,
                }
            )

        # Trim stale zone entries when the user reduces the zone count
        del zones_cfg[int(num_zones):]

        st.markdown("---")

        # Convert fixed system inputs to metres
        gutter_length_m = cls.to_meters(gutter_length_val, gutter_length_unit)
        gutter_width_m = cls.to_meters(gutter_width_val, gutter_width_unit)

        # Validate system inputs before computing
        if gutter_length_m <= 0 or gutter_width_m <= 0 or seeds_per_gutter <= 0:
            st.info(
                "Enter non-zero values for gutter length, gutter width, "
                "and seeds per gutter to see results."
            )
            return

        # Compute per-zone stats (time-weighted average density)
        rows = []
        total_weighted_density = 0.0
        total_days = 0.0

        for zi in zone_inputs:
            zone_spacing_m = cls.to_meters(zi["spacing_val"], zi["spacing_unit"])
            days = zi["days_in_zone"]

            seeds_per_m2 = cls.compute_seeds_per_m2(
                gutter_length_m,
                gutter_width_m,
                zone_spacing_m,
                seeds_per_gutter,
            )
            seeds_per_sqft = seeds_per_m2 / cls._SQM_TO_SQFT

            total_weighted_density += seeds_per_m2 * days
            total_days += days

            rows.append(
                {
                    "Zone": zi["name"],
                    "Days in zone": round(days, 1),
                    "Spacing (m)": round(zone_spacing_m, 3),
                    "Plants/m²": round(seeds_per_m2, 2),
                    "Plants/sqft": round(seeds_per_sqft, 3),
                }
            )

        # Display per-zone results
        st.markdown("### Per-zone results")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Overall time-weighted average density
        if total_days > 0:
            overall_avg_m2 = total_weighted_density / total_days
            overall_avg_sqft = overall_avg_m2 / cls._SQM_TO_SQFT

            st.markdown("### Overall time-weighted average density")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Plants/m²", f"{overall_avg_m2:.2f}")
            with res_col2:
                st.metric("Plants/sqft", f"{overall_avg_sqft:.3f}")

            with st.expander("Show system details", expanded=False):
                st.write(f"Gutter length (m): `{gutter_length_m:.3f}`")
                st.write(f"Gutter width (m): `{gutter_width_m:.3f}`")
                st.write(f"Seeds per gutter: `{seeds_per_gutter:.0f}`")
                st.write(f"Total days across all zones: `{total_days:.1f}`")
                st.write(
                    f"Time-weighted average density: `{overall_avg_m2:.2f} plants/m²`"
                )
        else:
            st.info("Enter valid days and spacings for each zone to see results.")


class MGSAnnualizedYieldCalculator:
    """
    MGS Annualized Yield Calculator.

    Calculates annualized fresh-weight yield (kg/m²/year) for a Mobile Gutter
    System using either gutter-level inputs or direct density / plant-weight inputs.

    Two results are shown:
    - Yield using **average density** (plants/m² weighted across all zones)
    - Yield using **final density** (plants/m² at the final/harvest zone)

    Cycle time and optional germination time are used to compute cycles per year.
    """

    _UNITS = ["m", "cm", "ft", "in"]
    _SQM_TO_SQFT = 10.7639
    _CFG_KEY = "mgs_yield_cfg"

    @classmethod
    def _cfg(cls) -> dict:
        """Return the persistent session-state config dict for this calculator."""
        if cls._CFG_KEY not in st.session_state:
            st.session_state[cls._CFG_KEY] = {}
        return st.session_state[cls._CFG_KEY]

    @staticmethod
    def to_meters(value: float, unit: str) -> float:
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
        return value

    @staticmethod
    def density_from_gutter(
        gutter_length_m: float,
        gutter_width_m: float,
        spacing_m: float,
        plants_per_gutter: float,
    ) -> float:
        """Plants/m² from gutter geometry."""
        pitch = gutter_width_m + spacing_m
        if pitch <= 0 or gutter_length_m <= 0 or plants_per_gutter <= 0:
            return 0.0
        return plants_per_gutter / (gutter_length_m * pitch)

    @classmethod
    def render(cls):
        st.subheader("MGS Annualized Yield")

        st.markdown(
            """
            Estimates the **annualized fresh-weight yield** (kg · m⁻² · year⁻¹) for a
            Mobile Gutter System.

            Enter weight using any available measurement (gutter weight, single plant weight,
            or weight per m²), along with average and/or final plant density.

            Two results are shown:
            - **Average density yield** — based on the weighted plant density across all zones.
            - **Final density yield** — based on the plant density at the final (harvest) zone.
            """
        )

        cfg = cls._cfg()

        # --- Weight inputs ---
        st.markdown("#### Weight inputs")

        wcol1, wcol2, wcol3 = st.columns(3)

        with wcol1:
            gutter_weight_g = st.number_input(
                "Gutter weight (g)",
                min_value=0.0,
                value=cfg.get("gutter_weight_g", 0.0),
                step=50.0,
                help="Total fresh weight of all plants in one gutter at harvest.",
                key="mgs_yield_gutter_weight",
            )
            cfg["gutter_weight_g"] = gutter_weight_g
            plants_per_gutter = st.number_input(
                "Plants per gutter",
                min_value=0.0,
                value=cfg.get("plants_per_gutter", 22.0),
                step=1.0,
                help="Number of plants in one gutter (used with gutter weight).",
                key="mgs_yield_plants_per_gutter",
            )
            cfg["plants_per_gutter"] = plants_per_gutter

        with wcol2:
            plant_weight_g = st.number_input(
                "Single plant weight (g)",
                min_value=0.0,
                value=cfg.get("plant_weight_g", 0.0),
                step=5.0,
                help="Average fresh weight per plant at harvest.",
                key="mgs_yield_plant_weight",
            )
            cfg["plant_weight_g"] = plant_weight_g

        with wcol3:
            weight_per_m2_kg = st.number_input(
                "Fresh weight per m² (kg/m²)",
                min_value=0.0,
                value=cfg.get("weight_per_m2_kg", 0.0),
                step=0.5,
                help="Total fresh weight per square metre of growing area at harvest. "
                     "Used directly — density inputs are not required.",
                key="mgs_yield_weight_per_m2",
            )
            cfg["weight_per_m2_kg"] = weight_per_m2_kg

        # --- Density inputs ---
        st.markdown("#### Density inputs")

        dcol1, dcol2 = st.columns(2)

        with dcol1:
            avg_density = st.number_input(
                "Average density (plants/m²)",
                min_value=0.0,
                value=cfg.get("avg_density", 0.0),
                step=10.0,
                help="Weighted average density across all zones (propagation → harvest).",
                key="mgs_yield_avg_density",
            )
            cfg["avg_density"] = avg_density

        with dcol2:
            final_density = st.number_input(
                "Final density (plants/m²)",
                min_value=0.0,
                value=cfg.get("final_density", 0.0),
                step=5.0,
                help="Plant density in the final (harvest) zone.",
                key="mgs_yield_final_density",
            )
            cfg["final_density"] = final_density

        # --- Cycle time ---
        st.markdown("---")
        st.markdown("#### Cycle time")

        col_ct1, col_ct2 = st.columns(2)

        with col_ct1:
            cycle_time_days = st.number_input(
                "Total cycle time (days, seed to harvest)",
                min_value=1.0,
                value=cfg.get("cycle_time_days", 35.0),
                step=1.0,
                key="mgs_yield_cycle",
            )
            cfg["cycle_time_days"] = cycle_time_days
            germ_time_hrs = st.number_input(
                "Germination time (hrs)",
                min_value=0.0,
                value=cfg.get("germ_time_hrs", 48.0),
                step=1.0,
                help="Hours from seeding to end of germination. Part of total cycle time.",
                key="mgs_yield_germ",
            )
            cfg["germ_time_hrs"] = germ_time_hrs

        with col_ct2:
            exclude_germ = st.checkbox(
                "Exclude germination time from cycle",
                value=cfg.get("exclude_germ", False),
                help=(
                    "When checked, the germination period is not counted in the "
                    "productive cycle. Use this if the germination area is not "
                    "part of the MGS floor space being measured."
                ),
                key="mgs_yield_excl_germ",
            )
            cfg["exclude_germ"] = exclude_germ

            germ_time_days_equiv = germ_time_hrs / 24.0
            if exclude_germ:
                effective_cycle = max(cycle_time_days - germ_time_days_equiv, 1.0)
                st.caption(
                    f"Effective cycle = {cycle_time_days:.0f} days – {germ_time_hrs:.0f} hrs "
                    f"({germ_time_days_equiv:.2f} days) = **{effective_cycle:.2f} days**"
                )
            else:
                effective_cycle = cycle_time_days
                st.caption(f"Effective cycle = **{effective_cycle:.0f} days**")

        # --- Determine per-plant weight ---
        if gutter_weight_g > 0 and plants_per_gutter > 0:
            effective_plant_weight_kg = (gutter_weight_g / plants_per_gutter) / 1000.0
            weight_source = f"gutter weight ({gutter_weight_g:.0f} g) ÷ plants per gutter ({plants_per_gutter:.0f})"
        elif plant_weight_g > 0:
            effective_plant_weight_kg = plant_weight_g / 1000.0
            weight_source = f"single plant weight ({plant_weight_g:.1f} g)"
        else:
            effective_plant_weight_kg = None
            weight_source = None

        # --- Results ---
        st.markdown("---")
        st.markdown("### Results")

        if effective_cycle <= 0:
            st.info("Enter a valid cycle time to see results.")
            return

        cycles_per_year = 365.0 / effective_cycle

        results = []

        if weight_per_m2_kg > 0:
            yield_direct = weight_per_m2_kg * cycles_per_year
            results.append(
                (
                    "Annualized yield (weight/m²)",
                    f"{yield_direct:.2f} kg/m²/year",
                    "Based on fresh weight per m² × cycles per year.",
                )
            )

        if effective_plant_weight_kg and avg_density > 0:
            yield_avg = avg_density * effective_plant_weight_kg * cycles_per_year
            results.append(
                (
                    "Annualized yield — average density",
                    f"{yield_avg:.2f} kg/m²/year",
                    "Based on the weighted average density across all MGS zones.",
                )
            )

        if effective_plant_weight_kg and final_density > 0:
            yield_final = final_density * effective_plant_weight_kg * cycles_per_year
            results.append(
                (
                    "Annualized yield — final density",
                    f"{yield_final:.2f} kg/m²/year",
                    "Based on the plant density in the final (harvest) zone.",
                )
            )

        if not results:
            st.info(
                "Enter valid weight and density inputs above to see results. "
                "Provide at least one weight input (gutter weight, single plant weight, "
                "or fresh weight per m²) and at least one density input."
            )
            return

        if len(results) == 1:
            st.metric(results[0][0], results[0][1], help=results[0][2])
        elif len(results) == 2:
            r1, r2 = st.columns(2)
            with r1:
                st.metric(results[0][0], results[0][1], help=results[0][2])
            with r2:
                st.metric(results[1][0], results[1][1], help=results[1][2])
        else:
            r1, r2, r3 = st.columns(3)
            for col, item in zip([r1, r2, r3], results):
                with col:
                    st.metric(item[0], item[1], help=item[2])

        with st.expander("Show calculation details", expanded=False):
            if weight_source:
                plant_wt_display = effective_plant_weight_kg * 1000
                st.write(
                    f"Plant fresh weight: `{plant_wt_display:.1f} g` = "
                    f"`{effective_plant_weight_kg:.4f} kg` (from {weight_source})"
                )
            if weight_per_m2_kg > 0:
                st.write(f"Fresh weight per m²: `{weight_per_m2_kg:.3f} kg/m²`")
            st.write(f"Total cycle time: `{cycle_time_days:.0f} days`")
            st.write(f"Germination time: `{germ_time_hrs:.0f} hrs` = `{germ_time_days_equiv:.2f} days`")
            st.write(
                f"Germination excluded from cycle: `{'Yes' if exclude_germ else 'No'}`"
            )
            st.write(f"Effective cycle time: `{effective_cycle:.2f} days`")
            st.write(f"Cycles per year: `{cycles_per_year:.2f}`")
            if avg_density > 0 and effective_plant_weight_kg:
                st.markdown("---")
                st.write(f"Average density: `{avg_density:.2f} plants/m²`")
                yield_avg_det = avg_density * effective_plant_weight_kg * cycles_per_year
                st.write(
                    f"Yield (avg density) = {avg_density:.2f} × "
                    f"{effective_plant_weight_kg:.4f} × {cycles_per_year:.2f} "
                    f"= **{yield_avg_det:.2f} kg/m²/year**"
                )
            if final_density > 0 and effective_plant_weight_kg:
                st.markdown("---")
                st.write(f"Final density: `{final_density:.2f} plants/m²`")
                yield_final_det = final_density * effective_plant_weight_kg * cycles_per_year
                st.write(
                    f"Yield (final density) = {final_density:.2f} × "
                    f"{effective_plant_weight_kg:.4f} × {cycles_per_year:.2f} "
                    f"= **{yield_final_det:.2f} kg/m²/year**"
                )


class DWCDensityCalculator:
    """
    DWC (Deep Water Culture) density calculator.

    Calculates per-stage and time-weighted average plant density for DWC systems
    using rafts.

    Input options:
    - Raft dimensions + seeds per raft → density computed per stage
    - Direct density (plants/m²) per stage

    Transplant stages:
    - 0 transplants → one stage (seed to harvest on the same raft)
    - N transplants → N+1 stages (e.g. propagation + harvest)

    Overall average density is the time-weighted mean across all stages.
    Final density is the density in the last (harvest) stage.
    """

    _UNITS = ["m", "cm", "ft", "in"]
    _SQM_TO_SQFT = 10.7639
    _CFG_KEY = "dwc_density_cfg"

    @classmethod
    def _cfg(cls) -> dict:
        """Return the persistent session-state config dict for this calculator."""
        if cls._CFG_KEY not in st.session_state:
            st.session_state[cls._CFG_KEY] = {}
        return st.session_state[cls._CFG_KEY]

    @staticmethod
    def to_meters(value: float, unit: str) -> float:
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
        return value

    @classmethod
    def render(cls):
        st.subheader("DWC Density")

        st.markdown(
            """
            This calculator estimates **plants per m²** for a
            **Deep Water Culture (DWC)** system using rafts.

            Choose how to express density (raft dimensions + seeds, or direct entry),
            set the number of transplant stages, and enter the days spent in each stage.  
            Results show per-stage density and an overall **time-weighted average**.
            """
        )

        cfg = cls._cfg()

        # --- Density input method ---
        _dm_opts = ["Raft dimensions + seeds per raft", "Direct density (plants/m²)"]
        _dm_stored = cfg.get("density_mode", _dm_opts[0])
        density_mode = st.radio(
            "Density input method",
            _dm_opts,
            index=_dm_opts.index(_dm_stored) if _dm_stored in _dm_opts else 0,
            horizontal=True,
            key="dwc_density_mode",
        )
        cfg["density_mode"] = density_mode

        # --- Raft size (shown only in raft mode) ---
        raft_area_m2 = 0.0
        if density_mode == _dm_opts[0]:
            st.markdown("#### Raft dimensions")
            rc1, rc2, rc3, rc4, rc5 = st.columns([2, 1, 2, 1, 2])
            with rc1:
                raft_length_val = st.number_input(
                    "Raft length",
                    min_value=0.0,
                    value=cfg.get("raft_length_val", 1.2),
                    step=0.05,
                    key="dwc_raft_length_val",
                )
                cfg["raft_length_val"] = raft_length_val
            with rc2:
                _rl_unit = cfg.get("raft_length_unit", "m")
                raft_length_unit = st.selectbox(
                    "Unit",
                    cls._UNITS,
                    index=cls._UNITS.index(_rl_unit) if _rl_unit in cls._UNITS else 0,
                    key="dwc_raft_length_unit",
                )
                cfg["raft_length_unit"] = raft_length_unit
            with rc3:
                raft_width_val = st.number_input(
                    "Raft width",
                    min_value=0.0,
                    value=cfg.get("raft_width_val", 0.6),
                    step=0.05,
                    key="dwc_raft_width_val",
                )
                cfg["raft_width_val"] = raft_width_val
            with rc4:
                _rw_unit = cfg.get("raft_width_unit", "m")
                raft_width_unit = st.selectbox(
                    "Unit",
                    cls._UNITS,
                    index=cls._UNITS.index(_rw_unit) if _rw_unit in cls._UNITS else 0,
                    key="dwc_raft_width_unit",
                )
                cfg["raft_width_unit"] = raft_width_unit
            with rc5:
                raft_area_m2_disp = st.empty()

            rl_m = cls.to_meters(raft_length_val, raft_length_unit)
            rw_m = cls.to_meters(raft_width_val, raft_width_unit)
            raft_area_m2 = rl_m * rw_m
            raft_area_m2_disp.metric("Raft area (m²)", f"{raft_area_m2:.4f}")

        # --- Number of transplants / stages ---
        st.markdown("---")
        st.markdown("#### Transplant stages")

        num_transplants = st.number_input(
            "Number of transplants",
            min_value=0,
            max_value=10,
            value=cfg.get("num_transplants", 0),
            step=1,
            help="0 = seed directly to harvest raft (no transplant). "
                 "Each transplant adds one intermediate stage.",
            key="dwc_num_transplants",
        )
        cfg["num_transplants"] = int(num_transplants)
        num_stages = int(num_transplants) + 1

        # --- Per-stage inputs ---
        stage_inputs = []
        stages_cfg = cfg.setdefault("stages", [])
        for i in range(num_stages):
            while len(stages_cfg) <= i:
                stages_cfg.append({})
            sc = stages_cfg[i]

            default_name = "Harvest" if i == num_stages - 1 else f"Stage {i + 1}"
            st.markdown(f"**{default_name}**")

            if density_mode == _dm_opts[0]:
                # Raft dimensions mode: per-stage seeds per raft
                sp1, sp2 = st.columns(2)
                with sp1:
                    stage_name = st.text_input(
                        "Stage name",
                        value=sc.get("name", default_name),
                        key=f"dwc_stage_name_{i}",
                    )
                    sc["name"] = stage_name
                with sp2:
                    days_in_stage = st.number_input(
                        "Days in stage",
                        min_value=0.0,
                        value=sc.get("days", 14.0),
                        step=1.0,
                        key=f"dwc_stage_days_{i}",
                    )
                    sc["days"] = days_in_stage
                seeds_per_raft = st.number_input(
                    "Seeds per raft",
                    min_value=0.0,
                    value=sc.get("seeds_per_raft", 24.0),
                    step=1.0,
                    key=f"dwc_stage_seeds_{i}",
                )
                sc["seeds_per_raft"] = seeds_per_raft
                density_m2 = (seeds_per_raft / raft_area_m2) if raft_area_m2 > 0 else 0.0
            else:
                # Direct density mode
                sp1, sp2, sp3 = st.columns(3)
                with sp1:
                    stage_name = st.text_input(
                        "Stage name",
                        value=sc.get("name", default_name),
                        key=f"dwc_stage_name_{i}",
                    )
                    sc["name"] = stage_name
                with sp2:
                    days_in_stage = st.number_input(
                        "Days in stage",
                        min_value=0.0,
                        value=sc.get("days", 14.0),
                        step=1.0,
                        key=f"dwc_stage_days_{i}",
                    )
                    sc["days"] = days_in_stage
                with sp3:
                    density_m2 = st.number_input(
                        "Density (plants/m²)",
                        min_value=0.0,
                        value=sc.get("density_m2", 30.0),
                        step=1.0,
                        key=f"dwc_stage_density_{i}",
                    )
                    sc["density_m2"] = density_m2
                seeds_per_raft = None

            stage_inputs.append(
                {
                    "name": stage_name,
                    "days": days_in_stage,
                    "density_m2": density_m2,
                    "seeds_per_raft": seeds_per_raft,
                }
            )

        # Trim stale stage entries when the user reduces the transplant count
        del stages_cfg[num_stages:]

        st.markdown("---")

        # --- Compute results ---
        rows = []
        total_weighted_density = 0.0
        total_days = 0.0

        for si in stage_inputs:
            density_m2 = si["density_m2"]
            density_sqft = density_m2 / cls._SQM_TO_SQFT
            days = si["days"]

            total_weighted_density += density_m2 * days
            total_days += days

            row = {
                "Stage": si["name"],
                "Days in stage": round(days, 1),
                "Plants/m²": round(density_m2, 2),
                "Plants/sqft": round(density_sqft, 3),
            }
            if si["seeds_per_raft"] is not None:
                row["Seeds per raft"] = round(si["seeds_per_raft"], 1)
            rows.append(row)

        st.markdown("### Per-stage results")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        if total_days > 0:
            overall_avg_m2 = total_weighted_density / total_days
            overall_avg_sqft = overall_avg_m2 / cls._SQM_TO_SQFT
            final_density_m2 = stage_inputs[-1]["density_m2"] if stage_inputs else 0.0
            final_density_sqft = final_density_m2 / cls._SQM_TO_SQFT

            st.markdown("### Overall time-weighted average density")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Average density (plants/m²)", f"{overall_avg_m2:.2f}")
            with c2:
                st.metric("Average density (plants/sqft)", f"{overall_avg_sqft:.3f}")

            st.markdown("### Final (harvest) stage density")
            c3, c4 = st.columns(2)
            with c3:
                st.metric("Final density (plants/m²)", f"{final_density_m2:.2f}")
            with c4:
                st.metric("Final density (plants/sqft)", f"{final_density_sqft:.3f}")

            if density_mode == _dm_opts[0] and raft_area_m2 > 0:
                with st.expander("Show raft details", expanded=False):
                    st.write(f"Raft area: `{raft_area_m2:.4f} m²`")
                    st.write(f"Total days across all stages: `{total_days:.1f}`")
                    st.write(
                        f"Time-weighted average density: `{overall_avg_m2:.2f} plants/m²`"
                    )
        else:
            st.info("Enter valid days for each stage to see results.")


class DWCAnnualizedYieldCalculator:
    """
    DWC (Deep Water Culture) Annualized Yield Calculator.

    Calculates annualized fresh-weight yield (kg/m²/year) for a DWC raft system.

    Weight can be expressed as:
    - Raft weight (g) + plants per raft → per-plant weight
    - Single plant weight (g)
    - Fresh weight per m² (kg/m²) directly

    Density inputs (average and/or final) combined with per-plant weight give
    two separate annualized yield results.
    """

    _SQM_TO_SQFT = 10.7639
    _CFG_KEY = "dwc_yield_cfg"

    @classmethod
    def _cfg(cls) -> dict:
        """Return the persistent session-state config dict for this calculator."""
        if cls._CFG_KEY not in st.session_state:
            st.session_state[cls._CFG_KEY] = {}
        return st.session_state[cls._CFG_KEY]

    @classmethod
    def render(cls):
        st.subheader("DWC Annualized Yield")

        st.markdown(
            """
            Estimates the **annualized fresh-weight yield** (kg · m⁻² · year⁻¹) for a
            Deep Water Culture system.

            Enter weight using any available measurement (raft weight, single plant weight,
            or weight per m²), along with average and/or final plant density.

            Two results are shown:
            - **Average density yield** — based on the time-weighted average density.
            - **Final density yield** — based on the harvest stage density.
            """
        )

        cfg = cls._cfg()

        # --- Weight inputs ---
        st.markdown("#### Weight inputs")

        wcol1, wcol2, wcol3 = st.columns(3)

        with wcol1:
            raft_weight_g = st.number_input(
                "Raft weight (g)",
                min_value=0.0,
                value=cfg.get("raft_weight_g", 0.0),
                step=100.0,
                help="Total fresh weight of all plants on one raft at harvest.",
                key="dwc_yield_raft_weight",
            )
            cfg["raft_weight_g"] = raft_weight_g
            plants_per_raft = st.number_input(
                "Plants per raft",
                min_value=0.0,
                value=cfg.get("plants_per_raft", 24.0),
                step=1.0,
                help="Number of plants on one raft (used with raft weight).",
                key="dwc_yield_plants_per_raft",
            )
            cfg["plants_per_raft"] = plants_per_raft

        with wcol2:
            plant_weight_g = st.number_input(
                "Single plant weight (g)",
                min_value=0.0,
                value=cfg.get("plant_weight_g", 0.0),
                step=5.0,
                help="Average fresh weight per plant at harvest.",
                key="dwc_yield_plant_weight",
            )
            cfg["plant_weight_g"] = plant_weight_g

        with wcol3:
            weight_per_m2_kg = st.number_input(
                "Fresh weight per m² (kg/m²)",
                min_value=0.0,
                value=cfg.get("weight_per_m2_kg", 0.0),
                step=0.5,
                help="Total fresh weight per square metre of raft area at harvest. "
                     "Used directly — density inputs are not required.",
                key="dwc_yield_weight_per_m2",
            )
            cfg["weight_per_m2_kg"] = weight_per_m2_kg

        # --- Density inputs ---
        st.markdown("#### Density inputs")

        dcol1, dcol2 = st.columns(2)

        with dcol1:
            avg_density = st.number_input(
                "Average density (plants/m²)",
                min_value=0.0,
                value=cfg.get("avg_density", 0.0),
                step=5.0,
                help="Time-weighted average density across all stages.",
                key="dwc_yield_avg_density",
            )
            cfg["avg_density"] = avg_density

        with dcol2:
            final_density = st.number_input(
                "Final density (plants/m²)",
                min_value=0.0,
                value=cfg.get("final_density", 0.0),
                step=5.0,
                help="Plant density in the final (harvest) stage.",
                key="dwc_yield_final_density",
            )
            cfg["final_density"] = final_density

        # --- Cycle time ---
        st.markdown("---")
        st.markdown("#### Cycle time")

        col_ct1, col_ct2 = st.columns(2)

        with col_ct1:
            cycle_time_days = st.number_input(
                "Total cycle time (days, seed to harvest)",
                min_value=1.0,
                value=cfg.get("cycle_time_days", 28.0),
                step=1.0,
                key="dwc_yield_cycle",
            )
            cfg["cycle_time_days"] = cycle_time_days
            germ_time_hrs = st.number_input(
                "Germination time (hrs)",
                min_value=0.0,
                value=cfg.get("germ_time_hrs", 48.0),
                step=1.0,
                help="Hours from seeding to end of germination. Part of total cycle time.",
                key="dwc_yield_germ",
            )
            cfg["germ_time_hrs"] = germ_time_hrs

        with col_ct2:
            exclude_germ = st.checkbox(
                "Exclude germination time from cycle",
                value=cfg.get("exclude_germ", False),
                help=(
                    "When checked, the germination period is not counted in the "
                    "productive cycle."
                ),
                key="dwc_yield_excl_germ",
            )
            cfg["exclude_germ"] = exclude_germ

            germ_time_days_equiv = germ_time_hrs / 24.0
            if exclude_germ:
                effective_cycle = max(cycle_time_days - germ_time_days_equiv, 1.0)
                st.caption(
                    f"Effective cycle = {cycle_time_days:.0f} days – {germ_time_hrs:.0f} hrs "
                    f"({germ_time_days_equiv:.2f} days) = **{effective_cycle:.2f} days**"
                )
            else:
                effective_cycle = cycle_time_days
                st.caption(f"Effective cycle = **{effective_cycle:.0f} days**")

        # --- Determine per-plant weight ---
        if raft_weight_g > 0 and plants_per_raft > 0:
            effective_plant_weight_kg = (raft_weight_g / plants_per_raft) / 1000.0
            weight_source = (
                f"raft weight ({raft_weight_g:.0f} g) ÷ plants per raft ({plants_per_raft:.0f})"
            )
        elif plant_weight_g > 0:
            effective_plant_weight_kg = plant_weight_g / 1000.0
            weight_source = f"single plant weight ({plant_weight_g:.1f} g)"
        else:
            effective_plant_weight_kg = None
            weight_source = None

        # --- Results ---
        st.markdown("---")
        st.markdown("### Results")

        if effective_cycle <= 0:
            st.info("Enter a valid cycle time to see results.")
            return

        cycles_per_year = 365.0 / effective_cycle

        results = []

        if weight_per_m2_kg > 0:
            yield_direct = weight_per_m2_kg * cycles_per_year
            results.append(
                (
                    "Annualized yield (weight/m²)",
                    f"{yield_direct:.2f} kg/m²/year",
                    "Based on fresh weight per m² × cycles per year.",
                )
            )

        if effective_plant_weight_kg and avg_density > 0:
            yield_avg = avg_density * effective_plant_weight_kg * cycles_per_year
            results.append(
                (
                    "Annualized yield — average density",
                    f"{yield_avg:.2f} kg/m²/year",
                    "Based on the time-weighted average density across all DWC stages.",
                )
            )

        if effective_plant_weight_kg and final_density > 0:
            yield_final = final_density * effective_plant_weight_kg * cycles_per_year
            results.append(
                (
                    "Annualized yield — final density",
                    f"{yield_final:.2f} kg/m²/year",
                    "Based on the plant density in the final (harvest) stage.",
                )
            )

        if not results:
            st.info(
                "Enter valid weight and density inputs above to see results. "
                "Provide at least one weight input (raft weight, single plant weight, "
                "or fresh weight per m²) and at least one density input."
            )
            return

        if len(results) == 1:
            st.metric(results[0][0], results[0][1], help=results[0][2])
        elif len(results) == 2:
            r1, r2 = st.columns(2)
            with r1:
                st.metric(results[0][0], results[0][1], help=results[0][2])
            with r2:
                st.metric(results[1][0], results[1][1], help=results[1][2])
        else:
            r1, r2, r3 = st.columns(3)
            for col, item in zip([r1, r2, r3], results):
                with col:
                    st.metric(item[0], item[1], help=item[2])

        with st.expander("Show calculation details", expanded=False):
            if weight_source:
                plant_wt_display = effective_plant_weight_kg * 1000
                st.write(
                    f"Plant fresh weight: `{plant_wt_display:.1f} g` = "
                    f"`{effective_plant_weight_kg:.4f} kg` (from {weight_source})"
                )
            if weight_per_m2_kg > 0:
                st.write(f"Fresh weight per m²: `{weight_per_m2_kg:.3f} kg/m²`")
            st.write(f"Total cycle time: `{cycle_time_days:.0f} days`")
            st.write(
                f"Germination time: `{germ_time_hrs:.0f} hrs` = `{germ_time_days_equiv:.2f} days`"
            )
            st.write(
                f"Germination excluded from cycle: `{'Yes' if exclude_germ else 'No'}`"
            )
            st.write(f"Effective cycle time: `{effective_cycle:.2f} days`")
            st.write(f"Cycles per year: `{cycles_per_year:.2f}`")
            if avg_density > 0 and effective_plant_weight_kg:
                st.markdown("---")
                st.write(f"Average density: `{avg_density:.2f} plants/m²`")
                y = avg_density * effective_plant_weight_kg * cycles_per_year
                st.write(
                    f"Yield (avg density) = {avg_density:.2f} × "
                    f"{effective_plant_weight_kg:.4f} × {cycles_per_year:.2f} "
                    f"= **{y:.2f} kg/m²/year**"
                )
            if final_density > 0 and effective_plant_weight_kg:
                st.markdown("---")
                st.write(f"Final density: `{final_density:.2f} plants/m²`")
                y = final_density * effective_plant_weight_kg * cycles_per_year
                st.write(
                    f"Yield (final density) = {final_density:.2f} × "
                    f"{effective_plant_weight_kg:.4f} × {cycles_per_year:.2f} "
                    f"= **{y:.2f} kg/m²/year**"
                )
