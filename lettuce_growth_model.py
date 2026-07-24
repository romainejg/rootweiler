"""
Lettuce Growth Model – calculation engine and Streamlit UI.

Pure-function calculation layer
────────────────────────────────
All public functions in this module take plain numeric inputs and return plain
numeric outputs.  They import constants from lettuce_growth_model_constants
and do NOT import streamlit.

UI layer
────────
LettuceGrowthModelCalculator.render() is the Streamlit entry point and is the
only place that imports / calls st.*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lettuce_growth_model_constants import (
    ARCHITECTURE_PRESETS,
    DEFAULT_ARCHITECTURE,
    DEFAULT_DAYS,
    DEFAULT_DENSITY,
    DEFAULT_GROWTH_ENV,
    DEFAULT_INPUT_MODE,
    DEFAULT_WEIGHT,
    DIAMETER_HIGH,
    DIAMETER_LOW,
    DIAMETER_MIDPOINT,
    DIAMETER_RATE,
    E_MAX_CALIBRATED,
    E_MIN_CALIBRATED,
    GROWTH_PRESETS,
    W_MAX_CALIBRATED,
    W_MIN_CALIBRATED,
    WEIGHT_A,
    WEIGHT_B,
    WEIGHT_C,
    WEIGHT_P,
)

# ──────────────────────────────────────────────────────────────────────────────
# Core mathematical functions
# ──────────────────────────────────────────────────────────────────────────────


def weight_from_effective_density(E: float) -> float:
    """
    Estimate fresh plant weight W (g/plant) from effective crowding density E
    (plants/m²) using the market-calibrated logistic model.

    W(E) = WEIGHT_C + WEIGHT_A / (1 + (E / WEIGHT_B)^WEIGHT_P)
    """
    return WEIGHT_C + WEIGHT_A / (1.0 + math.pow(E / WEIGHT_B, WEIGHT_P))


def effective_density_from_weight(W: float) -> float:
    """
    Analytical inverse of weight_from_effective_density.

    Raises ValueError when W is outside the model domain (W ≤ WEIGHT_C).
    """
    inner = WEIGHT_A / (W - WEIGHT_C) - 1.0
    if inner <= 0.0:
        raise ValueError(
            f"Weight {W:.3f} g is outside the model domain "
            f"(must be greater than {WEIGHT_C:.4f} g)."
        )
    return WEIGHT_B * math.pow(inner, 1.0 / WEIGHT_P)


def base_diameter_from_effective_days(tau: float) -> float:
    """
    Estimate average-architecture plant diameter dBase (mm) from effective
    physiological grow days tau using the refitted logistic model.

    dBase(tau) = DIAMETER_LOW + (DIAMETER_HIGH - DIAMETER_LOW) /
                 (1 + exp(-DIAMETER_RATE * (tau - DIAMETER_MIDPOINT)))
    """
    return DIAMETER_LOW + (DIAMETER_HIGH - DIAMETER_LOW) / (
        1.0 + math.exp(-DIAMETER_RATE * (tau - DIAMETER_MIDPOINT))
    )


def effective_days_from_base_diameter(dBase: float) -> float:
    """
    Analytical inverse of base_diameter_from_effective_days.

    Raises RangeError when dBase is at or beyond the logistic asymptotes.
    """
    if dBase <= DIAMETER_LOW or dBase >= DIAMETER_HIGH:
        raise ValueError(
            f"Base diameter {dBase:.3f} mm is outside the logistic model domain "
            f"({DIAMETER_LOW:.4f} mm < dBase < {DIAMETER_HIGH:.1f} mm)."
        )
    return DIAMETER_MIDPOINT + math.log(
        (dBase - DIAMETER_LOW) / (DIAMETER_HIGH - dBase)
    ) / DIAMETER_RATE


def width_from_actual_density(N: float) -> float:
    """
    Convert actual plant density N (plants/m²) to equivalent square-grid
    spacing d (mm).

    d = 1000 / sqrt(N)
    """
    return 1000.0 / math.sqrt(N)


def actual_density_from_width(d: float) -> float:
    """
    Convert equivalent square-grid spacing d (mm) to actual plant density N
    (plants/m²).

    N = 1_000_000 / d²
    """
    return 1_000_000.0 / (d * d)


# ──────────────────────────────────────────────────────────────────────────────
# Result and validation helpers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class GrowthModelResult:
    """Holds all outputs produced by the three solver modes."""

    W: float              # fresh weight, g/plant
    N: float              # actual density, plants/m²
    E: float              # effective crowding density, plants/m²
    d: float              # actual plant diameter, mm
    t: float              # chronological grow days
    tau: float            # effective physiological grow days
    warnings: list[str]   # human-readable warnings (empty = all in calibrated range)
    category: str         # product category label


def _collect_warnings(
    W: float,
    E: float,
    d: float,
    dBase: float,
    t: float,
    g: float,
) -> list[str]:
    """Return a list of out-of-range warning strings (empty = none)."""
    warnings: list[str] = []

    if E < E_MIN_CALIBRATED:
        warnings.append(
            f"Effective density {E:.1f} plants/m² is below the calibrated minimum "
            f"({E_MIN_CALIBRATED:.0f} plants/m²). Result is an extrapolation."
        )
    if E > E_MAX_CALIBRATED:
        warnings.append(
            f"Effective density {E:.1f} plants/m² exceeds the calibrated maximum "
            f"({E_MAX_CALIBRATED:.0f} plants/m²). Result is an extrapolation."
        )
    if W < W_MIN_CALIBRATED:
        warnings.append(
            f"Estimated weight {W:.1f} g/plant is below the calibrated minimum "
            f"({W_MIN_CALIBRATED:.0f} g). Result is an extrapolation."
        )
    if W > W_MAX_CALIBRATED:
        warnings.append(
            f"Estimated weight {W:.1f} g/plant exceeds the calibrated maximum "
            f"({W_MAX_CALIBRATED:.0f} g). Result is an extrapolation."
        )
    if dBase <= DIAMETER_LOW or dBase >= DIAMETER_HIGH:
        warnings.append(
            f"Base diameter {dBase:.1f} mm is at or beyond the logistic asymptotes "
            f"({DIAMETER_LOW:.1f} – {DIAMETER_HIGH:.0f} mm). "
            "Grow days estimate is unreliable and not shown."
        )
    if not math.isnan(t) and t < 1:
        warnings.append(
            "Calculated grow days are very short (< 1 day). "
            "Check inputs and modifiers."
        )

    return warnings


def _product_category(W: float, N: float) -> str:
    """
    Assign an approximate product category based on weight and density.

    Categories:
      Belgian / full head : 250–450 g,  low density
      Full head            : 160–249 g
      Teen leaf            : 80–159 g
      Small leaf           : ~15–30 g or N ≈ 200–400 plants/m²
      Baby leaf            : ~8–10 g   or N ≈ 800–1 200 plants/m²
      Transition           : between defined ranges
    """
    if 250.0 <= W <= 450.0 and N <= 30.0:
        return "Belgian / full head"
    if 160.0 <= W <= 449.0:
        return "Full head"
    if 80.0 <= W <= 159.9:
        return "Teen leaf"
    if (15.0 <= W <= 30.0) or (200.0 <= N <= 400.0):
        return "Small leaf"
    if (8.0 <= W <= 10.0) or (800.0 <= N <= 1200.0):
        return "Baby leaf"
    return "Transition"


# ──────────────────────────────────────────────────────────────────────────────
# Three solver modes
# ──────────────────────────────────────────────────────────────────────────────


def _safe_days(dBase: float, g: float) -> tuple[float, float]:
    """
    Return (tau, t) for a given base diameter and growth factor.

    When dBase is outside the logistic domain, returns (nan, nan) instead of
    raising.  The caller should check warnings for the diameter-asymptote flag.
    """
    try:
        tau = effective_days_from_base_diameter(dBase)
        return tau, tau / g
    except ValueError:
        return float("nan"), float("nan")


def solve_from_density(N: float, g: float, c: float) -> GrowthModelResult:
    """
    Mode 1: Actual density N is the primary input.

    E = N × c²
    W = weight_from_effective_density(E)
    d = width_from_actual_density(N)
    dBase = d / c
    tau = effective_days_from_base_diameter(dBase)
    t = tau / g
    """
    E = N * c * c
    W = weight_from_effective_density(E)
    d = width_from_actual_density(N)
    dBase = d / c
    tau, t = _safe_days(dBase, g)
    warnings = _collect_warnings(W, E, d, dBase, t, g)
    return GrowthModelResult(
        W=W, N=N, E=E, d=d, t=t, tau=tau,
        warnings=warnings,
        category=_product_category(W, N),
    )


def solve_from_weight(W: float, g: float, c: float) -> GrowthModelResult:
    """
    Mode 2: Desired fresh weight W is the primary input.

    E = effective_density_from_weight(W)
    N = E / c²
    dBase = width_from_actual_density(E)
    d = c × dBase
    tau = effective_days_from_base_diameter(dBase)
    t = tau / g
    """
    E = effective_density_from_weight(W)
    N = E / (c * c)
    dBase = width_from_actual_density(E)
    d = c * dBase
    tau, t = _safe_days(dBase, g)
    warnings = _collect_warnings(W, E, d, dBase, t, g)
    return GrowthModelResult(
        W=W, N=N, E=E, d=d, t=t, tau=tau,
        warnings=warnings,
        category=_product_category(W, N),
    )


def solve_from_days(t: float, g: float, c: float) -> GrowthModelResult:
    """
    Mode 3: Chronological grow days t is the primary input.

    tau = g × t
    dBase = base_diameter_from_effective_days(tau)
    d = c × dBase
    N = actual_density_from_width(d)
    E = N × c²
    W = weight_from_effective_density(E)
    """
    tau = g * t
    dBase = base_diameter_from_effective_days(tau)
    d = c * dBase
    N = actual_density_from_width(d)
    E = N * c * c
    W = weight_from_effective_density(E)
    warnings = _collect_warnings(W, E, d, dBase, t, g)
    return GrowthModelResult(
        W=W, N=N, E=E, d=d, t=t, tau=tau,
        warnings=warnings,
        category=_product_category(W, N),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Chart helper
# ──────────────────────────────────────────────────────────────────────────────


def build_weight_density_chart(
    result: GrowthModelResult,
    highlight_color: str = "#45C96B",
) -> go.Figure:
    """
    Return a Plotly figure showing the W(E) calibration curve with the current
    result highlighted.
    """
    e_vals = np.linspace(E_MIN_CALIBRATED, E_MAX_CALIBRATED, 400)
    w_vals = [weight_from_effective_density(e) for e in e_vals]

    fig = go.Figure()

    # Calibration curve
    fig.add_trace(
        go.Scatter(
            x=list(e_vals),
            y=w_vals,
            mode="lines",
            name="W(E) curve",
            line=dict(color="#8C8BFF", width=2),
        )
    )

    # Calibrated-range shading
    e_range = e_vals[(e_vals >= E_MIN_CALIBRATED) & (e_vals <= E_MAX_CALIBRATED)]
    w_range = [weight_from_effective_density(e) for e in e_range]
    fig.add_trace(
        go.Scatter(
            x=list(e_range),
            y=w_range,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(69,201,107,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Calibrated range",
            showlegend=True,
        )
    )

    # Current result point
    fig.add_trace(
        go.Scatter(
            x=[result.E],
            y=[result.W],
            mode="markers",
            marker=dict(color=highlight_color, size=12, symbol="circle"),
            name=f"Current: E={result.E:.1f}, W={result.W:.1f} g",
        )
    )

    # N line (actual density)
    if abs(result.N - result.E) > 0.5:
        fig.add_vline(
            x=result.N,
            line_dash="dot",
            line_color="#FFD750",
            annotation_text=f"N={result.N:.0f}",
            annotation_position="top right",
        )

    fig.update_layout(
        xaxis_title="Effective crowding density E (plants/m²)",
        yaxis_title="Fresh weight W (g/plant)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=40),
        height=300,
        template="simple_white",
        font=dict(family="Rubik, system-ui, sans-serif"),
    )
    fig.update_xaxes(type="log", range=[math.log10(8), math.log10(E_MAX_CALIBRATED * 1.3)])
    fig.update_yaxes(range=[0, W_MAX_CALIBRATED * 1.1])

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

_RESULT_CARD_CSS = """
<style>
.lgm-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.lgm-card {
    background: #F5FAFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 0.9rem 1rem 0.7rem 1rem;
    text-align: center;
}
.lgm-card-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.3rem;
}
.lgm-card-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #111111;
    line-height: 1.1;
}
.lgm-card-unit {
    font-size: 0.78rem;
    color: #6B7280;
    margin-top: 0.1rem;
}
.lgm-category-badge {
    display: inline-block;
    background: #45C96B;
    color: #ffffff;
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    margin: 0.5rem 0 0.75rem 0;
}
.lgm-env-card {
    border: 2px solid transparent;
    border-radius: 10px;
    padding: 0.65rem 0.8rem;
    cursor: pointer;
    transition: border-color 0.15s;
    background: #F9FAFB;
}
.lgm-env-card-selected {
    border-color: #45C96B;
    background: #F0FFF4;
}
</style>
"""


class LettuceGrowthModelCalculator:
    """
    Interactive Lettuce Growth Model Streamlit component.

    Separates three input modes (density / weight / days), applies
    growth-environment and plant-architecture modifiers, and displays
    result cards, a product-category indicator, and a W(E) chart.
    """

    _SESSION_PREFIX = "lgm_"

    # ── Session-state helpers ─────────────────────────────────────────────────

    @classmethod
    def _sk(cls, key: str) -> str:
        """Namespace a session-state key."""
        return cls._SESSION_PREFIX + key

    @classmethod
    def _reset_defaults(cls) -> None:
        """Restore all inputs to their default values."""
        sk = cls._sk
        st.session_state[sk("input_mode")] = DEFAULT_INPUT_MODE
        st.session_state[sk("density_val")] = DEFAULT_DENSITY
        st.session_state[sk("weight_val")] = DEFAULT_WEIGHT
        st.session_state[sk("days_val")] = DEFAULT_DAYS
        st.session_state[sk("growth_env")] = DEFAULT_GROWTH_ENV
        st.session_state[sk("architecture")] = DEFAULT_ARCHITECTURE

    # ── Render ────────────────────────────────────────────────────────────────

    @classmethod
    def render(cls) -> None:
        """Main Streamlit render function."""
        st.markdown(_RESULT_CARD_CSS, unsafe_allow_html=True)
        st.subheader("Lettuce Growth Model")

        sk = cls._sk

        # Initialize defaults on first load
        for key, default in [
            (sk("input_mode"), DEFAULT_INPUT_MODE),
            (sk("density_val"), DEFAULT_DENSITY),
            (sk("weight_val"), DEFAULT_WEIGHT),
            (sk("days_val"), DEFAULT_DAYS),
            (sk("growth_env"), DEFAULT_GROWTH_ENV),
            (sk("architecture"), DEFAULT_ARCHITECTURE),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        # ── Input mode selector ───────────────────────────────────────────────
        col_mode, col_reset = st.columns([4, 1])
        with col_mode:
            input_mode = st.radio(
                "Primary input",
                ["Density", "Weight", "Days"],
                horizontal=True,
                key=sk("input_mode"),
                help=(
                    "Choose which quantity you want to enter directly. "
                    "The other two are calculated automatically."
                ),
            )
        with col_reset:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            if st.button("↺ Reset", key=sk("reset"), use_container_width=True):
                cls._reset_defaults()
                st.rerun()

        # ── Numeric input ─────────────────────────────────────────────────────
        if input_mode == "Density":
            primary_value = st.number_input(
                "Final planting density (plants/m²)",
                min_value=0.1,
                max_value=5000.0,
                value=float(st.session_state[sk("density_val")]),
                step=1.0,
                format="%.1f",
                key=sk("density_val"),
            )
        elif input_mode == "Weight":
            primary_value = st.number_input(
                "Desired fresh plant weight (g/plant)",
                min_value=0.1,
                max_value=1000.0,
                value=float(st.session_state[sk("weight_val")]),
                step=1.0,
                format="%.1f",
                key=sk("weight_val"),
            )
        else:  # Days
            primary_value = st.number_input(
                "Grow days",
                min_value=0.1,
                max_value=365.0,
                value=float(st.session_state[sk("days_val")]),
                step=0.5,
                format="%.1f",
                key=sk("days_val"),
            )

        # ── Modifiers ─────────────────────────────────────────────────────────
        st.markdown("#### Modifiers")
        col_env, col_arch = st.columns(2)

        with col_env:
            st.markdown("**Growth environment**")
            growth_opts = list(GROWTH_PRESETS.keys())
            growth_labels = [GROWTH_PRESETS[k]["label"] for k in growth_opts]
            current_env = st.session_state.get(sk("growth_env"), DEFAULT_GROWTH_ENV)
            if current_env not in growth_opts:
                current_env = DEFAULT_GROWTH_ENV
            growth_env_key = st.radio(
                "Growth environment",
                growth_opts,
                format_func=lambda k: GROWTH_PRESETS[k]["label"],
                index=growth_opts.index(current_env),
                key=sk("growth_env"),
                label_visibility="collapsed",
                help=(
                    "High means favorable warm conditions, not heat stress. "
                    "Lettuce yield can decline and tipburn can increase at excessive temperatures."
                ),
            )
            g = GROWTH_PRESETS[growth_env_key]["speedFactor"]
            st.caption(GROWTH_PRESETS[growth_env_key]["description"])

        with col_arch:
            st.markdown("**Plant architecture**")
            arch_opts = list(ARCHITECTURE_PRESETS.keys())
            current_arch = st.session_state.get(sk("architecture"), DEFAULT_ARCHITECTURE)
            if current_arch not in arch_opts:
                current_arch = DEFAULT_ARCHITECTURE
            arch_key = st.radio(
                "Plant architecture",
                arch_opts,
                format_func=lambda k: ARCHITECTURE_PRESETS[k]["label"],
                index=arch_opts.index(current_arch),
                key=sk("architecture"),
                label_visibility="collapsed",
            )
            c = ARCHITECTURE_PRESETS[arch_key]["diameterFactor"]
            st.caption(ARCHITECTURE_PRESETS[arch_key]["description"])

        # ── Solve ─────────────────────────────────────────────────────────────
        result: Optional[GrowthModelResult] = None
        solve_error: Optional[str] = None

        try:
            if input_mode == "Density":
                result = solve_from_density(N=primary_value, g=g, c=c)
            elif input_mode == "Weight":
                result = solve_from_weight(W=primary_value, g=g, c=c)
            else:
                result = solve_from_days(t=primary_value, g=g, c=c)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            solve_error = str(exc)

        st.markdown("---")

        # ── Result cards ──────────────────────────────────────────────────────
        if solve_error:
            st.error(f"Calculation error: {solve_error}")
        elif result is not None:
            # Warnings
            for w in result.warnings:
                st.warning(w)

            # Four metric cards
            st.markdown("#### Results")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Fresh weight", f"{result.W:.1f} g/plant")
            with c2:
                st.metric("Density", f"{round(result.N):.0f} plants/m²")
            with c3:
                days_str = "—" if math.isnan(result.t) else f"{result.t:.1f} d"
                st.metric("Grow days", days_str)
            with c4:
                st.metric(
                    "Plant width",
                    f"{result.d:.1f} mm",
                    delta=f"{result.d / 10.0:.1f} cm",
                    delta_color="off",
                )

            # Spacing note
            st.caption(
                f"Equivalent square spacing: **{result.d:.1f} mm** "
                f"({result.d / 10.0:.1f} cm) — based on a square grid. "
                "Staggered or hexagonal layouts are not currently modelled."
            )

            # Product category badge
            badge_color = {
                "Belgian / full head": "#8C8BFF",
                "Full head": "#45C96B",
                "Teen leaf": "#FFD750",
                "Small leaf": "#ED695D",
                "Baby leaf": "#111111",
            }.get(result.category, "#6B7280")
            st.markdown(
                f'<span class="lgm-category-badge" '
                f'style="background:{badge_color}">{result.category}</span>',
                unsafe_allow_html=True,
            )

            # ── Chart ─────────────────────────────────────────────────────────
            with st.expander("📈 W(E) density–weight curve", expanded=True):
                fig = build_weight_density_chart(result)
                st.plotly_chart(fig, use_container_width=True, key="lgm_chart")

            # ── Assumptions and calibration ───────────────────────────────────
            with st.expander("📋 Assumptions and calibration", expanded=False):
                _render_assumptions(g, c, growth_env_key, arch_key, result)

        else:
            st.info("Enter a value above to see results.")


def _render_assumptions(
    g: float,
    c: float,
    growth_env_key: str,
    arch_key: str,
    result: GrowthModelResult,
) -> None:
    """Render the expandable assumptions/methodology section."""
    env_preset = GROWTH_PRESETS[growth_env_key]
    arch_preset = ARCHITECTURE_PRESETS[arch_key]

    tau_str = "—" if math.isnan(result.tau) else f"{result.tau:.2f} d"
    st.markdown(
        f"""
**Active modifiers**

| Modifier | Preset | Value |
|---|---|---|
| Growth environment | {env_preset['label']} (DLI ≈ {env_preset['dli']} mol·m⁻²·day⁻¹) | g = {g:.3f} |
| Plant architecture | {arch_preset['label']} | c = {c:.2f} |

**Derived quantities**

| Symbol | Description | Value |
|---|---|---|
| E | Effective crowding density | {result.E:.2f} plants/m² |
| dBase | Average-architecture diameter | {result.d / c:.1f} mm |
| tau | Physiological grow days | {tau_str} |

**Calibrated ranges**

* Effective density: {E_MIN_CALIBRATED:.0f}–{E_MAX_CALIBRATED:.0f} plants/m²
* Fresh weight: {W_MIN_CALIBRATED:.0f}–{W_MAX_CALIBRATED:.0f} g/plant
* Grow days: model is valid from seedling to full maturity

---

**Methodology and scientific notes**

This tool uses two market-calibrated logistic models:

1. **W(E)**: Fresh weight as a function of effective crowding density.
2. **dBase(tau)**: Average-architecture diameter as a function of physiological grow
   days (tau = g × t).

Architecture is applied as d = c × dBase and E = N × c², because crowding is
proportional to occupied horizontal area (which scales as diameter squared).

**Growth environment factor g**

The speed factor is derived from a saturating approximation:
g(I) = (1 − exp(−I/12)) / (1 − exp(−20/12)), normalised at DLI 20.
These are initial calibration factors, not universally validated biological
constants. DLI, temperature, CO₂, cultivar, humidity, nutrient management,
spectrum, and photoperiod can all change actual crop timing.

> ⚠️ "High" temperature means favorable warm conditions, not heat stress.
> Lettuce yield can decline and tipburn can increase at excessive temperatures.
> High DLI or high temperature preset may increase tipburn risk under poor
> airflow, humidity control, or excessive temperature.

**Architecture factor c**

The ±15 % values are initial cultivar-calibration assumptions. Light,
temperature, and spectrum can interactively alter lettuce leaf expansion and
architecture, supporting a separate compactness modifier without a universal
coefficient.

---

**Scientific references**

* Yan et al. (2019) — Lettuce fresh/dry biomass increases approximately linearly
  with DLI at lower light, with diminishing returns at higher DLI (DLI 5.04–15.12
  mol·m⁻²·day⁻¹).
  https://journals.ashs.org/view/journals/hortsci/54/10/article-p1737.xml

* Katzin et al. (2020) — Increasing light reduced time to market weight, with
  diminishing benefit at high intensity; ~24 °C produced highest yield; excessive
  temperature reduced yield and increased tipburn risk.
  https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2020.592171/full

* Graamans et al. (2024) — Light, temperature, and spectrum can interactively
  alter lettuce leaf expansion and architecture.
  https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1497672/full

* Market weight and density categories based on CEA leafy-greens overview:
  https://wikifarmer.com/library/en/article/inside-the-modern-leafy-greens-market-in-cea-systems

---

> **Disclaimer:** This is a planning and calibration tool, not a guarantee of
> crop performance. Validate all estimates against your own facility data before
> making production decisions.
        """
    )
