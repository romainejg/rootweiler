"""
Lettuce Growth Model – model constants and configuration.

All numerical constants and preset tables are defined here so they can be
reviewed, recalibrated, or extended in one place without touching calculation
or UI code.

Symbols used throughout the model
──────────────────────────────────
W     – fresh plant weight, g/plant
N     – actual physical density, plants/m²
E     – effective crowding density, plants/m²
d     – actual plant diameter, mm
dBase – average-architecture diameter, mm
t     – chronological grow days
tau   – effective physiological grow days  (tau = g × t)
g     – environmental growth-speed factor
c     – plant compactness / diameter factor
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Fresh-weight model  W(E)
#   W = WEIGHT_C + WEIGHT_A / (1 + (E / WEIGHT_B) ^ WEIGHT_P)
#
# Market-calibrated anchor points (average architecture, E ≈ N):
#   E 12  → W ≈ 350 g
#   E 43  → W ≈ 180 g
#   E 200 → W ≈  30 g
#   E 400 → W ≈  14.8 g
#   E 800 → W ≈   9.7 g
#   E 1000→ W ≈   9 g
# ──────────────────────────────────────────────────────────────────────────────

WEIGHT_C: float = 7.3304183
WEIGHT_A: float = 396.39861833
WEIGHT_B: float = 36.76850027
WEIGHT_P: float = 1.65468912

# ──────────────────────────────────────────────────────────────────────────────
# Diameter-versus-physiological-time model  dBase(tau)
#   dBase = DIAMETER_LOW + (DIAMETER_HIGH – DIAMETER_LOW) /
#           (1 + exp(–DIAMETER_RATE × (tau – DIAMETER_MIDPOINT)))
#
# Refitted logistic: asymptotes at DIAMETER_LOW ≈ 4 mm and DIAMETER_HIGH = 300 mm.
# ──────────────────────────────────────────────────────────────────────────────

DIAMETER_LOW: float = 3.96382847       # lower asymptote, mm
DIAMETER_HIGH: float = 300.0           # upper asymptote, mm
DIAMETER_RATE: float = 0.12671467      # logistic growth rate, day⁻¹
DIAMETER_MIDPOINT: float = 36.81336719 # inflection point, physiological days

# ──────────────────────────────────────────────────────────────────────────────
# Calibrated effective-density range
# Results outside these limits are flagged as lower-confidence extrapolations.
# ──────────────────────────────────────────────────────────────────────────────

E_MIN_CALIBRATED: float = 12.0    # plants/m²
E_MAX_CALIBRATED: float = 1000.0  # plants/m²

W_MIN_CALIBRATED: float = 9.0    # g/plant
W_MAX_CALIBRATED: float = 350.0  # g/plant

# ──────────────────────────────────────────────────────────────────────────────
# Growth-environment presets
#
# speedFactor (g) is derived from the saturating engineering approximation
# normalised at DLI 20 mol·m⁻²·day⁻¹:
#   g(I) = (1 − exp(−I / 12)) / (1 − exp(−20 / 12))
#
# These are initial calibration factors representing the linear response at
# low DLI and diminishing growth response at higher DLI.  They are NOT
# universally validated biological constants; recalibrate per cultivar and
# facility as data become available.
# Normalized at DLI 20 mol·m⁻²·day⁻¹.
# ──────────────────────────────────────────────────────────────────────────────

GROWTH_PRESETS: dict[str, dict] = {
    "high": {
        "label": "High",
        "dli": 27,
        "speedFactor": 1.103,
        "description": (
            "DLI around 27 with favorable warm, non-stressful temperature."
        ),
    },
    "average": {
        "label": "Average",
        "dli": 20,
        "speedFactor": 1.0,
        "description": (
            "DLI around 20 with typical favorable lettuce temperature."
        ),
    },
    "low": {
        "label": "Low",
        "dli": 13,
        "speedFactor": 0.816,
        "description": (
            "DLI around 13 and/or cooler, growth-limiting temperature."
        ),
    },
    "minimal": {
        "label": "Minimal",
        "dli": 5,
        "speedFactor": 0.420,
        "description": (
            "DLI around 5 and/or strongly growth-limiting temperature."
        ),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Plant-architecture presets
#
# diameterFactor (c) is a cultivar-calibration assumption, not a universal
# biological constant.  Approximately ±15 % around average is an initial
# estimate; recalibrate per cultivar as measurements become available.
# ──────────────────────────────────────────────────────────────────────────────

ARCHITECTURE_PRESETS: dict[str, dict] = {
    "compact": {
        "label": "Compact / upright",
        "diameterFactor": 0.85,
        "description": (
            "Approximately 15% narrower than the average architecture "
            "at equal physiological maturity."
        ),
    },
    "average": {
        "label": "Average",
        "diameterFactor": 1.0,
        "description": "Baseline plant architecture.",
    },
    "open": {
        "label": "Open / spreading",
        "diameterFactor": 1.15,
        "description": (
            "Approximately 15% wider than the average architecture "
            "at equal physiological maturity."
        ),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Default input values (used by the Reset button)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT_MODE: str = "Density"      # "Density" | "Weight" | "Days"
DEFAULT_DENSITY: float = 43.0            # plants/m²
DEFAULT_WEIGHT: float = 180.0            # g/plant
DEFAULT_DAYS: float = 36.9              # grow days
DEFAULT_GROWTH_ENV: str = "average"
DEFAULT_ARCHITECTURE: str = "average"
