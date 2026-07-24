"""
Unit tests for the Lettuce Growth Model calculation engine.

Run with:
    pytest test_lettuce_growth_model.py -v

All floating-point comparisons use tolerances rather than exact equality.
"""

from __future__ import annotations

import math
import pytest

from lettuce_growth_model import (
    actual_density_from_width,
    base_diameter_from_effective_days,
    effective_days_from_base_diameter,
    effective_density_from_weight,
    solve_from_days,
    solve_from_density,
    solve_from_weight,
    weight_from_effective_density,
    width_from_actual_density,
    _product_category,
)
from lettuce_growth_model_constants import (
    ARCHITECTURE_PRESETS,
    GROWTH_PRESETS,
)

# ──────────────────────────────────────────────────────────────────────────────
# Tolerances
# ──────────────────────────────────────────────────────────────────────────────
TOL_WEIGHT = 0.5   # g/plant  – acceptable absolute error on weight
TOL_DAYS = 0.5     # days     – acceptable absolute error on grow days
TOL_WIDTH = 0.5    # mm       – acceptable absolute error on diameter
TOL_DENSITY = 0.5  # plants/m² – acceptable absolute error on density


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def g_avg() -> float:
    return GROWTH_PRESETS["average"]["speedFactor"]

def c_avg() -> float:
    return ARCHITECTURE_PRESETS["average"]["diameterFactor"]


# ──────────────────────────────────────────────────────────────────────────────
# Core function unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestWeightModel:
    """Tests for weight_from_effective_density and its inverse."""

    def test_weight_at_12(self):
        assert abs(weight_from_effective_density(12) - 350) < TOL_WEIGHT * 2

    def test_weight_at_43(self):
        assert abs(weight_from_effective_density(43) - 180) < TOL_WEIGHT

    def test_weight_at_200(self):
        assert abs(weight_from_effective_density(200) - 30) < TOL_WEIGHT

    def test_weight_at_400(self):
        assert abs(weight_from_effective_density(400) - 14.8) < TOL_WEIGHT

    def test_weight_at_800(self):
        assert abs(weight_from_effective_density(800) - 9.7) < TOL_WEIGHT

    def test_weight_at_1000(self):
        assert abs(weight_from_effective_density(1000) - 9.0) < TOL_WEIGHT

    def test_inverse_roundtrip(self):
        for E in [12, 43, 100, 200, 400, 800, 1000]:
            W = weight_from_effective_density(E)
            E2 = effective_density_from_weight(W)
            assert abs(E2 - E) < TOL_DENSITY, f"Round-trip failed at E={E}: got {E2}"

    def test_inverse_raises_below_asymptote(self):
        """W at or below WEIGHT_C is outside the domain."""
        with pytest.raises(ValueError):
            effective_density_from_weight(7.33)  # near WEIGHT_C

    def test_inverse_raises_negative(self):
        with pytest.raises(ValueError):
            effective_density_from_weight(0.0)


class TestDiameterModel:
    """Tests for base_diameter_from_effective_days and its inverse."""

    def test_diameter_positive_for_positive_days(self):
        for tau in [10, 20, 30, 40, 50]:
            d = base_diameter_from_effective_days(tau)
            assert d > 0, f"Diameter should be positive at tau={tau}"

    def test_diameter_increases_with_days(self):
        d_values = [base_diameter_from_effective_days(t) for t in [10, 20, 30, 40, 50]]
        for i in range(len(d_values) - 1):
            assert d_values[i] < d_values[i + 1], "Diameter should be monotonically increasing"

    def test_inverse_roundtrip(self):
        for dBase in [20, 50, 100, 150, 200, 250]:
            tau = effective_days_from_base_diameter(dBase)
            dBase2 = base_diameter_from_effective_days(tau)
            assert abs(dBase2 - dBase) < TOL_WIDTH, (
                f"Round-trip failed at dBase={dBase}: got {dBase2}"
            )

    def test_inverse_raises_at_lower_asymptote(self):
        from lettuce_growth_model_constants import DIAMETER_LOW
        with pytest.raises(ValueError):
            effective_days_from_base_diameter(DIAMETER_LOW)

    def test_inverse_raises_at_upper_asymptote(self):
        from lettuce_growth_model_constants import DIAMETER_HIGH
        with pytest.raises(ValueError):
            effective_days_from_base_diameter(DIAMETER_HIGH)


class TestSpacingFunctions:
    """Tests for the spacing <-> density conversion."""

    def test_width_from_density_at_43(self):
        d = width_from_actual_density(43)
        assert abs(d - 152.5) < TOL_WIDTH

    def test_round_trip_density_to_width(self):
        for N in [12, 43, 100, 200, 400, 800, 1000]:
            d = width_from_actual_density(N)
            N2 = actual_density_from_width(d)
            assert abs(N2 - N) < TOL_DENSITY, f"Round-trip failed at N={N}: got {N2}"


# ──────────────────────────────────────────────────────────────────────────────
# Baseline solver tests (average environment, average architecture)
# ──────────────────────────────────────────────────────────────────────────────

class TestBaselineDensityMode:
    """
    Verify Mode 1 (density input) against the calibrated reference values.

    Average growth (g=1.0) and Average architecture (c=1.0).
    """

    CASES = [
        (16,   323.8, 49.4, 250.0),
        (43,   180.0, 36.9, 152.5),
        (50,   156.2, 35.7, 141.4),
        (200,   30.0, 27.1,  70.7),
        (400,   14.8, 23.5,  50.0),
        (600,   11.2, 21.4,  40.8),
        (1000,   9.0, 18.9,  31.6),
    ]

    def test_all_cases(self):
        g = g_avg()
        c = c_avg()
        for N, expected_W, expected_t, expected_d in self.CASES:
            result = solve_from_density(N=N, g=g, c=c)
            assert abs(result.W - expected_W) < TOL_WEIGHT, (
                f"N={N}: weight {result.W:.2f} g ≠ expected {expected_W} g"
            )
            assert abs(result.t - expected_t) < TOL_DAYS, (
                f"N={N}: days {result.t:.2f} ≠ expected {expected_t} d"
            )
            assert abs(result.d - expected_d) < TOL_WIDTH, (
                f"N={N}: width {result.d:.2f} mm ≠ expected {expected_d} mm"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Round-trip tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRoundTrips:
    """Verify solver modes are mutually consistent."""

    def test_density_to_weight_to_density(self):
        g, c = g_avg(), c_avg()
        for N in [16, 43, 100, 200, 400]:
            r1 = solve_from_density(N=N, g=g, c=c)
            r2 = solve_from_weight(W=r1.W, g=g, c=c)
            assert abs(r2.N - N) < TOL_DENSITY, (
                f"Density→weight→density: N={N}, got back {r2.N:.2f}"
            )

    def test_density_to_days_to_density(self):
        g, c = g_avg(), c_avg()
        for N in [16, 43, 100, 200, 400]:
            r1 = solve_from_density(N=N, g=g, c=c)
            r2 = solve_from_days(t=r1.t, g=g, c=c)
            assert abs(r2.N - N) < TOL_DENSITY, (
                f"Density→days→density: N={N}, got back {r2.N:.2f}"
            )

    def test_weight_to_density_to_weight(self):
        g, c = g_avg(), c_avg()
        for W in [30, 80, 160, 250, 323]:
            r1 = solve_from_weight(W=W, g=g, c=c)
            r2 = solve_from_density(N=r1.N, g=g, c=c)
            assert abs(r2.W - W) < TOL_WEIGHT, (
                f"Weight→density→weight: W={W}, got back {r2.W:.2f}"
            )

    def test_days_to_density_to_days(self):
        g, c = g_avg(), c_avg()
        for t in [20, 30, 36.9, 45, 50]:
            r1 = solve_from_days(t=t, g=g, c=c)
            r2 = solve_from_density(N=r1.N, g=g, c=c)
            assert abs(r2.t - t) < TOL_DAYS, (
                f"Days→density→days: t={t}, got back {r2.t:.2f}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Modifier behaviour tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGrowthEnvironmentModifiers:
    """High growth must shorten calendar days; Low/Minimal must lengthen them."""

    def test_high_growth_reduces_days_at_fixed_density(self):
        c = c_avg()
        r_avg = solve_from_density(N=43, g=GROWTH_PRESETS["average"]["speedFactor"], c=c)
        r_high = solve_from_density(N=43, g=GROWTH_PRESETS["high"]["speedFactor"], c=c)
        assert r_high.t < r_avg.t, "High growth should reduce calendar days"

    def test_low_growth_increases_days_at_fixed_density(self):
        c = c_avg()
        r_avg = solve_from_density(N=43, g=GROWTH_PRESETS["average"]["speedFactor"], c=c)
        r_low = solve_from_density(N=43, g=GROWTH_PRESETS["low"]["speedFactor"], c=c)
        assert r_low.t > r_avg.t, "Low growth should increase calendar days"

    def test_minimal_growth_increases_days_at_fixed_density(self):
        c = c_avg()
        r_avg = solve_from_density(N=43, g=GROWTH_PRESETS["average"]["speedFactor"], c=c)
        r_min = solve_from_density(N=43, g=GROWTH_PRESETS["minimal"]["speedFactor"], c=c)
        assert r_min.t > r_avg.t, "Minimal growth should increase calendar days"

    def test_high_growth_reduces_days_at_fixed_weight(self):
        c = c_avg()
        r_avg = solve_from_weight(W=180, g=GROWTH_PRESETS["average"]["speedFactor"], c=c)
        r_high = solve_from_weight(W=180, g=GROWTH_PRESETS["high"]["speedFactor"], c=c)
        assert r_high.t < r_avg.t, "High growth should reduce calendar days for fixed weight"

    def test_ordering_high_avg_low_minimal(self):
        """Calendar days should increase from High → Average → Low → Minimal."""
        c = c_avg()
        days = {}
        for env in ["high", "average", "low", "minimal"]:
            g = GROWTH_PRESETS[env]["speedFactor"]
            days[env] = solve_from_density(N=43, g=g, c=c).t
        assert days["high"] < days["average"] < days["low"] < days["minimal"]


class TestArchitectureModifiers:
    """
    Compact architecture must allow greater actual density for a fixed weight.
    Open architecture must require lower actual density.

    Reference values for W = 180 g:
      Compact  → N ≈ 59.5 plants/m², d ≈ 129.6 mm
      Average  → N ≈ 43.0 plants/m², d ≈ 152.5 mm
      Open     → N ≈ 32.5 plants/m², d ≈ 175.4 mm
    """

    def test_compact_higher_density_than_average(self):
        g = g_avg()
        r_compact = solve_from_weight(W=180, g=g, c=ARCHITECTURE_PRESETS["compact"]["diameterFactor"])
        r_avg = solve_from_weight(W=180, g=g, c=ARCHITECTURE_PRESETS["average"]["diameterFactor"])
        assert r_compact.N > r_avg.N, "Compact should give higher density for same weight"

    def test_open_lower_density_than_average(self):
        g = g_avg()
        r_open = solve_from_weight(W=180, g=g, c=ARCHITECTURE_PRESETS["open"]["diameterFactor"])
        r_avg = solve_from_weight(W=180, g=g, c=ARCHITECTURE_PRESETS["average"]["diameterFactor"])
        assert r_open.N < r_avg.N, "Open should give lower density for same weight"

    def test_compact_reference_values(self):
        """Compact, W=180 g: N ≈ 59.5 plants/m², d ≈ 129.6 mm."""
        g = g_avg()
        c = ARCHITECTURE_PRESETS["compact"]["diameterFactor"]
        result = solve_from_weight(W=180, g=g, c=c)
        assert abs(result.N - 59.5) < 1.0, f"Compact N: expected ~59.5, got {result.N:.2f}"
        assert abs(result.d - 129.6) < 2.0, f"Compact d: expected ~129.6, got {result.d:.2f}"

    def test_average_reference_values(self):
        """Average, W=180 g: N ≈ 43 plants/m², d ≈ 152.5 mm."""
        g = g_avg()
        c = ARCHITECTURE_PRESETS["average"]["diameterFactor"]
        result = solve_from_weight(W=180, g=g, c=c)
        assert abs(result.N - 43.0) < 1.0, f"Average N: expected ~43, got {result.N:.2f}"
        assert abs(result.d - 152.5) < 2.0, f"Average d: expected ~152.5, got {result.d:.2f}"

    def test_open_reference_values(self):
        """Open, W=180 g: N ≈ 32.5 plants/m², d ≈ 175.4 mm."""
        g = g_avg()
        c = ARCHITECTURE_PRESETS["open"]["diameterFactor"]
        result = solve_from_weight(W=180, g=g, c=c)
        assert abs(result.N - 32.5) < 1.0, f"Open N: expected ~32.5, got {result.N:.2f}"
        assert abs(result.d - 175.4) < 2.0, f"Open d: expected ~175.4, got {result.d:.2f}"

    def test_same_weight_same_days_regardless_of_architecture(self):
        """
        At the same target weight and same grow environment, all architectures
        should produce the same physiological grow days (tau), because the
        effective density driving weight is the same relative to architecture.
        """
        g = g_avg()
        results = {
            k: solve_from_weight(W=180, g=g, c=ARCHITECTURE_PRESETS[k]["diameterFactor"])
            for k in ["compact", "average", "open"]
        }
        # tau = g * t must be the same (because dBase is the same for equal W)
        taus = [r.tau for r in results.values()]
        assert max(taus) - min(taus) < TOL_DAYS, (
            f"Physiological days should be equal across architectures for same W; "
            f"got taus = {taus}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Validation / warnings
# ──────────────────────────────────────────────────────────────────────────────

class TestValidation:
    """Verify that out-of-range inputs produce warnings, not silent clamping."""

    def test_high_density_triggers_warning(self):
        result = solve_from_density(N=1200, g=g_avg(), c=c_avg())
        assert len(result.warnings) > 0, "Should warn when E > 1000"

    def test_low_density_triggers_warning(self):
        result = solve_from_density(N=5, g=g_avg(), c=c_avg())
        assert len(result.warnings) > 0, "Should warn when E < 12"

    def test_very_high_weight_triggers_warning(self):
        # Solve from low density that gives weight > 350 g
        result = solve_from_density(N=8, g=g_avg(), c=c_avg())
        assert len(result.warnings) > 0, "Should warn when W > 350 g"

    def test_no_warning_in_calibrated_range(self):
        # Density 43, average environment and architecture → all in range
        result = solve_from_density(N=43, g=g_avg(), c=c_avg())
        assert result.warnings == [], (
            f"No warning expected for N=43 avg/avg, got: {result.warnings}"
        )

    def test_weight_domain_error(self):
        with pytest.raises(ValueError):
            solve_from_weight(W=7.0, g=g_avg(), c=c_avg())


# ──────────────────────────────────────────────────────────────────────────────
# Product category
# ──────────────────────────────────────────────────────────────────────────────

class TestProductCategory:
    def test_full_head(self):
        assert _product_category(W=200, N=43) == "Full head"

    def test_teen_leaf(self):
        assert _product_category(W=100, N=150) == "Teen leaf"

    def test_small_leaf_by_weight(self):
        result = solve_from_density(N=250, g=g_avg(), c=c_avg())
        assert result.category in ("Small leaf", "Teen leaf", "Transition")

    def test_baby_leaf_by_density(self):
        result = solve_from_density(N=900, g=g_avg(), c=c_avg())
        assert result.category in ("Baby leaf", "Transition")

    def test_belgian_full_head(self):
        assert _product_category(W=300, N=15) == "Belgian / full head"
