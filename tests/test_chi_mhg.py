"""Tests for the chi_mhg module."""

import numpy as np
import pytest

# ── χ tests ───────────────────────────────────────────────────────────────


def test_import():
    """Module imports and exposes the public API."""
    from chi_mhg import chi0_heg, chi_mhg, delta_chi_mhg, delta_pi_mhg, pi_mhg

    assert callable(chi_mhg)
    assert callable(chi0_heg)
    assert callable(delta_chi_mhg)
    assert callable(pi_mhg)
    assert callable(delta_pi_mhg)


def test_chi0_large_r():
    """χ₀ should decay for large r (Friedel oscillations)."""
    from chi_mhg import chi0_heg

    r = np.linspace(20, 50, 100)
    chi0 = chi0_heg(r, rs=4.0)
    # Should be small in magnitude at large r
    assert np.all(np.abs(chi0) < 1.0)


def test_chi0_shape():
    """χ₀ returns array with same shape as input."""
    from chi_mhg import chi0_heg

    r = np.array([1.0, 2.0, 3.0])
    chi0 = chi0_heg(r, rs=2.0)
    assert chi0.shape == r.shape


def test_chi_mhg_shape():
    """chi_mhg returns array with same shape as input."""
    from chi_mhg import chi_mhg

    r = np.linspace(0.1, 30, 500)
    chi = chi_mhg(r, rs=5.0)
    assert chi.shape == r.shape
    assert np.all(np.isfinite(chi))


def test_delta_chi_decays():
    """Δχ should decay to zero at large r."""
    from chi_mhg import delta_chi_mhg

    r = np.linspace(50, 100, 100)
    dchi = delta_chi_mhg(r, rs=3.0)
    assert np.max(np.abs(dchi)) < 1e-4


def test_chi_decomposition():
    """chi_mhg ≈ chi0 + factor * delta_chi."""
    from chi_mhg import chi0_heg, chi_mhg, delta_chi_mhg
    from chi_mhg._helpers import _gas_params

    rs = 4.0
    r = np.linspace(0.5, 30, 200)
    kF, n0, NF = _gas_params(rs)
    factor = -6.0 * np.pi * n0 * NF

    chi = chi_mhg(r, rs)
    chi0 = chi0_heg(r, rs)
    dchi = delta_chi_mhg(r, rs)

    np.testing.assert_allclose(chi, chi0 + factor * dchi, rtol=1e-12)


def test_chi_rs_warning():
    """Should warn for rs outside [0.5, 10]."""
    from chi_mhg import chi_mhg

    r = np.array([1.0])
    with pytest.warns(UserWarning, match="outside the fitted range"):
        chi_mhg(r, rs=0.1)
    with pytest.warns(UserWarning, match="outside the fitted range"):
        chi_mhg(r, rs=20.0)


def test_chi_scalar_r():
    """Works with scalar r input."""
    from chi_mhg import chi_mhg

    chi = chi_mhg(5.0, rs=3.0)
    assert np.isfinite(chi)


def test_chi_several_rs():
    """Produces finite results across the valid rs range."""
    from chi_mhg import chi_mhg

    r = np.linspace(0.1, 20, 100)
    for rs in [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        chi = chi_mhg(r, rs)
        assert np.all(np.isfinite(chi)), f"NaN/Inf at rs={rs}"


# ── Π tests ───────────────────────────────────────────────────────────────


def test_pi_mhg_shape():
    """pi_mhg returns array with same shape as input."""
    from chi_mhg import pi_mhg

    r = np.linspace(0.1, 30, 500)
    pi = pi_mhg(r, rs=5.0)
    assert pi.shape == r.shape
    assert np.all(np.isfinite(pi))


def test_delta_pi_decays():
    """ΔΠ should decay to zero at large r."""
    from chi_mhg import delta_pi_mhg

    r = np.linspace(50, 100, 100)
    dpi = delta_pi_mhg(r, rs=3.0)
    assert np.max(np.abs(dpi)) < 1e-4


def test_pi_decomposition():
    """pi_mhg ≈ chi0 + factor * delta_pi."""
    from chi_mhg import chi0_heg, delta_pi_mhg, pi_mhg
    from chi_mhg._helpers import _gas_params

    rs = 4.0
    r = np.linspace(0.5, 30, 200)
    kF, n0, NF = _gas_params(rs)
    factor = -6.0 * np.pi * n0 * NF

    pi = pi_mhg(r, rs)
    chi0 = chi0_heg(r, rs)
    dpi = delta_pi_mhg(r, rs)

    np.testing.assert_allclose(pi, chi0 + factor * dpi, rtol=1e-12)


def test_pi_rs_warning():
    """Should warn for rs outside [0.2, 10]."""
    from chi_mhg import pi_mhg

    r = np.array([1.0])
    with pytest.warns(UserWarning, match="outside the fitted range"):
        pi_mhg(r, rs=0.1)
    with pytest.warns(UserWarning, match="outside the fitted range"):
        pi_mhg(r, rs=20.0)


def test_pi_scalar_r():
    """Works with scalar r input."""
    from chi_mhg import pi_mhg

    pi = pi_mhg(5.0, rs=3.0)
    assert np.isfinite(pi)


def test_pi_several_rs():
    """Produces finite results across the valid rs range."""
    from chi_mhg import pi_mhg

    r = np.linspace(0.1, 20, 100)
    for rs in [0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        pi = pi_mhg(r, rs)
        assert np.all(np.isfinite(pi)), f"NaN/Inf at rs={rs}"


def test_pi_differs_from_chi():
    """Π and χ should differ (different Coulomb kernel)."""
    from chi_mhg import chi_mhg, pi_mhg

    r = np.linspace(0.5, 20, 200)
    rs = 4.0
    chi = chi_mhg(r, rs)
    pi = pi_mhg(r, rs)
    # They share chi0 but have different corrections
    assert not np.allclose(chi, pi, atol=1e-10)
