"""Core implementation of the interpolated HEG response function χ(r, rₛ).

Physics
-------
The density-density response function of the homogeneous electron gas
is decomposed as:

    χ(r, rₛ) = χ₀(r, rₛ) + (-6π n₀ NF) Δχ(r, rₛ)

where χ₀ is the non-interacting Lindhard function in real space and Δχ
is a two-damped-cosine correction whose parameters are interpolated
from QMC-constrained fits using the modified Padé [2/3] form.
"""

from __future__ import annotations

import warnings
from math import factorial

import numpy as np

from ._data import COEFFICIENTS, PARAM_NAMES, RS_RANGE

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gas_params(rs: float) -> tuple[float, float, float]:
    """Return (kF, n₀, NF) for the HEG at given Wigner-Seitz radius *rs*."""
    n0 = 3.0 / (4.0 * np.pi * rs**3)
    kF = (3.0 * np.pi**2 * n0) ** (1.0 / 3.0)
    NF = kF / np.pi**2
    return kF, n0, NF


def _mpz23(rs: float, c: np.ndarray) -> float:
    """Evaluate the mPZ[2/3] form at *rs* with coefficients *c*.

    f(rs) = g + (a + b·rs + c·rs² + h·rs³) / (1 + d·rs + e·rs² + f·rs³)

    Parameters in *c*: [a, b, c, d, e, f, g, h].
    """
    a, b, cc, d, e, f, g, h = c
    return g + (a + b * rs + cc * rs**2 + h * rs**3) / (
        1.0 + d * rs + e * rs**2 + f * rs**3
    )


def _interpolate_params(rs: float) -> np.ndarray:
    """Interpolate the 6 physical parameters (α₀, f₀, φ₀, α₁, f₁, φ₁) at *rs*."""
    return np.array([_mpz23(rs, COEFFICIENTS[p]) for p in PARAM_NAMES])


def _J_n_m_kFr(n: int, k: float, gamma: float, phi: float, kF: float) -> float:
    r"""Analytic moment integral.

    .. math::
        J_n = (2n+2)!\;\mathrm{Re}\!\left[
            \frac{e^{i\varphi}}{(\gamma - ik)^{2n+3}\,k_F^{2n+3}}
        \right]
    """
    return factorial(2 * n + 2) * np.real(
        np.exp(1j * phi) / (gamma - 1j * k) ** (2 * n + 3) / kF ** (2 * n + 3)
    )


def _delta_C(n: int, kF: float, n0: float, NF: float) -> float:
    """Right-hand side of the moment constraint for order *n* (n = 0 or 1).

    δC(n) = [M_χ(n) − M_χ₀(n)] / (−6π n₀ NF)

    where M_χ(n), M_χ₀(n) are the n-th frequency moments of the interacting
    and non-interacting response functions respectively.
    """
    factor = -6.0 * np.pi * n0 * NF

    # Non-interacting (Lindhard) moments
    if n == 0:
        chi0_m = -kF / (4.0 * np.pi**3)
    elif n == 1:
        chi0_m = -1.0 / (8.0 * np.pi**3 * kF)
    else:
        raise ValueError(f"Only n=0,1 implemented; got n={n}")

    # Interacting (exact) moments
    if n == 0:
        chi_m = 0.0
    elif n == 1:
        chi_m = 3.0 / (8.0 * np.pi**2)

    return (chi_m - chi0_m) / factor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chi0_heg(r, rs: float):
    r"""Non-interacting (Lindhard) response function in real space.

    .. math::
        \chi_0(r) = -6\pi\,n_0\,N_F\,
            \frac{\sin(2k_F r) - 2k_F r\,\cos(2k_F r)}{(2k_F r)^4}

    Parameters
    ----------
    r : array_like
        Distance(s) in Bohr.
    rs : float
        Wigner-Seitz radius.

    Returns
    -------
    chi0 : ndarray
        χ₀(r, rₛ) in atomic units.
    """
    r = np.asarray(r, dtype=float)
    kF, n0, NF = _gas_params(rs)
    factor = -6.0 * np.pi * n0 * NF
    x = 2.0 * kF * r

    with np.errstate(divide="ignore", invalid="ignore"):
        chi0 = np.where(
            x == 0.0,
            0.0,
            factor * (np.sin(x) - x * np.cos(x)) / x**4,
        )
    return chi0


def delta_chi_mhg(r, rs: float):
    r"""Interacting correction Δχ(r, rₛ) from the two-damped-cosine model.

    .. math::
        \Delta\chi(r) = B_0\,e^{-\alpha_0 k_F r}\cos(2\pi f_0\,k_F r + \varphi_0)
                      + B_1\,e^{-\alpha_1 k_F r}\cos(2\pi f_1\,k_F r + \varphi_1)

    where *B₀, B₁* are determined from the zeroth and first moment
    constraints.

    Parameters
    ----------
    r : array_like
        Distance(s) in Bohr.
    rs : float
        Wigner-Seitz radius.

    Returns
    -------
    dchi : ndarray
        Dimensionless Δχ(r) (to be multiplied by −6π n₀ NF to get
        physical units).
    """
    r = np.asarray(r, dtype=float)
    kF, n0, NF = _gas_params(rs)
    params = _interpolate_params(rs)

    alpha0, f0, phi0, alpha1, f1, phi1 = params
    k0 = 2.0 * np.pi * f0
    k1 = 2.0 * np.pi * f1
    kFr = kF * r

    # Build 2×2 moment matrix and solve for amplitudes B0, B1
    J_mat = np.array(
        [
            [
                _J_n_m_kFr(1, k0, alpha0, phi0, kF),
                _J_n_m_kFr(1, k1, alpha1, phi1, kF),
            ],
            [
                _J_n_m_kFr(0, k0, alpha0, phi0, kF),
                _J_n_m_kFr(0, k1, alpha1, phi1, kF),
            ],
        ]
    )
    rhs = np.array([_delta_C(1, kF, n0, NF), _delta_C(0, kF, n0, NF)])
    B0, B1 = np.linalg.solve(J_mat, rhs)

    dchi = B0 * np.exp(-alpha0 * kFr) * np.cos(k0 * kFr + phi0) + B1 * np.exp(
        -alpha1 * kFr
    ) * np.cos(k1 * kFr + phi1)
    return dchi


def chi_mhg(r, rs: float):
    r"""Interpolated density-density response function of the homogeneous
    electron gas in real space.

    Combines the analytic Lindhard function χ₀(r) with a QMC-constrained
    interacting correction Δχ(r):

    .. math::
        \chi(r, r_s) = \chi_0(r, r_s) + (-6\pi\,n_0\,N_F)\,\Delta\chi(r, r_s)

    The six shape parameters of the two-damped-cosine model for Δχ are
    interpolated in *rₛ* using a modified Padé [2/3] form fitted to
    QMC-constrained data at 51 electron densities.

    Parameters
    ----------
    r : array_like
        Real-space distance(s) in Bohr.
    rs : float
        Wigner-Seitz radius. Recommended range: 0.5 ≤ rₛ ≤ 10.0.

    Returns
    -------
    chi : ndarray
        χ(r, rₛ) in atomic units.

    Notes
    -----
    48 meta-parameters (8 per physical quantity × 6 quantities) fully
    determine χ for any (r, rₛ) pair.  No Fourier transforms are used;
    evaluation is O(len(r)).

    References
    ----------
    .. Güneş, Holzmann, & Pedroza (2025). Interpolation of the
       density-density response function of the homogeneous electron gas.
    """
    r = np.asarray(r, dtype=float)

    if rs < RS_RANGE[0] or rs > RS_RANGE[1]:
        warnings.warn(
            f"rs={rs} is outside the fitted range [{RS_RANGE[0]}, {RS_RANGE[1]}]. "
            "Results may be unreliable.",
            stacklevel=2,
        )

    kF, n0, NF = _gas_params(rs)
    factor = -6.0 * np.pi * n0 * NF

    # --- χ₀ (Lindhard, analytic) ---
    x = 2.0 * kF * r
    with np.errstate(divide="ignore", invalid="ignore"):
        chi0 = np.where(
            x == 0.0,
            0.0,
            factor * (np.sin(x) - x * np.cos(x)) / x**4,
        )

    # --- Δχ (interpolated two-damped-cosine model) ---
    params = _interpolate_params(rs)
    alpha0, f0, phi0, alpha1, f1, phi1 = params
    k0 = 2.0 * np.pi * f0
    k1 = 2.0 * np.pi * f1
    kFr = kF * r

    # Moment constraint: solve for amplitudes B₀, B₁
    J_mat = np.array(
        [
            [
                _J_n_m_kFr(1, k0, alpha0, phi0, kF),
                _J_n_m_kFr(1, k1, alpha1, phi1, kF),
            ],
            [
                _J_n_m_kFr(0, k0, alpha0, phi0, kF),
                _J_n_m_kFr(0, k1, alpha1, phi1, kF),
            ],
        ]
    )
    rhs = np.array([_delta_C(1, kF, n0, NF), _delta_C(0, kF, n0, NF)])
    B0, B1 = np.linalg.solve(J_mat, rhs)

    dchi = B0 * np.exp(-alpha0 * kFr) * np.cos(k0 * kFr + phi0) + B1 * np.exp(
        -alpha1 * kFr
    ) * np.cos(k1 * kFr + phi1)

    return chi0 + factor * dchi
