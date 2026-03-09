"""Interpolated density-density response function χ(r, rₛ) of the HEG.

Physics
-------
The density-density response function of the homogeneous electron gas
is decomposed as:

    χ(r, rₛ) = χ₀(r, rₛ) + (-6π n₀ NF) Δχ(r, rₛ)

where χ₀ is the non-interacting Lindhard function in real space and Δχ
is a two-damped-cosine correction whose parameters are interpolated
from QMC-constrained fits using the modified Padé [2/3] form in √rₛ.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._data import CHI_COEFFICIENTS, CHI_RS_RANGE, PARAM_NAMES
from ._helpers import (
    _chi0_moment,
    _evaluate_two_mode,
    _gas_params,
    chi0_heg,  # noqa: F401 — re-exported
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mpz23_sqrt(rs: float, c: np.ndarray) -> float:
    """Evaluate the mPZ[2/3]√ form at *rs* with coefficients *c*.

    s = √rs
    f(rs) = g + (a + b·s + c·s² + h·s³) / (1 + d·s + e·s² + f·s³)

    Parameters in *c*: [a, b, c, d, e, f, g, h].
    """
    a, b, cc, d, e, f, g, h = c
    s = np.sqrt(rs)
    return g + (a + b * s + cc * s**2 + h * s**3) / (1.0 + d * s + e * s**2 + f * s**3)


def _interpolate_chi_params(rs: float) -> np.ndarray:
    """Interpolate the 6 physical parameters at *rs* for χ."""
    return np.array([_mpz23_sqrt(rs, CHI_COEFFICIENTS[p]) for p in PARAM_NAMES])


def _chi_delta_C(n: int, kF: float, n0: float, NF: float) -> float:
    """Moment constraint RHS for χ: (C_χ(n) − C_χ₀(n)) / (−6π n₀ NF)."""
    factor = -6.0 * np.pi * n0 * NF
    chi0_m = _chi0_moment(n, kF)
    if n == 0:
        chi_m = 0.0
    elif n == 1:
        chi_m = 3.0 / (8.0 * np.pi**2)
    else:
        raise ValueError(f"Only n=0,1 implemented; got n={n}")
    return (chi_m - chi0_m) / factor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    kF, n0, NF = _gas_params(rs)
    params = _interpolate_chi_params(rs)
    dc0 = _chi_delta_C(0, kF, n0, NF)
    dc1 = _chi_delta_C(1, kF, n0, NF)
    return _evaluate_two_mode(r, kF, params, dc0, dc1)


def chi_mhg(r, rs: float):
    r"""Interpolated density-density response function of the homogeneous
    electron gas in real space.

    Combines the analytic Lindhard function χ₀(r) with a QMC-constrained
    interacting correction Δχ(r):

    .. math::
        \chi(r, r_s) = \chi_0(r, r_s) + (-6\pi\,n_0\,N_F)\,\Delta\chi(r, r_s)

    The six shape parameters of the two-damped-cosine model for Δχ are
    interpolated in *rₛ* using a modified Padé [2/3] form in √rₛ fitted to
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

    if rs < CHI_RS_RANGE[0] or rs > CHI_RS_RANGE[1]:
        warnings.warn(
            f"rs={rs} is outside the fitted range [{CHI_RS_RANGE[0]}, {CHI_RS_RANGE[1]}]. "
            "Results may be unreliable.",
            stacklevel=2,
        )

    kF, n0, NF = _gas_params(rs)
    factor = -6.0 * np.pi * n0 * NF

    return chi0_heg(r, rs) + factor * delta_chi_mhg(r, rs)
