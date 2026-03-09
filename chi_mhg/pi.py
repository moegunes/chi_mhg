"""Interpolated polarisation function Π(r, rₛ) of the HEG.

Physics
-------
The polarisation function of the homogeneous electron gas is defined via
the screened Coulomb interaction v_c = 4π/(q² + κ²) with κ = 0.0225 :

    Π(q) = χ₀(q) / [1 − χ₀(q) (v_c(q) + f_xc(q))]

In real space the interpolation uses the same two-damped-cosine ansatz
as for χ but with moment constraints derived from the screened kernel:

    Π(r, rₛ) = χ₀(r, rₛ) + (−6π n₀ NF) ΔΠ(r, rₛ)

The six shape parameters are interpolated using the modified Padé [3/4]
form in rₛ (10 coefficients each, 60 total).
"""

from __future__ import annotations

import warnings

import numpy as np

from ._data import PARAM_NAMES, PI_COEFFICIENTS, PI_RS_RANGE
from ._helpers import (
    _chi0_moment,
    _evaluate_two_mode,
    _gas_params,
    chi0_heg,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_KAPPA: float = 0.0225  # screening parameter

_F_MIN: float = 0.02  # minimum frequency (physical constraint)
_ALPHA_MIN: float = 1e-4  # minimum damping (physical constraint)


# ---------------------------------------------------------------------------
# mPZ[3/4] parametric form
# ---------------------------------------------------------------------------


def _mpz34(rs: float, c: np.ndarray) -> float:
    """Evaluate mPZ[3/4] at *rs* with 10 coefficients *c*.

    f(rs) = j + (a + b·rs + c·rs² + d·rs³ + i·rs⁴)
                / (1 + e·rs + f·rs² + g·rs³ + h·rs⁴)

    Parameters in *c*: [a, b, c, d, e, f, g, h, i, j].
    """
    a, b, cc, d, e, f, g, h, i, j = c
    return j + (a + b * rs + cc * rs**2 + d * rs**3 + i * rs**4) / (
        1.0 + e * rs + f * rs**2 + g * rs**3 + h * rs**4
    )


def _interpolate_pi_params(rs: float) -> np.ndarray:
    """Interpolate the 6 physical parameters at *rs* for Π.

    Applies physical constraints: frequencies ≥ F_MIN, damping > 0.
    """
    params = []
    for p in PARAM_NAMES:
        val = _mpz34(rs, PI_COEFFICIENTS[p])
        if p in ("f0", "f1"):
            val = max(val, _F_MIN)
        if p in ("alpha0", "alpha1"):
            val = max(val, _ALPHA_MIN)
        params.append(val)
    return np.array(params)


# ---------------------------------------------------------------------------
# Corradini–PZ local-field factor  (private, self-contained)
# ---------------------------------------------------------------------------


def _diffv_cep(rs: float) -> float:
    """d(rₛ εc)/drₛ using Perdew–Zunger correlation."""
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    denom = beta1 * np.sqrt(rs) + beta2 * rs + 1.0
    return (beta1 * gamma * np.sqrt(rs)) / (2.0 * denom**2) + gamma / denom**2


def _diffvc(rho: float):
    """d²(n εc)/dn² for the Perdew–Zunger parametrisation."""
    third = 1.0 / 3.0
    a = 0.0311
    c = 0.0020
    d = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334

    rs = (3.0 / (4.0 * np.pi * rho)) ** third

    stor1 = (1.0 + beta1 * np.sqrt(rs) + beta2 * rs) ** (-3.0)
    stor2 = (
        -0.41666667 * beta1 * (rs ** (-0.5))
        - 0.5833333 * (beta1**2)
        - 0.66666667 * beta2
    )
    stor3 = -1.75 * beta1 * beta2 * np.sqrt(rs) - 1.3333333 * rs * (beta2**2)
    reshigh = gamma * stor1 * (stor2 + stor3)
    reslow = a / rs + 0.66666667 * (c * np.log(rs) + d) + 0.33333333 * c

    reshigh = reshigh * (-4.0 * np.pi / 9.0) * (rs**4)
    reslow = reslow * (-4.0 * np.pi / 9.0) * (rs**4)

    filterlow = rs < 1
    filterhigh = rs >= 1
    return reslow * filterlow + reshigh * filterhigh


def _corradini_pz(rs: float, q):
    """Corradini–Perdew–Zunger local-field factor G(q).

    Returns −4π/q² · G_cor(q), i.e. the exchange-correlation kernel f_xc(q).
    """
    q = np.asarray(q, dtype=float) + 1e-18
    rho = 3.0 / (4.0 * np.pi * rs**3)
    kF = (3.0 * np.pi**2 * rho) ** (1.0 / 3.0)
    Q = q / kF

    diff_mu = _diffvc(rho)
    A = 0.25 - (kF**2) / (4.0 * np.pi) * diff_mu

    diff_rse = _diffv_cep(rs)
    C = np.pi / (2.0 * kF) * (-diff_rse)

    a1, a2 = 2.15, 0.435
    b1, b2 = 1.57, 0.409
    x = np.sqrt(rs)
    B = (1.0 + a1 * x + a2 * x**3) / (3.0 + b1 * x + b2 * x**3)
    g = B / (A - C)
    alpha = 1.5 / (rs**0.25) * A / (B * g)
    beta = 1.2 / (B * g)

    Gcor = C * Q**2 + (B * Q**2) / (g + Q**2) + alpha * Q**4 * np.exp(-beta * Q**2)
    return -4.0 * np.pi / (q**2) * Gcor


# ---------------------------------------------------------------------------
# Π moment constraints
# ---------------------------------------------------------------------------


def _pi_moment(n: int, rs: float) -> float:
    """n-th moment of Π(q) = χ₀/(1 − χ₀(v_c^κ + f_xc)).

    Only n = 0 and n = 1 are implemented (sufficient for the 2×2 system).
    """
    kF = (9.0 * np.pi / 4.0) ** (1.0 / 3.0) / rs
    K = _KAPPA
    pi_ = np.pi

    f0_fxc = float(_corradini_pz(rs, 0.0))
    D = 4.0 * kF * pi_ + (f0_fxc * kF + pi_**2) * K**2

    if n == 0:
        return -kF * K**2 / (4.0 * pi_ * D)

    if n == 1:
        # Second derivative of f_xc at q = 0 (numerical)
        dq = 1e-3
        q3 = np.array([-dq, 0.0, dq])
        fvals = _corradini_pz(rs, q3)
        f2 = 0.5 * float(fvals[2] - 2.0 * fvals[1] + fvals[0]) / dq**2

        num = -(pi_**2 * K**4 + 12.0 * kF**3 * (-4.0 * pi_ + f2 * K**4))
        return float(num / (8.0 * kF * pi_ * D**2))

    raise ValueError(f"Only n=0,1 implemented; got n={n}")


def _pi_delta_C(n: int, kF: float, n0: float, NF: float) -> float:
    """Moment constraint RHS for Π: (C_Π(n) − C_χ₀(n)) / (−6π n₀ NF)."""
    factor = -6.0 * np.pi * n0 * NF
    rs = (9.0 * np.pi / 4.0) ** (1.0 / 3.0) / kF
    return (_pi_moment(n, rs) - _chi0_moment(n, kF)) / factor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def delta_pi_mhg(r, rs: float):
    r"""Interacting correction ΔΠ(r, rₛ) from the two-damped-cosine model.

    Same functional form as Δχ but with moment constraints derived from
    the screened polarisation function Π(q).

    Parameters
    ----------
    r : array_like
        Distance(s) in Bohr.
    rs : float
        Wigner-Seitz radius.

    Returns
    -------
    dpi : ndarray
        Dimensionless ΔΠ(r) (multiply by −6π n₀ NF for physical units).
    """
    kF, n0, NF = _gas_params(rs)
    params = _interpolate_pi_params(rs)
    dc0 = _pi_delta_C(0, kF, n0, NF)
    dc1 = _pi_delta_C(1, kF, n0, NF)
    return _evaluate_two_mode(r, kF, params, dc0, dc1)


def pi_mhg(r, rs: float):
    r"""Interpolated polarisation function of the homogeneous electron gas
    in real space.

    Combines the analytic Lindhard function χ₀(r) with a QMC-constrained
    correction ΔΠ(r):

    .. math::
        \Pi(r, r_s) = \chi_0(r, r_s) + (-6\pi\,n_0\,N_F)\,\Delta\Pi(r, r_s)

    The six shape parameters of the two-damped-cosine model for ΔΠ are
    interpolated in *rₛ* using a modified Padé [3/4] form fitted to
    QMC-constrained data at 51 electron densities.

    Parameters
    ----------
    r : array_like
        Real-space distance(s) in Bohr.
    rs : float
        Wigner-Seitz radius. Recommended range: 0.2 ≤ rₛ ≤ 10.0.

    Returns
    -------
    pi : ndarray
        Π(r, rₛ) in atomic units.

    Notes
    -----
    60 meta-parameters (10 per physical quantity × 6 quantities) fully
    determine Π for any (r, rₛ) pair.  No Fourier transforms are used;
    evaluation is O(len(r)).

    The screened Coulomb kernel uses κ = 0.0225.

    References
    ----------
    .. Güneş, Holzmann, & Pedroza (2025). Interpolation of the
       density-density response function of the homogeneous electron gas.
    """
    r = np.asarray(r, dtype=float)

    if rs < PI_RS_RANGE[0] or rs > PI_RS_RANGE[1]:
        warnings.warn(
            f"rs={rs} is outside the fitted range [{PI_RS_RANGE[0]}, {PI_RS_RANGE[1]}]. "
            "Results may be unreliable.",
            stacklevel=2,
        )

    kF, n0, NF = _gas_params(rs)
    factor = -6.0 * np.pi * n0 * NF

    return chi0_heg(r, rs) + factor * delta_pi_mhg(r, rs)
