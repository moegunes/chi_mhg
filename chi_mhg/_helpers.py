"""Shared building blocks for χ and Π interpolation.

Contains the non-interacting Lindhard function and moment-integral
machinery used by both ``chi.py`` and ``pi.py``.
"""

from __future__ import annotations

from math import factorial

import numpy as np

# ---------------------------------------------------------------------------
# HEG gas parameters
# ---------------------------------------------------------------------------


def _gas_params(rs: float) -> tuple[float, float, float]:
    """Return (kF, n₀, NF) for the HEG at Wigner-Seitz radius *rs*."""
    n0 = 3.0 / (4.0 * np.pi * rs**3)
    kF = (3.0 * np.pi**2 * n0) ** (1.0 / 3.0)
    NF = kF / np.pi**2
    return kF, n0, NF


# ---------------------------------------------------------------------------
# Moment integrals
# ---------------------------------------------------------------------------


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


def _chi0_moment(n: int, kF: float) -> float:
    """Non-interacting (Lindhard) n-th frequency moment."""
    if n == 0:
        return -kF / (4.0 * np.pi**3)
    if n == 1:
        return -1.0 / (8.0 * np.pi**3 * kF)
    raise ValueError(f"Only n=0,1 implemented; got n={n}")


# ---------------------------------------------------------------------------
# Two-damped-cosine model evaluation
# ---------------------------------------------------------------------------


def _evaluate_two_mode(
    r,
    kF: float,
    params: np.ndarray,
    delta_c0: float,
    delta_c1: float,
) -> np.ndarray:
    r"""Evaluate Δχ (or Δπ) with the two-damped-cosine ansatz.

    .. math::
        \Delta(r) = B_0\,e^{-\alpha_0 k_F r}\cos(k_0 k_F r + \varphi_0)
                  + B_1\,e^{-\alpha_1 k_F r}\cos(k_1 k_F r + \varphi_1)

    where *B₀, B₁* are determined by the zeroth (*delta_c0*) and first
    (*delta_c1*) moment constraint values.

    Parameters
    ----------
    r : array_like
        Real-space distances in Bohr.
    kF : float
        Fermi wave vector.
    params : array of 6 floats
        [α₀, f₀, φ₀, α₁, f₁, φ₁].
    delta_c0, delta_c1 : float
        Right-hand sides of the 0th and 1st moment constraints.
    """
    r = np.asarray(r, dtype=float)
    alpha0, f0, phi0, alpha1, f1, phi1 = params
    k0 = 2.0 * np.pi * f0
    k1 = 2.0 * np.pi * f1
    kFr = kF * r

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
    rhs = np.array([delta_c1, delta_c0])
    B0, B1 = np.linalg.solve(J_mat, rhs)

    return B0 * np.exp(-alpha0 * kFr) * np.cos(k0 * kFr + phi0) + B1 * np.exp(
        -alpha1 * kFr
    ) * np.cos(k1 * kFr + phi1)


# ---------------------------------------------------------------------------
# Non-interacting (Lindhard) response — public API
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
