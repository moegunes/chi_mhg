"""chi_mhg — Interpolated real-space response functions of the homogeneous electron gas.

Usage::

    from chi_mhg import chi_mhg, pi_mhg
    import numpy as np

    r = np.linspace(0.01, 30, 1000)
    chi = chi_mhg(r, rs=4.0)   # density-density response (bare Coulomb)
    pi  = pi_mhg(r, rs=4.0)    # polarisation function (screened Coulomb)

"""

from ._helpers import chi0_heg
from .chi import chi_mhg, delta_chi_mhg
from .pi import delta_pi_mhg, pi_mhg

__all__ = ["chi_mhg", "chi0_heg", "delta_chi_mhg", "pi_mhg", "delta_pi_mhg"]
__version__ = "0.2.0"
