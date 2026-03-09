"""chi_mhg — Interpolated real-space response function of the homogeneous electron gas.

Usage::

    from chi_mhg import chi_mhg
    import numpy as np

    r = np.linspace(0.01, 30, 1000)
    chi = chi_mhg(r, rs=4.0)

"""

from .chi import chi0_heg, chi_mhg, delta_chi_mhg
from .pi import delta_pi_mhg, pi_mhg

__all__ = ["chi_mhg", "pi_mhg", "chi0_heg", "delta_chi_mhg", "delta_pi_mhg"]
__version__ = "0.2.1"
