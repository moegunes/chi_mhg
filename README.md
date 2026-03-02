# chi_mhg

Interpolated real-space density-density response function $\chi(r, r_s)$ of the homogeneous electron gas (HEG).

## Installation

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

## Quick start

```python
from chi_mhg import chi_mhg
import numpy as np

r = np.linspace(0.01, 30, 1000)   # distance in Bohr
chi = chi_mhg(r, rs=4.0)          # χ(r) in atomic units
```

Three functions are exported:

| Function | Description |
|---|---|
| `chi_mhg(r, rs)` | Full interacting χ(r, rₛ) |
| `chi0_heg(r, rs)` | Non-interacting Lindhard χ₀(r, rₛ) |
| `delta_chi_mhg(r, rs)` | Dimensionless correction Δχ(r, rₛ) |

The full response is:

$$\chi(r, r_s) = \chi_0(r, r_s) + (-6\pi\,n_0\,N_F)\,\Delta\chi(r, r_s)$$

## Method

The interacting correction $\Delta\chi(r)$ is modeled as two damped cosines:

$$\Delta\chi(r) = B_0\,e^{-\alpha_0 k_F r}\cos(k_0k_F r + \varphi_0)
    + B_1\,e^{-\alpha_1 k_F r}\cos(k_1k_F r + \varphi_1)$$

where:
- The **six shape parameters** $(\alpha_0, f_0, \varphi_0, \alpha_1, f_1, \varphi_1)$ are fitted at each $r_s$ to reproduce the QMC-constrained response function (Corradini–Del Sole–Moroni–Perdew-Zunger local field factor).
- The **amplitudes** $B_0, B_1$ are fixed by the zeroth and first frequency-moment sum rules.
- Each shape parameter is **interpolated** across $r_s$ using a modified Padé [2/3] form in $\sqrt{r_s}$ (mPZ[2/3]√):
$$s=\sqrt{r_s},\qquad p(r_s) = g + \frac{a + b\,s + c\,s^2 + h\,s^3}{1 + d\,s + e\,s^2 + f\,s^3}$$

The current release uses this mPZ[2/3]√ interpolation as the default coefficient set.

This gives **48 meta-parameters** (8 coefficients × 6 quantities) that fully determine $\chi(r, r_s)$ for any $(r, r_s)$ pair without Fourier transforms.

## Valid range

$0.5 \leq r_s \leq 10.0$ (a warning is emitted outside this range).

## Dependencies

- NumPy ≥ 1.24

## Reference

Güneş, Holzmann, & Pedroza (2025). *Interpolation of the density-density response function of the homogeneous electron gas.*

## License

MIT
