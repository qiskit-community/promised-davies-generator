# Supplementary Material for "Thermal State Preparation via Rounding Promises"

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


By Patrick Rall, Chunhao Wang, and Pawel Wocjan. [arXiv:2210.01670](https://arxiv.org/abs/2210.01670).

This repository presents the software that performed the numerical analysis presented in the manuscript.

## Notebooks

Figure 3 presents the impact of attenuation on the spectral gap of promised Davies generators. The methodology is presented in [promised_Davies_generator.ipynb](promised_Davies_generator.ipynb).

Figure 4 shows the accuracy of the steady state of the approximate Davies generator with an adversarial Hamiltonian. The methodology is presented in [approximate_Davies_generator.ipynb](approximate_Davies_generator.ipynb).

## Supporting code

Both notebooks rely on some common functionality:

- [src/poly_construction.py](src/poly_construction.py) implements the eigenvalue transformations constructed in Appendix A. These are required for energy estimation and approximate projection onto the promised subspace.
- [src/lindblad_analysis.py](src/lindblad_analysis.py) implements some common functionality for the analyis of Davies generators: their construction given a notion of energy estimation, extraction of the steady state and spectral gap, as well as synthesis of coupling operators.
- [src/rounding_promises.py](src/rounding_promises.py) generates the fine rounding promises presented in Section 4. Additionally, it provides functionality for projecting into and out of the promised subspace.
- [src/caching.py](src/caching.py) implements some functionality for saving the results of lengthy computations of a file and extracting them as needed.
