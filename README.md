nmfqsofit
============

**An Efficient, Fast, and Reliable Automated Continuum Fitter for Low-Resolution Quasar Spectra Using Non-negative Matrix Factorization (NMF)**

[![github shields.io](https://img.shields.io/badge/GitHub-abhi0395%2Fnmfqsofit-blue.svg?style=flat)](https://github.com/abhi0395/nmfqsofit)
[![github shields.io](https://img.shields.io/badge/GitHub-abhi0395%2Fnmfeigenspectra-pink.svg?style=flat)](https://github.com/abhi0395/nmfeigenspectra)
[![arXiv-2504.20299](http://img.shields.io/badge/arXiv-2504.20299-orange.svg?style=flat)](https://arxiv.org/abs/2504.20299)
[![arXiv-2103.15842](http://img.shields.io/badge/arXiv-2103.15842-orange.svg?style=flat)](https://arxiv.org/abs/2103.15842)
[![arXiv-1612.06037](http://img.shields.io/badge/arXiv-1612.06037-orange.svg?style=flat)](https://arxiv.org/abs/1612.06037)
[![license shields.io](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/abhi0395/nmfqsofit/blob/main/LICENSE)


`nmfqsofit` is a fast, modular, and scalable Python package for estimating quasar continua using precomputed NMF eigenspectra. It supports both NNLS-based fitting vectorized NMF coefficient estimation, works with SDSS/DESI/4MOST/WEAVE-like spectra, and is designed for large spectroscopic surveys and HPC environments.

---

## Features

- NMF-based quasar continuum modeling  
- Alternatively, NNLS-based fitting method  
- Multiple redshift-dependent eigenspectra support  
- Automatic best-eigenset selection via minimum cost  
- Efficient parallel processing (multiprocessing)  
- Clean FITS I/O structure  
- Modular and extensible design  
- Unittest included 

---

## Designed For

- Large spectroscopic surveys (SDSS, DESI, 4MOST, WEAVE, etc.)
- HPC continuum fitting  
 
----

## Requirements

- Python ≥ 3.8
- numpy
- scipy
- astropy
- matplotlib
- psutil
- tqdm (for timing)
- NonnegMFPy (Zhu 2016 NMF implementation) (see [https://github.com/guangtunbenzhu/NonnegMFPy](https://github.com/guangtunbenzhu/NonnegMFPy))
- pytest (for testing)

Install dependencies manually if needed:

```bash
pip install numpy scipy astropy matplotlib psutil pytest NonnegMFPy tqdm
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/nmfqsofit.git
cd nmfqsofit

# Install a tagged version (stable and reproducible). Replace vX.Y.Z with the desired Git tag (for example v2.0.1).
pip install --upgrade "git+https://github.com/abhi0395/qsoabsfind.git@vX.Y.Z"

#Install in editable mode (for developers):
pip install -e .
```

## Run Unit tests and verify installation

```bash
python -m unittest discover -s tests
nmfqsofit --help
python -c "import qsoabsfind; print(qsoabsfind.__version__)"
python -c "from qsoabsfind.parallel_convolution import parallel_convolution_method_absorber_finder_QSO_spectra; print('Installation successful!')"
```

---

## NMF Eigenspectra (Necessary before you run the script)

The NMF eigenspectra are maintained in a separate [repository](https://github.com/abhi0395/nmfeigenspectra). This keeps the eigenspectra data independent from the `nmfqsofit` codebase, allowing both to evolve separately. Currently, the repository provides eigenspectra built from **SDSS DR14** quasar spectra only. In the future, eigenspectra from additional surveys (e.g., DESI DR1/DR2, 4MOST, WEAVE) will be added.

Before running `nmfqsofit`, download the eigenspectra repository:

```bash
git clone https://github.com/abhi0395/nmfeigenspectra.git

### For SDSS DR14 eigenspectra, use the sdss/ directory:
/path/to/nmfeigenspectra/sdss/
```

Files follow the naming convention:

```
DR14_QSO_NMF_zQSO_000_100_basis.fits
```

The pipeline automatically:
- Selects appropriate eigenspectra based on quasar redshift  
- Fits using all matching redshift bins  
- Chooses the solution with minimum cost  
---

##  How It Works

1. Load quasar spectra (FLUX, IVAR, WAVELENGTH, METADATA with Z).
2. Load all available eigenspectra from directory.
3. For each quasar:
   - Identify matching redshift bins.
   - Fit continuum using NNLS or NMF.
   - If multiple bins valid it will select solution with minimum cost.
4. Apply median filtering correction.
5. Save coefficients, continua, and fit statistics to FITS.

---

## Command Line Usage

### Basic Run

```bash
nmfqsofit \
  --spectra-file spectra.fits \
  --eigenspectra /path/to/nmfeigenspectra/sdss \
  --kernel-size 71 \
  --method nnls \
  --ncpus 8 \
  --output continuum_output.fits
  --headers AUTHOR=Abhijeet SURVEY=DESI VERSION=1.0
  --n-qso 100
```

### Using NMF (or nnls) Method. Just replace *nmf* to *nnls*

```bash
--method nmf
```

Output FITS Structure
------------

The output FITS file contains:

| HDU | Description |
|------|------------|
| Primary | Header only |
| COEFFICIENTS | NMF coefficients (nspec × ncomp) |
| FIRST_CONTINUUM | Initial reconstructed continuum |
| CONTINUUM | Final median-filter corrected continuum |
| METADATA | FIRST_COST, FINAL_COST, EIGVECTOR_TYPE |

---

Plotting Example
------------

```python
from nmfqsofit.utils import plot_flux_and_continuum

plot_flux_and_continuum(
    wave,
    flux,
    continuum,
    flux_kwargs={"color": "black"},
    continuum_kwargs={"color": "red", "lw": 2}
)
```

---

Citations
------------

If you use this code, please cite:

- [Anand et al. 2025](https://arxiv.org/abs/2103.15842), DESI DR1 and DR2 continuum
- [Anand et al. 2021](https://arxiv.org/abs/2103.15842), SDSS DR14 Eigenspectra
- [Zhu 2016](https://arxiv.org/abs/1612.06037), NMF implementation

---

Contribution
------------

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

License
-------

Copyright (c) 2021-2025 Abhijeet Anand.

**nmfqsofit** is a free software made available under the MIT License. For details, see the LICENSE file.

Thanks,  
Abhijeet Anand  
IUCAA, Pune &
Lawrence Berkeley National Lab