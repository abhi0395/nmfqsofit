# nmfqsofit/nmfqsofit.py

from __future__ import annotations

import time
from tqdm import tqdm
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
from astropy.io import fits
from scipy.optimize import nnls
from scipy.signal import medfilt

# Zhu 2016 based vectorized NMF (if you use it)
from NonnegMFPy import nmf

from .utils import linear_interpolation, parse_qso_sequence
from .io import QSOSpecRead, load_all_eigenspectra, write_continuum

@dataclass(frozen=True)
class Eigenset:
    """Container for one redshift-bin eigenset."""
    rest_wave: np.ndarray          # (n_rest_wave,)
    eigvec: np.ndarray             # (ncomp, n_rest_wave)


class NMFContinuum:
    """
    Fit a quasar continuum using fixed NMF eigenspectra and non-negative coefficients.

    The workflow is:
      1) Choose the appropriate eigenset(s) based on QSO redshift z.
         If bins overlap, fit all matching eigensets and keep the lowest final cost.
      2) Interpolate rest-frame eigenspectra to the observed wavelength grid.
      3) Solve for non-negative coefficients using either:
         - Weighted NNLS
         - NonnegMFPy with H-only solve
      4) Apply a median-filter correction on flux/continuum ratio for smooth calibration.

    Attributes:
        wave (np.ndarray): Observed-frame wavelength grid (nwave,).
        flux (np.ndarray): Observed flux (nwave,).
        ivar (np.ndarray): Inverse variance (nwave,).
        z (float): Quasar redshift.
        kernel_size (int): Median filter kernel size (must be odd; will be enforced).
        method (str): 'nnls' or 'nmf'.
        eigenspectra_dict (dict): {(zmin, zmax): Eigenset or {"wave":..., "eigvec":...}}.
    """

    def __init__(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        ivar: np.ndarray,
        z: float,
        eigenspectra: Dict[Tuple[float, float], Any],
        kernel_size: int = 71,
        method: str = "nnls",
        maxiters: int = None,
    ):
        self.wave = np.asarray(wave)
        self.flux = np.asarray(flux)
        self.ivar = np.asarray(ivar)
        self.z = float(z)
        self.maxiters = maxiters

        self.delta_lambda = np.nanmedian(self.wave[1:]-self.wave[:-1])

        if kernel_size is not None:
            # kernel size for medfilt must be odd
            kernel_size = int(kernel_size)
            if (kernel_size % 2 == 0) or (kernel_size < 3):
                raise ValueError("kernel size must be odd and > 3")

        self.kernel_size = kernel_size

        if method not in {"nnls", "nmf"}:
            raise ValueError("method must be either 'nnls' or 'nmf'")
        self.method = method

        self.eigenspectra_dict = eigenspectra

        # Outputs
        self.coeff =  None            # (ncomp,)
        self.first_continuum = None  # (nwave,)
        self.continuum = None        # (nwave,)
        self.first_cost = None
        self.cost = None
        self.eigvec_range = None

        # Mask array from ivar
        self.mask = np.isfinite(self.flux) & np.isfinite(self.ivar) & (self.ivar > 0)

    def _keys_as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return zmin and zmax arrays from eigenspectra_dict keys.

        Returns:
            (np.ndarray, np.ndarray): zmins, zmaxs each shape (nbins,).
        """
        keys = list(self.eigenspectra_dict.keys())
        if len(keys) == 0:
            raise ValueError("eigenspectra_dict is empty")

        zmins = np.array([float(k[0]) for k in keys], dtype=float)
        zmaxs = np.array([float(k[1]) for k in keys], dtype=float)
        return zmins, zmaxs

    def _matching_bins(self) -> np.ndarray:
        """
        Return indices of all redshift bins that contain self.z.

        Uses [zmin, zmax) convention.

        Returns:
            np.ndarray: indices into zmins/zmaxs arrays.
        """
        zmins, zmaxs = self._keys_as_arrays()
        return np.where((self.z >= zmins) & (self.z < zmaxs))[0]

    def _get_eigenset_by_index(self, idx: int) -> Tuple[float, float, Eigenset]:
        """
        Get (zmin, zmax, eigenset) for a given matching index.

        Returns:
            (float, float, Eigenset)
        """
        zmins, zmaxs = self._keys_as_arrays()
        zmin = float(zmins[idx])
        zmax = float(zmaxs[idx])

        item = self.eigenspectra_dict[(zmin, zmax)]

        if isinstance(item, Eigenset):
            eig = item
        else:
            # Or {"wave": rest_wave, "eigvec": eigvec}
            rest_wave = np.asarray(item["wave"])
            eigvec = np.asarray(item["eigvec"])
            eig = Eigenset(rest_wave=rest_wave, eigvec=eigvec)

        return zmin, zmax, eig

    # -------------------------
    # Main functions to run the continuum fitting
    # -------------------------

    def _interpolate_eigvec_to_observed(self, eig: Eigenset) -> np.ndarray:
        """
        Interpolate rest-frame eigenspectra to this spectrum's observed wavelength grid.

        Assumes:
            obs_wave = rest_wave * (1 + z)
            f_lambda(obs) = f_lambda(rest) / (1 + z)   (simple flux-density scaling)

        Args:
            eig (Eigenset): Rest-frame wavelength and eigvec for one bin.

        Returns:
            np.ndarray: Interpolated eigenspectra (ncomp, nwave_obs).
        """
        cz = 1.0 + self.z
        rest_wave = eig.rest_wave
        eigvec = eig.eigvec  # (ncomp, n_rest)

        obs_wave = rest_wave * cz

        ncomp = eigvec.shape[0]
        out = np.empty((ncomp, self.wave.size), dtype=float)

        for i in range(ncomp):
            # Convert basis to observed frame (simple f_lambda scaling)
            obs_flux = eigvec[i, :] / cz
            out[i, :] = linear_interpolation(self.wave, obs_wave, obs_flux)

        return out

    def _chi2_reduced(self, model: np.ndarray) -> float:
        """
        Compute reduced chi^2 using ivar and valid-mask pixels.

        Args:
            model (np.ndarray): Model flux (nwave,).

        Returns:
            float: Reduced chi^2.
        """
        good = self.mask & np.isfinite(model)
        n = int(np.count_nonzero(good))
        if n == 0:
            return np.inf

        diff = self.flux[good] - model[good]
        chi2 = np.sum(self.ivar[good] * diff * diff)
        return float(chi2 / n)

    def _fit_coeff_nnls(self, A_interp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit coefficients with weighted NNLS using scipy.

        Args:
            A_interp (np.ndarray): Interpolated eigvec (ncomp, nwave).

        Returns:
            (coeff, model, chi2red)
        """
        good = self.mask & np.all(np.isfinite(A_interp), axis=0)
        if int(np.count_nonzero(good)) < A_interp.shape[0]:
            raise ValueError("Not enough valid pixels to fit NNLS coefficients.")

        w = np.sqrt(self.ivar[good])                        # (ngood,)
        A = (A_interp[:, good].T) * w[:, None]              # (ngood, ncomp)
        b = self.flux[good] * w                             # (ngood,)

        coeff, _ = nnls(A, b, maxiter=self.maxiters)
        model = coeff @ A_interp                             # (nwave,)
        chi2red = self._chi2_reduced(model)
        return coeff, model, chi2red

    def _fit_coeff_nmf(self, A_interp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit coefficients with NonnegMFPy by solving H only using Zhu 2016.
        Repo: https://github.com/guangtunbenzhu/NonnegMFPy/blob/master/NonnegMFPy/nmf.py

        Note:
            This assumes NonnegMFPy expects X with shape (n_pixels, n_spectra).
            Here we fit a single spectrum -> n_spectra=1.

        Args:
            A_interp (np.ndarray): Interpolated eigvec (ncomp, nwave).

        Returns:
            (coeff, model, chi2red)
        """
        good = self.mask & np.all(np.isfinite(A_interp), axis=0)
        if int(np.count_nonzero(good)) < A_interp.shape[0]:
            raise ValueError("Not enough valid pixels to fit NMF coefficients.")

        # X, V, M shapes: (n_pixels, 1)
        X = self.flux[good].reshape(-1, 1)
        V = self.ivar[good].reshape(-1, 1)
        M = np.ones_like(X, dtype=bool)

        # W shape in NonnegMFPy: (n_pixels, n_components)
        W = A_interp[:, good].T

        solver = nmf.NMF(X=X, V=V, M=M, W=W, n_components=W.shape[1])
        solver.SolveNMF(H_only=True, maxiters=self.maxiters)

        # H is (n_components, n_spectra=1)
        coeff = np.asarray(solver.H).reshape(-1)
        model = coeff @ A_interp
        chi2red = self._chi2_reduced(model)
        return coeff, model, chi2red

    def _apply_smooth_correction(self, first_continuum: np.ndarray) -> np.ndarray:
        """
        Apply median-filter smooth correction to the first-pass continuum.

        Uses ratio r = flux / first_continuum, median-filters it, and multiplies:
            continuum = first_continuum * smooth_residual

        Args:
            first_continuum (np.ndarray): First-pass continuum (nwave,).

        Returns:
            np.ndarray: Final continuum (nwave,).
        """
        cont = np.asarray(first_continuum)
        good = self.mask & np.isfinite(cont) & (cont > 0)

        ratio = np.full_like(cont, np.nan, dtype=float)
        ratio[good] = self.flux[good] / cont[good]

        if self.kernel_size is not None:
            smooth = medfilt(ratio, kernel_size=self.kernel_size)
        else:
            smooth = np.ones(ratio.size)

        # Where smooth is non-finite, do not modify continuum
        out = cont.copy()
        ok = np.isfinite(smooth)
        out[ok] = cont[ok] * smooth[ok]
        return out

    def fit(self) -> None:
        """
        Select eigenspectra for this z, fit coefficients, and build continuum.

        If multiple redshift bins match (overlapping bins), fits each candidate
        eigenset and selects the solution with the lowest final cost.
        """
        zmins, zmaxs = self._keys_as_arrays()
        match_idx = np.where((self.z >= zmins) & (self.z < zmaxs))[0]

        if match_idx.size == 0:
            raise ValueError("QSO is outside NMF eigenspectra range.")

        best = {
            "cost": np.inf,
            "coeff": None,
            "first_cont": None,
            "cont": None,
            "first_cost": None,
            "zmin": None,
            "zmax": None,
        }

        for ii in match_idx:
            zmin, zmax, eig = self._get_eigenset_by_index(int(ii))
            A_interp = self._interpolate_eigvec_to_observed(eig)

            if self.method == "nnls":
                coeff, first_cont, first_cost = self._fit_coeff_nnls(A_interp)
            else:
                coeff, first_cont, first_cost = self._fit_coeff_nmf(A_interp)

            cont = self._apply_smooth_correction(first_cont)

            cost = self._chi2_reduced(cont)

            if cost < best["cost"]:
                best.update(
                    {
                        "cost": cost,
                        "coeff": coeff,
                        "first_cont": first_cont,
                        "cont": cont,
                        "first_cost": first_cost,
                        "zmin": zmin,
                        "zmax": zmax,
                    }
                )

        self.coeff = best["coeff"]
        self.first_continuum = best["first_cont"]
        self.continuum = best["cont"]
        self.first_cost = float(best["first_cost"])
        self.cost = float(best["cost"])
        self.eigvec_range = f"z_{best['zmin']:.2f}_{best['zmax']:.2f}"


# -------------------------
# Parallel running of the NMF continuum fit
# -------------------------

def _process_one(args):
    """Worker for multiprocessing."""
    wave, flux1, ivar1, z1, eigenspectra, kernel_size, method, maxiters = args
    fitter = NMFContinuum(
        wave=wave,
        flux=flux1,
        ivar=ivar1,
        z=z1,
        eigenspectra=eigenspectra,
        kernel_size=kernel_size,
        method=method,
        maxiters=maxiters
    )
    fitter.fit()

    return (
        fitter.coeff,
        fitter.first_continuum,
        fitter.continuum,
        fitter.first_cost,
        fitter.cost,
        fitter.eigvec_range,
    )

def run_parallel_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    z: np.ndarray,
    eigenspectra: Dict[Tuple[float, float], Any],
    kernel_size: int,
    method: str,
    n_jobs: int = -1,
    maxiters: int=100):

    """
    Fit continua for many QSOs in parallel.

    Args:
        wave (np.ndarray): Observed wavelength grid (nwave,).
        flux (np.ndarray): Flux array (nqso, nwave).
        ivar (np.ndarray): IVAR array (nqso, nwave).
        z (np.ndarray): Redshifts (nqso,).
        eigenspectra (dict): Dict of eigensets keyed by (zmin, zmax).
        kernel_size (int): Median filter kernel size.
        method (str): 'nnls' or 'nmf'.
        n_jobs (int): Number of processes. If -1, uses mp.cpu_count().
        maxiters (int): Number of iterations (default is None)

    Returns:
        tuple:
            coefficient_matrix (np.ndarray): (nqso, ncomp) float32
            first_continuum_matrix (np.ndarray): (nqso, nwave) float32
            final_continuum_matrix (np.ndarray): (nqso, nwave) float32
            first_cost (np.ndarray): (nqso,) float32
            final_cost (np.ndarray): (nqso,) float32
            eigvec_range (np.ndarray): (nqso,) fixed-width bytes
    """
    flux = np.asarray(flux)
    ivar = np.asarray(ivar)
    z = np.asarray(z)

    if flux.ndim != 2:
        raise ValueError("flux must be 2D with shape (nqso, nwave)")
    if ivar.shape != flux.shape:
        raise ValueError("ivar must have the same shape as flux")
    if z.shape[0] != flux.shape[0]:
        raise ValueError("z must have length nqso")

    nqso, nwave = flux.shape

    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    n_jobs = int(max(1, n_jobs))

    tasks = [
        (wave, flux[i], ivar[i], float(z[i]), eigenspectra, kernel_size, method, maxiters)
        for i in range(nqso)
    ]

    t0 = time.time()
    chunksize = max(1, nqso // (10 * n_jobs))
    print(f'INFO: chunk size for parallel run: {chunksize}')

    with mp.Pool(processes=n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(_process_one, tasks, chunksize=chunksize),
                total=nqso,
                desc="Fitting continua",
                unit="qso",
            )
        )
    t1 = time.time()

    print(f"Total continuum computation time for {nqso} QSOs: {t1 - t0:.2f} s")

    coeff_list, first_cont_list, cont_list, first_cost_list, cost_list, range_list = zip(*results)

    coefficient_matrix = np.vstack(coeff_list).astype(np.float32)
    first_continuum_matrix = np.vstack(first_cont_list).astype(np.float32)
    final_continuum_matrix = np.vstack(cont_list).astype(np.float32)

    first_cost = np.asarray(first_cost_list, dtype=np.float32)
    final_cost = np.asarray(cost_list, dtype=np.float32)

    # Store as fixed-width bytes for FITS friendliness
    eigvec_range = np.asarray(range_list, dtype="S32")

    return (
        coefficient_matrix,
        first_continuum_matrix,
        final_continuum_matrix,
        first_cost,
        final_cost,
        eigvec_range,
    )