# nmfqsofit/nmfcontinuum.py

from __future__ import annotations

import time
from tqdm import tqdm
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
from scipy.optimize import nnls
from scipy.signal import medfilt

# Zhu 2016 based vectorized NMF (if you use it)
from NonnegMFPy import nmf

from .utils import interpolation1D
from .logger import get_logger

logger = get_logger(__name__)
LARGE_CHI2 = 999999.0  # for failure cases

@dataclass(frozen=True)
class Eigenset:
    """Container for one redshift-bin eigenset."""
    rest_wave: np.ndarray          # (n_rest_wave,)
    eigvec: np.ndarray             # (ncomp, n_rest_wave)


def compute_normalization(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    lam_min_obs: float,
    lam_max_obs: float,
    stat: str = "median",
) -> float:
    """
    Compute normalization factor in an observed-frame wavelength window.

    Normalization is computed using only valid pixels:
      finite(flux), finite(ivar), ivar > 0, and lam_min_obs < wave < lam_max_obs.

    Args:
        wave (np.ndarray): Observed wavelength grid (nwave,).
        flux (np.ndarray): Observed flux array (nwave,).
        ivar (np.ndarray): Observed inverse variance array (nwave,).
        lam_min_obs (float): Observed-frame lower bound of normalization window (Angstrom).
        lam_max_obs (float): Observed-frame upper bound of normalization window (Angstrom).
        stat (str): "median" or "mean".

    Returns:
        float: Normalization factor. If insufficient valid pixels, returns np.nan.
    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    ivar = np.asarray(ivar)

    sel = (
        (wave > lam_min_obs)
        & (wave < lam_max_obs)
        & np.isfinite(flux)
        & np.isfinite(ivar)
        & (ivar > 0.0)
    )

    if int(np.count_nonzero(sel)) < 5:
        return np.nan

    if stat == "median":
        return float(np.nanmedian(flux[sel]))
    elif stat == "mean":
        return float(np.nanmean(flux[sel]))
    else:
        raise ValueError("stat must be 'median' or 'mean'")


class NMFContinuum:
    """
    Fit a quasar continuum using fixed NMF eigenspectra and non-negative coefficients.

    IMPORTANT (Normalization-aware fitting):
      - Eigenspectra are assumed to be trained on normalized spectra.
      - For each candidate eigenset (zmin,zmax,lam_min,lam_max,stat), we:
          (1) normalize the observed spectrum using that window/stat
          (2) fit coefficients in normalized space
          (3) scale the fitted continuum back to observed flux units
          (4) compute chi^2 costs against the observed spectrum
      - If multiple bins overlap in z, we try all and keep the lowest final cost.

    Attributes:
        wave (np.ndarray): Observed-frame wavelength grid (nwave,).
        flux (np.ndarray): Observed flux (nwave,).
        ivar (np.ndarray): Observed inverse variance (nwave,).
        z (float): Quasar redshift.
        eigenspectra_dict (dict): keys are (zmin,zmax,lam_min,lam_max,stat).
        kernel_large (int): Median filter kernel size (odd; enforced).
        method (str): 'nnls' or 'nmf'.
        maxiters (int): Maximum iterations for solver (None means default).
        interp_kind (str): Interpolation method for eigenspectra ('linear' or 'nearest'; default: 'linear').

    Outputs:
        coeff (np.ndarray): best-fit coefficients (ncomp,).
        first_continuum (np.ndarray): first-pass continuum in observed units (nwave,).
        continuum (np.ndarray): final continuum in observed units (nwave,).
        first_cost (float): reduced chi^2 for first_continuum against observed spectrum.
        cost (float): reduced chi^2 for continuum against observed spectrum.
        eigvec_range (str): "z_{zmin:.2f}_{zmax:.2f}"
        norm (float): normalization factor used for the best-fit eigenset.
    """

    def __init__(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        ivar: np.ndarray,
        z: float,
        eigenspectra: Dict[Tuple[float, float, float, float, str], Any],
        kernel_large: int,
        kernel_small: int,
        method: str,
        maxiters: int,
        smoothing_niter: int,
        interp_kind: str,
        fit_niter: int = 3,
        fit_nsigma: float = 3.0,
    ):
        """
        Initialize NMFContinuum fitter.

        Args:
            wave (np.ndarray): Observed-frame wavelength grid (nwave,).
            flux (np.ndarray): Observed flux array (nwave,).
            ivar (np.ndarray): Observed inverse variance array (nwave,).
            z (float): Quasar redshift.
            eigenspectra (dict): Eigenspectra dictionary with keys (zmin, zmax, lam_min, lam_max, stat).
            kernel_large (int): Median filter kernel size (odd; default: 71).
            method (str): 'nnls' or 'nmf' (default: 'nnls').
            maxiters (int, optional): Maximum iterations for solver (default: None).
            smoothing_niter (int, optional): Maximum iterations for median filtering (default: None)
        """
        self.wave = np.asarray(wave)
        self.flux = np.asarray(flux)
        self.ivar = np.asarray(ivar)
        self.z = float(z)
        self.maxiters = maxiters
        self.smoothing_niter = smoothing_niter
        self.interp_kind = interp_kind
        self.fit_niter = int(fit_niter)
        self.fit_nsigma = float(fit_nsigma)

        self.delta_lambda = np.nanmedian(self.wave[1:] - self.wave[:-1])

        if kernel_large is not None and kernel_large > 0:
            kernel_large = int(kernel_large)
            if (kernel_large % 2 == 0) or (kernel_large < 1) or (kernel_small < 1):
                raise ValueError("kernel size must be odd and >= 1")
        else:
            logger.info("Kernel size is negative or None, so no smoothing will be done")

        self.kernel_large = kernel_large
        self.kernel_small = kernel_small

        if method not in {"nnls", "nmf"}:
            raise ValueError("method must be either 'nnls' or 'nmf'")
        self.method = method

        self.eigenspectra_dict = eigenspectra

        # Outputs
        self.coeff = None             # (ncomp,)
        self.first_continuum = None   # (nwave,)
        self.continuum = None         # (nwave,)
        self.first_cost = None
        self.cost = None
        self.eigvec_range = None
        self.norm = None

        # Mask array from observed ivar (DESI-friendly)
        self.mask = np.isfinite(self.flux) & np.isfinite(self.ivar) & (self.ivar > 0)

    def _keys_as_arrays(self):
        """
        Return arrays of zmin, zmax, lam_min, lam_max, stat from eigenspectra_dict keys.

        Returns:
            tuple of arrays: zmins, zmaxs, lam_mins, lam_maxs, stats
        """
        keys = list(self.eigenspectra_dict.keys())
        if len(keys) == 0:
            raise ValueError("eigenspectra_dict is empty")

        zmins = np.array([float(k[0]) for k in keys], dtype=float)
        zmaxs = np.array([float(k[1]) for k in keys], dtype=float)
        lam_mins = np.array([float(k[2]) for k in keys], dtype=float)
        lam_maxs = np.array([float(k[3]) for k in keys], dtype=float)
        stats = np.array([str(k[4]) for k in keys], dtype=object)
        return zmins, zmaxs, lam_mins, lam_maxs, stats

    def _matching_bins(self) -> np.ndarray:
        """
        Return indices of all redshift bins that contain self.z.

        Uses [zmin, zmax) convention.

        Returns:
            np.ndarray: indices into key arrays.
        """
        zmins, zmaxs, _, _, _ = self._keys_as_arrays()
        return np.where((self.z >= zmins) & (self.z < zmaxs))[0]

    def _get_eigenset_by_index(self, idx: int):
        """
        Get (zmin, zmax, lam_min, lam_max, stat, eigenset) for a given matching index.

        Args:
            idx (int): Index into the eigenspectra dictionary.

        Returns:
            tuple: (zmin, zmax, lam_min, lam_max, stat, eigenset)
        """
        keys = list(self.eigenspectra_dict.keys())
        key = keys[idx]
        zmin, zmax, lam_min, lam_max, stat = key

        item = self.eigenspectra_dict[key]

        if isinstance(item, Eigenset):
            eig = item
        else:
            rest_wave = np.asarray(item["wave"])
            eigvec = np.asarray(item["eigvec"])
            eig = Eigenset(rest_wave=rest_wave, eigvec=eigvec)

        return float(zmin), float(zmax), float(lam_min), float(lam_max), str(stat), eig

    def _interpolate_eigvec_to_observed(self, eig: Eigenset) -> np.ndarray:
        """
        Interpolate rest-frame eigenspectra to this spectrum's observed wavelength grid.

        Assumes:
            obs_wave = rest_wave * (1 + z)
            f_lambda(obs) = f_lambda(rest) / (1 + z)   (simple flux-density scaling)

        Args:
            eig (Eigenset): Eigenset container with rest_wave and eigvec.

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
            obs_flux = eigvec[i, :] / cz
            foo = np.nanmedian(obs_flux)
            out[i, :] = interpolation1D(self.wave, obs_wave, obs_flux, kind=self.interp_kind, fill_value=foo)

        return out

    def _chi2_reduced_against_observed(self, model_obs: np.ndarray, n_comp: int) -> float:
        """
        Reduced chi^2 computed against the observed spectrum using all valid pixels.

        Args:
            model_obs (np.ndarray): Model flux in observed units (nwave,).

        Returns:
            float: Reduced chi^2 value.
        """
        good = self.mask & np.isfinite(model_obs)
        n = int(np.count_nonzero(good)) - n_comp  #degrees of freedome
        if n <= 0:
            return np.inf

        diff = self.flux[good] - model_obs[good]
        chi2 = np.sum(self.ivar[good] * diff * diff)
        return float(chi2 / n)

    def _fit_coeff_nnls(self, A_interp: np.ndarray, flux_fit: np.ndarray, ivar_fit: np.ndarray, mask_fit: np.ndarray):
        """
        Fit coefficients with weighted NNLS using scipy (on provided flux/ivar/mask).

        Args:
            A_interp (np.ndarray): Interpolated eigenspectra matrix (ncomp, nwave).
            flux_fit (np.ndarray): Flux to fit (nwave,).
            ivar_fit (np.ndarray): Inverse variance (nwave,).
            mask_fit (np.ndarray): Boolean mask for valid pixels (nwave,).

        Returns:
            tuple: (coeff, model) where coeff is (ncomp,) and model is (nwave,).
        """
        good = mask_fit & np.all(np.isfinite(A_interp), axis=0)
        if int(np.count_nonzero(good)) < A_interp.shape[0]:
            raise ValueError("Not enough valid pixels to fit NNLS coefficients.")

        w = np.sqrt(ivar_fit[good])                       # (ngood,)
        A = (A_interp[:, good].T) * w[:, None]            # (ngood, ncomp)
        b = flux_fit[good] * w                            # (ngood,)

        coeff, _ = nnls(A, b, maxiter=self.maxiters)
        model = coeff @ A_interp                           # model in the same space as flux_fit
        return coeff, model

    def _fit_coeff_nmf(self, A_interp: np.ndarray, flux_fit: np.ndarray, ivar_fit: np.ndarray, mask_fit: np.ndarray):
        """
        Fit coefficients with NonnegMFPy by solving H only (single spectrum).

        Args:
            A_interp (np.ndarray): Interpolated eigenspectra matrix (ncomp, nwave).
            flux_fit (np.ndarray): Flux to fit (nwave,).
            ivar_fit (np.ndarray): Inverse variance (nwave,).
            mask_fit (np.ndarray): Boolean mask for valid pixels (nwave,).

        Note:
            NonnegMFPy expects X with shape (n_pixels, n_spectra).
            Here n_spectra=1.

        Returns:
            tuple: (coeff, model) where coeff is (ncomp,) and model is (nwave,).
        """
        good = mask_fit & np.all(np.isfinite(A_interp), axis=0)
        if int(np.count_nonzero(good)) < A_interp.shape[0]:
            raise ValueError("Not enough valid pixels to fit NMF coefficients.")

        X = flux_fit[good].reshape(-1, 1)
        V = ivar_fit[good].reshape(-1, 1)
        M = np.ones_like(X, dtype=bool)

        W = A_interp[:, good].T  # (n_pixels, n_components)

        solver = nmf.NMF(X=X, V=V, M=M, W=W, n_components=W.shape[1])
        solver.SolveNMF(H_only=True, maxiters=self.maxiters)

        coeff = np.asarray(solver.H).reshape(-1)
        model = coeff @ A_interp
        return coeff, model

    def _run_solver(self, A_interp, flux_fit, ivar_fit, good):
        """Run solver (nnls or nmf)."""
        if self.method == "nnls":
            return self._fit_coeff_nnls(A_interp, flux_fit, ivar_fit, good)
        return self._fit_coeff_nmf(A_interp, flux_fit, ivar_fit, good)

    def _fit_coeff_iterative(self, A_interp: np.ndarray, flux_fit: np.ndarray, ivar_fit: np.ndarray, mask_fit: np.ndarray):
        """
        Iterative sigma-rejection wrapper around the coefficient solver.

        Performs an initial fit, then repeats up to fit_niter rounds of
        absorption masking using sigma clipping followed by a refit.
        Only pixels with pull < -fit_nsigma * sigma are masked, so emission
        features stay as they are.

        Args:
            A_interp (np.ndarray): Interpolated eigenspectra matrix (ncomp, nwave).
            flux_fit (np.ndarray): Normalized flux to fit (nwave,).
            ivar_fit (np.ndarray): Normalized inverse variance (nwave,).
            mask_fit (np.ndarray): Boolean mask for valid pixels (nwave,).

        Returns:
            tuple: (coeff, model) from the final iteration.
        """
        good = mask_fit.copy()
        coeff, model = self._run_solver(A_interp, flux_fit, ivar_fit, good)

        for ntr in range(self.fit_niter):
            # Normalized residuals at currently accepted pixels
            pull = np.full_like(flux_fit, np.nan)
            ok = good & np.isfinite(model) & (ivar_fit > 0)
            pull[ok] = (flux_fit[ok] - model[ok]) * np.sqrt(ivar_fit[ok])

            # Estimate sigma from emission-side pixels only.

            above = pull[ok & (pull > 0.0)]
            sigma = self._mad_sigma(above) if above.size >= 2 else self._mad_sigma(pull[ok])
            if not np.isfinite(sigma) or sigma <= 0.0:
                break

            # Mask only potential absorption pixels
            new_good = good & np.where(np.isfinite(pull), pull > -self.fit_nsigma * sigma, True)
            n_changed = int(np.count_nonzero(new_good != good))
            good = new_good

            logger.debug(f'fit_coeff_iterative iter={ntr}: sigma={sigma:.4f}, n_masked={n_changed}')

            if n_changed == 0:
                logger.debug(f'fit_coeff_iterative: converged (no new masked pixels) after iteration {ntr}')
                break

            coeff, model = self._run_solver(A_interp, flux_fit, ivar_fit, good)

        return coeff, model

    @staticmethod
    def _neutral_fill(arr: np.ndarray) -> np.ndarray:
        """Return a copy of arr with non-finite values replaced by 1.0."""
        out = arr.copy()
        out[~np.isfinite(out)] = 1.0
        return out

    @staticmethod
    def _mad_sigma(arr: np.ndarray) -> float:
        """Robust sigma estimate via Median Absolute Deviation (MAD).

        sigma_MAD = 1.4826 * median(|x - median(x)|)

        NaN values are ignored.  Returns np.nan if fewer than 2 finite values.
        """
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            return np.nan
        return 1.4826 * float(np.median(np.abs(finite - np.median(finite))))

    def _apply_smooth_correction(self, first_continuum, kernel_large, kernel_small, smoothing_niter, nsigma=0.5):

        """Apply iterative median-filter correction to remove intermediate
        and small scale fluctuations.

        This implements the idea described in Zhu (e.g., Zhu & Ménard 2013/2016):
        0) Pre-masking: apply a single coarse median filter (kernel_large) to the
            raw ratio to identify and mask absorption dips before the main loop,
            so that even the first iteration does not feed absorber pixels into medfilt.
        1) Construct the ratio r = flux / first_continuum.
        2) Smooth r with an intermediate-scale median filter (kernel_large).
        3) Remove smaller-scale power with an additional median filter (kernel_small).
        4) Mask pixels likely affected by narrow absorption by keeping only
            fluctuations within nsigma * sigma of the smoothed ratio.
        5) Repeat steps (2)-(4) niter times.

        The output is a multiplicative correction applied to the first-pass continuum:
            continuum_final = first_continuum * smooth_ratio

        Notes:
        - This correction is performed in observed frame (same grid as flux).
        - The masking uses a robust sigma estimate (MAD) on (r - smooth_ratio).
        - Pixels with ivar <= 0, non-finite flux/ivar, or non-positive continuum are excluded from the correction.

        Args:
            flux (np.ndarray): Observed flux array (nwave,).
            ivar (np.ndarray): Observed inverse variance array (nwave,).
            first_continuum (np.ndarray): First-pass continuum (nwave,).
            kernel_large (int): Median-filter kernel size for intermediate scales.
                Must be odd.
            kernel_small (int): Median-filter kernel size for small scales.
                Must be odd.
            nsigma (float): Sigma threshold for masking narrow absorption features.
            smoothing_niter (int): Number of iterationsfor median filtering.

        Returns:
            np.ndarray: Final continuum array (nwave,).
        """

        cont0 = np.asarray(first_continuum)

        if kernel_large is None or kernel_small is None:
            kernel_large = None
            kernel_small = None
        elif kernel_large < 0 or kernel_small < 0:
            kernel_large = None
            kernel_small = None

        if (kernel_large is not None) and (kernel_large > 0):
            if kernel_large % 2 == 0:
                raise ValueError("kernel_large must be odd")

        if (kernel_small is not None) and (kernel_small > 0):
            if kernel_small % 2 == 0:
                raise ValueError("kernel_small must be odd")

        # Valid pixels mask
        good0 = (
            np.isfinite(self.flux)
            & np.isfinite(self.ivar)
            & (self.ivar > 0.0)
            & np.isfinite(cont0)
            & (cont0 > 0.0)
        )

        # Ratio in observed frame
        ratio = np.full_like(cont0, np.nan, dtype=float)
        ratio[good0] = self.flux[good0] / cont0[good0]

        good = good0.copy()
        smooth = np.ones_like(cont0, dtype=float)

        # Pre-masking: run a single coarse median filter pass on the raw ratio
        # to identify absorption dips before the iterative loop begins.

        if (kernel_large is not None) and (kernel_large > 0):
            r_pre = self._neutral_fill(ratio)
            r_pre[~good0] = 1.0

            r_smooth_pre = medfilt(r_pre, kernel_size=kernel_large)
            r_smooth_pre = self._neutral_fill(r_smooth_pre)
            r_smooth_pre[r_smooth_pre <= 0.0] = 1.0

            delta_pre = np.full_like(ratio, np.nan)
            ok_pre = good0 & np.isfinite(ratio) & (r_smooth_pre > 0)
            delta_pre[ok_pre] = ratio[ok_pre] / r_smooth_pre[ok_pre] - 1.0

            sigma_pre = self._mad_sigma(delta_pre[ok_pre])
            if np.isfinite(sigma_pre) and (sigma_pre > 0.0):
                good = ok_pre & (delta_pre > -nsigma * sigma_pre)

        if kernel_large is None or kernel_small is None:
            return cont0 * smooth

        for it in range(int(smoothing_niter)):

            prev_good = good.copy()

            # Fill masked/invalid pixels with 1.0 before medfilt
            r_fill = self._neutral_fill(ratio)
            r_fill[~good] = 1.0

            # Intermediate-scale smoothing
            r1 = medfilt(r_fill, kernel_size=kernel_large)
            r1 = self._neutral_fill(r1)
            r1[r1 <= 0.0] = 1.0

            # Small-scale smoothing on residual ratio
            r2 = medfilt(r_fill / r1, kernel_size=kernel_small)
            r2 = self._neutral_fill(r2)
            r2[r2 <= 0.0] = 1.0

            smooth = r1 * r2

            # Robust masking against narrow absorption features in ratio
            delta = np.full_like(ratio, np.nan, dtype=float)
            ok = good0 & np.isfinite(ratio) & np.isfinite(smooth) & (smooth > 0)
            delta[ok] = ratio[ok] / smooth[ok] - 1.0

            # Use MAD sigma for absorber masking
            sigma = self._mad_sigma(delta[good])
            if (not np.isfinite(sigma)) or (sigma <= 0.0):
                break

            cut = -nsigma * sigma
            good = ok & (delta > cut)

            n_changed = int(np.count_nonzero(good != prev_good))

            if n_changed ==0:
                break

        return cont0 * smooth


    def fit(self) -> None:
        """
        Select eigenspectra for this z, fit coefficients in normalized space,
        scale continuum back to observed space, and choose best solution (lowest final cost).
        """
        match_idx = self._matching_bins()
        if match_idx.size == 0:
            logger.warning(f"QSO (redshift: {self.z}) is outside NMF eigenspectra range.")


        best = {
            "final_cost": LARGE_CHI2,
            "coefficients": None,
            "first_continuum": None,
            "continuum": None,
            "first_cost": LARGE_CHI2,
            "zmin": None,
            "zmax": None,
            "norm": None,
            "n_comp": None,
            "method": self.method,
            "coverage": -1,
        }

        # Keep observed arrays fixed (do not mutate self.flux/ivar)
        flux_obs = self.flux
        ivar_obs = self.ivar

        for ii in match_idx:
            zmin, zmax, lam_min, lam_max, stat, eig = self._get_eigenset_by_index(int(ii))

            n_comp = min(eig.eigvec.shape)

            # Interpolate eigenspectra to observed grid (still "normalized basis" in observed units)
            A_interp = self._interpolate_eigvec_to_observed(eig)

            # Normalization window is defined in REST frame in the key, convert to OBS frame
            lam_min_obs = lam_min * (1.0 + self.z)
            lam_max_obs = lam_max * (1.0 + self.z)

            norm = compute_normalization(
                wave=self.wave,
                flux=flux_obs,
                ivar=ivar_obs,
                lam_min_obs=lam_min_obs,
                lam_max_obs=lam_max_obs,
                stat=stat,
            )

            # Coverage: valid pixels within the eigenset's observed-frame wavelength range
            obs_min = eig.rest_wave.min() * (1.0 + self.z)
            obs_max = eig.rest_wave.max() * (1.0 + self.z)
            mask_coverage = self.mask & np.isfinite(self.wave) & (self.wave >= obs_min) & (self.wave <= obs_max)
            coverage = int(np.count_nonzero(mask_coverage))

            # Skip if norm is not usable
            if (not np.isfinite(norm)) or (norm <=0.0):
                logger.info(f"Skipping eigenset z[{zmin:.2f}, {zmax:.2f}) lam[{lam_min:.1f}, {lam_max:.1f}] stat={stat} due to invalid norm={norm:.3e}")
                logger.warning(f"Continuum failed for the quasar with redshfit: {self.z}, probably due to negative norm...")
                logger.warning(f"Normalization details: lam_min_obs={lam_min_obs:.1f}, lam_max_obs={lam_max_obs:.1f}, computed norm={norm:.3e}")
                continue

            # Build normalized spectrum for fitting
            flux_fit = flux_obs / norm
            ivar_fit = ivar_obs * (norm ** 2)
            mask_fit = np.isfinite(flux_fit) & np.isfinite(ivar_fit) & (ivar_fit > 0)

            # Fit coefficients in normalized space (with iterative absorption rejection)
            try:
                coeff, first_cont_norm = self._fit_coeff_iterative(A_interp, flux_fit, ivar_fit, mask_fit)
            except Exception as exc:
                logger.exception(
                    "Exception during coefficient fitting at z=%.4f for eigenset "
                    "z[%.2f, %.2f) lam[%.1f, %.1f] stat=%s; skipping this configuration",
                    self.z, zmin, zmax, lam_min, lam_max, stat
                )
                continue

            # Scale first-pass continuum back to OBSERVED units
            first_cont_obs = first_cont_norm * norm

            # Smooth correction in observed units (uses self.flux/self.mask which are observed)
            cont_obs = self._apply_smooth_correction(first_cont_obs, kernel_large=self.kernel_large, kernel_small=self.kernel_small, smoothing_niter=self.smoothing_niter)

            # Chi2 on all valid pixels for both continua
            first_cost = self._chi2_reduced_against_observed(first_cont_obs, n_comp)
            cost = self._chi2_reduced_against_observed(cont_obs, n_comp)

            # Keep smooth-corrected continuum only if it improves chi2
            if cost > first_cost:
                cost = first_cost
                cont_obs = first_cont_obs.copy()

            if coverage > best["coverage"] or (
                coverage == best["coverage"] and cost < best["final_cost"]
            ):
                best.update(
                    {
                        "final_cost": cost,
                        "coefficients": coeff,
                        "first_continuum": first_cont_obs,
                        "continuum": cont_obs,
                        "first_cost": first_cost,
                        "zmin": zmin,
                        "zmax": zmax,
                        "norm": norm,
                        "n_comp": n_comp,
                        "coverage": coverage,
                    }
                )

        if best["coefficients"] is None:
            # Extract ncomp from eigenspectra for proper array sizing
            first_eig = next(iter(self.eigenspectra_dict.values()))
            ncomp = first_eig['eigvec'].shape[0] if isinstance(first_eig, dict) else first_eig.eigvec.shape[0]

            # Initialize all outputs with zero values and chi2 would be very large for failure cases
            best["coefficients"] = np.zeros(ncomp)
            best["first_continuum"] = np.zeros(self.wave.size)
            best["continuum"] = np.zeros(self.wave.size)
            best["first_cost"] = LARGE_CHI2
            best["final_cost"] = LARGE_CHI2
            best["norm"] = 0.0
            best["zmin"] = -1.0
            best["zmax"] = -1.0
            best["n_comp"] = 0


            logger.warning(f"Continuum fitting failed for QSO at z={self.z}, returning zero arrays")

        self.coeff = best["coefficients"]
        self.first_continuum = best["first_continuum"]
        self.continuum = best["continuum"]
        self.first_cost = float(best["first_cost"])
        self.cost = float(best["final_cost"])
        self.norm = float(best["norm"])
        self.eigvec_range = f"z_{int(best['zmin']*100):03d}_{int(best['zmax']*100):03d}"
        self.zmin = float(best["zmin"])
        self.zmax = float(best["zmax"])
        self.n_comp = int(best["n_comp"])

# -------------------------
# Parallel running of the NMF continuum fit
# -------------------------

def _process_one(args):
    """
    Worker function for multiprocessing pool.

    Args:
        args (tuple): Tuple of (wave, flux, ivar, z, eigenspectra, kernel_large, method, maxiters).

    Returns:
        tuple: (coeff, first_continuum, continuum, first_cost, cost, eigvec_range, norm)
    """
    wave, flux1, ivar1, z1, eigenspectra, kernel_large, kernel_small, method, maxiters, interp_kind, smoothing_niter, fit_niter, fit_nsigma = args
    fitter = NMFContinuum(
        wave=wave,
        flux=flux1,
        ivar=ivar1,
        z=z1,
        eigenspectra=eigenspectra,
        kernel_large=kernel_large,
        kernel_small=kernel_small,
        method=method,
        maxiters=maxiters,
        interp_kind=interp_kind,
        smoothing_niter=smoothing_niter,
        fit_niter=fit_niter,
        fit_nsigma=fit_nsigma,
    )
    fitter.fit()

    return (
        fitter.coeff,
        fitter.first_continuum,
        fitter.continuum,
        fitter.first_cost,
        fitter.cost,
        fitter.eigvec_range,
        fitter.zmin,
        fitter.zmax,
        fitter.norm,
        fitter.n_comp
        )


def run_parallel_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    z: np.ndarray,
    eigenspectra: Dict[Tuple[float, float, float, float, str], Any],
    kernel_large: int,
    kernel_small: int,
    method: str,
    n_jobs: int = -1,
    maxiters: int = 100,
    interp_kind: str = "linear",
    smoothing_niter: int = 3,
    fit_niter: int = 3,
    fit_nsigma: float = 3.0,
):
    """
    Fit continua for many QSOs in parallel.

    Args:
        wave (np.ndarray): Observed-frame wavelength grid (nwave,).
        flux (np.ndarray): Observed flux array (nqso, nwave).
        ivar (np.ndarray): Observed inverse variance array (nqso, nwave).
        z (np.ndarray): Quasar redshifts (nqso,).
        eigenspectra (dict): Eigenspectra dictionary with keys (zmin, zmax, lam_min, lam_max, stat).
        kernel_large (int): Median filter kernel size (odd integer) for intermediate scale fluctuations.
        kernel_small (int): Median filter kernel size (odd integer) for small scale fluctuations.
        method (str): 'nnls' or 'nmf'.
        n_jobs (int): Number of parallel jobs (-1 for all CPUs; default: -1).
        maxiters (int): Maximum iterations for solver (default: 100).
        interp_kind (str): Interpolation method for eigenspectra ('linear' or 'nearest'; default: 'linear').
        smoothing_niter (int): Maximum iteration for median filtering
        fit_niter (int): Number of sigma-rejection iterations during coefficient fitting (default: 3).
        fit_nsigma (float): Sigma threshold for absorption masking during fitting (default: 3.0).

    Returns:
        tuple:
            coefficient_matrix (np.ndarray): (nqso, ncomp) float32
            first_continuum_matrix (np.ndarray): (nqso, nwave) float32 (OBS units)
            final_continuum_matrix (np.ndarray): (nqso, nwave) float32 (OBS units)
            first_cost (np.ndarray): (nqso,) float32  (OBS-based)
            final_cost (np.ndarray): (nqso,) float32  (OBS-based)
            eigvec_range (np.ndarray): (nqso,) fixed-width bytes
            norm_factor (np.ndarray): (nqso,) float32
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

    nqso, _ = flux.shape

    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    n_jobs = int(max(1, n_jobs))

    tasks = [
        (wave, flux[i], ivar[i], float(z[i].item()), eigenspectra, kernel_large, kernel_small, method, maxiters, interp_kind, smoothing_niter, fit_niter, fit_nsigma)
        for i in range(nqso)
    ]

    t0 = time.time()
    chunksize = max(1, nqso // (10 * n_jobs))
    logger.info(f"chunk size for parallel run: {chunksize}")

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
    logger.info(f"Total continuum computation time for {nqso} QSOs: {t1 - t0:.2f} s")

    coeff_list, first_cont_list, cont_list, first_cost_list, cost_list, range_list, zmin_list, zmax_list, norm_list, n_comp_list = zip(*results)

    coefficient_matrix = np.vstack(coeff_list).astype(np.float32)
    first_continuum_matrix = np.vstack(first_cont_list).astype(np.float32)
    final_continuum_matrix = np.vstack(cont_list).astype(np.float32)

    first_cost = np.asarray(first_cost_list, dtype=np.float32)
    final_cost = np.asarray(cost_list, dtype=np.float32)

    eigvec_range = np.asarray(range_list, dtype="S12")
    zmins = np.asarray(zmin_list, dtype=np.float32)
    zmaxs = np.asarray(zmax_list, dtype=np.float32)

    norm_factor = np.asarray(norm_list, dtype=np.float32)
    n_comp = np.asarray(n_comp_list, dtype=np.int32)

    method_array = np.array([method.upper().encode('utf-8')] * nqso, dtype='S4')

    return {
        "z": z,
        "coefficients":coefficient_matrix,
        "first_continuum":first_continuum_matrix,
        "continuum":final_continuum_matrix,
        "first_cost":first_cost,
        "final_cost":final_cost,
        "eigvector_range":eigvec_range,
        "zmin":zmins,
        "zmax":zmaxs,
        "norm_factor":norm_factor,
        "n_comp":n_comp,
        "method":method_array
    }
