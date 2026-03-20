"""Unit tests for nmfqsofit/nmfcontinuum.py (no network / no eigenspectra repo needed)"""

import unittest
import numpy as np

from nmfqsofit.nmfcontinuum import (
    compute_normalization,
    NMFContinuum,
    Eigenset,
    run_parallel_continuum,
    LARGE_CHI2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eigenspectra_dict(n_comp=3, n_wave=80,
                             zmin=0.5, zmax=2.0,
                             lam_min=1000.0, lam_max=3000.0,
                             stat="median"):
    """Return a minimal eigenspectra dict for NMFContinuum."""
    np.random.seed(42)
    rest_wave = np.linspace(lam_min, lam_max, n_wave)
    # Non-negative eigen-vectors
    eigvec = np.abs(np.random.rand(n_comp, n_wave)) + 0.1
    key = (zmin, zmax, lam_min, lam_max, stat)
    return {key: {"wave": rest_wave, "eigvec": eigvec, "headers": {}}}


def _make_fitter(z=1.0, n_wave=200, n_comp=3,
                 kernel_large=11, kernel_small=5,
                 method="nnls", norm_flux=2.0):
    """Build an NMFContinuum with a simple synthetic spectrum."""
    eigenspectra = _make_eigenspectra_dict(
        n_comp=n_comp, n_wave=80, zmin=0.5, zmax=2.0,
        lam_min=1000.0, lam_max=3000.0, stat="median"
    )
    wave_obs = np.linspace(2500.0, 8000.0, n_wave)
    flux = np.full(n_wave, norm_flux, dtype=float)
    ivar = np.ones(n_wave, dtype=float)
    fitter = NMFContinuum(
        wave=wave_obs,
        flux=flux,
        ivar=ivar,
        z=z,
        eigenspectra=eigenspectra,
        kernel_large=kernel_large,
        kernel_small=kernel_small,
        method=method,
        maxiters=50,
        smoothing_niter=2,
        interp_kind="linear",
    )
    return fitter


# ---------------------------------------------------------------------------
# Tests for compute_normalization
# ---------------------------------------------------------------------------

class TestComputeNormalization(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(3500.0, 9500.0, 200)
        self.flux = np.full(200, 3.0)
        self.ivar = np.ones(200)

    def test_median_normalization(self):
        result = compute_normalization(self.wave, self.flux, self.ivar,
                                       lam_min_obs=4000.0, lam_max_obs=8000.0,
                                       stat="median")
        self.assertAlmostEqual(result, 3.0)

    def test_mean_normalization(self):
        result = compute_normalization(self.wave, self.flux, self.ivar,
                                       lam_min_obs=4000.0, lam_max_obs=8000.0,
                                       stat="mean")
        self.assertAlmostEqual(result, 3.0)

    def test_invalid_stat_raises(self):
        with self.assertRaises(ValueError):
            compute_normalization(self.wave, self.flux, self.ivar,
                                  lam_min_obs=4000.0, lam_max_obs=8000.0,
                                  stat="rms")

    def test_too_few_valid_pixels_returns_nan(self):
        # Only 2 pixels in range and all ivar=0 - fewer than 5 valid
        ivar = np.zeros(200)
        result = compute_normalization(self.wave, self.flux, ivar,
                                       lam_min_obs=4000.0, lam_max_obs=8000.0)
        self.assertTrue(np.isnan(result))

    def test_narrow_window_returns_nan(self):
        # Window so narrow it has fewer than 5 pixels
        result = compute_normalization(self.wave, self.flux, self.ivar,
                                       lam_min_obs=5000.0, lam_max_obs=5000.5)
        self.assertTrue(np.isnan(result))


# ---------------------------------------------------------------------------
# Tests for NMFContinuum initialisation
# ---------------------------------------------------------------------------

class TestNMFContinuumInit(unittest.TestCase):
    def test_invalid_method_raises(self):
        eigenspectra = _make_eigenspectra_dict()
        with self.assertRaises(ValueError):
            NMFContinuum(
                wave=np.linspace(3000, 9000, 100),
                flux=np.ones(100),
                ivar=np.ones(100),
                z=1.0,
                eigenspectra=eigenspectra,
                kernel_large=11,
                kernel_small=5,
                method="svd",
                maxiters=50,
                smoothing_niter=2,
                interp_kind="linear",
            )

    def test_even_kernel_raises(self):
        eigenspectra = _make_eigenspectra_dict()
        with self.assertRaises(ValueError):
            NMFContinuum(
                wave=np.linspace(3000, 9000, 100),
                flux=np.ones(100),
                ivar=np.ones(100),
                z=1.0,
                eigenspectra=eigenspectra,
                kernel_large=10,  # even!
                kernel_small=5,
                method="nnls",
                maxiters=50,
                smoothing_niter=2,
                interp_kind="linear",
            )

    def test_none_kernel_accepted(self):
        eigenspectra = _make_eigenspectra_dict()
        fitter = NMFContinuum(
            wave=np.linspace(3000, 9000, 100),
            flux=np.ones(100),
            ivar=np.ones(100),
            z=1.0,
            eigenspectra=eigenspectra,
            kernel_large=None,
            kernel_small=5,
            method="nnls",
            maxiters=50,
            smoothing_niter=2,
            interp_kind="linear",
        )
        self.assertIsNone(fitter.kernel_large)


# ---------------------------------------------------------------------------
# Tests for internal helper methods
# ---------------------------------------------------------------------------

class TestNMFContinuumHelpers(unittest.TestCase):
    def setUp(self):
        self.fitter = _make_fitter()

    def test_keys_as_arrays(self):
        zmins, zmaxs, lam_mins, lam_maxs, stats = self.fitter._keys_as_arrays()
        self.assertEqual(len(zmins), 1)
        self.assertAlmostEqual(zmins[0], 0.5)
        self.assertAlmostEqual(zmaxs[0], 2.0)

    def test_matching_bins_in_range(self):
        # z=1.0 should be in [0.5, 2.0)
        indices = self.fitter._matching_bins()
        self.assertEqual(len(indices), 1)

    def test_matching_bins_out_of_range(self):
        self.fitter.z = 5.0
        indices = self.fitter._matching_bins()
        self.assertEqual(len(indices), 0)

    def test_get_eigenset_by_index_returns_eigenset(self):
        zmin, zmax, lam_min, lam_max, stat, eig = self.fitter._get_eigenset_by_index(0)
        self.assertIsInstance(eig, Eigenset)
        self.assertAlmostEqual(zmin, 0.5)
        self.assertAlmostEqual(zmax, 2.0)

    def test_get_eigenset_from_eigenset_object(self):
        # Replace dict-style entry with an actual Eigenset object
        key = list(self.fitter.eigenspectra_dict.keys())[0]
        item = self.fitter.eigenspectra_dict[key]
        eig_obj = Eigenset(
            rest_wave=np.asarray(item["wave"]),
            eigvec=np.asarray(item["eigvec"]),
        )
        self.fitter.eigenspectra_dict[key] = eig_obj
        _, _, _, _, _, eig = self.fitter._get_eigenset_by_index(0)
        self.assertIsInstance(eig, Eigenset)

    def test_interpolate_eigvec_shape(self):
        key = list(self.fitter.eigenspectra_dict.keys())[0]
        item = self.fitter.eigenspectra_dict[key]
        eig = Eigenset(rest_wave=np.asarray(item["wave"]),
                        eigvec=np.asarray(item["eigvec"]))
        result = self.fitter._interpolate_eigvec_to_observed(eig)
        n_comp = eig.eigvec.shape[0]
        n_wave = self.fitter.wave.size
        self.assertEqual(result.shape, (n_comp, n_wave))

    def test_chi2_reduced_finite(self):
        model = np.full(self.fitter.wave.size, 2.0)
        chi2 = self.fitter._chi2_reduced_against_observed(model, n_comp=3)
        self.assertTrue(np.isfinite(chi2))
        self.assertGreaterEqual(chi2, 0.0)

    def test_chi2_reduced_no_good_pixels_returns_inf(self):
        self.fitter.ivar = np.zeros_like(self.fitter.ivar)
        self.fitter.mask = np.zeros(self.fitter.wave.size, dtype=bool)
        model = np.full(self.fitter.wave.size, 2.0)
        chi2 = self.fitter._chi2_reduced_against_observed(model, n_comp=0)
        self.assertTrue(np.isinf(chi2))


# ---------------------------------------------------------------------------
# Tests for fitting methods
# ---------------------------------------------------------------------------

class TestFitCoeffMethods(unittest.TestCase):
    def setUp(self):
        self.fitter = _make_fitter(n_comp=3)

    def _get_a_interp(self):
        key = list(self.fitter.eigenspectra_dict.keys())[0]
        item = self.fitter.eigenspectra_dict[key]
        eig = Eigenset(rest_wave=np.asarray(item["wave"]),
                        eigvec=np.asarray(item["eigvec"]))
        return self.fitter._interpolate_eigvec_to_observed(eig)

    def test_nnls_coeff_nonneg(self):
        A = self._get_a_interp()
        flux_fit = self.fitter.flux / 2.0
        ivar_fit = self.fitter.ivar
        mask_fit = self.fitter.mask
        coeff, model = self.fitter._fit_coeff_nnls(A, flux_fit, ivar_fit, mask_fit)
        self.assertEqual(coeff.shape, (3,))
        self.assertTrue(np.all(coeff >= 0))
        self.assertEqual(model.shape, (self.fitter.wave.size,))

    def test_nmf_coeff_nonneg(self):
        self.fitter.method = "nmf"
        A = self._get_a_interp()
        flux_fit = self.fitter.flux / 2.0
        ivar_fit = self.fitter.ivar
        mask_fit = self.fitter.mask
        coeff, model = self.fitter._fit_coeff_nmf(A, flux_fit, ivar_fit, mask_fit)
        self.assertEqual(coeff.shape, (3,))
        self.assertTrue(np.all(coeff >= 0))

    def test_nnls_raises_when_too_few_pixels(self):
        A = self._get_a_interp()
        mask_fit = np.zeros(A.shape[1], dtype=bool)  # no valid pixels
        with self.assertRaises(ValueError):
            self.fitter._fit_coeff_nnls(A, self.fitter.flux,
                                         self.fitter.ivar, mask_fit)

    def test_nmf_raises_when_too_few_pixels(self):
        A = self._get_a_interp()
        mask_fit = np.zeros(A.shape[1], dtype=bool)
        with self.assertRaises(ValueError):
            self.fitter._fit_coeff_nmf(A, self.fitter.flux,
                                        self.fitter.ivar, mask_fit)


# ---------------------------------------------------------------------------
# Tests for _apply_smooth_correction
# ---------------------------------------------------------------------------

class TestApplySmoothCorrection(unittest.TestCase):
    def setUp(self):
        self.fitter = _make_fitter(n_wave=200, kernel_large=11, kernel_small=5)

    def test_returns_array_same_shape(self):
        first_cont = np.full(200, 2.0)
        result = self.fitter._apply_smooth_correction(
            first_cont, kernel_large=11, kernel_small=5, smoothing_niter=2
        )
        self.assertEqual(result.shape, first_cont.shape)

    def test_negative_kernel_skips_smoothing(self):
        first_cont = np.full(200, 2.0)
        result = self.fitter._apply_smooth_correction(
            first_cont, kernel_large=-1, kernel_small=-1, smoothing_niter=2
        )
        # With None kernels the smooth ratio stays 1 - result == first_cont
        np.testing.assert_allclose(result, first_cont)

    def test_even_kernel_raises(self):
        first_cont = np.full(200, 2.0)
        with self.assertRaises(ValueError):
            self.fitter._apply_smooth_correction(
                first_cont, kernel_large=10, kernel_small=5, smoothing_niter=1
            )

    def test_output_finite_when_input_finite(self):
        first_cont = np.full(200, 2.0)
        result = self.fitter._apply_smooth_correction(
            first_cont, kernel_large=11, kernel_small=5, smoothing_niter=3
        )
        self.assertTrue(np.all(np.isfinite(result)))


# ---------------------------------------------------------------------------
# Tests for NMFContinuum.fit (end-to-end for a single spectrum)
# ---------------------------------------------------------------------------

class TestNMFContinuumFit(unittest.TestCase):
    def test_fit_nnls_produces_valid_output(self):
        fitter = _make_fitter(method="nnls")
        fitter.fit()
        self.assertIsNotNone(fitter.coeff)
        self.assertEqual(fitter.coeff.shape, (3,))
        self.assertTrue(np.all(fitter.coeff >= 0))
        self.assertEqual(fitter.continuum.shape, (200,))
        self.assertTrue(np.isfinite(fitter.first_cost))
        self.assertTrue(np.isfinite(fitter.cost))

    def test_fit_nmf_produces_valid_output(self):
        fitter = _make_fitter(method="nmf")
        fitter.fit()
        self.assertIsNotNone(fitter.coeff)
        self.assertEqual(fitter.continuum.shape, (200,))

    def test_fit_out_of_range_z_uses_zero_fallback(self):
        fitter = _make_fitter(z=10.0)  # z far outside any bin
        fitter.fit()
        np.testing.assert_array_equal(fitter.coeff, np.zeros(3))
        np.testing.assert_array_equal(fitter.continuum, np.zeros(200))
        self.assertAlmostEqual(fitter.first_cost, LARGE_CHI2)

    def test_fit_invalid_norm_uses_zero_fallback(self):
        # Make ivar=0 everywhere - norm = nan - no valid eigenset - fallback
        fitter = _make_fitter()
        fitter.ivar = np.zeros_like(fitter.ivar)
        fitter.mask = np.zeros(fitter.wave.size, dtype=bool)
        fitter.fit()
        np.testing.assert_array_equal(fitter.coeff, np.zeros(3))

    def test_fit_eigvec_range_string(self):
        fitter = _make_fitter()
        fitter.fit()
        # Should be a string of the form z_NNN_NNN
        self.assertIsInstance(fitter.eigvec_range, str)
        self.assertTrue(fitter.eigvec_range.startswith("z_"))


# ---------------------------------------------------------------------------
# Tests for run_parallel_continuum input validation
# ---------------------------------------------------------------------------

class TestRunParallelContinuumValidation(unittest.TestCase):
    def setUp(self):
        self.eigenspectra = _make_eigenspectra_dict()
        n_wave = 200
        self.wave = np.linspace(2500.0, 8000.0, n_wave)

    def test_flux_not_2d_raises(self):
        with self.assertRaises(ValueError):
            run_parallel_continuum(
                wave=self.wave,
                flux=np.ones(200),
                ivar=np.ones(200),
                z=np.array([1.0]),
                eigenspectra=self.eigenspectra,
                kernel_large=11,
                kernel_small=5,
                method="nnls",
                n_jobs=1,
            )

    def test_ivar_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            run_parallel_continuum(
                wave=self.wave,
                flux=np.ones((2, 200)),
                ivar=np.ones((3, 200)),
                z=np.array([1.0, 1.5]),
                eigenspectra=self.eigenspectra,
                kernel_large=11,
                kernel_small=5,
                method="nnls",
                n_jobs=1,
            )

    def test_z_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            run_parallel_continuum(
                wave=self.wave,
                flux=np.ones((2, 200)),
                ivar=np.ones((2, 200)),
                z=np.array([1.0]),   # one too short
                eigenspectra=self.eigenspectra,
                kernel_large=11,
                kernel_small=5,
                method="nnls",
                n_jobs=1,
            )

    def test_n_jobs_minus_one_accepted(self):
        # n_jobs=-1 should not raise (uses all CPUs)
        out = run_parallel_continuum(
            wave=self.wave,
            flux=np.ones((1, 200)) * 2.0,
            ivar=np.ones((1, 200)),
            z=np.array([1.0]),
            eigenspectra=self.eigenspectra,
            kernel_large=11,
            kernel_small=5,
            method="nnls",
            n_jobs=-1,
        )
        self.assertEqual(out["continuum"].shape, (1, 200))

    def test_basic_run_output_keys(self):
        out = run_parallel_continuum(
            wave=self.wave,
            flux=np.ones((2, 200)) * 2.0,
            ivar=np.ones((2, 200)),
            z=np.array([1.0, 1.2]),
            eigenspectra=self.eigenspectra,
            kernel_large=11,
            kernel_small=5,
            method="nnls",
            n_jobs=1,
        )
        for key in ("coefficients", "first_continuum", "continuum",
                     "first_cost", "final_cost", "eigvector_range",
                     "norm_factor", "z", "zmin", "zmax", "n_comp"):
            self.assertIn(key, out)
        self.assertEqual(out["continuum"].shape, (2, 200))
        self.assertEqual(out["coefficients"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
