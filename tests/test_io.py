"""Unit tests for nmfqsofit/io.py"""

import os
import tempfile
import unittest
from pathlib import Path
from io import BytesIO

import numpy as np
from astropy.io import fits
from astropy.table import Table

from nmfqsofit.io import (
    _find_hdu_name,
    read_fits_file,
    write_continuum,
    QSOSpecRead,
    load_all_eigenspectra,
    read_continuum_file,
    ContinuumSpec,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic FITS files
# ---------------------------------------------------------------------------

def _make_qso_spectra_fits(tmpdir, n_qso=3, n_wave=50, rename_z=False):
    """Create a minimal QSO spectra FITS file and return the path."""
    wave = np.linspace(3500.0, 9500.0, n_wave).astype(np.float32)
    flux = np.ones((n_qso, n_wave), dtype=np.float32)
    ivar = np.ones((n_qso, n_wave), dtype=np.float32)

    meta = Table()
    z_col = "Z_QSO" if rename_z else "Z"
    meta[z_col] = np.linspace(0.5, 2.0, n_qso)

    path = os.path.join(tmpdir, "test_qso.fits")
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=wave, name="WAVELENGTH"),
        fits.ImageHDU(data=flux, name="FLUX"),
        fits.ImageHDU(data=ivar, name="IVAR"),
        fits.BinTableHDU(meta, name="METADATA"),
    ])
    hdul.writeto(path, overwrite=True)
    return path


def _make_eigenspectra_fits(tmpdir, zmin=50, zmax=150, lam_min=1000.0,
                             lam_max=3000.0, stat="median", n_wave=50, n_comp=3,
                             filename=None):
    """Create a minimal eigenspectra FITS file and return the path."""
    if filename is None:
        filename = f"eigspec_{zmin:03d}_{zmax:03d}.fits"
    rest_wave = np.linspace(lam_min, lam_max, n_wave).astype(np.float64)
    eigvec = np.random.rand(n_wave, n_comp).astype(np.float64)

    hdr = fits.Header()
    hdr["LAMSTART"] = lam_min
    hdr["LAMEND"] = lam_max
    hdr["NORMSTAT"] = stat

    path = os.path.join(tmpdir, filename)
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(data=rest_wave, name="REST_WAVE"),
        fits.ImageHDU(data=eigvec, name="EIGENVEC"),
    ])
    hdul.writeto(path, overwrite=True)
    return path


def _make_continuum_fits(tmpdir, n_qso=3, n_wave=50, n_comp=3):
    """Create a minimal continuum output FITS file and return the path."""
    coeff = np.ones((n_qso, n_comp), dtype=np.float32)
    first_cont = np.ones((n_qso, n_wave), dtype=np.float32)
    cont = np.ones((n_qso, n_wave), dtype=np.float32) * 1.1

    meta = Table()
    meta["FIRST_COST"] = np.ones(n_qso, dtype=np.float32)
    meta["FINAL_COST"] = np.ones(n_qso, dtype=np.float32) * 0.9
    meta["EIGVECTOR_RANGE"] = [b"z_050_150"] * n_qso

    path = os.path.join(tmpdir, "test_continuum.fits")
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=coeff, name="COEFFICIENTS"),
        fits.ImageHDU(data=first_cont, name="FIRST_CONTINUUM"),
        fits.ImageHDU(data=cont, name="CONTINUUM"),
        fits.BinTableHDU(meta, name="METADATA"),
    ])
    hdul.writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# Tests for _find_hdu_name
# ---------------------------------------------------------------------------

class TestFindHduName(unittest.TestCase):
    def _make_hdul(self, names):
        hdus = [fits.PrimaryHDU()]
        for n in names:
            hdus.append(fits.ImageHDU(name=n))
        return fits.HDUList(hdus)

    def test_exact_match(self):
        hdul = self._make_hdul(["FLUX", "IVAR", "WAVELENGTH"])
        self.assertEqual(_find_hdu_name(hdul, "FLUX"), "FLUX")

    def test_case_insensitive_exact(self):
        # FITS normalises extension names to uppercase; _find_hdu_name should still match
        hdul = self._make_hdul(["FLUX", "IVAR"])
        result = _find_hdu_name(hdul, "flux")
        self.assertEqual(result.upper(), "FLUX")

    def test_substring_fallback(self):
        # Uses substring match when no exact match
        hdul = self._make_hdul(["B_FLUX_EXT", "IVAR_EXT"])
        result = _find_hdu_name(hdul, "FLUX")
        self.assertIn("FLUX", result.upper())

    def test_keyword_not_found_raises(self):
        hdul = self._make_hdul(["IVAR", "WAVELENGTH"])
        with self.assertRaises(KeyError):
            _find_hdu_name(hdul, "FLUX")


# ---------------------------------------------------------------------------
# Tests for read_fits_file
# ---------------------------------------------------------------------------

class TestReadFitsFile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_fits_file("/nonexistent/file.fits")

    def test_read_all_rows(self):
        path = _make_qso_spectra_fits(self.tmpdir, n_qso=3, n_wave=50)
        header, flux, ivar, wave, meta = read_fits_file(path)
        self.assertEqual(flux.shape, (3, 50))
        self.assertEqual(ivar.shape, (3, 50))
        self.assertEqual(wave.shape, (50,))
        self.assertEqual(len(meta), 3)

    def test_read_single_int_index(self):
        path = _make_qso_spectra_fits(self.tmpdir, n_qso=3, n_wave=50)
        header, flux, ivar, wave, meta = read_fits_file(path, index=1)
        self.assertEqual(flux.ndim, 1)
        self.assertEqual(flux.shape[0], 50)
        self.assertEqual(len(meta), 1)

    def test_read_numpy_integer_index(self):
        path = _make_qso_spectra_fits(self.tmpdir, n_qso=3, n_wave=50)
        header, flux, ivar, wave, meta = read_fits_file(path, index=np.int64(2))
        self.assertEqual(flux.ndim, 1)
        self.assertEqual(len(meta), 1)

    def test_read_array_index(self):
        path = _make_qso_spectra_fits(self.tmpdir, n_qso=3, n_wave=50)
        header, flux, ivar, wave, meta = read_fits_file(path, index=[0, 2])
        self.assertEqual(flux.shape, (2, 50))
        self.assertEqual(len(meta), 2)

    def test_z_qso_renamed_to_z(self):
        path = _make_qso_spectra_fits(self.tmpdir, n_qso=2, n_wave=50, rename_z=True)
        _, _, _, _, meta = read_fits_file(path)
        self.assertIn("Z", meta.colnames)
        self.assertNotIn("Z_QSO", meta.colnames)


# ---------------------------------------------------------------------------
# Tests for write_continuum
# ---------------------------------------------------------------------------

class TestWriteContinuum(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        n_qso, n_wave, n_comp = 3, 50, 4
        self.results = {
            "coefficients": np.random.rand(n_qso, n_comp).astype(np.float32),
            "first_continuum": np.random.rand(n_qso, n_wave).astype(np.float32),
            "continuum": np.random.rand(n_qso, n_wave).astype(np.float32),
            "first_cost": np.ones(n_qso, dtype=np.float32),
            "final_cost": np.ones(n_qso, dtype=np.float32) * 0.9,
            "eigvector_range": np.array([b"z_050_150"] * n_qso),
            "z": np.linspace(0.5, 2.0, n_qso),
            "zmin": np.full(n_qso, 0.5, dtype=np.float32),
            "zmax": np.full(n_qso, 1.5, dtype=np.float32),
            "norm_factor": np.ones(n_qso, dtype=np.float32),
            "n_comp": np.full(n_qso, n_comp, dtype=np.int32),
        }

    def test_write_and_verify(self):
        path = os.path.join(self.tmpdir, "out.fits")
        write_continuum(self.results, {"SURVEY": "TEST"}, path)
        self.assertTrue(os.path.exists(path))
        with fits.open(path) as hdul:
            self.assertIn("COEFFICIENTS", hdul)
            self.assertIn("FIRST_CONTINUUM", hdul)
            self.assertIn("CONTINUUM", hdul)
            self.assertIn("METADATA", hdul)
            self.assertEqual(hdul[0].header["SURVEY"], "TEST")

    def test_write_no_headers(self):
        path = os.path.join(self.tmpdir, "out_noheader.fits")
        write_continuum(self.results, None, path)
        self.assertTrue(os.path.exists(path))

    def test_raises_if_coefficients_not_2d(self):
        bad_results = dict(self.results)
        bad_results["coefficients"] = np.ones(4)
        with self.assertRaises(ValueError):
            write_continuum(bad_results, {}, os.path.join(self.tmpdir, "x.fits"))

    def test_raises_if_continuum_shape_mismatch(self):
        bad_results = dict(self.results)
        bad_results["first_continuum"] = np.ones((3, 60), dtype=np.float32)
        with self.assertRaises(ValueError):
            write_continuum(bad_results, {}, os.path.join(self.tmpdir, "y.fits"))

    def test_raises_if_continuum_not_2d(self):
        bad_results = dict(self.results)
        bad_results["first_continuum"] = np.ones((3, 50, 2), dtype=np.float32)
        bad_results["continuum"] = np.ones((3, 50, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            write_continuum(bad_results, {}, os.path.join(self.tmpdir, "z.fits"))

    def test_raises_if_coeff_nspec_mismatch(self):
        bad_results = dict(self.results)
        bad_results["coefficients"] = np.ones((5, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            write_continuum(bad_results, {}, os.path.join(self.tmpdir, "w.fits"))

    def test_raises_if_cost_length_mismatch(self):
        bad_results = dict(self.results)
        bad_results["first_cost"] = np.ones(5, dtype=np.float32)
        with self.assertRaises(ValueError):
            write_continuum(bad_results, {}, os.path.join(self.tmpdir, "v.fits"))

    def test_raises_if_eigvector_range_mismatch(self):
        bad_results = dict(self.results)
        bad_results["eigvector_range"] = np.array([b"z_050_150"] * 5)
        with self.assertRaises(ValueError):
            write_continuum(bad_results, {}, os.path.join(self.tmpdir, "u.fits"))


# ---------------------------------------------------------------------------
# Tests for QSOSpecRead
# ---------------------------------------------------------------------------

class TestQSOSpecRead(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = _make_qso_spectra_fits(self.tmpdir, n_qso=3, n_wave=50)

    def test_autoload_true(self):
        spec = QSOSpecRead(self.path, autoload=True, verbose=False)
        self.assertIsNotNone(spec.flux)
        self.assertIsNotNone(spec.wavelength)
        self.assertEqual(spec.flux.shape, (3, 50))

    def test_autoload_false(self):
        spec = QSOSpecRead(self.path, autoload=False)
        self.assertIsNone(spec.flux)

    def test_autoload_with_index(self):
        spec = QSOSpecRead(self.path, index=[0, 1], autoload=True, verbose=False)
        self.assertEqual(spec.flux.shape, (2, 50))

    def test_verbose_true_does_not_crash(self):
        # verbose=True uses the logger, just check no exception
        spec = QSOSpecRead(self.path, autoload=True, verbose=True)
        self.assertIsNotNone(spec.flux)

    def test_file_not_found_raises(self):
        spec = QSOSpecRead("/nonexistent/path.fits", autoload=False)
        with self.assertRaises(IOError):
            spec.read_fits()

    def test_read_fits_manually(self):
        spec = QSOSpecRead(self.path, autoload=False)
        spec.read_fits()
        self.assertIsNotNone(spec.flux)


# ---------------------------------------------------------------------------
# Tests for load_all_eigenspectra
# ---------------------------------------------------------------------------

class TestLoadAllEigenspectra(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_path_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_all_eigenspectra("/nonexistent/path")

    def test_empty_directory_raises(self):
        with self.assertRaises(ValueError):
            load_all_eigenspectra(self.tmpdir)

    def test_directory_no_valid_fits_raises(self):
        # Put a FITS file with no z-bin pattern in name
        path = os.path.join(self.tmpdir, "no_pattern.fits")
        fits.PrimaryHDU().writeto(path)
        with self.assertRaises(ValueError):
            load_all_eigenspectra(self.tmpdir)

    def test_load_from_directory(self):
        _make_eigenspectra_fits(self.tmpdir, zmin=50, zmax=150)
        result = load_all_eigenspectra(self.tmpdir)
        self.assertEqual(len(result), 1)
        key = list(result.keys())[0]
        self.assertAlmostEqual(key[0], 0.50)
        self.assertAlmostEqual(key[1], 1.50)

    def test_load_from_single_file(self):
        path = _make_eigenspectra_fits(self.tmpdir, zmin=100, zmax=200)
        result = load_all_eigenspectra(path)
        self.assertEqual(len(result), 1)

    def test_load_multiple_files(self):
        _make_eigenspectra_fits(self.tmpdir, zmin=50, zmax=150, filename="eigspec_050_150.fits")
        _make_eigenspectra_fits(self.tmpdir, zmin=150, zmax=250, filename="eigspec_150_250.fits")
        result = load_all_eigenspectra(self.tmpdir)
        self.assertEqual(len(result), 2)

    def test_duplicate_key_raises(self):
        # Two files that resolve to the same key (same zmin, zmax, lam_min, lam_max, stat)
        _make_eigenspectra_fits(self.tmpdir, zmin=50, zmax=150, filename="eigspec_050_150_a.fits")
        _make_eigenspectra_fits(self.tmpdir, zmin=50, zmax=150, filename="eigspec_050_150_b.fits")
        with self.assertRaises(ValueError):
            load_all_eigenspectra(self.tmpdir)

    def test_missing_required_hdus_raises(self):
        # Build a FITS without REST_WAVE / EIGENVEC
        hdr = fits.Header()
        hdr["LAMSTART"] = 1000.0
        hdr["LAMEND"] = 3000.0
        hdr["NORMSTAT"] = "median"
        path = os.path.join(self.tmpdir, "eigspec_050_150.fits")
        fits.HDUList([fits.PrimaryHDU(header=hdr)]).writeto(path, overwrite=True)
        with self.assertRaises(KeyError):
            load_all_eigenspectra(path)

    def test_eigvec_shape_transposed_correctly(self):
        n_wave, n_comp = 50, 3
        _make_eigenspectra_fits(self.tmpdir, n_wave=n_wave, n_comp=n_comp,
                                filename="eigspec_050_150.fits")
        result = load_all_eigenspectra(self.tmpdir)
        key = list(result.keys())[0]
        # eigvec should be (ncomp, nwave) after .T
        self.assertEqual(result[key]["eigvec"].shape, (n_comp, n_wave))


# ---------------------------------------------------------------------------
# Tests for read_continuum_file
# ---------------------------------------------------------------------------

class TestReadContinuumFile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = _make_continuum_fits(self.tmpdir, n_qso=3, n_wave=50, n_comp=4)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_continuum_file("/nonexistent/file.fits")

    def test_read_all(self):
        header, coeff, first_cont, cont, meta = read_continuum_file(self.path)
        self.assertEqual(coeff.shape, (3, 4))
        self.assertEqual(first_cont.shape, (3, 50))
        self.assertEqual(cont.shape, (3, 50))
        self.assertEqual(len(meta), 3)

    def test_read_int_index(self):
        header, coeff, first_cont, cont, meta = read_continuum_file(self.path, index=1)
        self.assertEqual(coeff.ndim, 1)
        self.assertEqual(first_cont.ndim, 1)
        self.assertEqual(len(meta), 1)

    def test_read_array_index(self):
        header, coeff, first_cont, cont, meta = read_continuum_file(self.path, index=[0, 2])
        self.assertEqual(coeff.shape, (2, 4))
        self.assertEqual(first_cont.shape, (2, 50))

    def test_missing_hdu_raises(self):
        # Build a FITS missing COEFFICIENTS
        path = os.path.join(self.tmpdir, "bad_cont.fits")
        meta = Table()
        meta["X"] = np.ones(3)
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((3, 50)), name="FIRST_CONTINUUM"),
            fits.ImageHDU(data=np.ones((3, 50)), name="CONTINUUM"),
            fits.BinTableHDU(meta, name="METADATA"),
        ])
        hdul.writeto(path, overwrite=True)
        with self.assertRaises(KeyError):
            read_continuum_file(path)


# ---------------------------------------------------------------------------
# Tests for ContinuumSpec
# ---------------------------------------------------------------------------

class TestContinuumSpec(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = _make_continuum_fits(self.tmpdir, n_qso=3, n_wave=50, n_comp=4)

    def test_autoload_true(self):
        spec = ContinuumSpec(self.path, autoload=True, verbose=False)
        self.assertIsNotNone(spec.continuum)
        self.assertEqual(spec.continuum.shape, (3, 50))

    def test_autoload_false_no_data(self):
        spec = ContinuumSpec(self.path, autoload=False)
        self.assertIsNone(spec.continuum)

    def test_read_fits_manually(self):
        spec = ContinuumSpec(self.path, autoload=False, verbose=False)
        spec.read_fits()
        self.assertIsNotNone(spec.continuum)

    def test_file_not_found_raises(self):
        spec = ContinuumSpec("/nonexistent/path.fits", autoload=False)
        with self.assertRaises(FileNotFoundError):
            spec.read_fits()

    def test_verbose_prints_timing(self):
        import logging
        with self.assertLogs("nmfqsofit.io", level="INFO") as cm:
            spec = ContinuumSpec(self.path, autoload=True, verbose=True)
        self.assertTrue(any("Time taken" in msg for msg in cm.output))

    def test_with_index(self):
        spec = ContinuumSpec(self.path, index=0, autoload=True, verbose=False)
        self.assertEqual(spec.continuum.ndim, 1)


if __name__ == "__main__":
    unittest.main()
