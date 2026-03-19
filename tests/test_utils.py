"""Unit tests for nmfqsofit/utils.py"""

import os
import tempfile
import time
import unittest
from unittest.mock import patch
import argparse

import numpy as np
from astropy.io import fits
from astropy.table import Table

from nmfqsofit.utils import (
    elapsed,
    parse_qso_sequence,
    read_nqso_from_header,
    interpolation1D,
    _parse_headers,
    plot_flux_and_continuum,
)


# ---------------------------------------------------------------------------
# Tests for elapsed
# ---------------------------------------------------------------------------

class TestElapsed(unittest.TestCase):
    def test_returns_current_time(self):
        start = time.time()
        result = elapsed(start, "test msg")
        self.assertGreaterEqual(result, start)

    def test_prints_message(self):
        import io, sys
        buf = io.StringIO()
        start = time.time()
        sys.stdout = buf
        try:
            elapsed(start, "hello elapsed")
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("hello elapsed", buf.getvalue())

    def test_uses_logger_when_requested(self):
        import logging
        start = time.time()
        with self.assertLogs("nmfqsofit.utils", level="INFO") as cm:
            elapsed(start, "logged msg", use_logger=True)
        self.assertTrue(any("logged msg" in m for m in cm.output))

    def test_start_none_does_not_print(self):
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        try:
            result = elapsed(None, "should not appear")
        finally:
            sys.stdout = sys.__stdout__
        self.assertEqual(buf.getvalue(), "")
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# Tests for parse_qso_sequence
# ---------------------------------------------------------------------------

class TestParseQsoSequence(unittest.TestCase):
    def test_integer_input(self):
        result = parse_qso_sequence(5)
        np.testing.assert_array_equal(result, np.arange(5))

    def test_string_digit(self):
        result = parse_qso_sequence("10")
        np.testing.assert_array_equal(result, np.arange(10))

    def test_range_no_step(self):
        result = parse_qso_sequence("2-5")
        np.testing.assert_array_equal(result, np.arange(2, 6))

    def test_range_with_step(self):
        result = parse_qso_sequence("0-10:2")
        np.testing.assert_array_equal(result, np.arange(0, 11, 2))

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            parse_qso_sequence("abc-xyz")

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            parse_qso_sequence("not-a-range-with-letters")


# ---------------------------------------------------------------------------
# Tests for read_nqso_from_header
# ---------------------------------------------------------------------------

class TestReadNqsoFromHeader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_fits_with_metadata(self, n_rows=5):
        meta = Table()
        meta["Z"] = np.linspace(0.5, 2.0, n_rows)
        path = os.path.join(self.tmpdir, "test.fits")
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU(meta, name="METADATA"),
        ])
        hdul.writeto(path, overwrite=True)
        return path

    def test_reads_naxis2(self):
        path = self._make_fits_with_metadata(7)
        result = read_nqso_from_header(path)
        self.assertEqual(result, 7)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_nqso_from_header("/nonexistent/file.fits")

    def test_missing_hdu_raises(self):
        path = os.path.join(self.tmpdir, "no_meta.fits")
        fits.HDUList([fits.PrimaryHDU()]).writeto(path, overwrite=True)
        with self.assertRaises(ValueError):
            read_nqso_from_header(path)

    def test_custom_hdu_name(self):
        meta = Table()
        meta["X"] = np.arange(4)
        path = os.path.join(self.tmpdir, "custom_hdu.fits")
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU(meta, name="MYDATA"),
        ])
        hdul.writeto(path, overwrite=True)
        result = read_nqso_from_header(path, hdu_name="MYDATA")
        self.assertEqual(result, 4)


# ---------------------------------------------------------------------------
# Tests for interpolation1D
# ---------------------------------------------------------------------------

class TestInterpolation1D(unittest.TestCase):
    def setUp(self):
        self.xold = np.linspace(0.0, 10.0, 100)
        self.yold = np.sin(self.xold)

    def test_linear_interpolation(self):
        xnew = np.array([0.0, 5.0, 10.0])
        result = interpolation1D(xnew, self.xold, self.yold, kind="linear")
        self.assertEqual(result.shape, (3,))
        # End-points should be within interpolation range → finite
        self.assertTrue(np.isfinite(result[0]))

    def test_outside_range_gives_fill_value(self):
        xnew = np.array([-1.0, 20.0])
        result = interpolation1D(xnew, self.xold, self.yold, kind="linear",
                                  fill_value=np.nan)
        self.assertTrue(np.all(np.isnan(result)))

    def test_custom_fill_value(self):
        xnew = np.array([-5.0, 50.0])
        result = interpolation1D(xnew, self.xold, self.yold, kind="linear",
                                  fill_value=-999.0)
        np.testing.assert_array_equal(result, [-999.0, -999.0])

    def test_cubic_interpolation(self):
        # Use cubic spline
        xnew = np.linspace(1.0, 9.0, 20)
        result = interpolation1D(xnew, self.xold, self.yold, kind="cubic")
        expected = np.sin(xnew)
        # Should be close to sin(x)
        np.testing.assert_allclose(result, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Tests for _parse_headers
# ---------------------------------------------------------------------------

class TestParseHeaders(unittest.TestCase):
    def _make_args(self, headers=None):
        args = argparse.Namespace()
        args.headers = headers or []
        args.method = "nnls"
        args.eigenspectra = "/some/path"
        args.kernel_size = 141
        args.kernel_small = 71
        args.maxiters = 100
        args.smoothing_niter = 3
        return args

    def test_basic_parse(self):
        args = self._make_args(headers=["SURVEY=SDSS", "RELEASE=DR16"])
        result = _parse_headers(args)
        self.assertEqual(result["SURVEY"], "SDSS")
        self.assertEqual(result["RELEASE"], "DR16")
        self.assertIn("METHOD", result)
        self.assertIn("EIGSPEC", result)

    def test_invalid_header_no_equals_raises(self):
        args = self._make_args(headers=["BADHEADER"])
        with self.assertRaises(ValueError):
            _parse_headers(args)

    def test_invalid_header_empty_key_raises(self):
        args = self._make_args(headers=["=VALUE"])
        with self.assertRaises(ValueError):
            _parse_headers(args)

    def test_multiple_equals_raises(self):
        args = self._make_args(headers=["KEY=A=B"])
        with self.assertRaises(ValueError):
            _parse_headers(args)

    def test_version_keys_present(self):
        args = self._make_args()
        result = _parse_headers(args)
        # Should include DEPNAM00 / DEPVER00 etc.
        self.assertIn("DEPNAM00", result)
        self.assertIn("DEPVER00", result)

    def test_origauth_and_gitrepo_always_set(self):
        args = self._make_args()
        result = _parse_headers(args)
        self.assertIn("ORIGAUTH", result)
        self.assertIn("GITREPO", result)


# ---------------------------------------------------------------------------
# Tests for plot_flux_and_continuum
# ---------------------------------------------------------------------------

class TestPlotFluxAndContinuum(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend

    def _make_data(self):
        wave = np.linspace(3500, 9500, 100)
        flux = np.random.rand(100) + 1.0
        continuum = np.ones(100) * 1.2
        return wave, flux, continuum

    def test_returns_axes_when_no_save(self):
        import matplotlib.axes
        wave, flux, cont = self._make_data()
        ax = plot_flux_and_continuum(wave, flux, cont)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_saves_file(self):
        wave, flux, cont = self._make_data()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = plot_flux_and_continuum(wave, flux, cont, save_file=path)
            self.assertIsNone(result)  # should return None when saving
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_with_title(self):
        import matplotlib.axes
        wave, flux, cont = self._make_data()
        ax = plot_flux_and_continuum(wave, flux, cont, title="Test QSO")
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_with_custom_kwargs(self):
        import matplotlib.axes
        wave, flux, cont = self._make_data()
        ax = plot_flux_and_continuum(
            wave, flux, cont,
            flux_kwargs={"color": "blue", "lw": 2},
            continuum_kwargs={"color": "green"},
            ax_kwargs={"xlim": (4000, 8000)},
        )
        self.assertIsInstance(ax, matplotlib.axes.Axes)


if __name__ == "__main__":
    unittest.main()
