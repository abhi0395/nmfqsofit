# tests/test_nmfqsofit.py

import unittest
import tempfile
import os
from pathlib import Path
import subprocess

import numpy as np

from nmfqsofit.io import QSOSpecRead, load_all_eigenspectra
from nmfqsofit.nmfcontinuum import run_parallel_continuum
from nmfqsofit.utils import read_nqso_from_header

EIGEN_REPO = "https://github.com/abhi0395/nmfeigenspectra.git"

class TestNMFContinuum(unittest.TestCase):
    """
    Minimal integration test.

    This test is intentionally small and only checks that:
      - packaged/checked-in small test spectra FITS can be read
      - eigenspectra directory can be loaded
      - continuum fitting runs end-to-end without crashing
      - output array shapes are consistent

    It does NOT validate scientific accuracy.
    """

    def setUp(self):

        self.tmpdir = Path(tempfile.mkdtemp())

        # Local clone path inside temp dir
        self.test_eigenspectra_repo = self.tmpdir / "nmfeigenspectra"

        # IMPORTANT: in that repo, the FITS files live under "sdss/"
        self.test_eigenspectra_dir = self.test_eigenspectra_repo / "sdss"

        # Clone repo
        subprocess.run(
            ["git", "clone", "--depth", "1", EIGEN_REPO, str(self.test_eigenspectra_repo)],
            check=True,
        )

        self.fi = np.random.randint(0,2)
        if self.fi > 1:
            self.fi = 1

        repo_root = Path(__file__).resolve().parents[1]

        self.test_spectra_files = [
            repo_root / "data" / "test_desi_spectra.fits",
            repo_root / "data" / "test_sdss_spectra.fits",
        ]

        print("DEBUG test files:", self.test_spectra_files)
        print("DEBUG test eigenspectra:", self.test_eigenspectra_dir)

        for f in self.test_spectra_files:
            self.assertTrue(f.exists(), f"File {f} does not exist")

        self.assertTrue(
            self.test_eigenspectra_dir.exists(),
            f"Missing test eigenspectra directory: {self.test_eigenspectra_dir}"
        )

    def test_nmf_end_to_end_fit_runs(self):
        # Read a small number of QSOs (using NMF)
        nqso_total = int(read_nqso_from_header(str(self.test_spectra_files[self.fi])))
        n_take = min(2, nqso_total) # keep it minimal
        indices = np.arange(n_take)

        spec = QSOSpecRead(
            str(self.test_spectra_files[self.fi]),
            index=indices,
            autoload=True,
            verbose=False,
        )

        eigenspectra = load_all_eigenspectra(str(self.test_eigenspectra_dir))

        z = np.asarray(spec.metadata["Z"], dtype=float)

        # Run with 1 process in tests
        out = run_parallel_continuum(
            wave=spec.wavelength,
            flux=spec.flux,
            ivar=spec.ivar,
            z=z,
            eigenspectra=eigenspectra,
            kernel_size=71,
            method="nmf",
            n_jobs=1,
        )

        (
            coeff_mat,
            first_cont_mat,
            cont_mat,
            first_cost,
            cost,
            eigvec_range,
        ) = out

        # Basic shape checks (main goal: ensure nothing is broken)
        self.assertEqual(coeff_mat.shape[0], n_take)
        self.assertEqual(first_cont_mat.shape, spec.flux.shape)
        self.assertEqual(cont_mat.shape, spec.flux.shape)
        self.assertEqual(first_cost.shape, (n_take,))
        self.assertEqual(cost.shape, (n_take,))
        self.assertEqual(eigvec_range.shape, (n_take,))

        # Sanity: costs must be finite
        self.assertTrue(np.all(np.isfinite(first_cost)))
        self.assertTrue(np.all(np.isfinite(cost)))

        # Continuum should be finite at least where ivar>0
        good = spec.ivar > 0
        self.assertTrue(np.all(np.isfinite(cont_mat[good])))

    def test_nnls_end_to_end_fit_runs(self):
        # Read a small number of QSOs (using NNLS)
        nqso_total = int(read_nqso_from_header(str(self.test_spectra_files[self.fi])))
        n_take = min(2, nqso_total)  # keep it minimal
        indices = np.arange(n_take)

        spec = QSOSpecRead(
            str(self.test_spectra_files[self.fi]),
            index=indices,
            autoload=True,
            verbose=False,
        )

        eigenspectra = load_all_eigenspectra(str(self.test_eigenspectra_dir))

        z = np.asarray(spec.metadata["Z"], dtype=float)

        # Run with 1 process in tests
        out = run_parallel_continuum(
            wave=spec.wavelength,
            flux=spec.flux,
            ivar=spec.ivar,
            z=z,
            eigenspectra=eigenspectra,
            kernel_size=71,
            method="nnls",
            n_jobs=1,
        )

        (
            coeff_mat,
            first_cont_mat,
            cont_mat,
            first_cost,
            cost,
            eigvec_range,
        ) = out

        # Basic shape checks (main goal: ensure nothing is broken)
        self.assertEqual(coeff_mat.shape[0], n_take)
        self.assertEqual(first_cont_mat.shape, spec.flux.shape)
        self.assertEqual(cont_mat.shape, spec.flux.shape)
        self.assertEqual(first_cost.shape, (n_take,))
        self.assertEqual(cost.shape, (n_take,))
        self.assertEqual(eigvec_range.shape, (n_take,))

        # Sanity: costs must be finite
        self.assertTrue(np.all(np.isfinite(first_cost)))
        self.assertTrue(np.all(np.isfinite(cost)))

        # Continuum should be finite at least where ivar>0
        good = spec.ivar > 0
        self.assertTrue(np.all(np.isfinite(cont_mat[good])))


if __name__ == "__main__":
    unittest.main()