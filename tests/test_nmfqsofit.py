# tests/test_nmfqsofit.py

import unittest
import numpy as np
from astropy.io import fits
from nmfqsofit.nmfqsofit import NMFContinuum, run_parallel_continuum, write_continuum, read_fits_data
from nmfqsofit.utils import plot_flux_and_continuum

class TestNMFQSOFit(unittest.TestCase):
    def setUp(self):
        self.flux = np.random.rand(100, 4500)  # Example flux matrix
        self.eigenspectra = np.random.rand(4500, 5)  # Example eigenspectra matrix

        # Create temporary FITS files for testing
        hdu_flux = fits.PrimaryHDU(self.flux)
        hdu_flux.writeto('test_flux.fits', overwrite=True)

        hdu_eigenspectra = fits.PrimaryHDU(self.eigenspectra)
        hdu_eigenspectra.writeto('test_eigenspectra.fits', overwrite=True)

    def tearDown(self):
        import os
        os.remove('test_flux.fits')
        os.remove('test_eigenspectra.fits')

    def test_create_continuum(self):
        nmf_continuum = NMFContinuum(self.flux, self.eigenspectra)
        coefficients, continuum = nmf_continuum.create_continuum()
        self.assertEqual(coefficients.shape, (100, 5))
        self.assertEqual(continuum.shape, (100, 4500))

    def test_run_parallel_continuum(self):
        coefficient_matrix, continuum_matrix = run_parallel_continuum(self.flux, self.eigenspectra, chunk_size=20, n_jobs=4)
        self.assertEqual(coefficient_matrix.shape, (100, 5))
        self.assertEqual(continuum_matrix.shape, (100, 4500))

    def test_write_continuum(self):
        coefficient_matrix, continuum_matrix = run_parallel_continuum(self.flux, self.eigenspectra, chunk_size=20, n_jobs=4)
        headers = {'AUTHOR': 'Test'}
        write_continuum(coefficient_matrix, continuum_matrix, headers, 'test_continuum.fits')
        self.assertTrue(True)  # If no exception, the test passes

    def test_plot_flux_and_continuum(self):
        coefficient_matrix, continuum_matrix = run_parallel_continuum(self.flux, self.eigenspectra, chunk_size=20, n_jobs=4)
        try:
            plot_flux_and_continuum(self.flux, continuum_matrix)
            self.assertTrue(True)  # If no exception, the test passes
        except Exception as e:
            self.fail(f"plot_flux_and_continuum raised an exception: {e}")

    def test_read_fits_data(self):
        flux, eigenspectra = read_fits_data('test_flux.fits', 'test_eigenspectra.fits')
        np.testing.assert_array_equal(flux, self.flux)
        np.testing.assert_array_equal(eigenspectra, self.eigenspectra)

if __name__ == '__main__':
    unittest.main()
