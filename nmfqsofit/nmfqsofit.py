# nmfqsofit/nmfqsofit.py

import numpy as np
from sklearn.decomposition import NMF
import multiprocessing
import time
import psutil
from astropy.io import fits

class NMFContinuum:
    def __init__(self, flux, eigenspectra):
        self.flux = np.array(flux)
        self.eigenspectra = np.array(eigenspectra)
        self.coefficients = None
        self.continuum = None

    def create_continuum(self, init='random', random_state=None):
        n_components = self.eigenspectra.shape[1]
        model = NMF(n_components=n_components, init=init, random_state=random_state)

        if self.flux.ndim == 1:
            W = model.fit_transform(self.flux.reshape(-1, 1))
            H = model.components_
            self.coefficients = W
            self.continuum = np.dot(W, H).flatten()
        elif self.flux.ndim == 2:
            W = model.fit_transform(self.flux)
            H = model.components_
            self.coefficients = W
            self.continuum = np.dot(W, H)
        else:
            raise ValueError("Flux must be either a 1D array or a 2D matrix.")

        return self.coefficients, self.continuum

def process_flux_chunk(args):
    flux_chunk, eigenspectra, init, random_state = args
    nmf_continuum = NMFContinuum(flux_chunk, eigenspectra)
    coefficients_chunk, continuum_chunk = nmf_continuum.create_continuum(init=init, random_state=random_state)
    return coefficients_chunk, continuum_chunk

def optimal_chunk_size(flux_rows, flux_cols, n_workers, n_cores):
    total_memory = psutil.virtual_memory().total
    memory_per_worker = total_memory // n_workers
    memory_per_row = flux_cols * 8  # flux_cols elements * 8 bytes per element
    rows_per_chunk = memory_per_worker // memory_per_row
    return max(1, min(flux_rows // (n_workers * n_cores * 2), rows_per_chunk))

def run_parallel_continuum(flux_matrix, eigenspectra, chunk_size=1000, n_jobs=-1, init='random', random_state=None):
    num_chunks = (flux_matrix.shape[0] + chunk_size - 1) // chunk_size

    pool = multiprocessing.Pool(processes=n_jobs)
    tasks = [
        (flux_matrix[i * chunk_size: (i + 1) * chunk_size], eigenspectra, init, random_state)
        for i in range(num_chunks)
    ]

    start_time = time.time()
    results = pool.map(process_flux_chunk, tasks)
    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds")

    pool.close()
    pool.join()

    coefficients_list, continuum_list = zip(*results)
    coefficient_matrix = np.vstack(coefficients_list)
    continuum_matrix = np.vstack(continuum_list)

    return coefficient_matrix, continuum_matrix

def write_continuum(coefficient_matrix, continuum_matrix, headers, filename):
    hdu_coefficients = fits.ImageHDU(data=coefficient_matrix, name='COEFFICIENTS')
    hdu_continuum = fits.ImageHDU(data=continuum_matrix, name='CONTINUUM')

    header = fits.Header()
    for key, value in headers.items():
        header[key] = value

    primary_hdu = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([primary_hdu, hdu_coefficients, hdu_continuum])
    hdul.writeto(filename, overwrite=True)
    print(f"Data written to {filename}")

def read_fits_data(flux_filename, eigenspectra_filename):
    with fits.open(flux_filename) as flux_hdul:
        flux = flux_hdul[0].data
    with fits.open(eigenspectra_filename) as eigenspectra_hdul:
        eigenspectra = eigenspectra_hdul[0].data
    return flux, eigenspectra

import argparse

def main():
    parser = argparse.ArgumentParser(description="NMF Continuum Estimation")
    parser.add_argument('--flux', type=str, required=True, help='Input FITS file containing FLUX extension')
    parser.add_argument('--eigenspectra', type=str, required=True, help='Input FITS file containing EIGENSPECTRA extension')
    parser.add_argument('--n_workers', type=int, default=256, help='Number of workers for parallel processing')
    parser.add_argument('--plot', action='store_true', help='Plot a random flux array and its continuum')
    parser.add_argument('--output', type=str, default='continuum.fits', help='Output FITS file name')
    parser.add_argument('--headers', type=str, nargs='+', help='Headers to include in the FITS file in KEY=VALUE format')

    args = parser.parse_args()

    flux_matrix, eigenspectra = read_fits_data(args.flux, args.eigenspectra)
    n_cores = multiprocessing.cpu_count()
    chunk_size = optimal_chunk_size(flux_matrix.shape[0], flux_matrix.shape[1], args.n_workers, n_cores)
    print(f"Optimal chunk size: {chunk_size}")

    coefficient_matrix, continuum_matrix = run_parallel_continuum(flux_matrix, eigenspectra, chunk_size=chunk_size, n_jobs=args.n_workers)

    headers = {}
    if args.headers:
        for header in args.headers:
            key, value = header.split('=')
            headers[key] = value

    write_continuum(coefficient_matrix, continuum_matrix, headers, args.output)

    if args.plot:
        from nmfqsofit.utils import plot_flux_and_continuum
        plot_flux_and_continuum(flux_matrix, continuum_matrix)
