"""
This script contains functions to read, append and write fits files.
"""
import os
import numpy as np
import time
import re
from astropy.io import fits
from astropy.table import Table
from pathlib import Path
from .utils import elapsed

def _find_hdu_name(hdul, keyword):
    """
    Find an HDU whose EXTNAME contains `keyword` (case-insensitive).

    Prefers an exact match (EXTNAME == keyword), otherwise returns the first
    HDU whose name contains the keyword.

    Args:
        hdul (fits.HDUList): Open FITS HDUList.
        keyword (str): Keyword to search for (e.g., 'FLUX').

    Returns:
        str: Matching HDU name.

    Raises:
        KeyError: If no match is found.
    """
    keyword_u = keyword.upper()

    for hdu in hdul:
        if hdu.name and hdu.name.upper() == keyword_u:
            return hdu.name

    # Or Fall back to substring match
    for hdu in hdul:
        if hdu.name and keyword_u in hdu.name.upper():
            return hdu.name

    available = [h.name for h in hdul]
    raise KeyError(f"No HDU found matching '{keyword}'. Available HDUs: {available}")


def read_fits_file(fits_file, index=None):
    """
    Read QSO spectra and metadata from a FITS file.

    This reader is survey-agnostic and does not assume strict HDU names.
    It searches HDU EXTNAMEs for keywords:
      - WAVELENGTH
      - FLUX
      - IVAR

    It also reads the METADATA table and standardizes the redshift column:
      - if 'Z_QSO' exists, it is renamed to 'Z'.

    Args:
        fits_file (str):
            Path to the FITS file containing QSO spectra.
        index (int, list, or np.ndarray, optional):
            Row index/indices to load. If None, loads all rows.

    Returns:
        tuple:
            header (astropy.io.fits.Header):
                Primary header.
            flux (numpy.ndarray):
                Flux array. Shape is (nwave,) for a single index, otherwise (nspec, nwave).
            ivar (numpy.ndarray):
                Inverse variance array. Shape is (nwave,) for a single index, otherwise (nspec, nwave).
            wavelength (numpy.ndarray):
                Wavelength array with shape (nwave,).
            metadata (astropy.table.Table):
                Metadata table. Sliced consistently with `index`.

    Raises:
        FileNotFoundError:
            If `fits_file` does not exist.
        KeyError:
            If required HDUs cannot be found.
        IndexError:
            If `index` is out of bounds.
    """
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    # Read metadata (and normalize column names)
    metadata = Table.read(str(fits_file), hdu="METADATA")
    if "Z_QSO" in metadata.colnames:
        metadata.rename_column("Z_QSO", "Z")

    with fits.open(str(fits_file), memmap=True) as hdul:
        header = hdul[0].header

        wave_name = _find_hdu_name(hdul, "WAVELENGTH")
        flux_name = _find_hdu_name(hdul, "FLUX")
        ivar_name = _find_hdu_name(hdul, "IVAR")

        wavelength = np.asarray(hdul[wave_name].data)
        flux_hdu = hdul[flux_name].data
        ivar_hdu = hdul[ivar_name].data

        if index is None:
            flux = np.asarray(flux_hdu)
            ivar = np.asarray(ivar_hdu)

        elif isinstance(index, (int, np.integer)):
            i = int(index)
            flux = np.asarray(flux_hdu[i]).ravel()
            ivar = np.asarray(ivar_hdu[i]).ravel()
            metadata = metadata[i : i + 1]

        else:
            idx = np.asarray(index, dtype=int)
            flux = np.asarray(flux_hdu[idx])
            ivar = np.asarray(ivar_hdu[idx])
            metadata = metadata[idx]

    return header, flux, ivar, wavelength, metadata

def write_continuum(coefficient_matrix, first_continuum_matrix, continuum_matrix, first_costs, costs, eigvector_type,  headers, filename):
    """
    Write continuum fitting results to a FITS file.

    The output FITS contains:
        - Primary HDU: header keywords from `headers`
        - Image HDU 'COEFFICIENTS': (nspec, ncomp)
        - Image HDU 'FIRST_CONTINUUM': (nspec, nwave)
        - Image HDU 'CONTINUUM': (nspec, nwave)
        - BinTable HDU 'METADATA': FIRST_COST, FINAL_COST, EIGVECTOR_TYPE

    Args:
        coefficient_matrix (numpy.ndarray):
            Coefficient matrix of shape (nspec, ncomp).
        first_continuum_matrix (numpy.ndarray):
            Continuum before median-filter correction, shape (nspec, nwave).
        continuum_matrix (numpy.ndarray):
            Final continuum after median-filter correction, shape (nspec, nwave).
        first_costs (numpy.ndarray):
            Cost values computed from the first continuum, shape (nspec,).
        costs (numpy.ndarray):
            Cost values computed from the final continuum, shape (nspec,).
        eigvector_type (numpy.ndarray):
            Array describing which eigenset/bin was selected per spectrum, shape (nspec,).
            Recommended dtype is fixed-width bytes (e.g., dtype='S32') for FITS compatibility.
        headers (dict, optional):
            Primary header keywords to include.
        filename (str or pathlib.Path):
            Output FITS filename.

    Returns:
        None
    """

    t0 = time.time()

    # Basic shape checks (fast and prevents silent FITS corruption)
    if coefficient_matrix.ndim != 2:
        raise ValueError("coefficient_matrix must be 2D (nspec, ncomp)")
    if first_continuum_matrix.shape != continuum_matrix.shape:
        raise ValueError("first_continuum_matrix and continuum_matrix must have the same shape")
    if first_continuum_matrix.ndim != 2:
        raise ValueError("continuum matrices must be 2D (nspec, nwave)")

    nspec = first_continuum_matrix.shape[0]
    if coefficient_matrix.shape[0] != nspec:
        raise ValueError("coefficient_matrix first dimension must match nspec")
    if first_costs.shape[0] != nspec or costs.shape[0] != nspec:
        raise ValueError("first_costs and costs must have length nspec")
    if eigvector_type.shape[0] != nspec:
        raise ValueError("eigvector_type must have length nspec")

    hdu_coefficients = fits.ImageHDU(data=coefficient_matrix, name="COEFFICIENTS")
    hdu_first_continuum = fits.ImageHDU(data=first_continuum_matrix, name="FIRST_CONTINUUM")
    hdu_continuum = fits.ImageHDU(data=continuum_matrix, name="CONTINUUM")

    meta = Table()
    meta["FIRST_COST"] = np.asarray(first_costs, dtype=np.float32)
    meta["FINAL_COST"] = np.asarray(costs, dtype=np.float32)
    meta["EIGVECTOR_TYPE"] = np.asarray(eigvector_type)

    metadata_hdu = fits.BinTableHDU(meta, name="METADATA")

    header = fits.Header()
    if headers:
        for key, value in headers.items():
            header[str(key)] = value

    primary_hdu = fits.PrimaryHDU(header=header)

    hdul = fits.HDUList([primary_hdu, hdu_coefficients, hdu_first_continuum, hdu_continuum, metadata_hdu])
    hdul.writeto(str(filename), overwrite=True)

    print(f"INFO: Data written to {filename}")
    print(f"INFO: Time taken to write {filename}: {time.time() - t0:.2f} s")

class QSOSpecRead:
    """
    A class to read and handle QSO spectra from a FITS file containing FLUX, IVAR, WAVELENGTH.
    """

    def __init__(self, fits_file, index=None, autoload=False, verbose=True):
        """
        Initializes the QSOSpecRead class.

        Args:
            fits_file (str): Path to the FITS file containing QSO spectra.
            index (int, list, or np.ndarray, optional): Index or indices of the rows to load. Default is None.
            autoload (bool): if True, class itself will load the data (default=False),
                             in True case, user does not need to use available class functions.
            verbose (bool): if want to print time info
        """
        self.fits_file = fits_file
        self.header = None
        self.flux = None
        self.ivar = None
        self.wavelength = None
        self.index = index
        self.verbose = verbose
        self.autoload = autoload
        self.metadata  = None
        if self.autoload:
            self.read_fits()

    def read_fits(self):
        """
        Reads the FITS file and measures the time taken for the operation.
        """
        if not os.path.exists(self.fits_file):
            raise IOError(f"ERROR: {self.fits_file} does not exist")
        start_time = time.time()
        self.header, self.flux, self.ivar, self.wavelength, self.metadata = read_fits_file(self.fits_file, self.index)
        if self.verbose:
            elapsed(start_time, f"INFO: Time taken to read {self.fits_file}")


def load_all_eigenspectra(data_dir):
    """
    Load all NMF eigenspectra FITS files in a directory into a dictionary.

    Files are keyed by (zmin, zmax), parsed from filenames that contain
    two 3-digit integers corresponding to zmin*100 and zmax*100, e.g.:
    ..._000_100_... -> (0.00, 1.00).

    The FITS files must contain extensions:
        - REST_WAVE
        - EIGENVEC

    Args:
        data_dir (str or Path):
            Directory containing eigenspectra FITS files.

    Returns:
        dict:
            Mapping (zmin, zmax) -> {"wave": wave, "eigvec": eigvec},
            where:
              wave has shape (nwave,)
              eigvec has shape (ncomp, nwave)
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"data_dir is not a directory: {data_dir}")

    nmf_dict = {}
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.fits"))

    for filepath in files:
        match = re.search(r"(\d{3})_(\d{3})", filepath.name)
        if match is None:
            continue

        zmin = int(match.group(1)) / 100.0
        zmax = int(match.group(2)) / 100.0

        with fits.open(filepath, memmap=True) as hdul:
            if "REST_WAVE" not in hdul or "EIGENVEC" not in hdul:
                raise KeyError(
                    f"{filepath.name} missing required HDUs: REST_WAVE and/or EIGENVEC"
                )

            wave = np.asarray(hdul["REST_WAVE"].data, dtype=float)
            eigvec = np.asarray(hdul["EIGENVEC"].data, dtype=float).T  # (ncomp, nwave)

        key = (zmin, zmax)
        if key in nmf_dict:
            raise ValueError(f"Duplicate redshift bin {key} from file {filepath.name}")

        nmf_dict[key] = {"wave": wave, "eigvec": eigvec}

    if len(nmf_dict) == 0:
        raise ValueError(f"No eigenspectra FITS files found in {data_dir}")

    return nmf_dict


#------------------------------------#
# Class to read continuum output file
#------------------------------------#

def read_continuum_file(
    fits_file,
    index= None):
    """
    Read continuum products from an nmfqsofit output FITS file.

    Expected HDUs:
        - 'COEFFICIENTS'    : (nspec, ncomp)
        - 'FIRST_CONTINUUM' : (nspec, nwave)
        - 'CONTINUUM'       : (nspec, nwave)
        - 'METADATA'        : table with FIRST_COST, FINAL_COST, EIGVECTOR_TYPE

    Args:
        fits_file (str):
            Path to continuum FITS file written by nmfqsofit.
        index (int or array-like, optional):
            - None: load all rows (2D arrays returned)
            - int: load one row (1D continua returned; coefficients 1D)
            - array-like: load selected rows

    Returns:
        tuple:
            header (fits.Header): Primary header.
            coefficients (np.ndarray): (ncomp,) or (nsel, ncomp)
            first_continuum (np.ndarray): (nwave,) or (nsel, nwave)
            continuum (np.ndarray): (nwave,) or (nsel, nwave)
            metadata (Table): table sliced consistently with index
    """
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    metadata = Table.read(fits_file, hdu="METADATA")

    with fits.open(fits_file, memmap=True) as hdul:
        for name in ("COEFFICIENTS", "FIRST_CONTINUUM", "CONTINUUM"):
            if name not in hdul:
                raise KeyError(f"Missing required HDU '{name}' in {fits_file}")

        header = hdul[0].header
        coeff_hdu = hdul["COEFFICIENTS"].data
        first_hdu = hdul["FIRST_CONTINUUM"].data
        cont_hdu = hdul["CONTINUUM"].data

        if index is None:
            coefficients = np.asarray(coeff_hdu)
            first_continuum = np.asarray(first_hdu)
            continuum = np.asarray(cont_hdu)

        elif isinstance(index, (int, np.integer)):
            i = int(index)
            coefficients = np.asarray(coeff_hdu[i]).ravel()
            first_continuum = np.asarray(first_hdu[i]).ravel()
            continuum = np.asarray(cont_hdu[i]).ravel()
            metadata = metadata[i : i + 1]

        else:
            idx = np.asarray(index, dtype=int)
            coefficients = np.asarray(coeff_hdu[idx])
            first_continuum = np.asarray(first_hdu[idx])
            continuum = np.asarray(cont_hdu[idx])
            metadata = metadata[idx]

    return header, coefficients, first_continuum, continuum, metadata

class ContinuumSpec:
    """
    Reader for nmfqsofit continuum output FITS files.

    Similar to QSOSpecRead, but for continuum products:
        - coefficients
        - first_continuum
        - continuum
        - metadata (FIRST_COST, FINAL_COST, EIGVECTOR_TYPE)
    """

    def __init__(
        self,
        fits_file,
        index = None,
        autoload = False,
        verbose = True):
        """
        Initialize ContinuumSpec.

        Args:
            fits_file (str):
                Continuum FITS file written by nmfqsofit.
            index (int or array-like, optional):
                Which rows to load (see `read_continuum_file`).
            autoload (bool):
                If True, reads data immediately.
            verbose (bool):
                If True, prints read timing information.
        """
        self.fits_file = fits_file
        self.index = index
        self.autoload = autoload
        self.verbose = verbose

        self.header =  None
        self.coefficients = None
        self.first_continuum = None
        self.continuum = None
        self.metadata = None

        if self.autoload:
            self.read_fits()

    def read_fits(self):
        """
        Read continuum products from disk into memory.

        Returns:
            None
        """
        if not os.path.exists(self.fits_file):
            raise FileNotFoundError(f"FITS file does not exist: {self.fits_file}")

        t0 = time.time()
        (
            self.header,
            self.coefficients,
            self.first_continuum,
            self.continuum,
            self.metadata,
        ) = read_continuum_file(self.fits_file, self.index)

        if self.verbose:
            dt = time.time() - t0
            print(f"INFO: Time taken to read {self.fits_file}: {dt:.2f} s")