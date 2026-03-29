"""
This script contains functions to read, append and write fits files.
"""
from fileinput import filename
import os
import re
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .utils import elapsed
from .logger import get_logger

logger = get_logger(__name__)

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
                Inverse variance array. Shape is (nwave,) for a single index,
                otherwise (nspec, nwave).
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

def write_continuum(
    results,
    headers,
    filename,
):
    """
    Write continuum fitting results to a FITS file.

    The output FITS contains:
        - Primary HDU: header keywords from `headers`
        - Image HDU 'COEFFICIENTS': (nspec, ncomp)
        - Image HDU 'FIRST_CONTINUUM': (nspec, nwave)
        - Image HDU 'CONTINUUM': (nspec, nwave)
        - BinTable HDU 'METADATA': FIRST_COST, FINAL_COST, eigvector_range, ZMIN, ZMAX, N_COMP, NORM_FACTOR

    Args:
        results (dict):
            result directory.
        headers (dict, optional):
            Primary header keywords to include.
        filename (str or pathlib.Path):
            Output FITS filename.
    """

    start_time = time.time()
    if results["coefficients"].ndim != 2:
        raise ValueError("coefficient_matrix must be 2D (nspec, ncomp)")
    if results["first_continuum"].shape != results["continuum"].shape:
        raise ValueError("first_continuum_matrix and continuum_matrix must have the same shape")
    if results["first_continuum"].ndim != 2:
        raise ValueError("continuum matrices must be 2D (nspec, nwave)")

    nspec = results["first_continuum"].shape[0]
    if results["coefficients"].shape[0] != nspec:
        raise ValueError("coefficient_matrix first dimension must match nspec")
    if results["first_cost"].shape[0] != nspec or results["final_cost"].shape[0] != nspec:
        raise ValueError("first_costs and costs must have length nspec")
    if results["eigvector_range"].shape[0] != nspec:
        raise ValueError("eigvector_range must have length nspec")

    hdu_coefficients = fits.ImageHDU(data=results["coefficients"], name="COEFFICIENTS")
    hdu_first_continuum = fits.ImageHDU(data=results["first_continuum"], name="FIRST_CONTINUUM")
    hdu_continuum = fits.ImageHDU(data=results["continuum"], name="CONTINUUM")

    meta = Table()
    for k, val in results.items():
        if k not in ["coefficients", "first_continuum", "continuum"]:
            meta[k.upper()] = np.asarray(val)

    metadata_hdu = fits.BinTableHDU(meta, name="METADATA")

    header = fits.Header()
    if headers:
        for key, value in headers.items():
            header[str(key)] = value

    primary_hdu = fits.PrimaryHDU(header=header)

    hdul = fits.HDUList(
        [primary_hdu, hdu_coefficients, hdu_first_continuum, hdu_continuum, metadata_hdu]
    )
    hdul.writeto(str(filename), overwrite=True)


class QSOSpecRead:
    """
    A class to read and handle QSO spectra from a FITS file containing FLUX, IVAR, WAVELENGTH.
    """

    def __init__(self, fits_file, index=None, autoload=False, verbose=True):
        """
        Initializes the QSOSpecRead class.

        Args:
            fits_file (str): Path to the FITS file containing QSO spectra.
            index (int, list, or np.ndarray, optional):
                Index or indices of the rows to load. Default is None.
            autoload (bool):
                if True, class itself will load the data (default=False),
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
        self.header, self.flux, self.ivar, self.wavelength, self.metadata = read_fits_file(
            self.fits_file, self.index
        )
        if self.verbose:
            elapsed_time = time.time() - start_time
            logger.info(f"Time taken to read {self.fits_file}: {elapsed_time:.2f} s")


def load_all_eigenspectra(data_path):
    """
    Load NMF eigenspectra FITS file(s) from a file or directory.

    Files are keyed by (zmin, zmax, lam_min, lam_max, stat), parsed from filenames that contain
    two 3-digit integers corresponding to zmin*100 and zmax*100, e.g.:
    ..._000_100_... -> (0.00, 1.00).

    The FITS files must contain extensions:
        - REST_WAVE
        - EIGENVEC

    Args:
        data_path (str or Path):
            Path to a single eigenspectra FITS file or directory containing
            eigenspectra FITS files.

    Returns:
        dict:
            Mapping (zmin, zmax, lam_min, lam_max, stat) -> {"wave": wave, "eigvec": eigvec},
            where:
              wave has shape (nwave,)
              eigvec has shape (ncomp, nwave)

    Raises:
        FileNotFoundError:
            If `data_path` does not exist.
        ValueError:
            If `data_path` is neither a file nor a directory, or if no valid
            eigenspectra files are found.
        KeyError:
            If a FITS file is missing required HDUs.
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path not found: {data_path}")

    start_time = time.time()
    data_path = Path(data_path)

    # Determine if path is a file or directory
    if data_path.is_file():
        logger.info(f"Loading eigenspectra from single file: {data_path}")
        files = [data_path]
    elif data_path.is_dir():
        logger.info(f"Loading eigenspectra from directory: {data_path}")
        files = sorted(data_path.glob("*.fits"))
        if not files:
            raise ValueError(f"No eigenspectra FITS files found in {data_path}")
    else:
        raise ValueError(f"Path is neither a file nor a directory: {data_path}")

    nmf_dict = {}
    files_loaded = 0

    for filepath in files:
        match = re.search(r"(\d{3})_(\d{3})", filepath.name)
        if match is None:
            logger.debug(f"Skipping file (no z-bin pattern found): {filepath.name}")
            continue

        zmin = int(match.group(1)) / 100.0
        zmax = int(match.group(2)) / 100.0

        with fits.open(filepath, memmap=True) as hdul:
            hdr = hdul[0].header
            lam_min, lam_max  = hdr["LAMSTART"], hdr["LAMEND"]
            stat = hdr["NORMSTAT"]
            if "REST_WAVE" not in hdul or "EIGENVEC" not in hdul:
                raise KeyError(
                    f"{filepath.name} missing required HDUs: REST_WAVE and/or EIGENVEC"
                )

            wave = np.asarray(hdul["REST_WAVE"].data, dtype=float)
            eigvec = np.asarray(hdul["EIGENVEC"].data, dtype=float).T  # (ncomp, nwave)

        key = (zmin, zmax, lam_min, lam_max, stat)
        if key in nmf_dict:
            raise ValueError(f"Duplicate redshift bin {key} from file {filepath.name}")

        nmf_dict[key] = {"wave": wave, "eigvec": eigvec, 'headers': hdr}
        files_loaded += 1
        logger.debug(
            f"Loaded eigenspectra from {filepath.name}: "
            f"z=[{zmin:.2f}, {zmax:.2f}], "
            f"lambda=[{lam_min:.1f}, {lam_max:.1f}], "
            f"ncomp={eigvec.shape[0]}, nwave={eigvec.shape[1]}"
        )

    if len(nmf_dict) == 0:
        raise ValueError(f"No eigenspectra FITS files loaded from {data_path}")

    elapsed_time = time.time() - start_time
    logger.info(
        f"Successfully loaded {files_loaded} eigenspectra file(s) with {len(nmf_dict)} "
        f"redshift bins. Time taken: {elapsed_time:.2f} s"
    )

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
        - 'METADATA'        : table with FIRST_COST, FINAL_COST, eigvector_range

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
        - metadata (FIRST_COST, FINAL_COST, eigvector_range)
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

        start_time = time.time()
        (
            self.header,
            self.coefficients,
            self.first_continuum,
            self.continuum,
            self.metadata,
        ) = read_continuum_file(self.fits_file, self.index)

        if self.verbose:
            elapsed_time = time.time() - start_time
            logger.info(f"Time taken to read {self.fits_file}: {elapsed_time:.2f} s")

