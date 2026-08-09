# nmfqsofit/utils.py

import numpy as np
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt
import time
import re
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
from typing import Dict, Any

from .logger import get_logger

logger = get_logger(__name__)

def elapsed(start, msg, use_logger=False):
    """
    Prints or logs the elapsed time since `start`.

    Args:
        start (float): The start time.
        msg (str): The message to print/log with the elapsed time.
        use_logger (bool): If True, use logger instead of print (default: False).

    Returns:
        float: The current time.
    """
    end = time.time()
    if start is not None:
        elapsed_msg = f"{msg} {end - start:.2f} seconds"
        if use_logger:
            logger.info(elapsed_msg)
        else:
            print(elapsed_msg)
    return end


def parse_qso_sequence(qso_sequence):
    """
    Parse a bash-like sequence or a single integer to generate QSO indices.

    Args:
        qso_sequence (str or int): Bash-like sequence (e.g., '1-1000', '1-1000:10') or an integer.

    Returns:
        numpy.array: Array of QSO indices.
    """
    if isinstance(qso_sequence, int):
        return np.arange(qso_sequence)

    # Handle string input
    if isinstance(qso_sequence, str):
        if qso_sequence.isdigit():
            return np.arange(int(qso_sequence))

        match = re.match(r"(\d+)-(\d+)(?::(\d+))?", qso_sequence)
        if match:
            start, end, step = match.groups()
            start, end = int(start), int(end)
            step = int(step) if step else 1
            return np.arange(start, end + 1, step)

    # If none of the conditions matched, raise an error
    raise ValueError(f"Invalid QSO sequence format: '{qso_sequence}'. Use 'start-end[:step]' or an integer.")


def read_nqso_from_header(file_path, hdu_name='METADATA'):
    """
    Read the NAXIS2 value from the header of a specified HDU in a FITS file.

    Args:
        file_path: str, path to the FITS file.
        hdu_name: str, name of the HDU from which to read NAXIS1 (default: 'METADATA').

    Returns:
        naxis2_value: int, value of NAXIS2 from the specified HDU header.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The FITS file {file_path} does not exist.")

    # Open the FITS file in read-only mode and load headers only
    with fits.open(file_path, mode='readonly') as hdul:
        # Attempt to access the specified HDU by name
        try:
            # Load only the header of the specified HDU
            header = hdul[hdu_name].header

            # Read the NAXIS1 value from the header
            naxis2_value = header.get('NAXIS2', None)

            if naxis2_value is None:
                raise KeyError(f"NAXIS2 not found in the '{hdu_name}' HDU header.")
            return naxis2_value

        except KeyError:
            raise ValueError(f"No '{hdu_name}' HDU found in {file_path}.")

def plot_flux_and_continuum(
    wave,
    flux,
    continuum,
    title=None,
    save_file=None,
    flux_kwargs=None,
    continuum_kwargs=None,
    ax_kwargs=None,
):
    """
    Plot observed flux and fitted continuum with major and minor ticks.

    Args:
        wave (np.ndarray):
            Observed wavelength array.

        flux (np.ndarray):
            Observed flux array.

        continuum (np.ndarray):
            Continuum model array.

        title (str, optional):
            Title of the plot.

        save_file (str, optional):
            If provided, saves the figure to this filename.
            If None, the Axes object is returned.

        flux_kwargs (dict, optional):
            Matplotlib kwargs for flux line (e.g., {'color': 'k', 'lw': 1}).

        continuum_kwargs (dict, optional):
            Matplotlib kwargs for continuum line.

        ax_kwargs (dict, optional):
            Additional axis-level settings (e.g., {'xlim': (3500, 9500)}).

    Returns:
        matplotlib.axes.Axes
    """
    flux_kwargs = flux_kwargs or {}
    continuum_kwargs = continuum_kwargs or {}
    ax_kwargs = ax_kwargs or {}

    # Default styles (only used if user doesn't override)
    default_flux_kwargs = {"lw": 1.5, "label": "Flux"}
    default_cont_kwargs = {"lw": 1.5, "color": "r", "label": "Continuum"}

    # Merge defaults with user kwargs (user overrides)
    flux_plot_kwargs = {**default_flux_kwargs, **flux_kwargs}
    cont_plot_kwargs = {**default_cont_kwargs, **continuum_kwargs}

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(wave, flux, **flux_plot_kwargs)
    ax.plot(wave, continuum, **cont_plot_kwargs)

    ax.set_xlabel("Obs Wavelength (Å)", fontsize=14)
    ax.set_ylabel("Flux (Survey unit)", fontsize=14)

    if title is not None:
        ax.set_title(title, fontsize=13)

    # Major & minor ticks
    ax.tick_params(axis="both", which="major",
                   direction="in", length=6, width=1, labelsize=12)
    ax.tick_params(axis="both", which="minor",
                   direction="in", length=3, width=0.8)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.grid(which="major", alpha=0.4)
    ax.grid(which="minor", alpha=0.2, linestyle="--")

    ax.legend(loc="best", prop={"size": 12})

    # Apply additional axis kwargs (xlim, ylim, etc.)
    for key, value in ax_kwargs.items():
        getattr(ax, f"set_{key}")(value)

    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file, dpi=300)
        plt.close(fig)
    else:
        return ax


def interpolation1D(xnew, xold, yold, kind='linear', fill_value=np.nan):
    """
    Perform 1D interpolation.

    Interpolates `yold` defined at coordinates `xold` onto new coordinates
    `xnew` using specified interpolation method. Values outside the interpolation
    range are set to NaN.

    Args:
        xnew (numpy.ndarray):
            New x-coordinates where interpolation is evaluated.
        xold (numpy.ndarray):
            Original x-coordinates of the data. Must be monotonically increasing.
        yold (numpy.ndarray):
            Original y-values corresponding to `xold`.
        kind (str):
            Interpolation method (default: 'linear'). Use 'linear' for linear interpolation.
        fill_value (float):
            Value to use for points outside the interpolation range (default: np.nan).

    Returns:
        numpy.ndarray:
            Interpolated values evaluated at `xnew`. Points outside the
            range of `xold` are assigned `fill_value`.
    """

    if kind == "linear":
        return np.interp(xnew, xold, yold, left=fill_value, right=fill_value)
    else:
        f = interp1d(
            xold,
            yold,
            kind=kind,
            bounds_error=False,
            fill_value=fill_value,
            assume_sorted=True,
        )
        return f(xnew)


def _parse_headers(args):
    """
    Parse KEY=VALUE header arguments into a dict.

    Args:
        args (argparse.Namespace):
            Parsed command-line arguments containing:
            - headers (list[str]): List of strings like ["KEY=VALUE", "KEY2=VALUE2"].
            - method (str): Fitting method.
            - eigenspectra (str): Eigenspectra directory.
            - kernel_large (int): Kernel size.
            - maxiters (int): Maximum iterations.

    Returns:
        dict:
            Dictionary of parsed header keywords and values, including package versions.

    Raises:
        ValueError:
            If any entry does not contain exactly one '='.
    """

    from importlib.metadata import version, PackageNotFoundError
    import sys

    try:
        from . import __author__ as package_author, __repo__ as package_repo
    except (ImportError, AttributeError):
        package_author = None
        package_repo = None

    headers: Dict[str, Any] = {}
    # Always set these keywords in the Primary Header
    headers['ORIGAUTH'] = (package_author or 'Unknown', 'Original author of repo')
    headers['GITREPO'] = (package_repo or 'https://github.com/abhi0395/nmfqsofit', 'Source code repository')

    for item in args.headers:
        if item.count("=") != 1:
            raise ValueError(f"Invalid header '{item}'. Expected format KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid header '{item}'. KEY cannot be empty.")
        headers[key] = value

    packages = ['python','nmfqsofit', 'numpy', 'astropy', 'scipy', 'NonnegMFPy', 'tqdm', 'pytest']
    versions = {}
    for pkg in packages:
        if pkg=='python':
            versions[pkg] = sys.version.split()[0]
        else:
            try:
                versions[pkg] = version(pkg)
            except PackageNotFoundError:
                versions[pkg] = 'not installed'

    headers["METHOD"] = (str(args.method) , 'Fitting Method')
    headers["EIGENVEC"] = (str(args.eigenspectra), 'Eigenvector (eigenspectra) configuration')
    headers["KERN_I"] = (int(args.kernel_large), 'kernel size for removing large fluctuation')
    headers["KERN_II"] = (int(args.kernel_small), 'kernel size for removing small fluctuation')

    # while still validating other invalid values with a clear error.
    if args.maxiters is None:
        maxiters_value = None
    else:
        try:
            maxiters_value = int(args.maxiters)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid value for maxiters: {args.maxiters!r}. Expected an integer or null."
            ) from exc

    headers["MAXITER"] = (maxiters_value, 'Maximum iteration for solver fitting')
    headers["FILT_ITR"] = (int(args.smoothing_niter), 'Number of iterations for median filtering')
    headers["SIG_CLIP"] = (float(args.fit_nsigma), 'Sigma clipping  (N * sigma), to mask absorption')
    headers["FIT_ITER"] = (int(args.fit_niter), 'Number of iterations for sigma clipping')
    headers["SM_SIGMA"] = (float(args.smooth_nsigma), 'Sigma clipping (N * sigma) for smoothing correction')

    for k, (pkg, ver) in enumerate(versions.items()):
        headers[f"DEPNAM{k:02d}"] = str(pkg)
        headers[f"DEPVER{k:02d}"] = str(ver)

    return headers


