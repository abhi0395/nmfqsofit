# nmfqsofit/parallel_main.py

import argparse
import logging
import multiprocessing as mp

import numpy as np

from .io import QSOSpecRead, load_all_eigenspectra, write_continuum
from .nmfcontinuum import run_parallel_continuum
from .utils import read_nqso_from_header, parse_qso_sequence, _parse_headers
from .logger import setup_logger

def main():
    """
    CLI entry point for parallel NMF continuum estimation.

    This script:
      1) Reads QSO spectra (wave/flux/ivar + redshifts) from an input FITS file.
      2) Loads all NMF eigenspectra from a directory into memory.
      3) Runs continuum fitting in parallel over the selected QSO indices.
      4) Writes coefficients, continua, costs, and metadata to an output FITS file.

    Notes:
      - `--n-qso` can be:
          * not provided -> process all QSOs in the input file
          * an integer string "100" -> first 100
          * a range "1-1000" -> inclusive range (depending on your parse_qso_sequence)
          * a stepped range "1-1000:10" -> every 10th
      - `--method` should be one of: "nnls", "nmf"
    """
    parser = argparse.ArgumentParser(description="NMF Continuum Estimation (Parallel)")

    parser.add_argument(
        "--spectra-file",
        type=str,
        required=True,
        help=(
            "Input FITS file containing FLUX, IVAR extensions and METADATA "
            "extension with redshift column/key 'Z'."
        ),
    )
    parser.add_argument(
        "--eigenspectra",
        type=str,
        required=True,
        help="Directory containing all eigenspectra FITS files.",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        default=8,
        help="Number of CPU processes to use (default: 8).",
    )
    parser.add_argument(
        "--n-qso",
        type=str,
        required=False,
        default=None,
        help=(
            "Number of QSO spectra to process or a sequence string "
            "(e.g., '100', '1-1000', '1-1000:10'). "
            "If not provided, all spectra are processed."
        ),
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=None,
        help="Median filter kernel size (odd integer; default: 71).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="nnls",
        choices=["nnls", "nmf"],
        help="Coefficient solver method (default: nnls).",
    )

    parser.add_argument(
        "--maxiters",
        type=int,
        default=None,
        help="number of maximum iteration for solver (default 200)",
    )

    parser.add_argument(
        "--interp-kind",
        type=str,
        required=False,
        default="linear",
        help="Interpolation method (all options valid in scipt.interp1d) (default: 'linear').",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output FITS filename.",
    )
    parser.add_argument(
        "--headers",
        type=str,
        nargs="+",
        default=None,
        help="Extra FITS headers to include in KEY=VALUE format.",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger("nmfqsofit", level=logging.INFO)

    args.kernel_small = int(args.kernel_size / 2)

    if args.kernel_small %2 == 0:
        args.kernel_small+=1 # just make it odd

    logger.info("==== USER PROVIDED ARGUMENTS ====")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info("================================")

    # Determine number/selection of QSOs
    if args.n_qso is None:
        nqso_total = int(read_nqso_from_header(args.spectra_file))
        qso_selector = str(nqso_total)
        logger.info(f"--n-qso not provided; found {nqso_total} QSOs in file. Processing all.")
    else:
        qso_selector = args.n_qso
        nqso_total = None  # may be unknown until we parse indices

    # Parse QSO sequence into explicit indices
    spec_indices = parse_qso_sequence(qso_selector)
    n_selected = len(spec_indices)
    logger.info(f"Number of QSOs selected for processing = {n_selected}")

    # Read spectra (autoload=True should load arrays into memory once)
    spec = QSOSpecRead(
        args.spectra_file,
        index=spec_indices,
        autoload=True,
        verbose=True,
    )

    z = np.asarray(spec.metadata["Z"], dtype=float).reshape(-1)
    if z.shape[0] != n_selected:
        raise ValueError(
            f"Metadata redshift length mismatch: len(Z)={z.shape[0]} "
            f"but selected QSOs={n_selected}"
        )

    # Load eigenspectra into memory once (shared read-only across workers by fork on linux)
    eigenspectra = load_all_eigenspectra(args.eigenspectra)

    # Choose CPU count safely (leave 1 core free)
    max_procs = max(1, mp.cpu_count() - 1)
    n_jobs = int(min(args.ncpus, max_procs))
    logger.info(f"CPUs available={mp.cpu_count()}, using n_jobs={n_jobs}")

    # Run continuum fitting in parallel

    results = (
        run_parallel_continuum(
            wave=spec.wavelength,
            flux=spec.flux,
            ivar=spec.ivar,
            z=z,
            eigenspectra=eigenspectra,
            kernel_large=args.kernel_size,
            kernel_small=args.kernel_small,
            method=args.method,
            n_jobs=n_jobs,
            maxiters=args.maxiters,
            interp_kind=args.interp_kind
        )
    )

    # Parsing user defined headers
    headers = _parse_headers(args)

    # Write output
    write_continuum(results,
        headers=headers,
        filename=args.output,
    )
    logger.info(f"Wrote output to {args.output}")