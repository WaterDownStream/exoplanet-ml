"""Inspect FITS HDUs and key light-curve columns."""
from __future__ import annotations

import argparse
from pathlib import Path

from astropy.io import fits

from src.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a FITS light-curve file")
    parser.add_argument("fits_path", help="Path to a FITS file")
    args = parser.parse_args()

    logger = get_logger()
    path = Path(args.fits_path)
    with fits.open(path) as hdul:
        logger.info("Opened %s with %d HDUs", path, len(hdul))
        for idx, hdu in enumerate(hdul):
            extname = hdu.header.get("EXTNAME", "PRIMARY")
            logger.info("HDU %d: %s", idx, extname)
            if getattr(hdu, "data", None) is not None and hasattr(hdu.data, "columns"):
                logger.info("  Columns: %s", list(hdu.data.columns.names))


if __name__ == "__main__":
    main()
