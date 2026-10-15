"""Load TESS light curves from CSV, FITS, or HDF5 into LightCurve objects."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import LightCurve, get_logger

logger = get_logger()


class DataLoader:
    """Multi-format light-curve loader.

    Physics background
    ------------------
    TESS produces calibrated flux time series. FITS files store ``TIME``,
    ``PDCSAP_FLUX`` (systematics-corrected), and ``PDCSAP_FLUX_ERR`` columns.
    """

    FITS_TIME = ("TIME",)
    FITS_FLUX = ("PDCSAP_FLUX", "SAP_FLUX", "FLUX")
    FITS_ERR = ("PDCSAP_FLUX_ERR", "SAP_FLUX_ERR", "FLUX_ERR")

    def load(self, path: str | Path) -> LightCurve:
        """Dispatch on file extension.

        Parameters
        ----------
        path : str | Path
            Input file (.csv, .fits/.fit, .hdf5/.h5).

        Returns
        -------
        LightCurve
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv(path)
        if suffix in (".fits", ".fit"):
            return self._load_fits(path)
        if suffix in (".hdf5", ".h5"):
            return self._load_hdf5(path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _load_csv(self, path: Path) -> LightCurve:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if "time" not in cols:
            raise ValueError(f"CSV {path} is missing a 'time' column")
        flux_col = cols.get("pdcsap_flux", cols.get("flux", cols.get("sap_flux")))
        if flux_col is None:
            raise ValueError(
                f"CSV {path} requires one of: pdcsap_flux, flux, sap_flux"
            )
        time = df[cols["time"]].to_numpy(dtype=float)
        flux = df[flux_col].to_numpy(dtype=float)
        err_col = cols.get(
            "pdcsap_flux_err",
            cols.get("flux_error", cols.get("flux_err", cols.get("sap_flux_err"))),
        )
        err = df[err_col].to_numpy() if err_col else None
        logger.info("Loaded CSV %s (%d points)", path.name, len(time))
        return LightCurve(time, flux, err, {"source": str(path)})

    def _load_fits(self, path: Path) -> LightCurve:
        """Read FITS explicitly so the selected flux product is unambiguous."""
        from astropy.io import fits

        with fits.open(path, memmap=True) as hdul:
            if len(hdul) < 2 or getattr(hdul[1], "data", None) is None:
                raise ValueError(f"FITS {path} has no light-curve table extension")
            data = hdul[1].data
            names = {n.upper(): n for n in data.columns.names}
            time = np.asarray(data[self._pick(names, self.FITS_TIME)])
            flux = np.asarray(data[self._pick(names, self.FITS_FLUX)])
            err_col = self._pick(names, self.FITS_ERR, required=False)
            err = np.asarray(data[err_col]) if err_col else None
            quality_col = names.get("QUALITY")
            meta = self._metadata_from_headers(path, hdul)
            if quality_col:
                meta["quality_flags"] = np.asarray(data[quality_col])
        logger.info("Loaded FITS %s using %s", path.name, self._pick(names, self.FITS_FLUX))
        return LightCurve(time, flux, err, meta)

    def _load_hdf5(self, path: Path) -> LightCurve:
        import h5py

        with h5py.File(path, "r") as f:
            time = f["time"][:]
            flux = f["flux"][:]
            err = f["flux_error"][:] if "flux_error" in f else None
        logger.info("Loaded HDF5 %s (%d points)", path.name, len(time))
        return LightCurve(time, flux, err, {"source": str(path)})

    @staticmethod
    def _pick(names: dict[str, str], candidates, required: bool = True):
        for c in candidates:
            if c in names:
                return names[c]
        if required:
            raise KeyError(f"None of {candidates} found in {list(names)}")
        return None

    @staticmethod
    def _metadata_from_headers(path: Path, hdul) -> dict:
        meta = {"source": str(path)}
        for hdu in hdul:
            header = getattr(hdu, "header", {})
            for key in (
                "TARGETID", "TICID", "OBJECT", "SECTOR", "CAMERA", "CCD",
                "CROWDSAP", "FLFRCSAP", "BJDREFI", "BJDREFF", "TIMESYS",
            ):
                if key in header:
                    meta[key.lower()] = header[key]
        stellar = {}
        header_aliases = {
            "radius": ("RADIUS", "RSTAR"),
            "radius_err": ("RADIUS_E", "RSTARERR"),
            "mass": ("MASS", "MSTAR"),
            "mass_err": ("MASS_E", "MSTARERR"),
            "teff": ("TEFF",),
            "teff_err": ("TEFF_E", "TEFFERR"),
        }
        for output_name, candidates in header_aliases.items():
            for hdu in hdul:
                header = getattr(hdu, "header", {})
                value = next((header[key] for key in candidates if key in header), None)
                if value is not None:
                    try:
                        stellar[output_name] = float(value)
                    except (TypeError, ValueError):
                        pass
                    break
        if stellar:
            meta["stellar"] = stellar
        if "crowdsap" in meta:
            meta["crowding"] = meta["crowdsap"]
            try:
                meta["contamination"] = 1.0 - float(meta["crowdsap"])
            except (TypeError, ValueError):
                pass
        return meta
