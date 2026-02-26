"""Shared helper utilities for the multi-variable satellite pickler."""

from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

from pickler_config import (
    CDOM_B0,
    CDOM_B1,
    CDOM_B2,
    LAT_BINS,
    LAT_MAX,
    LAT_MIN,
    LON_BINS,
    LON_MAX,
    LON_MIN,
    S275_295_COEFFS,
    S300_600_COEFFS,
)


def create_spatial_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create consistent spatial grid for binning."""
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, LAT_BINS + 1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, LON_BINS + 1)

    lat_centers = lat_edges[:-1] + np.diff(lat_edges) / 2
    lon_centers = lon_edges[:-1] + np.diff(lon_edges) / 2
    return lat_edges, lon_edges, lat_centers, lon_centers


def extract_date_from_filepath(filepath: str) -> Optional[datetime]:
    """Extract date from satellite filename."""
    import os
    import re

    basename = os.path.basename(filepath)
    patterns = [
        r"(\d{4})(\d{2})(\d{2})",  # YYYYMMDD
        r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
        r"(\d{4})(\d{3})",  # YYYYDDD
    ]

    for pattern in patterns:
        match = re.search(pattern, basename)
        if not match:
            continue
        try:
            if len(match.groups()) == 3:
                year, month, day = map(int, match.groups())
                return datetime(year, month, day)
            if len(match.groups()) == 2:
                year, day_of_year = map(int, match.groups())
                return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        except ValueError:
            continue
    return None


def bin_data_to_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
) -> np.ndarray:
    """Bin scattered data to 2D grid using mean statistic."""
    binned_data, _, _, _ = binned_statistic_2d(
        lat,
        lon,
        values,
        statistic="mean",
        bins=[lat_edges, lon_edges],
        range=[[LAT_MIN, LAT_MAX], [LON_MIN, LON_MAX]],
    )
    return binned_data


def gaussian_fill_nans(data: np.ndarray, sigma: float) -> np.ndarray:
    """Fill NaN gaps in a 2-D array using a NaN-aware Gaussian blur."""
    filled = np.where(np.isnan(data), 0.0, data)
    weights = np.where(np.isnan(data), 0.0, 1.0)

    blurred_data = gaussian_filter(filled, sigma=sigma)
    blurred_weights = gaussian_filter(weights, sigma=sigma)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(blurred_weights > 0, blurred_data / blurred_weights, np.nan)
    return result


def calculate_cdom(rrs_412: np.ndarray, rrs_555: np.ndarray) -> np.ndarray:
    """Calculate CDOM from Rrs_412 and Rrs_555."""
    cdom_values = np.full_like(rrs_412, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        term_ratio = rrs_412 / rrs_555
        term_numerator = term_ratio - CDOM_B0
        term_division = term_numerator / CDOM_B2
        valid_log_mask = term_division > 0

        if np.any(valid_log_mask):
            cdom_values[valid_log_mask] = (
                np.log(term_division[valid_log_mask]) / (-CDOM_B1)
            )

    return cdom_values


def calculate_cdom_slopes(rrs_443: np.ndarray, rrs_547: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate CDOM spectral slopes S_275:295 and S_300:600 (Mannino et al.)."""
    s275_295 = np.full_like(rrs_443, np.nan)
    s300_600 = np.full_like(rrs_443, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid_mask = (
            ~np.isnan(rrs_443)
            & ~np.isnan(rrs_547)
            & (rrs_443 > 0)
            & (rrs_547 > 0)
        )

        if np.any(valid_mask):
            ln_rrs_443 = np.log(rrs_443[valid_mask])
            ln_rrs_547 = np.log(rrs_547[valid_mask])

            ln_s275_295 = (
                S275_295_COEFFS["B0"]
                + S275_295_COEFFS["B1"] * ln_rrs_443
                + S275_295_COEFFS["B2"] * ln_rrs_547
            )
            ln_s300_600 = (
                S300_600_COEFFS["B0"]
                + S300_600_COEFFS["B1"] * ln_rrs_443
                + S300_600_COEFFS["B2"] * ln_rrs_547
            )

            s275_295[valid_mask] = np.exp(ln_s275_295)
            s300_600[valid_mask] = np.exp(ln_s300_600)

    return s275_295, s300_600
