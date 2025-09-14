# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 22:06:19 2025

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Photometry pipeline with background subtraction and user input
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter

# ===================== PARAMETERS =====================
# Aperture/annulus scaling relative to FWHM
APERTURE_SCALE = 1.5
ANNULUS_INNER_SCALE = 1.5
ANNULUS_OUTER_SCALE = 2.5

# Detection parameters
DAOFIND_FWHM = 3.0
SIGMA_CLIP = 3.0
DETECTION_THRESHOLD_SIGMA = 5.0
PEAK_MIN_STD = 10.0

# Background subtraction
BG_SUB_SIGMA = np.nanmedian  # function used to compute background level

# Default input values
DEFAULT_OBJ_NAME = "UnknownObject"
DEFAULT_DISTANCE = 10000.0
DEFAULT_A_V = 0.1
DEFAULT_E_BV = 0.05
DEFAULT_B_FITS = r"D:\example_B.fts"
DEFAULT_V_FITS = r"D:\example_V.fts"
# ======================================================

# ---------------- PATH NORMALIZATION ----------------
def _norm_path(p: str) -> str:
    """Remove extra quotes and normalize filesystem path."""
    p = p.strip().strip('"').strip("'")
    return os.path.normpath(p)

# ---------------- USER INPUT ----------------
def _ask(prompt, default, cast=str):
    """Ask user for input with default and type casting."""
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return default
    try:
        return cast(s)
    except Exception:
        return default

print("=== Photometry Input ===")
obj_name    = _ask("Object name", DEFAULT_OBJ_NAME, str)
distance    = _ask("Distance (pc)", DEFAULT_DISTANCE, float)
A_V         = _ask("Galactic extinction A_V (mag)", DEFAULT_A_V, float)
E_BV        = _ask("Galactic color excess E(B-V)", DEFAULT_E_BV, float)
fits_file_B = _norm_path(_ask("Path to B-band FITS", DEFAULT_B_FITS, str))
fits_file_V = _norm_path(_ask("Path to V-band FITS", DEFAULT_V_FITS, str))

_outdir = os.path.dirname(fits_file_B) if os.path.dirname(fits_file_B) else os.getcwd()

# ---------- Background subtraction ----------
def subtract_background_and_save(path):
    """Subtract median background from FITS and save new file."""
    data, hdr = fits.getdata(path, header=True)
    median_val = BG_SUB_SIGMA(data)
    data_sub = data - median_val
    out_path = os.path.splitext(path)[0] + "_bgsub.fits"
    fits.writeto(out_path, data_sub, hdr, overwrite=True)
    print(f"[bgsub] wrote {out_path} (median={median_val:.3f})")
    return out_path

fits_file_B = subtract_background_and_save(fits_file_B)
fits_file_V = subtract_background_and_save(fits_file_V)

# ---------------------------------------------------
def compute_fwhm(data, x, y, size=10):
    """Measure FWHM around a light source."""
    x_min, x_max = int(x-size), int(x+size)
    y_min, y_max = int(y-size), int(y+size)
    if x_min < 0 or y_min < 0 or x_max >= data.shape[1] or y_max >= data.shape[0]:
        print(f"Skipping source at ({x}, {y}) due to out-of-bounds sub-image.")
        return None

    sub_image = data[y_min:y_max, x_min:x_max]
    smoothed = gaussian_filter(sub_image, sigma=2)
    peak = np.max(smoothed)
    half_max = peak / 2
    above_half_max = smoothed > half_max
    indices = np.argwhere(above_half_max)
    if indices.size > 0:
        min_x, max_x = indices[:, 1].min(), indices[:, 1].max()
        min_y, max_y = indices[:, 0].min(), indices[:, 0].max()
        fwhm_x = max_x - min_x
        fwhm_y = max_y - min_y
        return np.mean([fwhm_x, fwhm_y])
    return None

def process_fits(filename, band):
    """Detect sources, perform aperture photometry and return results."""
    hdul = fits.open(filename)
    data = hdul[0].data
    hdul.close()

    mean, median, std = sigma_clipped_stats(data, sigma=SIGMA_CLIP)
    threshold = DETECTION_THRESHOLD_SIGMA * std
    daofind = DAOStarFinder(fwhm=DAOFIND_FWHM, threshold=threshold)
    sources = daofind(data - median)
    sources = sources[sources['peak'] > PEAK_MIN_STD * std]

    results = []
    for source in sources:
        x, y = source['xcentroid'], source['ycentroid']
        fwhm = compute_fwhm(data, x, y)
        if fwhm is not None:
            radius = APERTURE_SCALE * fwhm
            aperture = CircularAperture((x, y), r=radius)
            annulus_inner_radius = radius * ANNULUS_INNER_SCALE
            annulus_outer_radius = radius * ANNULUS_OUTER_SCALE
            annulus = CircularAnnulus((x, y), r_in=annulus_inner_radius, r_out=annulus_outer_radius)

            phot_table = aperture_photometry(data, [aperture, annulus])
            background_mean = phot_table['aperture_sum_1'][0] / annulus.area
            background_subtracted_flux = phot_table['aperture_sum_0'][0] - background_mean * aperture.area

            if background_subtracted_flux < 0:
                continue

            results.append([x, y, fwhm, radius, background_subtracted_flux,
                            band, annulus_inner_radius, annulus_outer_radius])
    return results

# ---------------- RUN ANALYSIS ----------------
all_results = []
all_results.extend(process_fits(fits_file_B, "B"))
all_results.extend(process_fits(fits_file_V, "V"))

df = pd.DataFrame(all_results, columns=[
    "X", "Y", "FWHM", "Aperture Radius", "Flux", "Band",
    "Annulus Inner Radius", "Annulus Outer Radius"
])

csv_filename = os.path.join(_outdir, f"{obj_name}_photometry_results.csv")
df.to_csv(csv_filename, index=False)

print(f"Data saved to {csv_filename}")
