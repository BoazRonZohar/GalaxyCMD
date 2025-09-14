# -*- coding: utf-8 -*-
"""
Galaxy_Photometry_4_fixed.py

Photometry pipeline with background subtraction, B/V matching,
reference star extraction from Vizier with tolerance filtering,
and magnitude calibration using N reference stars
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

# ===== added imports for reference stars =====
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
# =============================================

# ===================== PARAMETERS =====================
# Aperture/annulus scaling relative to FWHM
APERTURE_SCALE = 2
ANNULUS_INNER_SCALE = 2.5
ANNULUS_OUTER_SCALE = 3
# FWHM measurement window (pixels from center)
FWHM_WINDOW_SIZE = 20

# Detection parameters
DAOFIND_FWHM = 10.0
SIGMA_CLIP = 3.0
DETECTION_THRESHOLD_SIGMA = 0.5
PEAK_MIN_STD = 1.0

# Background subtraction
BG_SUB_FUNC = np.nanmedian  # function used to compute background level

# Matching tolerance B and V Filters (pixels)
MATCH_TOLERANCE = 10.0

# Reference star parameters
REF_MAG_LIMIT = 15.0
REF_CATALOG = "II/336/apass9"  # APASS9 (B,V magnitudes)

# Matching tolerance between catalog stars and detected sources (pixels)
CATALOG_MATCH_TOLERANCE = 10.0

# Matching tolerance between catalog stars and measured flux positions (pixels)
CATALOG_FLUX_MATCH_TOLERANCE = 10.0

# Calibration parameters
CALIB_NUM_STARS = None  # number of stars to use for calibration (user will be asked)
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
obj_name    = _ask("Galaxy name", "UnknownObject", str)
distance    = _ask("Distance (pc)", 10000.0, float)
A_V         = _ask("Galactic extinction A_V (mag)", 0.1, float)
E_BV        = _ask("Galactic color excess E(B-V)", 0.05, float)
fits_file_B = _norm_path(_ask("Path to B-band FITS", r"D:\example_B.fts", str))
fits_file_V = _norm_path(_ask("Path to V-band FITS", r"D:\example_V.fts", str))

_outdir = os.path.dirname(fits_file_B) if os.path.dirname(fits_file_B) else os.getcwd()

# ---------- Background subtraction ----------
def subtract_background_and_save(path):
    """Subtract background (median or other function) from FITS and save new file."""
    data, hdr = fits.getdata(path, header=True)
    bg_val = BG_SUB_FUNC(data)
    data_sub = data - bg_val
    out_path = os.path.splitext(path)[0] + "_bgsub.fits"
    fits.writeto(out_path, data_sub, hdr, overwrite=True)
    print(f"[bgsub] wrote {out_path} (bg={bg_val:.3f})")
    return out_path

fits_file_B = subtract_background_and_save(fits_file_B)
fits_file_V = subtract_background_and_save(fits_file_V)

# ------------Measure FWHM around a light source-----------
def compute_fwhm(data, x, y, size=FWHM_WINDOW_SIZE):
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

# ---------------- MATCHING FUNCTION ----------------
def match_sources(df_B, df_V, tol=MATCH_TOLERANCE):
    """Match B and V sources by nearest (X,Y) within tolerance."""
    matched_rows = []
    used_V = set()
    for _, rowB in df_B.iterrows():
        xB, yB = rowB["X"], rowB["Y"]
        dists = np.sqrt((df_V["X"] - xB)**2 + (df_V["Y"] - yB)**2)
        min_idx = dists.idxmin()
        if dists[min_idx] <= tol and min_idx not in used_V:
            rowV = df_V.loc[min_idx]
            merged = {
                "X_B": rowB["X"], "Y_B": rowB["Y"],
                "FWHM_B": rowB["FWHM"], "Flux_B": rowB["Flux"],
                "X_V": rowV["X"], "Y_V": rowV["Y"],
                "FWHM_V": rowV["FWHM"], "Flux_V": rowV["Flux"]
            }
            matched_rows.append(merged)
            used_V.add(min_idx)
    return pd.DataFrame(matched_rows)

# ---------------- REFERENCE STARS (Vizier) ----------------
def extract_reference_stars(fits_file, df_B, df_V,
                            mag_limit=15.0,
                            catalog="II/336/apass9",
                            tol=2.0,
                            flux_tol=3.0):
    """Query Vizier and return reference stars with catalog mags, measured fluxes and positions (B,V)."""
    hdr = fits.getheader(fits_file)
    wcs = WCS(hdr)

    ra_center, dec_center = wcs.wcs.crval
    naxis1, naxis2 = hdr["NAXIS1"], hdr["NAXIS2"]
    scale_deg = np.mean(np.abs(wcs.pixel_scale_matrix.diagonal()))
    fov_ra = naxis1 * scale_deg
    fov_dec = naxis2 * scale_deg

    Vizier.ROW_LIMIT = -1
    v = Vizier(columns=["RAJ2000","DEJ2000","Bmag","Vmag"],
               column_filters={"Vmag":"<%.2f" % mag_limit})
    result = v.query_region(
        SkyCoord(ra_center, dec_center, unit="deg"),
        width=f"{fov_ra}d", height=f"{fov_dec}d",
        catalog=catalog
    )

    if len(result) == 0:
        print("No reference stars found in Vizier catalog.")
        return pd.DataFrame()

    stars = result[0]
    coords = SkyCoord(stars["RAJ2000"], stars["DEJ2000"], unit="deg")
    x_pix, y_pix = wcs.world_to_pixel(coords)

    df_ref = pd.DataFrame({
        "RA": stars["RAJ2000"],
        "Dec": stars["DEJ2000"],
        "Bmag": stars["Bmag"],
        "Vmag": stars["Vmag"],
        "X_pix": x_pix,
        "Y_pix": y_pix
    })

    flux_B, flux_V = [], []
    XB_meas, YB_meas, XV_meas, YV_meas = [], [], [], []

    for _, row in df_ref.iterrows():
        dB = np.sqrt((df_B["X"] - row["X_pix"])**2 + (df_B["Y"] - row["Y_pix"])**2)
        dV = np.sqrt((df_V["X"] - row["X_pix"])**2 + (df_V["Y"] - row["Y_pix"])**2)

        if dB.min() <= flux_tol:
            idxB = dB.idxmin()
            fB = df_B.loc[idxB, "Flux"]
            XB, YB = df_B.loc[idxB, "X"], df_B.loc[idxB, "Y"]
        else:
            fB, XB, YB = np.nan, np.nan, np.nan

        if dV.min() <= flux_tol:
            idxV = dV.idxmin()
            fV = df_V.loc[idxV, "Flux"]
            XV, YV = df_V.loc[idxV, "X"], df_V.loc[idxV, "Y"]
        else:
            fV, XV, YV = np.nan, np.nan, np.nan

        flux_B.append(fB)
        flux_V.append(fV)
        XB_meas.append(XB)
        YB_meas.append(YB)
        XV_meas.append(XV)
        YV_meas.append(YV)

    df_ref["Flux_B_measured"] = flux_B
    df_ref["Flux_V_measured"] = flux_V
    df_ref["X_B"] = XB_meas
    df_ref["Y_B"] = YB_meas
    df_ref["X_V"] = XV_meas
    df_ref["Y_V"] = YV_meas

    # filter only stars with valid flux in both bands
    df_ref = df_ref.dropna(subset=["Flux_B_measured","Flux_V_measured"])

    return df_ref


# ---------------- CALIBRATION FUNCTION ----------------
# ---------------- CALIBRATION FUNCTION ----------------
def compute_zero_point(fluxes, mags):
    """Compute zero point given fluxes and catalog magnitudes, ignoring NaNs."""
    mask = (~np.isnan(fluxes)) & (~np.isnan(mags))
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(mags[mask] + 2.5 * np.log10(fluxes[mask]))


# ---------------- RUN ANALYSIS ----------------
results_B = process_fits(fits_file_B, "B")
results_V = process_fits(fits_file_V, "V")

df_B = pd.DataFrame(results_B, columns=[
    "X", "Y", "FWHM", "Aperture Radius", "Flux",
    "Band", "Annulus Inner Radius", "Annulus Outer Radius"
])
df_V = pd.DataFrame(results_V, columns=[
    "X", "Y", "FWHM", "Aperture Radius", "Flux",
    "Band", "Annulus Inner Radius", "Annulus Outer Radius"
])

df_matched = match_sources(df_B, df_V, tol=MATCH_TOLERANCE)

csv_filename = os.path.join(_outdir, f"{obj_name}_photometry_results.csv")
df_matched.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")

# ----- Run reference star extraction -----
df_ref = extract_reference_stars(fits_file_V, df_B, df_V,
                                 mag_limit=REF_MAG_LIMIT,
                                 catalog=REF_CATALOG,
                                 tol=CATALOG_MATCH_TOLERANCE,
                                 flux_tol=CATALOG_FLUX_MATCH_TOLERANCE)
if not df_ref.empty:
    csv_ref = os.path.join(_outdir, f"{obj_name}_reference_stars.csv")
    df_ref.to_csv(csv_ref, index=False)
    print(f"Reference stars saved to {csv_ref}")

    # ----- Calibration using N reference stars -----
    print(f"{len(df_ref)} reference stars available.")
    N = int(input("Enter number of reference stars to use for calibration: "))
    df_calib = df_ref.head(N)

    # compute zero points
    zp_B = compute_zero_point(df_calib["Flux_B_measured"].values, df_calib["Bmag"].values)
    zp_V = compute_zero_point(df_calib["Flux_V_measured"].values, df_calib["Vmag"].values)

    # add aperture radii
    df_matched["Aperture_Radius_B"] = df_matched["FWHM_B"] * APERTURE_SCALE
    df_matched["Aperture_Radius_V"] = df_matched["FWHM_V"] * APERTURE_SCALE

    # compute calibrated magnitudes
    df_matched["Mag_B"] = zp_B - 2.5 * np.log10(df_matched["Flux_B"])
    df_matched["Mag_V"] = zp_V - 2.5 * np.log10(df_matched["Flux_V"])

    # save calibrated photometry
    csv_calib = os.path.join(_outdir, f"{obj_name}_calibrated_photometry.csv")
    df_matched.to_csv(csv_calib, index=False)
    print(f"Calibrated photometry saved to {csv_calib}")
    
    # ----- Create V-band image with detected sources -----
data_V, hdr_V = fits.getdata(fits_file_V, header=True)

plt.figure(figsize=(10, 10))
plt.imshow(data_V, cmap="gray", origin="lower", vmin=np.percentile(data_V, 5), vmax=np.percentile(data_V, 99))
plt.colorbar(label="Counts")

# overlay measured sources from df_V
plt.scatter(df_V["X"], df_V["Y"], s=40, edgecolor="red", facecolor="none", label="Measured sources")

plt.title(f"{obj_name} - V band with detected sources")
plt.xlabel("X [pixels]")
plt.ylabel("Y [pixels]")
plt.legend()

out_png = os.path.join(_outdir, f"{obj_name}_V_sources.png")
plt.savefig(out_png, dpi=150)
plt.close()
print(f"V-band source map saved to {out_png}")

