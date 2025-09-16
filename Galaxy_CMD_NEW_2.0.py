# -*- coding: utf-8 -*-
"""
Created by: Dr. Boaz Ron Zohar
https://github.com/BoazRonZohar/GalaxyCMD/blob/main/Galaxy_CMD_NEW_2.0.py
Affiliation: Global Sky Forum / LCO Projects
Date: September 2025

This code was developed as part of students projects on
spiral galaxies.

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
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u


# ===================== PARAMETERS =====================
# Aperture/annulus scaling relative to FWHM
APERTURE_SCALE = 2
ANNULUS_INNER_SCALE = 2.5
ANNULUS_OUTER_SCALE = 3
# FWHM measurement window (pixels from center)
FWHM_WINDOW_SIZE = 20

# Detection parameters
DAOFIND_FWHM = 10.0              # Expected size (FWHM in pixels) of detected light patches
SIGMA_CLIP = 3.0                 # Sigma clipping level for background statistics
DETECTION_THRESHOLD_SIGMA = 0.5  # Detection threshold in units of background sigma
PEAK_MIN_STD = 1.0               # Minimum peak signal-to-noise ratio for a detection

# Background subtraction
BG_SUB_FUNC = np.nanmedian       # Function used to compute and subtract background level

# Matching tolerance B and V Filters (pixels)
MATCH_TOLERANCE = 10.0           # Pixel tolerance for matching detections between B and V images

# Reference star parameters
REF_MAG_LIMIT = 15.0             # Catalog magnitude limit for selecting reference stars
REF_CATALOG = "II/336/apass9"    # Reference star catalog to use (APASS9 with B,V magnitudes)

# Matching tolerance between catalog stars and detected sources (pixels)
CATALOG_MATCH_TOLERANCE = 10.0   # Pixel tolerance for matching catalog stars to detected sources

# Matching tolerance between catalog stars and measured flux positions (pixels)
CATALOG_FLUX_MATCH_TOLERANCE = 10.0  # Pixel tolerance for matching catalog stars to measured flux positions

# Calibration parameters
CALIB_NUM_STARS = None           # Number of reference stars for calibration (user will be asked)

# CMD plot range for color index (B-V)
CMD_COLOR_MIN = -0.5   # left limit of x-axis
CMD_COLOR_MAX = 0.5    # right limit of x-axis

# === Parameters ===
REMOVE_TOL = 5.0  # pixel tolerance for matching sources to catalog stars

Step_size = 20    # Resolution of The Blue Knots Density Diagram 

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
def compute_zero_point(fluxes, mags):
    """Compute zero point given fluxes and catalog magnitudes, ignoring NaNs."""
    mask = (~np.isnan(fluxes)) & (~np.isnan(mags))
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(mags[mask] + 2.5 * np.log10(fluxes[mask]))


# ----- Add color index (B-V) and create CMD for both full and cleaned catalogs -----

def make_cmd(df, base_file, label, suffix):
    if "Mag_B" in df.columns and "Mag_V" in df.columns:
        # Compute color index
        df = df.copy()
        df["B-V"] = df["Mag_B"] - df["Mag_V"]

        # Reorder columns: place B-V immediately before Mag_V
        cols = list(df.columns)
        if "B-V" in cols and "Mag_V" in cols:
            cols.remove("B-V")
            cols.remove("Mag_V")
            # put everything else first, then B-V, then Mag_V
            cols = cols + ["B-V", "Mag_V"]
            df = df[cols]

        # Save new file with color index
        out_file_color = base_file.replace(".csv", f"_with_color{suffix}.csv")
        df.to_csv(out_file_color, index=False)
        print(f"File with color index saved to {out_file_color}")

        # Create CMD plot
        plt.figure(figsize=(8, 10))
        plt.scatter(df["B-V"], df["Mag_V"],
                    s=30, edgecolor="black", facecolor="cyan", alpha=0.7)

        plt.gca().invert_yaxis()  # brighter objects at the top
        plt.xlabel("B - V (Color Index)")
        plt.ylabel("V magnitude")
        plt.title(f"CMD of {obj_name} {label} (B-V, V)")

        # >>> Control of x-axis (color index) range <<<
        plt.xlim(CMD_COLOR_MIN, CMD_COLOR_MAX)

        out_cmd = base_file.replace(".csv", f"_CMD{suffix}.png")
        plt.savefig(out_cmd, dpi=150)
        plt.close()
        print(f"CMD diagram saved to {out_cmd}")
    else:
        print(f"Mag_B or Mag_V not found in {label} catalog. CMD not created.")

# ---------------- RUN ANALYSIS ----------------
print("=== Photometry Input ===")
obj_name = _ask("Galaxy name", "M101", str)
distance    = _ask("Distance (Mpc)", 1, float)
A_V         = _ask("Galactic extinction A_V (mag)", 0.1, float)
E_BV        = _ask("Galactic color excess E(B-V)", 0.05, float)
fits_file_B = _norm_path(_ask("Path to B-band FITS", r"D:\example_B.fts", str))
fits_file_V = _norm_path(_ask("Path to V-band FITS", r"D:\example_V.fts", str))

_outdir = os.path.dirname(fits_file_B) if os.path.dirname(fits_file_B) else os.getcwd()
fits_file_B = subtract_background_and_save(fits_file_B)
fits_file_V = subtract_background_and_save(fits_file_V)

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

# This block loads the calibrated photometry results and the reference stars list
# It compares the X,Y positions of all measured sources with the reference stars
# Any source within a pixel tolerance (REMOVE_TOL) of a reference star is flagged as a star
# Those flagged sources are removed from the calibrated photometry table
# A new cleaned file is saved with the suffix "_calibrated_photometry_no_stars.csv"

# === Input files ===
calib_file = csv_calib
ref_file   = csv_ref

# === Load data ===
df_calib = pd.read_csv(calib_file)
df_ref   = pd.read_csv(ref_file)

# Ensure numeric coords
for col in ["X_B","Y_B","X_V","Y_V"]:
    if col in df_ref.columns:
        df_ref[col] = pd.to_numeric(df_ref[col], errors="coerce")

# === Filter out reference stars ===
mask_remove = []
for i, row in df_calib.iterrows():
    xb, yb = row.get("X_B", np.nan), row.get("Y_B", np.nan)
    xv, yv = row.get("X_V", np.nan), row.get("Y_V", np.nan)

    # check distance to all reference stars
    dB = np.sqrt((df_ref["X_B"] - xb)**2 + (df_ref["Y_B"] - yb)**2)
    dV = np.sqrt((df_ref["X_V"] - xv)**2 + (df_ref["Y_V"] - yv)**2)

    if (dB.min() <= REMOVE_TOL) or (dV.min() <= REMOVE_TOL):
        mask_remove.append(True)
    else:
        mask_remove.append(False)

df_clean = df_calib.loc[~pd.Series(mask_remove)].reset_index(drop=True)

# === Save new file ===
out_file = calib_file.replace("_calibrated_photometry.csv",
                              "_calibrated_photometry_no_stars.csv")
df_clean.to_csv(out_file, index=False)
print(f"Cleaned file saved to {out_file}")


# ----- Create V-band image with all measured sources (already exists above) No stars-----
# {obj_name}_V_sources.png is saved earlier in the code

# ----- Create V-band image with cleaned sources (no catalog stars) -----
plt.figure(figsize=(10, 10))
plt.imshow(data_V, cmap="gray", origin="lower",
           vmin=np.percentile(data_V, 5), vmax=np.percentile(data_V, 99))
plt.colorbar(label="Counts")

# overlay cleaned detections from the filtered catalog
plt.scatter(df_clean["X_B"], df_clean["Y_B"],
            s=40, edgecolor="blue", facecolor="none", label="Cleaned sources (no stars)")

plt.title(f"{obj_name} - V band with detected sources (no stars)")
plt.xlabel("X [pixels]")
plt.ylabel("Y [pixels]")
plt.legend()

out_png_clean = os.path.join(_outdir, f"{obj_name}_V_sources_no_stars.png")
plt.savefig(out_png_clean, dpi=150)
plt.close()
print(f"V-band cleaned source map saved to {out_png_clean}")

# Run CMD creation for full calibrated photometry
make_cmd(df_calib, calib_file, "(all sources)", "")

# Run CMD creation for cleaned catalog (no stars)
make_cmd(df_clean, out_file, "(no stars)", "_no_stars")


# ----- Create V-band image and CMD with cleaned sources filtered by color index (B-V range) -----
if "Mag_B" in df_clean.columns and "Mag_V" in df_clean.columns:
    df_clean_color = df_clean.copy()
    df_clean_color["B-V"] = df_clean_color["Mag_B"] - df_clean_color["Mag_V"]

    # filter by CMD_COLOR_MIN / CMD_COLOR_MAX
    mask_color = (df_clean_color["B-V"] >= CMD_COLOR_MIN) & (df_clean_color["B-V"] <= CMD_COLOR_MAX)
    df_color_filtered = df_clean_color[mask_color].reset_index(drop=True)

    # --- save filtered catalog as CSV ---
    out_csv_color = os.path.join(_outdir, f"{obj_name}_calibrated_photometry_no_stars_color_filtered.csv")
    df_color_filtered.to_csv(out_csv_color, index=False)
    print(f"Color-filtered photometry saved to {out_csv_color}")

    # --- create and save galaxy image with color-filtered sources ---
    plt.figure(figsize=(10, 10))
    plt.imshow(data_V, cmap="gray", origin="lower",
               vmin=np.percentile(data_V, 5), vmax=np.percentile(data_V, 99))
    plt.colorbar(label="Counts")

    plt.scatter(df_color_filtered["X_B"], df_color_filtered["Y_B"],
                s=40, edgecolor="green", facecolor="none",
                label=f"Sources in color range ({CMD_COLOR_MIN} ≤ B-V ≤ {CMD_COLOR_MAX})")

    plt.title(f"{obj_name} - V band with color-filtered sources (no stars)")
    plt.xlabel("X [pixels]")
    plt.ylabel("Y [pixels]")
    plt.legend()

    out_png_color = os.path.join(_outdir, f"{obj_name}_V_sources_color_filtered.png")
    plt.savefig(out_png_color, dpi=150)
    plt.close()
    print(f"V-band color-filtered source map saved to {out_png_color}")

    # --- create CMD for the color-filtered catalog ---
    make_cmd(df_color_filtered, out_csv_color, "(no stars, color filtered)", "_no_stars_color_filtered")

else:
    print("Mag_B or Mag_V not found in cleaned catalog. Color-filtered map not created.")
    

# ----- Add radial distances in px and pc and create radial profiles -----
if not df_color_filtered.empty:
    # Galaxy center in pixels
    x_center = data_V.shape[1] / 2.0
    y_center = data_V.shape[0] / 2.0

    # Radial distance in pixels
    df_color_filtered["Radial_Distance_px"] = np.sqrt(
        (df_color_filtered["X_B"] - x_center)**2 +
        (df_color_filtered["Y_B"] - y_center)**2
    )

    # Pixel scale from WCS (deg/pixel -> rad/pixel)
    wcs_V = WCS(hdr_V)
    pixscale_deg = np.mean(np.abs(wcs_V.pixel_scale_matrix.diagonal()))
    pixscale_rad = np.deg2rad(pixscale_deg)

    # Distance in parsec
    distance_pc = distance * 1.0e6

    # Conversion factor (pc/pixel)
    px_to_pc = pixscale_rad * distance_pc

    # Add distances in pc
    df_color_filtered["Radial_Distance_pc"] = df_color_filtered["Radial_Distance_px"] * px_to_pc

    # Sort by radial distance
    df_color_filtered = df_color_filtered.sort_values("Radial_Distance_px").reset_index(drop=True)

    # Save CSV with all distances
    out_csv_color_rad = os.path.join(
        _outdir,
        f"{obj_name}_calibrated_photometry_no_stars_color_filtered_with_radius.csv"
    )
    df_color_filtered.to_csv(out_csv_color_rad, index=False)
    print(f"Color-filtered photometry with radial distances saved to {out_csv_color_rad}")

    # -----------------------------------------------------------
    # Radial density profile in pixels
    # -----------------------------------------------------------
    bin_width_px = Step_size
    max_r_px = df_color_filtered["Radial_Distance_px"].max()
    bins_px = np.arange(0, max_r_px + bin_width_px, bin_width_px)

    counts_px, edges_px = np.histogram(df_color_filtered["Radial_Distance_px"], bins=bins_px)
    areas_px2 = np.pi * (edges_px[1:]**2 - edges_px[:-1]**2)
    densities_px = counts_px / areas_px2
    bin_centers_px = 0.5 * (edges_px[1:] + edges_px[:-1])

    # Save profile (px)
    df_profile_px = pd.DataFrame({
        "R_inner_px": edges_px[:-1],
        "R_outer_px": edges_px[1:],
        "R_center_px": bin_centers_px,
        "N_sources": counts_px,
        "Annulus_area_px2": areas_px2,
        "Density_per_px2": densities_px
    })
    out_csv_profile_px = os.path.join(_outdir, f"{obj_name}_radial_density_profile.csv")
    df_profile_px.to_csv(out_csv_profile_px, index=False)
    print(f"Radial density profile (px) saved to {out_csv_profile_px}")

    # Step profile (px)
    step_r_px, step_density_px, step_counts_px = [], [], []
    for i in range(len(densities_px)):
        r_in, r_out = edges_px[i], edges_px[i+1]
        step_r_px.extend([r_in, r_out])
        step_density_px.extend([densities_px[i], densities_px[i]])
        step_counts_px.extend([counts_px[i], counts_px[i]])

    df_step_px = pd.DataFrame({
        "R_step_px": step_r_px,
        "Density_step_per_px2": step_density_px,
        "N_sources_step": step_counts_px
    })
    out_csv_step_px = os.path.join(_outdir, f"{obj_name}_radial_density_profile_step.csv")
    df_step_px.to_csv(out_csv_step_px, index=False)
    print(f"Radial density step profile (px) saved to {out_csv_step_px}")

    plt.figure(figsize=(8,6))
    plt.step(step_r_px, step_density_px, where="post", color="blue", linewidth=2)
    plt.xlabel("Radial distance [pixels]")
    plt.ylabel("Source density [1/pixel²]")
    plt.title(f"Radial density step profile of {obj_name} (pixels)")
    out_png_step_px = os.path.join(_outdir, f"{obj_name}_radial_density_profile_step.png")
    plt.savefig(out_png_step_px, dpi=150)
    plt.close()
    print(f"Radial density step plot (px) saved to {out_png_step_px}")

    # -----------------------------------------------------------
    # Radial density profile in pc
    # -----------------------------------------------------------
    bin_width_pc = bin_width_px * px_to_pc
    max_r_pc = df_color_filtered["Radial_Distance_pc"].max()
    bins_pc = np.arange(0, max_r_pc + bin_width_pc, bin_width_pc)

    counts_pc, edges_pc = np.histogram(df_color_filtered["Radial_Distance_pc"], bins=bins_pc)
    areas_pc2 = np.pi * (edges_pc[1:]**2 - edges_pc[:-1]**2)
    densities_pc = counts_pc / areas_pc2
    bin_centers_pc = 0.5 * (edges_pc[1:] + edges_pc[:-1])

    # Save profile (pc)
    df_profile_pc = pd.DataFrame({
        "R_inner_pc": edges_pc[:-1],
        "R_outer_pc": edges_pc[1:],
        "R_center_pc": bin_centers_pc,
        "N_sources": counts_pc,
        "Annulus_area_pc2": areas_pc2,
        "Density_per_pc2": densities_pc
    })
    out_csv_profile_pc = os.path.join(_outdir, f"{obj_name}_radial_density_profile_pc.csv")
    df_profile_pc.to_csv(out_csv_profile_pc, index=False)
    print(f"Radial density profile (pc) saved to {out_csv_profile_pc}")

    # Step profile (pc)
    step_r_pc, step_density_pc, step_counts_pc = [], [], []
    for i in range(len(densities_pc)):
        r_in, r_out = edges_pc[i], edges_pc[i+1]
        step_r_pc.extend([r_in, r_out])
        step_density_pc.extend([densities_pc[i], densities_pc[i]])
        step_counts_pc.extend([counts_pc[i], counts_pc[i]])

    df_step_pc = pd.DataFrame({
        "R_step_pc": step_r_pc,
        "Density_step_per_pc2": step_density_pc,
        "N_sources_step": step_counts_pc
    })
    out_csv_step_pc = os.path.join(_outdir, f"{obj_name}_radial_density_profile_step_pc.csv")
    df_step_pc.to_csv(out_csv_step_pc, index=False)
    print(f"Radial density step profile (pc) saved to {out_csv_step_pc}")

    plt.figure(figsize=(8,6))
    plt.step(step_r_pc, step_density_pc, where="post", color="green", linewidth=2)
    plt.xlabel("Radial distance [pc]")
    plt.ylabel("Source density [1/pc²]")
    plt.title(f"Radial density step profile of {obj_name} (pc)")
    out_png_step_pc = os.path.join(_outdir, f"{obj_name}_radial_density_profile_step_pc.png")
    plt.savefig(out_png_step_pc, dpi=150)
    plt.close()
    print(f"Radial density step plot (pc) saved to {out_png_step_pc}")

else:
    print("No color-filtered sources available to compute radial distances.")
