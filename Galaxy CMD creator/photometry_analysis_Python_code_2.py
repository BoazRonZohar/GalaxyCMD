# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:49:04 2025

@author: Lenovo
"""

import os
import astropy
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter

# Function to calculate FWHM for each source
def compute_fwhm(data, x, y, size=10):
    """Measure FWHM around a light source."""
    # Check if the coordinates are within the image bounds
    x_min, x_max = int(x-size), int(x+size)
    y_min, y_max = int(y-size), int(y+size)

    # Ensure sub-image is within the bounds of the data array
    if x_min < 0 or y_min < 0 or x_max >= data.shape[1] or y_max >= data.shape[0]:
        print(f"Skipping source at ({x}, {y}) due to out-of-bounds sub-image.")
        return None

    # Extract sub-image centered around the source
    sub_image = data[y_min:y_max, x_min:x_max]
    smoothed = gaussian_filter(sub_image, sigma=2)  # Smoothing to reduce noise
    peak = np.max(smoothed)
    half_max = peak / 2

    # Find the width at half-max
    above_half_max = smoothed > half_max
    indices = np.argwhere(above_half_max)
    if indices.size > 0:
        min_x, max_x = indices[:, 1].min(), indices[:, 1].max()
        min_y, max_y = indices[:, 0].min(), indices[:, 0].max()
        fwhm_x = max_x - min_x
        fwhm_y = max_y - min_y
        return np.mean([fwhm_x, fwhm_y])
    return None

# Function to read a FITS file and perform source detection and measurements
def process_fits(filename, band):
    """Load a FITS image and detect and measure light sources."""
    hdul = fits.open(filename)
    data = hdul[0].data
    hdul.close()

    # Calculate background and standard deviation
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = 5.0 * std  # Threshold for source detection

    # Source detection using DAOStarFinder
    daofind = DAOStarFinder(fwhm=3.0, threshold=threshold)
    sources = daofind(data - median)

    # Noise filtering: Keep only sources with sufficient peak amplitude
    sources = sources[sources['peak'] > 10 * std]

    # List to store results
    results = []
    for source in sources:
        x, y = source['xcentroid'], source['ycentroid']
        fwhm = compute_fwhm(data, x, y)
        if fwhm is not None:
            radius = 1.5 * fwhm  # Aperture radius based on FWHM
            aperture = CircularAperture((x, y), r=radius)
            annulus_inner_radius = radius * 1.5
            annulus_outer_radius = radius * 2.5  # Annulus for background measurement
            annulus = CircularAnnulus((x, y), r_in=annulus_inner_radius, r_out=annulus_outer_radius)

            # Perform photometry
            phot_table = aperture_photometry(data, [aperture, annulus])

            # Calculate background-subtracted flux
            background_mean = phot_table['aperture_sum_1'][0] / annulus.area  # Background mean
            background_subtracted_flux = phot_table['aperture_sum_0'][0] - background_mean * aperture.area

            # Apply a condition to discard unrealistic flux values
            if background_subtracted_flux < 0:  # Discard sources with negative flux values
                continue

            # Append the result to the list
            results.append([x, y, fwhm, radius, background_subtracted_flux, band, annulus_inner_radius, annulus_outer_radius])

    return results

# Function to generate unique CSV filename
#def get_unique_filename(base_filename,file_path):
    #index = 1
    # Check if the file already exists
    #while os.path.exists(f"{base_filename}_{index}.csv"):
     #   index += 1
    #return f"{base_filename}_{index}.csv"

# File path for the "B" band image (integrated from your specified path)
#file_path = r"C:\Users\Lenovo\Desktop\M101 Median  of 3\M101 V 1 Meter Median  of 3.fts"
file_path = r"C:\Users\Lenovo\Desktop\M101 Ha Median of 3 15.12.24 SUB X 11.5 - rp.fts"

# Process the FITS image and obtain results
all_results = process_fits(file_path, file_path)

# Convert results to a pandas DataFrame
df = pd.DataFrame(all_results, columns=["X", "Y", "FWHM", "Aperture Radius", "Flux", "Band", "Annulus Inner Radius", "Annulus Outer Radius"])

# Generate a unique filename for the CSV file
#csv_filename = get_unique_filename("photometry_results")
#csv_filename = get_unique_filename("photometry_results", file_path)

end = "photometry_results"
index = file_path

csv_filename = f"{index}_{end}.csv"


# Save the DataFrame to the CSV file with the generated unique filename
df.to_csv(csv_filename, index=False)

print(f"Data saved to {csv_filename}")
