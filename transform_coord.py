#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: Line_Of_Sight
File   : transform_coord.py

Author: Pessel Arnaud
Date: 2025-07
Version: 1.1
GitHub: https://github.com/dunaar/Line_Of_Sight
License: MIT

Description:
    This module provides functions to convert between geographic coordinates (longitude, latitude, altitude) and Cartesian coordinates (x, y, z).
    It includes functions for both direct and inverse conversion, as well as functions to calculate
    great circle distances and straight-line distances between points on a sphere, adjusted for altitude.
    It uses NumPy for vectorized operations.

Dependencies:
    - NumPy: For handling array and scalar data types.

Usage:
    This module can be imported to provide functions for coordinate transformations and distance calculations.
    The `__main__` block can be run to test the functionality with generated data.
    >>> from transform_coord import geo_to_cart, cart_to_geo, great_circle_distances, straight_line_distances
    >>> x, y, z = geo_to_cart(lon, lat, alt)
    >>> lon, lat, alt = cart_to_geo(x, y, z)
    >>> d_gc = great_circle_distances(lon1, lat1, alt1, lon2, lat2, alt2)
    >>> d_sl = straight_line_distances(lon1, lat1, alt1, lon2, lat2, alt2)
"""

__version__ = "1.1"

# %% Imports
# -----------------------------------------------------------------------------
import os
import time
import numpy as np
from numba import jit

NUM_CORES = os.cpu_count()  # Detect number of CPU cores
# print('NUM_CORES', NUM_CORES)
# -----------------------------------------------------------------------------

# %% Constants
# -----------------------------------------------------------------------------
R_EARTH = np.float32(6371000.0)      # Mean radius of the Earth in meters
D2R     = np.float32(np.pi / 180.0)  # Degrees to radians
R2D     = np.float32(180.0 / np.pi)  # Radians to degrees
# -----------------------------------------------------------------------------

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Direct conversion: WGS84 → Cartesian coordinates

@jit(nopython=True, cache=True)
def geo_to_cart(lon, lat, alt, R=R_EARTH):
    """
    Converts geographic coordinates (lon, lat, alt) to Cartesian coordinates (x, y, z).
    Assumption: Spherical Earth
    Inputs: lon, lat (in degrees), alt (in meters); NumPy arrays of float32 or scalars
    Outputs: x, y, z (in meters); NumPy arrays of float32 or scalars
    """
    lon_rad = D2R * lon
    lat_rad = D2R * lat
    r_alt   = R + alt

    sin_lat         = np.sin(lat_rad)
    cos_lat_cos_lon = np.cos(lat_rad) * np.cos(lon_rad)
    cos_lat_sin_lon = np.cos(lat_rad) * np.sin(lon_rad)

    x = r_alt * cos_lat_cos_lon
    y = r_alt * cos_lat_sin_lon
    z = r_alt * sin_lat

    return x, y, z
# -----------------------------------------------------------------------------

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inverse conversion: Cartesian → WGS84 (Geographic coordinates)

@jit(nopython=True, cache=True)
def cart_to_geo(x, y, z, R=R_EARTH):
    """
    Converts Cartesian coordinates (x, y, z) to geographic coordinates (lon, lat, alt).
    Assumption: Spherical Earth
    Inputs: x, y, z (in meters), NumPy arrays of float32 or scalars
    Outputs: lon, lat (in degrees), alt (in meters); NumPy arrays of float32 or scalars
    """
    r_alt = np.sqrt(x**2 + y**2 + z**2)

    lon = R2D * np.arctan2(y, x)
    lat = R2D * np.arcsin(z / r_alt)
    alt = r_alt - R
    
    return lon, lat, alt
# -----------------------------------------------------------------------------

def great_circle_distances(lons1: np.ndarray, lats1: np.ndarray, alts1: np.ndarray,
                           lons2: np.ndarray, lats2: np.ndarray, alts2: np.ndarray,
                           R: float = R_EARTH) -> np.ndarray:
    """
    Calculates the great circle distance between multiple pairs of points on a sphere, adjusted for altitude.

    This function uses the haversine formula to compute the surface distance and incorporates the altitude difference to approximate the 3D distance.

    Parameters:
    - lons1: np.ndarray or float: Longitude of the first set of points in degrees.
    - lats1: np.ndarray or float: Latitude of the first set of points in degrees.
    - alts1: np.ndarray or float: Altitude of the first set of points in meters.
    - lons2: np.ndarray or float: Longitude of the second set of points in degrees.
    - lats2: np.ndarray or float: Latitude of the second set of points in degrees.
    - alts2: np.ndarray or float: Altitude of the second set of points in meters.
    - R: float, optional: Radius of the sphere in meters (default is Earth's mean radius: 6,371,000 meters).

    Returns:
    - distances: np.ndarray: Array of 3D distances in meters for each pair of points, combining surface distance and altitude difference.

    Notes:
    - The haversine formula computes the great circle distance, which is then adjusted for altitude using the Pythagorean theorem: sqrt(d_horiz^2 + delta_h^2).
    - Inputs are expected in degrees for latitude/longitude and meters for altitude.
    - Broadcasting is supported for arrays of different shapes, according to NumPy rules.
    - This is an approximation, as it assumes small altitude differences relative to Earth's radius.
    """
    # Convert degrees to radians
    lats1_rad = np.radians(lats1)
    lats2_rad = np.radians(lats2)

    # Differences in latitude and longitude
    delta_lats = lats2_rad - lats1_rad
    delta_lons = np.radians(lons2) - np.radians(lons1)

    # Haversine formula for surface distance
    a = np.sin(delta_lats / 2)**2 + np.cos(lats1_rad) * np.cos(lats2_rad) * np.sin(delta_lons / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d_horiz = R * c

    # Incorporate altitude difference
    delta_h = np.abs(alts2 - alts1)
    distances = np.sqrt(d_horiz**2 + delta_h**2)
    
    return distances

def straight_line_distances(lons1: np.ndarray, lats1: np.ndarray, alts1: np.ndarray,
                            lons2: np.ndarray, lats2: np.ndarray, alts2: np.ndarray,
                            R: float = R_EARTH) -> np.ndarray:
    """
    Calculates the straight-line (3D Euclidean) distance between multiple pairs of points using Cartesian coordinates.

    Parameters:
    - lons1: np.ndarray or float: Longitude of the first set of points in degrees.
    - lats1: np.ndarray or float: Latitude of the first set of points in degrees.
    - alts1: np.ndarray or float: Altitude of the first set of points in meters.
    - lons2: np.ndarray or float: Longitude of the second set of points in degrees.
    - lats2: np.ndarray or float: Latitude of the second set of points in degrees.
    - alts2: np.ndarray or float: Altitude of the second set of points in meters.
    - R: float, optional: Radius of the sphere in meters (default is Earth's mean radius: 6,371,000 meters).

    Returns:
    - distances: np.ndarray: 3D Euclidean distances in meters for each pair of points, including altitude.

    Example command for testing from command line:
    ```
    python -m Line_Of_Sight.transform_coord
    ```

    Notes:
    - Converts geographic coordinates (latitude, longitude, altitude) to Cartesian (x, y, z) coordinates.
    - Computes the Euclidean distance in 3D space.
    - Broadcasting is supported for arrays of different shapes, according to NumPy rules.
    - Assumes altitudes are relative to Earth's surface (e.g., above sea level).
    """
    # Convert degrees to radians
    xs1, ys1, zs1 = geo_to_cart(lons1, lats1, alts1, R)
    xs2, ys2, zs2 = geo_to_cart(lons2, lats2, alts2, R)

    # Compute Euclidean distance
    distances = np.sqrt((xs2 - xs1)**2 + (ys2 - ys1)**2 + (zs2 - zs1)**2)

    return distances

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main Execution
def main() -> None:
    # Generate test data (10,000 points, float32)
    n = 10000
    lon = np.random.uniform(-180, 180, n).astype(np.float32)
    lat = np.random.uniform(-90, 90, n).astype(np.float32)
    alt = np.random.uniform(-1000, 10000, n).astype(np.float32)
    
    # Measure performance
    results = {}
    REPEATS = 5000

    # Test NumPy implementation
    start = time.perf_counter_ns()
    for _ in range(REPEATS):
        x_np, y_np, z_np = geo_to_cart(lon, lat, alt)
    results['geo_to_cart_numpy'] = (time.perf_counter_ns() - start) / (REPEATS * 1e9)

    start = time.perf_counter_ns()
    for _ in range(REPEATS):
        lon_np, lat_np, alt_np = cart_to_geo(x_np, y_np, z_np)
    results['cart_to_geo_numpy'] = (time.perf_counter_ns() - start) / (REPEATS * 1e9)

    # Display results
    print(f"\nResults for {n} points (float32):")
    for impl, time_val in results.items():
        print(f"  {impl}: {time_val*1000.:.3f} ms")

    # Calculate speedups
    if 'geo_to_cart_numba' in results and 'geo_to_cart_numpy' in results:
        print(f"Speedup Numba vs. NumPy: {(results['geo_to_cart_numpy'] / results['geo_to_cart_numba']):.2f}x")

    if 'cart_to_geo_numba' in results and 'cart_to_geo_numpy' in results:
        print(f"Speedup Numba vs. NumPy: {(results['cart_to_geo_numpy'] / results['cart_to_geo_numba']):.2f}x")

if __name__ == '__main__':
    print(straight_line_distances(0., 80., 0., 0., 81., 0.))
    print(straight_line_distances(0., 80., 0., 1., 80., 0.))
    print(straight_line_distances(0., 80., 0., 0., 80., 1.))
    main()
# ====================================================================================================
