#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: Line_Of_Sight (Convert)

Author: Pessel Arnaud
Date: 2025-05-02
Version: 1.5
GitHub: https://github.com/dunaar/Line_Of_Sight
License: MIT

Description: Preprocesses SRTM15 NetCDF data into 1° x 1° tiles, applies resampling for |lat| > 30°,
and saves each latitude band as a single 1D NumPy array with tile start indices in a (180, 360) NumPy array in a ZIP archive with MessagePack serialization.
It is possible to download SRTM15 NetCDF file from website : https://topex.ucsd.edu/pub/
SRTM15 NetCDF file example: https://topex.ucsd.edu/pub/srtm15_plus/SRTM15_V2.7.nc

Example:
```
python -m Line_Of_Sight.convert_dtm SRTM15_V2.7.nc srtm15_tiles_compressed.zip
```
"""

__version__ = "1.5"

# === Built-in ===
import logging
import zipfile
from multiprocessing import Pool, cpu_count

# === Third-party ===
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

# === Local modules ===
from .np_msgspec_msgpack_utils import enc

def process_band(args):
    """
    Process a single 1° latitude band, creating a single 1D NumPy array of concatenated 1° x 1° tiles with start indices.

    Parameters:
        args (tuple): (input_filename, lat_idx, lat_inf, lat_sup, rows_per_tile, cols_per_tile, lons)
            - input_filename (str): Path to input NetCDF file.
            - lat_idx (int): Index of the latitude band (0 to 179).
            - lat_inf (float): Lower latitude of the band (°).
            - lat_sup (float): Upper latitude of the band (°).
            - rows_per_tile (int): Number of rows per tile (240).
            - cols_per_tile (int): Number of columns per tile (240).
            - lons (ndarray): Array of longitude values (-180° to 180°).

    Returns:
        tuple: (lat_idx, band_array, lon_indices_in_band)
            - lat_idx: Index of the latitude band.
            - band_array: Serialized (MessagePack-encoded) 1D array containing all tiles.
            - lon_indices_in_band: List of start_idx for each longitude tile in band_array.
    """
    input_filename, lat_idx, lat_inf, lat_sup, rows_per_tile, cols_per_tile, lons = args
    
    # Compute row indices for the band
    tile_y_beg = lat_idx * rows_per_tile
    tile_y_end = (lat_idx + 1) * rows_per_tile
    
    # Open NetCDF file and read the band
    with Dataset(input_filename, "r") as nc_data:
        # Force negative values to zero (remove bathymetry) and convert to uint16
        data_band = np.clip(nc_data.variables['z'][tile_y_beg:tile_y_end, :], 0, None).astype(np.uint16)  # Shape: (240, 86400)

    # Compute resampling factor for lats |lat| > 30°
    new_cols_per_tile = int(np.ceil(cols_per_tile * np.cos(np.radians(lat_inf if lat_inf >= 0 else lat_sup))))
    intile_new_stp    = cols_per_tile / new_cols_per_tile if new_cols_per_tile < cols_per_tile else 1
    intile_new_xs     = np.arange(0, new_cols_per_tile) * intile_new_stp

    # Calculate total size for 1D array
    #total_elements = 0
    #for lon_idx in range(len(lons)):
    #    if new_cols_per_tile >= cols_per_tile:
    #        total_elements += rows_per_tile * cols_per_tile
    #    else:
    #        total_elements += rows_per_tile * new_cols_per_tile if not np.all(data_band[:, lon_idx * cols_per_tile:(lon_idx + 1) * cols_per_tile] == 0) else 1
    
    # Initialize 1D array to store all tiles for this band
    lon_indices_in_band = np.empty(len(lons), dtype=np.uint64)  # 1D array to store 'Start indice' for each 1° longitude tile in band_array
    tiles_nrows_in_band = np.empty(len(lons), dtype=np.uint32)  # 1D array to store number of rows in each tile of the band
    tiles_ncols_in_band = np.empty(len(lons), dtype=np.uint32)  # 1D array to store number of cols in each tile of the band
    
    # Process each 1° longitude tile
    band_array  = np.empty(0, dtype=np.uint16)
    for lon_idx in range(len(lons)):
        # Store start index
        lon_indices_in_band[lon_idx] = band_array.size # current band_array size as band_array is continously appended

        # -- Extract tile data
        #    Compute column indices for the tile
        inband_x_beg = lon_idx * cols_per_tile
        inband_x_end = (lon_idx + 1) * cols_per_tile

        # Create tile only if any altitude > 0
        tile_data = data_band[:, inband_x_beg:inband_x_end]  # Shape: (240, 240)

        if np.all(tile_data == 0):
            # Store zero tile (1x1) as a single uint16
            tile_data = np.zeros((1, 1), dtype=np.uint16)  # Shape: (1, 1)
        elif new_cols_per_tile < cols_per_tile:
            # Resampling needed
            tile_data = np.empty((rows_per_tile, new_cols_per_tile), dtype=np.uint16)
            
            # Compute all lower and upper indices at once
            centers = inband_x_beg + intile_new_xs
            inband_x_inf = np.floor(centers - intile_new_stp / 2. + 0.5).astype(np.int32)
            inband_x_sup = np.floor(centers + intile_new_stp / 2. + 0.5).astype(np.int32)
            
            for newtile_x in range(new_cols_per_tile):
                if inband_x_inf[newtile_x] >= 0:
                    # Normal case: take max over the range
                    tile_data[:, newtile_x] = data_band[:, inband_x_inf[newtile_x]:inband_x_sup[newtile_x]].max(axis=1)
                else:
                    # Edge case near ±180°: concatenate ranges
                    negative_slice = data_band[:, inband_x_inf[newtile_x]:]
                    positive_slice = data_band[:, :inband_x_sup[newtile_x]]
                    combined = np.hstack((negative_slice, positive_slice))
                    tile_data[:, newtile_x] = combined.max(axis=1)

        # Flatten the tile data and append to band_array
        band_array = np.concatenate((band_array, tile_data.ravel()))
        tiles_nrows_in_band[lon_idx] = tile_data.shape[0]
        tiles_ncols_in_band[lon_idx] = tile_data.shape[1]

    # Control
    band_size  = (tiles_nrows_in_band * tiles_ncols_in_band).sum() # Total number of values in the band
    if band_array.size != band_size:
        logging.warning(f"Size mismatch for lat {lat_idx}: {band_array.size} != {band_size}")
        logging.warning(f"tiles_nrows: {tiles_nrows_in_band}")
        logging.warning(f"tiles_ncols: {tiles_ncols_in_band}")
    assert band_array.size == band_size, f"Size mismatch for lat {lat_idx}: {band_array.size} != {band_size}"

    # Serialize band_array before returning
    band_array = enc.encode(band_array)

    return lat_idx, band_array, lon_indices_in_band, tiles_nrows_in_band, tiles_ncols_in_band

def convert_file(input_filename, output_filename):
    """
    Preprocess SRTM15 NetCDF data into tiles and an index structure.

    Reads the NetCDF file, processes bands in parallel, creates a single 1D NumPy array per latitude band,
    and saves the results with tile start indices in a (180, 360) NumPy array to a zip archive.

    Parameters:
        input_filename (str): Path to input NetCDF file.
        output_filename (str): Path to output ZIP archive.

    Returns:
        None
    """
    # Define parameters
    cells_per_degree = 240  # Resolution: 15 arc-seconds = 1/240°
    
    # Open NetCDF file to read metadata
    with Dataset(input_filename, "r") as nc_data:
        lats_nc = nc_data.variables['lat'][:]  # -90° to 90°
        lons_nc = nc_data.variables['lon'][:]  # -180° to 180°
    
    # Verify NetCDF dimensions
    assert len(lats_nc) == 43200 and len(lons_nc) == 86400, "Invalid coordinate dimensions"
    
    # Define latitude and longitude arrays
    lat_stp, lon_stp = 1, 1
    lats = np.arange( -90,  90, lat_stp)  # 180 latitude bands
    lons = np.arange(-180, 180, lon_stp)  # 360 longitude tiles per band
    
    # Get number of CPU cores for parallelism
    n_cores = cpu_count()
    print('Number of cores:', n_cores)

    # Initialize tiles_lon_indices_in_bands array and tiles_nrows array
    tiles_lon_indices_in_bands = np.empty((len(lats), len(lons)), dtype=np.uint64)
    tiles_nrows                = np.empty((len(lats), len(lons)), dtype=np.uint32)
    tiles_ncols                = np.empty((len(lats), len(lons)), dtype=np.uint32)
    
    # Initialize metadata
    metadata = {
        'lat_range': [float(lats_nc[0]), float(lats_nc[-1])],
        'lon_range': [float(lons_nc[0]), float(lons_nc[-1])],
        'tiles_lon_indices_in_bands': tiles_lon_indices_in_bands,  # (180, 360) array for start indice for each 1°-lon in band (latitude)
        'tiles_nrows': tiles_nrows,  # Shape of tiles depending on band (latitude)
        'tiles_ncols': tiles_ncols,  # Shape of tiles depending on band (latitude)
    }

    # Process bands in batches of n_cores
    with zipfile.ZipFile(output_filename, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
        with Pool(n_cores) as pool:
            for batch_start in tqdm(range(0, len(lats), n_cores), total=len(lats)//n_cores + (1 if len(lats) % n_cores else 0), desc="Process bands in batches"):
                # Define indices for the current batch
                batch_indices = range(batch_start, min(batch_start + n_cores, len(lats)))
                
                # Prepare arguments for each band
                batch_args = [(input_filename, lat_idx, lats[lat_idx], lats[lat_idx] + lat_stp,
                               cells_per_degree, cells_per_degree, lons)
                              for lat_idx in batch_indices]
                
                # Launch parallel processing for the batch
                print('Lancement batch:', list(batch_indices))
                if n_cores > 1:
                    results = pool.map(process_band, batch_args)
                else:
                    results = [process_band(args) for args in batch_args]
                
                # Write results to zip archive and update lon_indices_array
                for lat_idx, band_array, lon_indices_in_band, tiles_nrows_in_band, tiles_ncols_in_band in results:
                    key = f'lat_{lat_idx:03d}'
                    zf.writestr(f"{key}.packed", band_array)
                    tiles_lon_indices_in_bands[lat_idx, :] = lon_indices_in_band
                    tiles_nrows[lat_idx, :] = tiles_nrows_in_band
                    tiles_ncols[lat_idx, :] = tiles_ncols_in_band
                    logging.info(f"Wrote {key}.packed to {output_filename}")

        # Save metadata to the zip archive
        zf.writestr('metadata.packed', enc.encode(metadata))
        logging.info(f"Wrote metadata.packed to {output_filename}")

    print(f"Preprocessing completed. Saved to: {output_filename}")

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(levelname)s-%(module)s-%(funcName)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(description='Preprocess SRTM15 NetCDF data into 1° x 1° tiles and save to a ZIP archive.')
    parser.add_argument('input' , type=str, help='Input NetCDF file name')
    parser.add_argument('output', type=str, help='Output ZIP file name'  )

    args = parser.parse_args()
    print(f'Input file: {args.input}')
    print(f'Output file: {args.output}')
    convert_file(args.input, args.output)

if __name__ == '__main__':
    main()
