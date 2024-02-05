import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import pandas as pd
import multiprocessing
import xarray as xr
import rioxarray
import os


###########convert grid to geodataframe#################
def grid_to_gdf(grid_path_total, grid_path_nres, gdf_final_dir):
    """
    Convert grid to geodataframe with res/nres.total area before chunking
    Input:
        grid_path_total: path to total area grid dataframe
        grid_path_nres: path to non-residential area grid dataframe
        gdf_final_dir: directory to save the final gdf
    Output:
        grid_total: total area grid dataframe
        gdf_final: final gdf with res/nres/total area
    """
    grid_total = pd.read_parquet(grid_path_total)
    gdf_total = grid_total.transpose().unstack().reset_index()
    gdf_total.columns = ["lat", "lon", "area_total"]

    grid_nres = pd.read_parquet(grid_path_nres)
    gdf_nres = grid_nres.transpose().unstack().reset_index()
    gdf_nres.columns = ["lat", "lon", "area_nres"]
    # merge gdf_total and gdf_nres
    gdf_total["area_nres"] = gdf_nres["area_nres"]
    gdf_total["area_res"] = gdf_total["area_total"] - gdf_total["area_nres"]
    gdf_total.to_parquet(
        os.path.join(gdf_final_dir, "gdf_before_chunk.parquet"), index=False
    )
    gdf_final = gdf_total.copy()

    return grid_total, gdf_final


#############Merge EUI and EF dataframes#################
def merge_eui_ef(eui_path, ef_path, dest_dir):
    """
    Merge eui and ef dataframes, save to destination directory
    Input:
        eui_path: path to eui dataframe
        ef_path: path to ef dataframe
        dest_dir: directory to save the merged dataframe
    Output:
        merged_eui_ef_gdf: merged geodataframe with eui and ef
    """
    eui = gpd.read_parquet(eui_path)
    eui.to_crs("ESRI:54009", inplace=True)
    # convert kwh to mj
    eui.loc[:, "Non-residential EUI (mj)"] = (
        eui.loc[:, "Non-residential EUI (kwh)"] * 3.6
    )
    eui.loc[:, "Residential EUI (mj)"] = eui.loc[:, "Residential EUI (kwh)"] * 3.6
    ef = gpd.read_parquet(ef_path)
    ef.to_crs("ESRI:54009", inplace=True)
    merged_eui_ef = eui.merge(ef, how="left", on="gid0", suffixes=("_eui", "_ef"))
    merged_eui_ef_gdf = gpd.GeoDataFrame(
        merged_eui_ef, geometry="geometry_eui", crs="ESRI:54009"
    )
    merged_eui_ef_gdf.drop(columns=["geometry_ef"], inplace=True)
    merged_eui_ef_gdf.to_parquet(os.path.join(dest_dir, "merged_eui_ef_gdf.parquet"))

    return merged_eui_ef_gdf


################## START: multiprocessing of chunking grid and gdf ##################
def chunk_grid_multiprocess(
    i,
    chunk_size,
    k,
    dest_dir_chunked_grid,
    dest_dir_chunked_gdf,
    dest_dir_chunked_full_gdf,
):
    """
    Child function to chunk grid and gdf into smaller dataframes, save to destination directory
    Input:
        i: starting row number
        chunk_size: number of rows in each chunk
        k: chunk number
        dest_dir_chunked_grid: directory to save grid chunks
        dest_dir_chunked_gdf: directory to save gdf chunks
        dest_dir_chunked_full_gdf: directory to save full gdf chunks
    Output:
        None
    """
    grid_chunk = grid_total.iloc[i : i + chunk_size]
    grid_chunk.to_parquet(
        os.path.join(dest_dir_chunked_grid, f"grid_chunk_{k}.parquet"), index=True
    )
    n_columns = grid_chunk.columns.size
    gdf_chunk = gdf_final.iloc[i * n_columns : (i + chunk_size) * n_columns]
    # keep only lat and lon columns
    full_gdf_chunk = gdf_chunk[["lat", "lon"]]
    full_gdf_chunk.to_parquet(
        os.path.join(dest_dir_chunked_full_gdf, f"gdf_full_chunk_{k}.parquet"),
        index=False,
    )
    gdf_chunk_subset = gdf_chunk[
        (gdf_chunk["area_total"] > 0) & (gdf_chunk["area_total"] < 4294967295)
    ]
    gdf_chunk_subset.to_parquet(
        os.path.join(dest_dir_chunked_gdf, f"gdf_chunk_{k}.parquet"), index=False
    )

    pass


def child_initialize_grid(_grid_total, _gdf_final):
    """
    Help function to make grid_total and gdf_final global variables
    Input:
        _grid_total: grid dataframe
        _gdf_final: gdf with total, nres and res area
    """
    global grid_total, gdf_final
    grid_total = _grid_total
    gdf_final = _gdf_final
    pass


def chunk_grid_gdf(
    grid_total,
    gdf_final,
    n_chunks,
    dest_dir_chunked_grid,
    dest_dir_chunked_gdf,
    dest_dir_chunked_full_gdf,
):
    """
    Multiprocessing chunking grid and gdf into smaller dataframes, save to destination directory
    Input:
        grid_total: grid dataframe
        gdf_final: gdf with total, nres and res area
        n_chunks: number of chunks
        dest_dir_chunked_grid: directory to save grid chunks
        dest_dir_chunked_gdf: directory to save gdf chunks
        dest_dir_chunked_full_gdf: directory to save full gdf chunks
    Output:
        None
    """

    # use multiprocessing to speed up
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(
        processes=num_cores,
        initializer=child_initialize_grid,
        initargs=(grid_total, gdf_final),
    )
    chunk_size = len(grid_total) // n_chunks

    k = 0
    for i in range(0, len(grid_total), chunk_size):
        pool.apply_async(
            chunk_grid_multiprocess,
            args=(
                i,
                chunk_size,
                k,
                dest_dir_chunked_grid,
                dest_dir_chunked_gdf,
                dest_dir_chunked_full_gdf,
            ),
        )
        k += 1
    pool.close()
    pool.join()

    pass


################## END: multiprocessing of chunking grid and gdf ##################


################## START: multiprocessing of calculate emissions ##################
def calculate_chunk(finest_gdf_chunk_eui_ef):
    """
    Calculate emissions for each grid chunk
    Input:
        finest_gdf_chunk_eui_ef: finest gdf chunk with eui and ef
    Output:
        calculated_chunk: calculated chunk with emissions
    """
    finest_gdf_chunk_eui_ef["nres_eui*area"] = (
        finest_gdf_chunk_eui_ef["area_nres"]
        * finest_gdf_chunk_eui_ef["Non-residential EUI (mj)"]
    )
    finest_gdf_chunk_eui_ef["res_eui*area"] = (
        finest_gdf_chunk_eui_ef["area_res"]
        * finest_gdf_chunk_eui_ef["Residential EUI (mj)"]
    )
    finest_gdf_chunk_eui_ef["nres_co2_emission"] = (
        finest_gdf_chunk_eui_ef["nres_eui*area"]
        * finest_gdf_chunk_eui_ef["Commercial_CO2"]
    )
    finest_gdf_chunk_eui_ef["nres_methane_emission"] = (
        finest_gdf_chunk_eui_ef["nres_eui*area"]
        * finest_gdf_chunk_eui_ef["Commercial_Methane"]
    )
    finest_gdf_chunk_eui_ef["nres_n20_emission"] = (
        finest_gdf_chunk_eui_ef["nres_eui*area"]
        * finest_gdf_chunk_eui_ef["Commercial_N2O"]
    )
    finest_gdf_chunk_eui_ef["res_co2_emission"] = (
        finest_gdf_chunk_eui_ef["res_eui*area"]
        * finest_gdf_chunk_eui_ef["Residential_CO2"]
    )
    finest_gdf_chunk_eui_ef["res_methane_emission"] = (
        finest_gdf_chunk_eui_ef["res_eui*area"]
        * finest_gdf_chunk_eui_ef["Residential_Methane"]
    )
    finest_gdf_chunk_eui_ef["res_n20_emission"] = (
        finest_gdf_chunk_eui_ef["res_eui*area"]
        * finest_gdf_chunk_eui_ef["Residential_N2O"]
    )
    return finest_gdf_chunk_eui_ef


def finest_eui(merged_gdf_chunk_eui_ef):
    """
    Find the finest-level of fidelity for each grid cell
    Input:
        merged_gdf_chunk_eui_ef: merged gdf chunk with eui and ef
    Output:
        finest_gdf_chunk_eui_ef: finest gdf chunk with eui and ef
    """
    merged_gdf_chunk_eui_ef.reset_index(inplace=True)
    finest_gdf_chunk_eui_ef = (
        merged_gdf_chunk_eui_ef.sort_values("level", ascending=False)
        .drop_duplicates("index")
        .sort_index()
        .reset_index(drop=True)
    )

    return finest_gdf_chunk_eui_ef


def layover_eui_ef(gdf_subchunk, results):
    """
    Match grid chunks to eui_ef datframe and do calculation to get emission for each grid chunk
    Input:
        gdf_subchunk: grid chunk
        results: list manager to store the calculated chunks
    Output:
        results: list manager to store the calculated chunks
    """
    merged_gdf_chunk_eui_ef = gpd.sjoin(
        gdf_subchunk, merged_eui_ef, how="left", predicate="intersects"
    )
    # find finest level of fidelity
    finest_gdf_chunk_eui_ef = finest_eui(merged_gdf_chunk_eui_ef)
    # drop geometry column to save memory
    finest_gdf_chunk_eui_ef.drop(columns=["geometry"], inplace=True)

    # calculation * 2(res/nres)* 4 columns(eui*area/co2/methane/n20)
    calculated_chunk = calculate_chunk(finest_gdf_chunk_eui_ef)
    # append to results
    results.append(calculated_chunk)
    return results
    # pass


def child_initialize(_merged_eui_ef):
    """
    Help function to make merged_eui_ef global variable
    Input:
        _merged_eui_ef: merged eui and ef dataframe
    """
    global merged_eui_ef
    merged_eui_ef = _merged_eui_ef
    pass


def chunk_data(data, n_chunks):
    """
    Function to chunk data for multiprocessing
    Input:
        data: data to be chunked
        n_chunks: number of chunks
    Output:
        chunked data
    """
    chunk_size = len(data) // n_chunks + 1
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def iterate_chunks(gdf_chunk, merged_eui_ef, n_chunks=60):
    """
    Match gdf chunks to eui and ef
    Input:
        gdf_chunk: grid chunk
        merged_eui_ef: merged eui and ef dataframe
        n_chunks: number of chunks
    Output:
        calculated_concated_chunks: calculated chunk with emissions
    """

    gdf_chunk_chunks = list(chunk_data(gdf_chunk, n_chunks))
    num_processes = multiprocessing.cpu_count()  # Use the number of available CPU cores
    pool = multiprocessing.Pool(
        processes=num_processes, initializer=child_initialize, initargs=(merged_eui_ef,)
    )
    manager = multiprocessing.Manager()
    results = manager.list()
    for gdf_subchunks in gdf_chunk_chunks:
        pool.apply_async(layover_eui_ef, args=(gdf_subchunks, results))
    pool.close()
    pool.join()

    if len(results) == 0:
        return None
    calculated_concated_chunks = pd.concat(results)

    # sort the concat result by latitude then longitude
    # calculated_concated_chunks = calculated_concated_chunks.sort_values(['lat', 'lon']).reset_index(drop = True)
    return calculated_concated_chunks


def cal_by_gdf_chunks(
    gdf_chunks_dir,
    merged_eui_ef,
    calculated_gdf_chunk_dir,
    n_chunks_within_chunk=60,
    range_min=0,
    range_max=300,
):
    """
    Calculate emissions based on grid area, eui and ef values
    Input:
        gdf_chunks_dir: directory of gdf chunks
        merged_eui_ef: merged eui and ef dataframe
        calculated_gdf_chunk_dir: directory to save calculated gdf chunks
        n_chunks_within_chunk: number of chunks within each chunk
        range_min: starting chunk number
        range_max: ending chunk number
    Output:
        None
    """
    for i in range(range_min, range_max):
        if i % 30 == 0:
            print(i)
        gdf_chunk = pd.read_parquet(f"{gdf_chunks_dir}/gdf_chunk_{i}.parquet")
        if len(gdf_chunk) == 0:
            continue
        gdf_chunk = gpd.GeoDataFrame(
            gdf_chunk,
            geometry=gpd.points_from_xy(gdf_chunk.lon, gdf_chunk.lat),
            crs="ESRI:54009",
        )
        calculated_chunk = iterate_chunks(
            gdf_chunk, merged_eui_ef, n_chunks_within_chunk
        )
        if calculated_chunk is not None:
            print(i)
            calculated_chunk.to_parquet(
                f"{calculated_gdf_chunk_dir}/calculated_gdf_chunk_{i}.parquet"
            )


################## END: multiprocessing of calculate emissions ##################


################## START: multiprocessing of map gdf to grid ##################
def process_chunk_to_grid(
    i,
    calculated_gdf_chunk_dir,
    dest_dir_chunked_full_gdf,
    calculated_grid_chunk_dir,
    calculated_full_gdf_dir,
):
    """
    Child process for multiprocessing to map gdf to grid
    Input:
        i: chunk number
        calculated_gdf_chunk_dir: directory of calculated gdf chunks
        dest_dir_chunked_full_gdf: directory of full gdf chunks
        calculated_grid_chunk_dir: directory to save calculated grid chunks
        calculated_full_gdf_dir: directory to save calculated full gdf
    """
    cal_gdf_chunk_path = f"{calculated_gdf_chunk_dir}/calculated_gdf_chunk_{i}.parquet"
    full_gdf_chunk_path = f"{dest_dir_chunked_full_gdf}/gdf_full_chunk_{i}.parquet"
    full_gdf = pd.read_parquet(full_gdf_chunk_path)
    # create column name list
    names = [
        "res_eui*area",
        "res_co2_emission",
        "res_methane_emission",
        "res_n2o_emission",
        "nres_eui*area",
        "nres_co2_emission",
        "nres_methane_emission",
        "nres_n2o_emission",
    ]
    # add these column to the existing fullgdf with 0 value
    for name in names:
        full_gdf[name] = 0

    if os.path.exists(cal_gdf_chunk_path):
        cal_gdf_chunk = pd.read_parquet(cal_gdf_chunk_path)
        # merge calculated chunk to full gdf
        for j in range(len(cal_gdf_chunk)):
            lat, lon = cal_gdf_chunk.iloc[j]["lat"], cal_gdf_chunk.iloc[j]["lon"]
            all_values = cal_gdf_chunk.iloc[j][
                [
                    "res_eui*area",
                    "res_co2_emission",
                    "res_methane_emission",
                    "res_n20_emission",
                    "nres_eui*area",
                    "nres_co2_emission",
                    "nres_methane_emission",
                    "nres_n20_emission",
                ]
            ].values.astype(np.float64)
            all_values = np.round(all_values, 2)
            # map value back to full_gdf
            full_gdf.loc[
                (full_gdf["lat"] == lat) & (full_gdf["lon"] == lon), names
            ] = all_values

        del cal_gdf_chunk

    # save the full gdf chunk
    full_gdf.to_parquet(
        f"{calculated_full_gdf_dir}/calculated_full_gdf_chunk_{i}.parquet"
    )
    # transform to grid format
    grid_res_eui_area = full_gdf.pivot(
        index="lat", columns="lon", values="res_eui*area"
    )
    grid_res_eui_area.to_parquet(
        f"{calculated_grid_chunk_dir}/res_eui_area/grid_res_eui_area_{i}.parquet"
    )
    del grid_res_eui_area
    grid_nres_eui_area = full_gdf.pivot(
        index="lat", columns="lon", values="nres_eui*area"
    )
    grid_nres_eui_area.to_parquet(
        f"{calculated_grid_chunk_dir}/nres_eui_area/grid_nres_eui_area_{i}.parquet"
    )
    del grid_nres_eui_area
    grid_res_co2 = full_gdf.pivot(index="lat", columns="lon", values="res_co2_emission")
    grid_res_co2.to_parquet(
        f"{calculated_grid_chunk_dir}/res_co2/grid_res_co2_{i}.parquet"
    )
    del grid_res_co2
    grid_nres_co2 = full_gdf.pivot(
        index="lat", columns="lon", values="nres_co2_emission"
    )
    grid_nres_co2.to_parquet(
        f"{calculated_grid_chunk_dir}/nres_co2/grid_nres_co2_{i}.parquet"
    )
    del grid_nres_co2
    grid_res_methane = full_gdf.pivot(
        index="lat", columns="lon", values="res_methane_emission"
    )
    grid_res_methane.to_parquet(
        f"{calculated_grid_chunk_dir}/res_methane/grid_res_methane_{i}.parquet"
    )
    del grid_res_methane
    grid_nres_methane = full_gdf.pivot(
        index="lat", columns="lon", values="nres_methane_emission"
    )
    grid_nres_methane.to_parquet(
        f"{calculated_grid_chunk_dir}/nres_methane/grid_nres_methane_{i}.parquet"
    )
    del grid_nres_methane
    grid_res_n20 = full_gdf.pivot(index="lat", columns="lon", values="res_n2o_emission")
    grid_res_n20.to_parquet(
        f"{calculated_grid_chunk_dir}/res_n2o/grid_res_n2o_{i}.parquet"
    )
    del grid_res_n20
    grid_nres_n20 = full_gdf.pivot(
        index="lat", columns="lon", values="nres_n2o_emission"
    )
    grid_nres_n20.to_parquet(
        f"{calculated_grid_chunk_dir}/nres_n2o/grid_nres_n2o_{i}.parquet"
    )
    del grid_nres_n20
    open(calculated_grid_chunk_dir + f"{i}.txt", "w").close()
    pass


def map_gdf_grid(
    calculated_gdf_chunk_dir,
    dest_dir_chunked_full_gdf,
    calculated_grid_chunk_dir,
    calculated_full_gdf_dir,
    range_min=0,
    range_max=300,
):
    """
    Multiprocessing mapping gdf to grid
    Input:
        calculated_gdf_chunk_dir: directory of calculated gdf chunks
        grid_chunk_dir: directory of grid chunks
        calculated_grid_chunk_dir: directory to save calculated grid chunks
        range_min: starting chunk number
        range_max: ending chunk number
    """
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    for i in range(range_min, range_max):
        pool.apply_async(
            process_chunk_to_grid,
            args=(
                i,
                calculated_gdf_chunk_dir,
                dest_dir_chunked_full_gdf,
                calculated_grid_chunk_dir,
                calculated_full_gdf_dir,
            ),
        )
    pool.close()
    pool.join()

    pass


################## END: multiprocessing of map gdf to grid ##################


################## START: multiprocessing of grid to tiff ##################
def process_grid2tiff(
    i, calculated_grid_chunk_dir, sub_dir, grid_name_header, grid_list, check_list
):
    """
    Concat grid lists
    Input:
        i: chunk number
        calculated_grid_chunk_dir: directory of calculated grid chunks(of different emissions)
        sub_dir: sub directory of calculated grid chunks
        grid_name_header: file name header of the final grid(in format of 'grid_res_co2_'
        grid_list: list manager to store the grid chunks
        check_list: list manager to store the chunk numbers
    """
    grid_path = os.path.join(
        calculated_grid_chunk_dir, sub_dir, f"{grid_name_header}{i}.parquet"
    )
    grid_chunk = pd.read_parquet(grid_path)
    grid_list.append(grid_chunk)


def gdf2tiff(
    calculated_grid_chunk_dir, sub_dir, grid_name_header, geotiff_dir, tiff_file_name
):
    """Concat grid chunk gdf to final grid and save as .tiff
    Input:
        calculated_grid_chunk_dir: directory of calculated grid chunks(of different emissions
        sub_dir: sub directory of calculated grid chunks
        grid_name_header: file name header of the final grid(in format of 'grid_res_co2_'
        geotiff_dir: directory to save the final grid tiff
        tiff_file_name: file name of the final grid tiff
    Output:
        grid: final grid
    """

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    manager = multiprocessing.Manager()
    grid_list = manager.list()
    check_list = manager.list()
    for i in range(300):
        pool.apply_async(
            process_grid2tiff,
            args=(
                i,
                calculated_grid_chunk_dir,
                sub_dir,
                grid_name_header,
                grid_list,
                check_list,
            ),
        )
    pool.close()
    pool.join()

    grid = pd.concat(grid_list, axis=0)
    grid.sort_index(inplace=True, ascending=False)
    grid.fillna(0, inplace=True)
    # convert for emission fator by * 10^(-9)
    grid = grid * (10 ** (-9))
    grid = grid.astype("float64")

    grid.to_parquet(
        os.path.join(geotiff_dir, f"grid_tiff/{tiff_file_name}.parquet"), index=True
    )

    tiff_path = os.path.join(geotiff_dir, f"{tiff_file_name}.tif")
    xarray_data = xr.DataArray(
        grid.values, coords=[grid.index, grid.columns], dims=["latitude", "longitude"]
    )
    xarray_data.rio.to_raster(tiff_path)

    return grid


################## END: multiprocessing of grid to tiff ##################


def other_grid_to_tiff(grid, geotiff_dir, tiff_file_name):
    """
    Generate other tiff files using basic grids
    Input:
        grid: grid dataframe
        geotiff_dir: directory to save the final grid tiff
        tiff_file_name: file name of the final grid tiff
    """
    grid.to_parquet(
        os.path.join(geotiff_dir, f"grid_tiff/{tiff_file_name}.parquet"), index=True
    )
    tiff_path = os.path.join(geotiff_dir, f"{tiff_file_name}.tif")
    xarray_data = xr.DataArray(
        grid.values, coords=[grid.index, grid.columns], dims=["latitude", "longitude"]
    )
    xarray_data.rio.to_raster(tiff_path)
    pass


def run():
    # set paths
    grid_path_total = "/shared/data/building-emissions/data/grid_template/area_ghs_total_dataframe.parquet"
    grid_path_nres = "/shared/data/building-emissions/data/grid_template/area_ghs_nres_dataframe.parquet"
    gdf_final_dir = "/shared/data/building-emissions/data/processing_files/"

    eui_path = "/shared/data/building-emissions/data/energyuseintensities/consolidated_eui_with_boundaries.parquet"
    ef_path = "/shared/data/building-emissions/data/emissionsfactors/global_emissionsfactors_nobiofuel.parquet"
    merged_eui_ef_dir = "/shared/data/building-emissions/data/processing_files/"

    dest_dir_chunked_gdf = (
        "/shared/data/building-emissions/data/processing_files/gdf_chunks"
    )
    dest_dir_chunked_full_gdf = "/shared/data/building-emissions/data/processing_files/gdf_chunks/full_gdf_chunks"
    dest_dir_chunked_grid = (
        "/shared/data/building-emissions/data/processing_files/grid_chunks"
    )

    gdf_chunks_dir = "/shared/data/building-emissions/data/processing_files/gdf_chunks"
    calculated_gdf_chunk_dir = (
        "/shared/data/building-emissions/data/processing_files/calculated_gdf_chunks"
    )
    calculated_grid_chunk_dir = (
        "/shared/data/building-emissions/data/processing_files/calculated_grid_chunks"
    )
    calculated_full_gdf_dir = (
        "/shared/data/building-emissions/data/processing_files/calculated_full_gdf"
    )

    geotiff_dir = "/shared/data/building-emissions/data/geotiff_float64"

    # generate grid and gdf chunks
    grid_total, gdf_final = grid_to_gdf(grid_path_total, grid_path_nres, gdf_final_dir)
    # merge eui and ef dataframes
    merged_eui_ef_gdf = merge_eui_ef(eui_path, ef_path, merged_eui_ef_dir)
    # chunk grid and gdf
    chunk_grid_gdf(
        grid_total,
        gdf_final,
        300,
        dest_dir_chunked_grid,
        dest_dir_chunked_gdf,
        dest_dir_chunked_full_gdf,
    )
    # calculate emissions
    cal_by_gdf_chunks(
        gdf_chunks_dir,
        merged_eui_ef_gdf,
        calculated_gdf_chunk_dir,
        n_chunks_within_chunk=60,
        range_min=0,
        range_max=300,
    )
    # map gdf to grid
    map_gdf_grid(
        calculated_gdf_chunk_dir,
        dest_dir_chunked_full_gdf,
        calculated_grid_chunk_dir,
        calculated_full_gdf_dir,
        range_min=0,
        range_max=300,
    )
    # map grid to tiff
    res_co2_grid = gdf2tiff(
        calculated_grid_chunk_dir, "res_co2", "grid_res_co2_", geotiff_dir, "res_co2"
    )
    nres_co2_grid = gdf2tiff(
        calculated_grid_chunk_dir, "nres_co2", "grid_nres_co2_", geotiff_dir, "nres_co2"
    )
    res_methane_grid = gdf2tiff(
        calculated_grid_chunk_dir,
        "res_methane",
        "grid_res_methane_",
        geotiff_dir,
        "res_methane",
    )
    nres_methane_grid = gdf2tiff(
        calculated_grid_chunk_dir,
        "nres_methane",
        "grid_nres_methane_",
        geotiff_dir,
        "nres_methane",
    )
    res_n2o_grid = gdf2tiff(
        calculated_grid_chunk_dir, "res_n2o", "grid_res_n2o_", geotiff_dir, "res_n2o"
    )
    nres_n2o_grid = gdf2tiff(
        calculated_grid_chunk_dir, "nres_n2o", "grid_nres_n2o_", geotiff_dir, "nres_n2o"
    )
    res_eui_area_grid = gdf2tiff(
        calculated_grid_chunk_dir,
        "res_eui_area",
        "grid_res_eui_area_",
        geotiff_dir,
        "res_eui_area",
    )
    nres_eui_area_grid = gdf2tiff(
        calculated_grid_chunk_dir,
        "nres_eui_area",
        "grid_nres_eui_area_",
        geotiff_dir,
        "nres_eui_area",
    )

    # generate other total grids and co2eq grids
    total_co2_grid = res_co2_grid + nres_co2_grid
    total_methane_grid = res_methane_grid + nres_methane_grid
    total_n2o_grid = res_n2o_grid + nres_n2o_grid
    toal_eui_area_grid = res_eui_area_grid + nres_eui_area_grid
    res_co2eq_20yr_grid = res_co2_grid + 82.5 * res_methane_grid + 273 * res_n2o_grid
    nres_co2eq_20yr_grid = (
        nres_co2_grid + 82.5 * nres_methane_grid + 273 * nres_n2o_grid
    )
    total_co2eq_20yr_grid = res_co2eq_20yr_grid + nres_co2eq_20yr_grid
    res_co2eq_100yr_grid = res_co2_grid + 29.8 * res_methane_grid + 273 * res_n2o_grid
    nres_co2eq_100yr_grid = (
        nres_co2_grid + 29.8 * nres_methane_grid + 273 * nres_n2o_grid
    )
    total_co2eq_100yr_grid = res_co2eq_100yr_grid + nres_co2eq_100yr_grid

    # save to tiff
    other_grid_to_tiff(total_co2_grid, geotiff_dir, "total_co2")
    other_grid_to_tiff(total_methane_grid, geotiff_dir, "total_methane")
    other_grid_to_tiff(total_n2o_grid, geotiff_dir, "total_n2o")
    other_grid_to_tiff(toal_eui_area_grid, geotiff_dir, "total_eui_area")
    other_grid_to_tiff(total_co2eq_20yr_grid, geotiff_dir, "total_co2eq_20yr")
    other_grid_to_tiff(total_co2eq_100yr_grid, geotiff_dir, "total_co2eq_100yr")
    other_grid_to_tiff(res_co2eq_20yr_grid, geotiff_dir, "res_co2eq_20yr")
    other_grid_to_tiff(res_co2eq_100yr_grid, geotiff_dir, "res_co2eq_100yr")
    other_grid_to_tiff(nres_co2eq_20yr_grid, geotiff_dir, "nres_co2eq_20yr")
    other_grid_to_tiff(nres_co2eq_100yr_grid, geotiff_dir, "nres_co2eq_100yr")


if __name__ == "__main__":
    run()
