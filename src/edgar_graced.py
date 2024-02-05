# from osgeo import gdal, osr
import os, struct
import numpy as np
import subprocess
import geopandas as gpd
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import rioxarray
from scipy.spatial import cKDTree
import xarray as xr


# function to project GRACED data to ESRI:54009: START
def crs_transform(graced, out_crs = "ESRI:54009"):
    """
    Transform GRACED data to ESRI:54009
    Input:
        graced: GRACED dataframe
        out_crs: output crs
    Output:
        graced: GRACED dataframe with geometry in ESRI:54009
    """
    graced = gpd.read_parquet(graced_path)
    graced.set_geometry('geometry', inplace=True)
    graced = graced.to_crs('ESRI:54009')
    # pull out the latitude and longitude from the gemoetry column
    graced['latitude_54009'] = graced.geometry.y
    graced['longitude_54009'] = graced.geometry.x
    # make the latitude_54009 and longitude_54009 columns integers
    graced['latitude_54009'] = graced['latitude_54009'].astype(int)
    graced['longitude_54009'] = graced['longitude_54009'].astype(int)

    return graced


# function to chunk grid and gdf into smaller dataframes: START
def chunk_grid_multiprocess(
    i,chunk_size,k,chunk_gdf_dir):
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
    if k == 9999:
        ghs_gdf_chunk = ghs_gdf.iloc[i:]
    else:
        ghs_gdf_chunk = ghs_gdf.iloc[i : i + chunk_size]
    ghs_gdf_chunk.to_parquet(
        os.path.join(chunk_gdf_dir, f"ghs_gdf_chunk_{k}.parquet"), index = False)

    pass


def child_initialize_grid(_ghs_gdf):
    """
    """
    global ghs_gdf
    ghs_gdf = _ghs_gdf
    pass


def chunk_grid_gdf(
    ghs_gdf,
    n_chunks,
    chunk_gdf_dir,
):
    """
    Multiprocessing chunking grid and gdf into smaller dataframes, save to destination directory
    Input:
    Output:
        None
    """
    print("Start chunking grid and gdf into smaller dataframes")
    # use multiprocessing to speed up
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(
        processes=num_cores,
        initializer=child_initialize_grid,
        initargs=(ghs_gdf,))

    chunk_size = len(ghs_gdf) // n_chunks

    k = 0
    for i in range(0, len(ghs_gdf), chunk_size):
        pool.apply_async(chunk_grid_multiprocess,args=(i,chunk_size,k,chunk_gdf_dir))

        k += 1
    pool.close()
    pool.join()
    
    print("Chunking grid and gdf into smaller dataframes done")
    pass
# function to chunk grid and gdf into smaller dataframes: END


# function to re-allocate emissions to grid cells: START

def nearest_point(chunk_path, chunk_gdf_graced_dir, i):
    # print(f"Processing chunk: {chunk_path}")
    ghs_chunk = pd.read_parquet(chunk_path)
    ghs_chunk['graced_index'] = 0
    # ghs_chunk['graced_polygons'] = None
    ghs_chunk['total_emissions_2019'] = None
    ghs_chunk['lat'] = None
    ghs_chunk['lon'] = None
    for j in range(len(ghs_chunk)):
        given_point = (ghs_chunk.iloc[j].y, ghs_chunk.iloc[j].x)
        npi = kdtree.query(given_point)[1]
        graced_row = graced.iloc[npi]
        ghs_chunk.loc[j, ['graced_index', 'total_emissions_2019', 'lat', 'lon', 'lat_original', 'lon_original']] = \
            npi, graced_row['total_emissions_2019'], graced_row['latitude_54009'], graced_row['longitude_54009'], graced_row['latitude'], graced_row['longitude']
    ghs_chunk.to_parquet(os.path.join(chunk_gdf_graced_dir, f'ghs_graced_chunk_{i}.parquet'), index = False)

        
def child_initialize_tree(_kdtree, _graced):
    """
    """
    global kdtree, graced
    kdtree = _kdtree
    graced = _graced
    pass


def grid_match_graced(graced, chunk_gdf_dir, chunk_gdf_graced_dir):
    """
    Multiprocessing re-allocate emissions to grid cells
    Input:
        graced: graced dataframe
        chunk_gdf_dir: directory to load gdf chunks
        chunk_gdf_graced_dir: directory to save gdf chunks with re-allocated emissions
    Output:
        None
    """
    print("Start re-allocate emissions to grid cells")
    graced_points = list(zip(graced.latitude_54009, graced.longitude_54009))
    kdtree = cKDTree(graced_points)

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(
        processes=num_cores,
        initializer=child_initialize_tree,
        initargs=(kdtree,graced,))

    for i in range(10000):
        chunk_path = os.path.join(chunk_gdf_dir, f'ghs_gdf_chunk_{i}.parquet')
        pool.apply_async(nearest_point,args=(chunk_path, chunk_gdf_graced_dir, i))
    pool.close()
    pool.join()
    print("Re-allocate emissions to grid cells done")
    pass
# function to re-allocate emissions to grid cells: END


# function to concat the re-allocated emission chunk: START
def subprocess_concat(chunk_path, graced_grid_list):
    graced_chunk = pd.read_parquet(chunk_path)
    graced_chunk = gpd.GeoDataFrame(graced_chunk, geometry= gpd.points_from_xy(graced_chunk.x, graced_chunk.y), crs = 'ESRI:54009')
    graced_grid_list.append(graced_chunk)


def generate_data(graced_grid_list):
    for item in graced_grid_list:
        yield item

def concat_gdf_graced_chunk(chunk_gdf_graced_dir, concat_gdf_graced_path):
    """
    Multiprocessing concat the re-allocated emission chunk
    Input:
        chunk_gdf_graced_dir: directory to load gdf chunks with re-allocated emissions
        concat_gdf_graced_path: path to save concatenated gdf with re-allocated GRACE emissions
    Output:
        graced_fine_gdf: concatenated gdf with re-allocated emissions
    """
    print('start to concat')
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(
    processes=num_cores)
    manager = multiprocessing.Manager()
    graced_grid_list = manager.list()
    for i in range(10000):
        chunk_path = os.path.join(chunk_gdf_graced_dir, f'ghs_graced_chunk_{i}.parquet')
        pool.apply_async(subprocess_concat, args = (chunk_path, graced_grid_list))
    
    pool.close()
    pool.join()
    print('processing done, start to concat')
    generator = generate_data(graced_grid_list)
    del graced_grid_list
    graced_fine_gdf = gpd.GeoDataFrame(pd.concat(generator, ignore_index=True, axis= 0))
    print('concat done, save to file')
    # save graced_grid
    graced_fine_gdf.to_parquet(concat_gdf_graced_path)
    print('save concat file done')
    return graced_fine_gdf
# function to concat the re-allocated emission chunk: END



# function to get proportional emissions: START
def get_proportional_emission(graced_fine_gdf, re_allocated_proportinal_graced_path):
    """
    Get proportional emissions
    Input:
        graced_fine_gdf: concatenated gdf with re-allocated emissions
        re_allocated_proportinal_graced_path: path to save re-allocated GRACE emissions with proportional emissions
    Output:
        None
    """
    print('start to get proportional emissions')
    # reanme data colum to area to avoid confusion
    graced_fine_gdf.rename(columns={'data': 'area'}, inplace = True)

    # get building area* graced emission
    graced_fine_gdf['area_emission'] = graced_fine_gdf['area'] * graced_fine_gdf['total_emissions_2019']
    # group by graced index and get sum area correspoding to graced index
    graced_fine_gdf['sum_area'] = graced_fine_gdf.groupby('graced_index')['area'].transform('sum') 
    # get proportional emission: area* total_emissions_2019/sum_area, avoid zero division
    graced_fine_gdf['area_emission_portion'] = np.where(
        graced_fine_gdf['sum_area'] != 0,
        graced_fine_gdf['area_emission'] / graced_fine_gdf['sum_area'],0)
    
    
    # get new proportional emission to equally distribute emission if sum_area == 0
    # add a column "count_non_water_body" to count the number of non-number body in each graced_index
    graced_fine_gdf['count_non_water_body'] =graced_fine_gdf.groupby('graced_index')['non_water_body'].transform('sum')
    # calculate the area_emission_portion_w_0_distribution: if non_water_body == 1, then area_emission_portion * (1/count_non_water_body)
    graced_fine_gdf['area_emission_portion_w_0_distribution'] = np.where(
        (graced_fine_gdf['non_water_body'] == 1 and graced_fine_gdf['sum_area'] == 0),
        graced_fine_gdf['total_emissions_2019'] /graced_fine_gdf['count_non_water_body'], graced_fine_gdf['area_emission_portion'])
    # redistribute the emission to all-water-body grid cells
    # add a colum count_grids to count the number of grids in each graced_index
    graced_fine_gdf['count_grids'] = graced_fine_gdf.groupby('graced_index')['graced_index'].transform('count')
    # redistribute emissions by count_grids where total_emissions_2019 != 0 and sum_area == 0 and count_non_water_body ==0
    graced_fine_gdf['area_emission_portion_w_0_distribution'] =\
        np.where((graced_fine_gdf['total_emissions_2019'] != 0) & (graced_fine_gdf['sum_area'] == 0) & (graced_fine_gdf['count_non_water_body'] == 0),\
                    graced_fine_gdf['total_emissions_2019'] / graced_fine_gdf['count_grids'], graced_fine_gdf['area_emission_portion_w_0_distribution'])
        
    graced_fine_gdf.to_parquet(re_allocated_proportinal_graced_path)
    print('get proportional emissions done')
    return graced_fine_gdf
# function to get proportional emissions: END

# function to transformation to .tif file: START
def to_tif(graced_reallocated, reallocated_tif_path):
    """
    Transformation to .tif file
    Input:
        graced_fine_gdf: concatenated gdf with re-allocated emissions
        re_allocated_proportinal_graced_path: path to save re-allocated GRACE emissions with proportional emissions
    Output:
        None
    """
    print("Start transformation to .tif file")

    graced_reallocated_new = graced_reallocated[
        ["y", "x", "area_emission_portion_w_0_distribution"]
    ].copy()

    sorted_graced = graced_reallocated_new.sort_values(
        by=["y", "x"], ascending=[False, True]
    )

    latitudes = sorted_graced.y.unique()
    longitudes = sorted_graced.x.unique()
    print(latitudes, longitudes)

    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")

    graced_2d = sorted_graced["area_emission_portion_w_0_distribution"].values.reshape(
        lat_grid.shape
    )

    graced_xr = xr.DataArray(
        graced_2d,
        dims=["latitude", "longitude"],
        coords={"latitude": latitudes, "longitude": longitudes},
    )

    graced_xr.rio.write_crs("ESRI:54009", inplace=True)
    graced_xr.rio.to_raster(reallocated_tif_path)
    print("transformation to .tif file done")
    pass

if "__name__" == "__main__":
    area_path = '/shared/data/building-emissions/data/ghsl/building_total_floor_area.tif'
    # edgar_path = '/shared/data/building-emissions/comparisons/EDGAR/2015/edgar_2015_with_polygons.parquet'
    graced_path = '/shared/data/building-emissions/comparisons/GRACED/residential_2019/GRACED_2019_Residential-with_polygons.parquet'
    chunk_gdf_dir = '/shared/data/building-emissions/data/processing_file_edgar_graced/graced_building_total_floor_area/chunk_ghs_gdfs' # dir to save gdf chunks
    chunk_gdf_graced_dir = '/shared/data/building-emissions/data/processing_file_edgar_graced/graced_building_total_floor_area/chunk_ghs_graced' # dir to save gdf chunks with re-allocated emissions
    concat_gdf_graced_path = '/shared/data/building-emissions/data/processing_file_edgar_graced/graced_building_total_floor_area/graced_fine_gdf.parquet' # path to save concatenated gdf with re-allocated emissions
    re_allocated_proportinal_graced_path = '/shared/data/building-emissions/data/processing_file_edgar_graced/graced_building_total_floor_area/reallocated_graced.parquet'
    reallocated_tif_path = '/shared/data/building-emissions/data/processing_file_edgar_graced/graced_building_total_floor_area/reallocated_graced_building_area.tif'
    # area_path = '/shared/data/ghsl_data/ghs_built_s_2015_1km/GHS_BUILT_S_E2015_GLOBE_R2023A_54009_1000_V1_0/GHS_BUILT_S_E2015_GLOBE_R2023A_54009_1000_V1_0.tif'
    # # edgar_path = '/shared/data/building-emissions/comparisons/EDGAR/2015/edgar_2015_with_polygons.parquet'
    # graced_path = '/shared/data/building-emissions/comparisons/GRACED/residential_2019/GRACED_2019_Residential-with_polygons.parquet'
    # building_path = '/shared/data/building-emissions/data/archive/grid_template/area_ghs_total_dataframe.parquet'
    # wkdir = '/shared/data/building-emissions/data/processing_file_edgar_graced/graced_building_area/'
    # chunk_gdf_dir = '/shared/data/building-emissions/data/processing_file_edgar_graced/graced_building_area/chunk_ghs_gdfs' # dir to save gdf chunks
    # chunk_gdf_graced_dir = '/shared/data/building-emissions/data/processing_file_edgar_graced/graced_building_area/chunk_ghs_graced' # dir to save gdf chunks with re-allocated emissions
    # concat_gdf_graced_path = '/shared/data/building-emissions/data/processing_file_edgar_graced/graced_building_area/graced_fine_gdf.parquet' # path to save concatenated gdf with re-allocated emissions
    # re_allocated_proportinal_graced_path = '/shared/data/building-emissions/data/processing_file_edgar_graced/graced_building_area/reallocated_graced.parquet'

    # read file that need to be re-allocated
    graced = gpd.read_parquet(graced_path)
    graced = crs_transform(graced, out_crs = "ESRI:54009")

    # read HSL finer resolution data
    rds = rioxarray.open_rasterio(area_path)
    rds.name = "data"
    ghs = rds.squeeze().to_dataframe().reset_index()
    ghs_new = ghs[['y', 'x', 'data']].copy()
    # replace water body and keep track of water body
    ghs_new['non_water_body'] = np.where(ghs_new['data'] == 4294967295, 0, 1)
    ghs_new['data'] = ghs_new['data'].replace(4294967295, 0) # replace water body 

    # chunk grid and gdf into smaller dataframes
    chunk_grid_gdf(ghs_new,10000,chunk_gdf_dir)
    del ghs_new

    # re-allocate emissions to grid cells
    grid_match_graced(graced, chunk_gdf_dir, chunk_gdf_graced_dir) # 205 mins

    # concat the re-allocated emission chunk
    graced_fine_gdf = concat_gdf_graced_chunk(chunk_gdf_graced_dir, concat_gdf_graced_path) # 30 mins

    # get proportional emissions
    graced_fine_gdf = get_proportional_emission(graced_fine_gdf, re_allocated_proportinal_graced_path) # 3 mins

    # transformation to .tif file
    to_tif(graced_fine_gdf, reallocated_tif_path) # 1 min


    # done