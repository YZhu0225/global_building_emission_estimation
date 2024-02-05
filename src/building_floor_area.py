import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show
import rioxarray
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


def tiff_to_df(tiff_path, df_path, col_name):
    """
    Convert tiff file to dataframe and save as parquet file
    Input:
        tiff_path: path to tiff file
        df_path: path to save parquet file
        col_name: name of the column
    Output:
        df_new: dataframe with 3 columns: lat, lon, col_name
    """
    tiff = rioxarray.open_rasterio(tiff_path)
    tiff.name = "data"
    df = tiff.squeeze().to_dataframe().reset_index()
    df_new = df[["y", "x", "data"]]
    df_new.columns = ["lat", "lon", col_name]
    df_new.to_parquet(df_path)
    return df_new


def df_to_tiff(df, tiff_path, col_name):
    """
    Convert dataframe to tiff file and save
    Input:
        df: dataframe with 3 columns: lat, lon, col_name
        tiff_path: path to save tiff file
        col_name: name of the column
    Output:
        tiff file
    """
    sorted_df = df.sort_values(by=['lat', 'lon'], ascending=[False, True])
    
    latitudes = sorted_df["lat"].unique()
    longitudes = sorted_df["lon"].unique()

    # Create 2D arrays of latitudes and longitudes
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")

    # Reshape your building floor area data to match the 2D grid
    df_2d = sorted_df[col_name].values.reshape(lat_grid.shape)

    # Create DataArray
    df_xr = xr.DataArray(
        df_2d,
        dims=["latitude", "longitude"],
        coords={"latitude": latitudes, "longitude": longitudes},
    )

    # Set CRS
    df_xr.rio.write_crs("ESRI:54009", inplace=True)
    
    # save to tiff
    df_xr.rio.to_raster(tiff_path)


def plt_building_footprint_building_floor(df):
    # Set default template to plotly_white
    pio.templates.default = "plotly_white"

    # Create figure
    fig = go.Figure()

    # Add scatter trace
    sample_df = df.sample(n=1000000)  # Adjust the sample size as needed

    # Calculate the coefficients of the linear regression line
    slope, intercept = np.polyfit(
        sample_df["total_area"], sample_df["building_floor_area"], 1
    )

    # Generate x values for your regression line (from min to max total_area)
    x_reg = np.linspace(
        sample_df["total_area"].min(), sample_df["total_area"].max(), 100
    )

    # Calculate the y values based on the slope and intercept
    y_reg = slope * x_reg + intercept

    # Create the scatter plot
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=sample_df["total_area"],
            y=sample_df["building_floor_area"],
            mode="markers",
            marker=dict(size=3),
            name="Data Points",
        )
    )

    # Add the regression line
    fig.add_trace(go.Scatter(x=x_reg, y=y_reg, mode="lines", name="Regression Line"))

    fig.update_layout(
        title="Correlation between total building area and building total floor area",
        xaxis_title="Total building area",
        yaxis_title="Total building floor area",
    )

    # Show the plot
    fig.show()


if __name__ == "__main__":
    # define paths
    building_volume_tiff_path = "/shared/data/building-emissions/data/ghsl/GHS_BUILT_V_E2015_GLOBE_R2023A_54009_1000_V1_0.tif"
    building_volume_df_path = (
        "/shared/data/building-emissions/data/ghsl/building_total_volume.parquet"
    )
    building_floor_area_df_path = "/shared/data/building-emissions/data/ghsl/building_floor_area_with_water_body.parquet"
    building_floor_area_tiff_path = "/shared/data/building-emissions/data/ghsl/building_floor_area_with_water_body.tif"
    building_area_tiff_path = "/shared/data/building-emissions/data/ghsl/GHS_BUILT_S_E2015_GLOBE_R2023A_54009_1000_V1_0.tif"
    building_area_df_path = "/shared/data/building-emissions/data/ghsl/building_footprint_total_area.parquet"

    # start processing
    building_volume_df = tiff_to_df(
        building_volume_tiff_path, building_volume_df_path, "building_volume"
    )

    # generate building floor area dataframe
    # assuming average floor height is 3m
    building_floor_area_df = building_volume_df.copy()
    # if water body
    building_floor_area_df.loc[
        building_floor_area_df["building_volume"] == 4294967295, "building_floor_area"
    ] = 4294967295
    # if not water body
    building_floor_area_df.loc[
        building_floor_area_df["building_volume"] != 4294967295, "building_floor_area"
    ] = (building_floor_area_df["building_volume"] / 3)
    building_floor_area_df.drop(columns=["building_volume"], inplace=True)
    building_floor_area_df.to_parquet(building_floor_area_df_path)

    # save to tiff
    df_to_tiff(
        building_floor_area_df, building_floor_area_tiff_path, "building_floor_area"
    )

    # check correlation
    building_area_df = tiff_to_df(
        building_area_tiff_path, building_area_df_path, "total_area"
    )
    df_merged = building_area_df.merge(
        building_floor_area_df, on=["lat", "lon"], how="left"
    )
    correlation = df_merged["total_area"].corr(df_merged["building_floor_area"])
    print(
        f"Correlation between building footprint area and building floor area is {correlation}"
    )

    # plot
    plt_building_footprint_building_floor(df_merged)
