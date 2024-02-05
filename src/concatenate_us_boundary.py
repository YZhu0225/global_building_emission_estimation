import pandas as pd
import geopandas as gpd
import numpy as np
import os

states = [
    "Arkansas",
    "Mississippi",
    "Nebraska",
    "Nevada",
    "New_Hampshire",
    "New_Jersey",
    "New_Mexico",
    "New_York",
    "North_Carolina",
    "North_Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode_Island",
    "South_Carolina",
    "South_Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West_Virginia",
    "Wisconsin",
    "Wyoming",
]

state_dic = {
    "Arkansas": {
        "STATE_NAME": "Arkansas",
        "NAME": "city_name",
        "GEOMETRY": "geometry",
        "INDEX": "objectid",
    },
    "Mississippi": {
        "STATE_NAME": "Mississippi",
        "NAME": "NAME",
        "GEOMETRY": "geometry",
        "INDEX": "GEOID",
    },
    "Nebraska": {
        "STATE_NAME": "Nebraska",
        "NAME": "NAME",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "Nevada": {
        "STATE_NAME": "Nevada",
        "NAME": "NAME",
        "GEOMETRY": "geometry",
        "INDEX": "GEOID",
    },
    "New_Hampshire": {
        "STATE_NAME": "New_Hampshire",
        "NAME": "pbpNAME",
        "GEOMETRY": "geometry",
        "INDEX": "pbpOBJECTI",
    },
    "New_Jersey": {
        "STATE_NAME": "New_Jersey",
        "NAME": "NAME",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "New_Mexico": {
        "STATE_NAME": "New_Mexico",
        "NAME": "NAME",
        "GEOMETRY": "geometry",
        "INDEX": "Shape_Area",
    },
    "New_York": {
        "STATE_NAME": "New_York",
        "NAME": "COUNTY",
        "GEOMETRY": "geometry",
        "INDEX": "GNIS_ID",
    },
    "North_Carolina": {
        "STATE_NAME": "North_Carolina",
        "NAME": "MunicipalB",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "North_Dakota": {
        "STATE_NAME": "North_Dakota",
        "NAME": "CITY_NAME",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "Ohio": {
        "STATE_NAME": "Ohio",
        "NAME": "NAME",
        "GEOMETRY": "geometry",
        "INDEX": "GEOID",
    },
    "Oklahoma": {
        "STATE_NAME": "Oklahoma",
        "NAME": "CITYNAME",
        "GEOMETRY": "geometry",
        "INDEX": "FID",
    },
    "Oregon": {
        "STATE_NAME": "Oregon",
        "NAME": "instname",
        "GEOMETRY": "geometry",
        "INDEX": "objectid",
    },
    "Pennsylvania": {
        "STATE_NAME": "Pennsylvania",
        "NAME": "MUNICIPAL1",
        "GEOMETRY": "geometry",
        "INDEX": "GEOID",
    },
    "Rhode_Island": {
        "STATE_NAME": "Rhode_Island",
        "NAME": "NAME",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "South_Carolina": {
        "STATE_NAME": "South_Carolina",
        "NAME": "CITY_NAME",
        "GEOMETRY": "geometry",
        "INDEX": "ID",
    },
    "South_Dakota": {
        "STATE_NAME": "South_Dakota",
        "NAME": "name",
        "GEOMETRY": "geometry",
        "INDEX": "geoid",
    },
    "Tennessee": {
        "STATE_NAME": "Tennessee",
        "NAME": "NAME",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "Texas": {
        "STATE_NAME": "Texas",
        "NAME": "CITY_NM",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "Utah": {
        "STATE_NAME": "Utah",
        "NAME": "NAME",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "Vermont": {
        "STATE_NAME": "Vermont",
        "NAME": "TOWNNAME",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "Virginia": {
        "STATE_NAME": "Virginia",
        "NAME": "NAME",
        "GEOMETRY": "geometry",
        "INDEX": "STPLFIPS",
    },
    "Washington": {
        "STATE_NAME": "Washington",
        "NAME": "CityName",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "West_Virginia": {
        "STATE_NAME": "West_Virginia",
        "NAME": "NAMELSAD",
        "GEOMETRY": "geometry",
        "INDEX": "GEOID",
    },
    "Wisconsin": {
        "STATE_NAME": "Wisconsin",
        "NAME": "MCD_NAME",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
    "Wyoming": {
        "STATE_NAME": "Wyoming",
        "NAME": "ENTITYNAME",
        "GEOMETRY": "geometry",
        "INDEX": "OBJECTID",
    },
}


def read_concatenate(path, state_dic, state_name, concatenated_gdf=None):
    """
    Read in the shapefile for all states, and concatenate it to the concatenated_gdf
    input:
        path: the path to the shapefile
        state_dic: the dictionary of state name to the column names
        state_name: the name of the state
        concatenated_gdf: the concatenated geodataframe
    output:
        concatenated_gdf: the concatenated geodataframe
    """
    p = os.path.join(path, state_name)
    dic = state_dic[state_name]
    gdf = gpd.read_file(p)
    # pull out the columns we need, NAME, geometry, INDEX
    gdf = gdf[[dic["INDEX"], dic["NAME"], dic["GEOMETRY"]]]
    # rename the columns
    gdf.columns = ["INDEX", "NAME", "geometry"]
    # add a column of state name to the front of the dataframe
    gdf.insert(0, "STATE_NAME", dic["STATE_NAME"])
    gdf = gdf.to_crs("EPSG:4326")

    # concatenate the dataframe
    if concatenated_gdf is None:
        concatenated_gdf = gdf
    else:
        concatenated_gdf = gpd.GeoDataFrame(
            pd.concat([concatenated_gdf, gdf], ignore_index=True),
            crs=concatenated_gdf.crs,
        )

    return concatenated_gdf


if __name__ == "__main__":
    path = "/Users/yj/Downloads/Energy Data Analytics Lab/US_Boundary"
    for i in range(len(states)):
        if i == 0:
            concatenated_gdf = None
        concatenated_gdf = read_concatenate(
            path, state_dic, states[i], concatenated_gdf
        )

    # save the concatenated geodataframe
    dest_dir = "/Users/yj/Downloads/Energy Data Analytics Lab/US_Boundary"
    concatenated_gdf.to_parquet(os.path.join(dest_dir, "US_Boundary_AtoZ.parquet"))
