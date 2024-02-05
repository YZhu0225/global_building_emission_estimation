# Data Processing Summary

## Geospatial Summary
Generally, we use WGS-84 (EPSG:4326) as our standard global coordinate reference system. All distances/areas are calculated in units of meters. If an equal-area projection is needed, the World_Mollweide (ESRI:54009) projection is used.

## Pre-Processing
### World Water Bodies
From the world water bodies dataset, we first took the original .gdb dataset from v108 and imported it into Python as a pandas dataframe. Some of the multipolygon geometries in the original dataset were malformed and unable to be read into Python as proper Shapely objects. We found that 86 out of the ~2.8M rows in the original dataset had malformed geometries, the largest of which was a .17 square meter body of water in the Arctic Ocean. We chose to omit those 86 bodies of water from v1.0 of the Duke EDAL Building Emissions model. 

## Other adjustments
### Validation
- Because Data Portal for Cities is a POINT location and boundary regions are independently-procured POLYGON locations, when we assign a POINT to a POLYGON it is possible (and, in fact, we have seen) that multiple POINTs can map within the same POLYGON. As we don't know a priori which is the "correct" POINT-POLYGON assignment, we choose to only keep those municipalities as part of the validation process which have a unique 1:1 mapping between POINT and POLYGON. 

## Units
### Duke EDAL Model
- DPFC: Metric tonnes (t CO2e)
    - Source: https://www.cdp.net/en/cities/ghg-emissions-tools-and-datasets-guide-for-cities/data-portal-for-cities 
### GRACED
- Each Cell: kgC/hr
    - Source: https://carbonmonitor-graced.com/datasets.html
- Conversions:
    - kgC/hr to metric tonnes: cell * 24 (24 hrs --> 1 day) * 1000 (kgC to metric tonnes)
    - Uses grid cell of size 0.1 degree x 0.1 degree so add buffer of 0.05 degree from center point to match
### EDGAR
- Conversions:
    - Uses grid cell of size 0.1 degree x 0.1 degree so add buffer of 0.05 degree from center point to match
    - EDGAR gridded data is published in tons so NO UNIT CONVERSION needed (source: https://edgar.jrc.ec.europa.eu/dataset_ghg80)
        - "Notes: Emission gridmaps are expressed in ton substance / 0.1degree x 0.1degree / year for the .txt and .nc files." (source: https://edgar.jrc.ec.europa.eu/dataset_ghg80)