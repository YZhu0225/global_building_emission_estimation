
# Municipality boundaries
- **Mexico and Indonesia**. Non-US boundaries ('Level 2' for Mexico and Indonesia is municipal-level): https://gadm.org/download_country.html
    - These data have multiple levels level-0, which is typically country, level-1, which is typically state or province, levels 2+ appears to vary by country both in terms of availability and their interpretation
    - Mexico (level-2): https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_MEX_2.json
    - Indonesia (level-2, regency, which is the same level as city): https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IDN_2.json.zip
- **U.S.** There is not municipal-level boundaries here for the US, I think 'Level 2' is county-level in US
- Michigan. THIS REQUIRES MANUAL DOWNLOADING from this website https://gis-michigan.opendata.arcgis.com/datasets/Michigan::minor-civil-divisions-cities-townships-/explore?location=44.860989%2C-86.415900%2C6.88. The geojson link for this file is as follows, but this HAS MISSING MUNICIPALITIES: https://gisagocss.state.mi.us/arcgis/rest/services/OpenData/michigan_geographic_framework/MapServer/2/query?outFields=*&where=1%3D1&f=geojson
- Massachusetts: https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/townssurvey_shp.zip
- For other US states (coverage varies by state): https://catalog.data.gov/dataset?collection_package_id=42a97b4c-ea50-45e0-be86-33d0ad326d8e

# EDGAR
## v8.0 - Most Recent Data
https://edgar.jrc.ec.europa.eu/dataset_ghg80
https://edgar.jrc.ec.europa.eu/gallery?release=v80ghg&substance=GWP_100_AR5_GHG&sector=RCO

In v8.0 of EDGAR, they release CO2e using the GWP_100_AR5 values. We use these from 2015 to align with Data Portal for Cities evaluation data. 

As there are several elements of these data that are unclear, we make the following assumptions: 
- These data used WGS-84 as the CRS
- Each lat/lon pair refers to the grid centroid
- These data include biomass
- "Buildings" as a sector includes both residential and non-residential buildings
- EDGAR seemingly includes Agricultural, Forestry, and Fisheries as a category under 1A4 but neither our model nor DPFC have that

## v7.0 - Archival Data (Preliminary results)
https://edgar.jrc.ec.europa.eu/dataset_ghg70
https://edgar.jrc.ec.europa.eu/gallery?release=v70ghg&substance=CO2_excl_short-cycle_org_C&sector=RCO

Specifically, 2015 data to align with Data Portal for Cities evaluation data.

Short cycle CO2 are these sectors: combusting biofuels, agricultural waste burning or field 30 burning (EDGAR v4.3.2 Global Atlas of the three major greenhouse gas emissions for the period 1970–2012)

Therefore, EDGAR's Energy for Buildings CO2_excl_short-cycle_org_C dataset is of greater relevance for The Duke model between the two options.

Nota that for all preliminary results incorporated into the pilot study, these data were used

# GRACED
https://carbonmonitor-graced.com/datasets.html

Specifically: 2019 month-by-month data for the Residential sector (which, per the first GRACED paper, includes both residential and industrial buildings)