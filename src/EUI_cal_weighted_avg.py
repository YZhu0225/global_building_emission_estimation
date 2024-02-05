import numpy as np
import pandas as pd


def get_non_residential_pct(df):
    """
    Calculate the percentage of each non-residential building type.
        Input: df - dataframe of building type percentage
        Output: non_residential_pct - dictionary of building type percentage
    """
    non_residential_pct = {}
    total_non_residential = df.loc[
        (df["Building Type"] != "Home") & (df["Building Type"] != "Other"),
        "Percentages \n(based on US data)",
    ].sum()
    non_residential_pct["Hospital"] = (
        df.loc[
            df["Building Type"] == "Hospital", "Percentages \n(based on US data)"
        ].values[0]
        / total_non_residential
    )
    non_residential_pct["Hotel"] = (
        df.loc[
            df["Building Type"] == "Hotel", "Percentages \n(based on US data)"
        ].values[0]
        / total_non_residential
    )
    non_residential_pct["Office"] = (
        df.loc[
            df["Building Type"] == "Office", "Percentages \n(based on US data)"
        ].values[0]
        / total_non_residential
    )
    non_residential_pct["Retail"] = (
        df.loc[
            df["Building Type"] == "Retail", "Percentages \n(based on US data)"
        ].values[0]
        / total_non_residential
    )
    non_residential_pct["Warehouse"] = (
        df.loc[
            df["Building Type"] == "Warehouse", "Percentages \n(based on US data)"
        ].values[0]
        / total_non_residential
    )
    return non_residential_pct


def cal_wtd_res_nonres_EUI(residential_weights, non_residential_weights, df):
    """
    Calculate the weighted residential and non-residential EUI for each city.
        Input: residential_weights - list of residential building type weights
               non_residential_weights - dictionary of non-residential building type weights
               df - dataframe of EUI
        Output: ddataframe of EUI output with three columns: Geonames ID, Residential EUI, Non-residential EUI
    """
    df_output = pd.DataFrame(
        columns=[
            "City",
            "Geonames ID",
            "Country",
            "Residential EUI",
            "Non-residential EUI",
        ]
    )
    df_2 = df[
        [
            "City",
            "Geonames ID",
            "Country",
            "Primary",
            "Heating + Hot Water EUI",
            "Source",
        ]
    ]
    df_2 = df_2.dropna(subset=["Geonames ID"])

    for idx, geonames_id in enumerate(df_2["Geonames ID"].unique()):
        # store the geonames ID, city, country
        df_output.loc[idx, "City"] = df_2.loc[
            df_2["Geonames ID"] == geonames_id, "City"
        ].values[0]
        df_output.loc[idx, "Geonames ID"] = geonames_id
        df_output.loc[idx, "Country"] = df_2.loc[
            df_2["Geonames ID"] == geonames_id, "Country"
        ].values[0]

        df_sub = df_2.loc[df_2["Geonames ID"] == geonames_id]
        df_sub_residential = df_sub.loc[df_sub["Primary"] == "Homes"]
        df_sub_non_residential = df_sub.loc[df_sub["Primary"] != "Homes"]

        if len(df_sub_residential) == 16:
            print(
                "Geonames ID: {} has duplicate sources. Choose EDGE.".format(
                    geonames_id
                )
            )
            df_sub_residential = df_sub_residential.loc[
                df_sub_residential["Source"] == "EDGE", :
            ]
            df_sub_non_residential = df_sub_non_residential.loc[
                df_sub_non_residential["Source"] == "EDGE", :
            ]

        # store the residential EUI
        if len(df_sub_residential) == 8:
            EUI_residential = 0
            for i in range(8):
                EUI_residential += (
                    df_sub_residential.iloc[i, 4] * residential_weights[i]
                )
            df_output.loc[idx, "Residential EUI"] = EUI_residential

        else:
            print("Geonames ID: {} doesn't have 8 residential EUI".format(geonames_id))

        # store the non-residential EUI
        EUI_non_residential = 0
        for _, row in df_sub_non_residential.iterrows():
            EUI_non_residential += (
                row["Heating + Hot Water EUI"] * non_residential_weights[row["Primary"]]
            )
        df_output.loc[idx, "Non-residential EUI"] = EUI_non_residential

    return df_output


if __name__ == "__main__":
    # read data
    df_EUI = pd.read_excel(
        "CURB Tool Energy Use Intensity Dataset - Duke Labeled.xlsx",
        sheet_name="Sheet1",
    )
    df_building_pct = pd.read_excel(
        "CURB Tool Energy Use Intensity Dataset - Duke Labeled.xlsx",
        sheet_name="Building Use %",
        header=2,
    )
    # get the percentage of each non-residential building type
    non_residential_pct = get_non_residential_pct(df_building_pct)
    # calculate the residential and non-residential EUI for each city
    residential_pct = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]
    df_output = cal_wtd_res_nonres_EUI(residential_pct, non_residential_pct, df_EUI)
    # drop the geonames ID with "?"
    df_output = df_output.loc[df_output["Geonames ID"] != "?", :]
    # save the output
    df_output.to_csv("EUI_output.csv", index=False)
