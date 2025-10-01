"""Load the regions of interest and fi too large split them in smaller pieces."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

datapath = Path("../data/")


def main() -> None:
    """Do the main."""

    adms = gpd.read_file(datapath / "gadm_410.gpkg").set_index("UID", drop=True)
    adms["UID"] = adms.index
    adms = adms[adms["CONTINENT"] == "Europe"]
    adms = adms[
        [c for c in adms.columns if c[:3] == "GID" or c[:4] == "NAME" or c in {"geometry", "UID"}]
    ]
    print(adms)
    # Fix missing data
    adms.loc[328778, "GID_1"] = "UKR.11_1"
    adms.loc[328778, "GID_2"] = "UKR.11.1_1"
    adms.loc[330111, "GID_1"] = "GBR.1_1"
    adms.loc[330122, "GID_1"] = "GBR.1_1"
    adms.loc[330128, "GID_1"] = "GBR.1_1"
    adms.loc[338143, "GID_1"] = "GBR.3_1"
    right_adms = []

    col = "GID_0"
    for level in range(0, 5):
        # GID 0 to 4
        col = f"GID_{level}"
        print(col)

        # col length
        length = adms[col].sum()
        if length == 0 or len(length) == 0:
            print("WARN: no adm level", col)
            col = f"GID_{level - 1}"
            break
        adm = adms.dissolve(by=col)
        adm = adm[adm.area <= 15.0]
        adm[col] = adm.index
        print(set(adm["GID_0"].to_list()))
        adms = adms[~(adms[col].isin(adm.index))]
        right_adms.append(adm)

    # the remaining
    print(adms)
    adms.index = pd.Index(
        [" ".join([row[f"GID_{i}"] for i in range(5)]).split()[-1] for id, row in adms.iterrows()]
    )
    right_adms.append(adms)
    right = pd.concat(right_adms)
    right.index = pd.Index([str(i) for i in right.index])

    right = gpd.GeoDataFrame(right.sort_index(), crs=4326)
    right.geometry = right.simplify_coverage(tolerance=0.01)
    right.to_file("../data/regions_europe.geojson")


if __name__ == "__main__":
    main()
