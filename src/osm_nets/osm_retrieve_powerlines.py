from pathlib import Path

import geopandas as geopd
import osm
import osm_utils
import pandas as pd
import shapely

datapath = osm.DATA / Path("graphs_powerlines_EU")
datapath.mkdir(parents=True, exist_ok=True)

osm.logconfig.setup_logging("INFO")


def retrieve():
    countries = osm_utils.load_regions(buffer_size=2.0, test=False).sort_values(by="code")
    print(countries)

    for id, region in countries.iterrows():
        osm.logging.info(f"{id}/{len(countries)}, {region['code']}")
        if isinstance(region["geometry"], str):
            osm.logging.info(region["geometry"])
        pl_path = datapath / f"graph_{region['code']}_graph.gpkg"

        if pl_path.is_file():
            osm.log.warning(f"File {pl_path.name} already present.")
            continue
        else:
            pl, pp = osm.osm_powerlines(
                region["geometry"],
                osm_dump_file="?",
                substation_distance=500,
                node_prefix=region["code"],
            )
            pl.write(pl_path)
            pp.data.to_file(pl_path, driver="GPKG", layer="powerplants", mode="w")


def merge() -> None:
    """Do the main."""
    # Load the countries

    graph: osm.Graph | None = None
    powerplants: geopd.GeoDataFrame | None = None

    for pl_path in sorted(datapath.glob("graph_[A-Z]*_graph.gpkg")):
        osm.log.info(pl_path)

        pl = osm.Graph.read(pl_path)

        if len(pl) == 0:
            continue

        if graph is None:
            graph = pl
        else:
            graph.merge(pl)

        pp = geopd.read_file(pl_path, layer="powerplants")

        if powerplants is None:
            powerplants = pp
        else:
            powerplants = geopd.GeoDataFrame(
                pd.concat([powerplants, pp], ignore_index=True), crs=powerplants.crs
            )

    if graph is not None:
        graph.write(datapath / "graph_all_graph.gpkg")
    if powerplants is not None:
        powerplants.to_file(
            datapath / "graph_all_graph.gpkg", driver="GPKG", layer=powerplants, index=True
        )


if __name__ == "__main__":
    # retrieve()
    merge()
    # test_merge()
