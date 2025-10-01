from pathlib import Path

import geopandas as geopd
import osm
import pandas as pd


def main() -> None:
    """Do the main."""
    # Load the countries
    countries = geopd.read_file("../data/regions_europe.geojson").rename(
        columns={"index": "code", "geometry": "region"}
    )[["code", "region"]]
    datapath = Path("../data/graphs")

    graph: osm.Graph | None = None
    powerplants: geopd.GeoDataFrame | None = None

    for id, country in countries.iterrows():
        osm.logging.info(f"{id}, {country['code']}")
        if isinstance(country["region"], str):
            osm.logging.info(country["region"])
        pp_path = datapath / f"graph_{country['code']}_powerplants.geojson"
        pl_path = datapath / f"graph_{country['code']}_graph.gpkg"

        if pp_path.is_file() and pl_path.is_file():
            pl = osm.Graph.read(pl_path)
            pp = geopd.read_file(pp_path)
        else:
            pl, pp = osm.osm_powerlines(
                country["region"], substation_distance=500, node_prefix=country["code"] + "_"
            )
            pl.write(pl_path)

        if graph is None:
            graph = osm.Graph(edges=pl.edges.copy(), region=pl.region)
            graph.nodes = pl.nodes.copy()
        else:
            graph = graph.merge(pl)

        if powerplants is None:
            powerplants = pp
        else:
            powerplants = geopd.GeoDataFrame(pd.concat([powerplants, pp]), crs=powerplants.crs)

        graph.write(datapath / "graph_all_graph.gpkg")
        powerplants.to_file(datapath / "graph_all_powerplants.geojson")
        pp.to_file(pp_path)


if __name__ == "__main__":
    main()
    # print(load_regions())
