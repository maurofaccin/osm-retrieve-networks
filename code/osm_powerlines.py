from pathlib import Path

import geopandas as geopd
import osm
import pandas as pd


def main() -> None:
    """Do the main."""
    # Load the countries
    countries = pd.read_csv("../data/countries.csv")
    # Consider only Europe (for now)
    countries = countries[countries.region == "Europe"]

    datapath = Path("../data/graphs")

    graph: osm.Graph | None = None
    powerplants: geopd.GeoDataFrame | None = None

    for id, country in countries.iterrows():
        osm.logging.info(f"{id}, {country['name']}, {country['sub-region']}")
        pp_path = datapath / f"graph_{country['alpha-3']}_powerplants.geojson"

        if pp_path.is_file():
            pl = osm.Graph.load(datapath, "graph_{}_{{}}.geojson".format(country["alpha-3"]))
            pp = geopd.read_file(pp_path)
        else:
            pl, pp = osm.osm_powerlines(country["name"])

        if graph is None:
            graph = osm.Graph(edges=pl.edges.copy(), region=pl.region)
            graph.nodes = pl.nodes.copy()
        else:
            graph = graph.merge(pl)

        if powerplants is None:
            powerplants = pp
        else:
            powerplants = geopd.GeoDataFrame(pd.concat([powerplants, pp]), crs=powerplants.crs)

        graph.write(datapath, "graph_all_{}.geojson")
        powerplants.to_file(datapath / "graph_all_powerplants.geojson")
        pl.write(datapath, "graph_{}_{{}}.geojson".format(country["alpha-3"]))
        pp.to_file(pp_path)


def test():
    edges = geopd.read_file("ciccio.geojson").set_index("index", drop=True)
    print(edges)
    nodes = geopd.read_file("./ciccio_pasticcio.geojson").set_index("id", drop=True)
    print(nodes)

    new_edges = osm.add_fuzzy_nodes(nodes, edges, distance=500)
    print(new_edges)


if __name__ == "__main__":
    main()
    # test()
