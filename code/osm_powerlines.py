from pathlib import Path

import geopandas as geopd
import osm
import pandas as pd
import shapely
import tqdm
import numpy as np

datapath = Path("../data/graphs")
datapath.mkdir(parents=True, exist_ok=True)

osm.logconfig.setup_logging("INFO")


def retrieve_countries() -> None:
    countries = load_countries()
    _retrieve(countries)


def retrieve_buffer() -> None:
    buffer = get_buffer_region(buf_size=2.0)

    bounds = buffer.iloc[0].geometry.bounds
    nx = int(bounds[2] - bounds[0]) // 2 + 2
    ny = int(bounds[3] - bounds[1]) // 2 + 2

    xes = np.linspace(bounds[0], bounds[2], nx)
    yes = np.linspace(bounds[1], bounds[3], ny)

    tiles = geopd.GeoDataFrame(
        [
            {"geometry": shapely.box(x1, y1, x2, y2)}
            for x1, x2 in zip(xes, xes[1:])
            for y1, y2 in zip(yes, yes[1:])
        ]
    )
    tiles["code"] = [f"BUFFER_{i:04d}" for i in tiles.index]
    tiles.geometry = tiles.intersection(buffer.iloc[0].geometry)
    # keep only the overlapping areas
    tiles = tiles[tiles.area > 0]
    _retrieve(tiles)


def _retrieve(data: pd.DataFrame) -> None:
    for id, region in data.iterrows():
        osm.logging.info(f"{id}, {region['code']}")
        if isinstance(region["geometry"], str):
            osm.logging.info(region["geometry"])
        pp_path = datapath / f"graph_{region['code']}_powerplants.geojson"
        pl_path = datapath / f"graph_{region['code']}_graph.gpkg"

        if pp_path.is_file() and pl_path.is_file():
            continue
        else:
            pl, pp = osm.osm_powerlines(
                region["geometry"], substation_distance=500, node_prefix=region["code"] + "_"
            )
            pl.write(pl_path)
            pp.to_file(pp_path)


def merge() -> None:
    """Do the main."""
    # Load the countries

    graph: osm.Graph | None = None
    powerplants: geopd.GeoDataFrame | None = None

    for pl_path in sorted(datapath.glob("graph_[A-Z]*_graph.gpkg")):
        pp_path = datapath / pl_path.name.replace("_graph.gpkg", "_powerplants.geojson")
        osm.log.info(pl_path)
        osm.log.info(pp_path)

        pl = osm.Graph.read(pl_path)
        pp = geopd.read_file(pp_path)

        if len(pl) == 0:
            continue

        if graph is None:
            graph = osm.Graph(edges=pl.edges.copy(), region=pl.region)
            graph.nodes = pl.nodes.copy()
        else:
            graph = graph.merge_edges(pl)

        if powerplants is None:
            powerplants = pp
        else:
            powerplants = geopd.GeoDataFrame(
                pd.concat([powerplants, pp], ignore_index=True), crs=powerplants.crs
            )

        graph.write(datapath / "graph_all_graph.gpkg")
        powerplants.to_file(datapath / "graph_all_powerplants.geojson")


def test_merge():
    graph: osm.Graph | None = None
    powerplants: geopd.GeoDataFrame | None = None
    for pl_path in [
        "../data/graphs/graph_RUS.44_1_graph.gpkg",
        "../data/graphs/graph_RUS.72_1_graph.gpkg",
    ]:
        pl_path = Path(pl_path)
        pp_path = datapath / pl_path.name.replace("_graph.gpkg", "_powerplants.geojson")
        osm.log.info(pl_path)
        osm.log.info(pp_path)

        pl = osm.Graph.read(pl_path)
        print("ESP.12_1_2" in pl.nodes.index)
        pp = geopd.read_file(pp_path)

        if len(pl) == 0:
            continue

        if graph is None:
            graph = osm.Graph(edges=pl.edges.copy(), region=pl.region)
            graph.nodes = pl.nodes.copy()
        else:
            graph = graph.merge_edges(pl)

        if powerplants is None:
            powerplants = pp
        else:
            powerplants = geopd.GeoDataFrame(
                pd.concat([powerplants, pp], ignore_index=True), crs=powerplants.crs
            )

        # graph.write(Path("/tmp/graph_all_graph.gpkg"))
        # powerplants.to_file(Path("/tmp/graph_all_powerplants.geojson"))


def get_buffer_region(buf_size: float = 2.0) -> geopd.GeoDataFrame:
    cache = Path("../data/regions_europe_buffer.geojson")

    if cache.is_file():
        return geopd.read_file(cache)

    countries = load_countries()
    buf = None
    for country in tqdm.tqdm(countries.geometry):
        if buf is None:
            buf = country.buffer(buf_size)
        else:
            buf = buf.union(country.buffer(buf_size), grid_size=0.01)

    if buf is not None:
        buffer = geopd.GeoDataFrame(
            [{"region": "EU"}],
            geometry=[shapely.difference(buf, countries.union_all(), grid_size=0.01)],
            crs=4326,
        )

    else:
        buffer = geopd.GeoDataFrame([], geometry=[])

    # cache it
    buffer.to_file(cache)
    return buffer


def load_countries() -> geopd.GeoDataFrame:
    return geopd.read_file("../data/regions_europe.geojson").rename(columns={"index": "code"})[
        ["code", "geometry"]
    ]


if __name__ == "__main__":
    # retrieve_buffer()
    # retrieve_countries()
    merge()
    # test_merge()
