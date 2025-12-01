from pathlib import Path

import geopandas as geopd
import osm
import pandas as pd
import shapely

datapath = osm.DATA / Path("graphs_powerlines_EU")
datapath.mkdir(parents=True, exist_ok=True)

osm.logconfig.setup_logging("INFO")


def retrieve():
    regions = osm.load_regions(buffer_size=2.0)
    _retrieve(regions)


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
        pl = fix_edges(pl, pl_path)
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

    if graph is not None:
        graph.write(datapath / "graph_all_graph.gpkg")
    if powerplants is not None:
        powerplants.to_file(datapath / "graph_all_powerplants.geojson")


def fix_edges(graph: osm.Graph, path: Path) -> osm.Graph:
    if "ITA.14" in path.name:
        new_edges = [("ITA.14_1_152", "ITA.14_1_4")]
        for n1, n2 in new_edges:
            assert n1 in graph.nodes.index
            assert n2 in graph.nodes.index
        graph.edges = geopd.GeoDataFrame(
            pd.concat(
                [
                    graph.edges,
                    geopd.GeoDataFrame(
                        pd.DataFrame(
                            [
                                {
                                    "power": "cable",
                                    "voltage": 500000,
                                    "source": source,
                                    "target": target,
                                    "geometry": shapely.LineString(
                                        [
                                            graph.nodes.loc[source, "geometry"],
                                            graph.nodes.loc[target, "geometry"],
                                        ]
                                    ),
                                }
                                for source, target in new_edges
                            ]
                        ),
                        crs=graph.edges.crs,
                    ),
                ]
            ),
            crs=graph.edges.crs,
        )

    return graph


def test_merge():
    graph: osm.Graph | None = None
    powerplants: geopd.GeoDataFrame | None = None
    for pl_path in [
        "../data/graphs/graph_FRA.5_1_graph.gpkg",
        "../data/graphs/graph_ITA.14_1_graph.gpkg",
        "../data/graphs/graph_BUFFER_0594_graph.gpkg",
    ]:
        pl_path = Path(pl_path)
        pp_path = datapath / pl_path.name.replace("_graph.gpkg", "_powerplants.geojson")
        osm.log.info(pl_path)
        osm.log.info(pp_path)

        pl = osm.Graph.read(pl_path)
        pl = fix_edges(pl, pl_path)
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

    if graph is not None:
        graph.write(Path("/tmp/graph_all_graph.gpkg"))
    if powerplants is not None:
        powerplants.to_file(Path("/tmp/graph_all_powerplants.geojson"))


if __name__ == "__main__":
    # retrieve()
    merge()
    # test_merge()
