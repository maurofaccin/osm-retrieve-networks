from pathlib import Path

import osm
import osm_utils

datapath = osm.DATA / Path("graphs_railways_EU")
datapath.mkdir(parents=True, exist_ok=True)

osm.logconfig.setup_logging("INFO")


def retrieve() -> None:
    countries = osm_utils.load_regions(grow_regions=0.1, test=False).sort_values(by="code")
    print(countries)

    for id, region in countries.iterrows():
        osm.log.info(f"{id}/{len(countries)}, {region['code']}")
        if isinstance(region["geometry"], str):
            osm.logging.info(region["geometry"])
        pl_path = datapath / f"graph_{region['code']}_railways.gpkg"

        if pl_path.is_file() and False:
            osm.log.warning(f"File {pl_path.name} already present.")
            continue
        else:
            pl = osm.osm_railways(
                region["geometry"],
                osm_dump_file=(
                    "~/curro/working_data/geodata/osm_dump/EU_railways_nodes.gpkg",
                    "~/curro/working_data/geodata/osm_dump/EU_railways_edges.gpkg",
                ),
                node_prefix=str(region["code"]) + "_",
            )
            if len(pl) > 0:
                pl.write(pl_path)


def merge() -> None:
    """Do the main."""
    # Load the countries

    graph: osm.Graph | None = None

    for pl_path in sorted(datapath.glob("graph_[A-Z]*_railways.gpkg")):
        osm.log.info(pl_path.name)

        pl = osm.Graph.read(pl_path)
        # pl = fix_edges(pl, pl_path)

        if len(pl) == 0:
            continue

        if graph is None:
            graph = pl
        else:
            # graph = graph.merge_edges(pl)
            graph.nodes = graph.nodes.append(pl.nodes)
            graph.edges = graph.edges.append(pl.edges)

    if graph is not None:
        graph.write(datapath / "graph_all_graph.gpkg")


def blend():
    g1 = osm.Graph.read(
        "~/curro/working_data/osm_retrieve_networks/graphs_railways_EU/graph_PD_railways.gpkg"
    )
    g2 = osm.Graph.read(
        "~/curro/working_data/osm_retrieve_networks/graphs_railways_EU/graph_RO_railways.gpkg"
    )
    g1.blend(g2)


if __name__ == "__main__":
    retrieve()
    # merge()
