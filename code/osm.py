"""Base functions and utilities.

Base functions for retrieving and post-processing data from OpenStreetMap
to build a network from the infrastructures.
"""

from __future__ import annotations

import logging
from collections import namedtuple
from collections.abc import Hashable
from dataclasses import dataclass, field
from itertools import product
from numbers import Number
from pathlib import Path
from typing import Self

import geopandas as geopd
import logconfig
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import shapely
import tqdm
from matplotlib import axes
from shapely import geometry, ops, plotting
from sklearn import cluster

CACHE = Path("~/.cache").expanduser() / "osm_retrieve_networks"
CACHE.mkdir(parents=True, exist_ok=True)

DATA = Path("~/curro/working_data/").expanduser() / "osm_retrieve_networks"
DATA.mkdir(parents=True, exist_ok=True)

PRJ_DEG = pyproj.CRS.from_epsg(4326)
PRJ_MET = pyproj.CRS.from_epsg(3857)
DEG2MET = pyproj.Transformer.from_crs(PRJ_DEG, PRJ_MET, always_xy=True).transform
MET2DEG = pyproj.Transformer.from_crs(PRJ_MET, PRJ_DEG, always_xy=True).transform


ox.settings.cache_folder = CACHE
logconfig.setup_logging("INFO")
log = logging.getLogger(__name__)


def osm_railways(
    place: str | geometry.Polygon | geometry.MultiPolygon, node_prefix: str = ""
) -> Graph:
    """Retrieve the power-line network.

    Parameters
    ----------
    place: str | geometry.Polygon | geometry.MultiPolygon
        Retrieve data from this place: e.g. `Padova`, `London`, `France`… or a `[Multi]Polygon`.
    node_prefix :  str
        A prefix for the nodes.

    Returns
    -------
    graph : Graph
        nodes and edges
    powerplants : GeoDataFrame
        the powerplants
    """
    # Retrieving the enclosing poligon.
    enclosing_polygon = get_geolocation(place=place) if isinstance(place, str) else place
    if enclosing_polygon.area > 10:
        log.warning(f"Area of size {enclosing_polygon.area}")
    else:
        log.info(f"Area of size {enclosing_polygon.area}")
    # Retrieve lines from 'OpenStreetMap'.
    log.info("Retrieving edges from OpenStreetMap.")
    edges = retrieve_edges(
        keys={"railway": ["rail", "construction", "preserved", "narrow_gauge"]},
        polygon=enclosing_polygon,
    )
    print(edges.columns)
    edges = edges[
        [
            "id",
            "operator",
            "ref",
            "name",
            "electrified",
            "frequency",
            "gauge",
            "maxspeed",
            "geometry",
        ]
    ]
    if len(edges) > 0:
        edges = split_edges_when_touching(edges)

    # - add nodes as line boundaries
    # - associate nodes to links
    # - add nodes as stations
    # - split links close to stations, re-associate nodes to links accordingly

    # get train stations
    nodes = retrieve_nodes({"railway": ["station", "halt", "stop"]}, polygon=enclosing_polygon)
    nodes["id"] = [f"{node_prefix}{i}" for i in range(len(nodes))]
    nodes = nodes.set_index("id", drop=True)
    nodes["__keep__"] = True
    print(edges)
    edges = split_lines_at_points(nodes=nodes, edges=edges, max_distance=100)

    # get nodes from edges
    edgeBoundary_nodes = edges.boundary
    edgeBoundary_nodes = geopd.GeoDataFrame(
        {
            "__keep__": False,
            "id_s": [f"s{i}" for i in range(len(edgeBoundary_nodes))],
            "id_t": [f"t{i}" for i in range(len(edgeBoundary_nodes))],
            "geometry": edgeBoundary_nodes,
        },
        crs=edges.crs,
    )
    print("t411" in edgeBoundary_nodes["id_t"])

    # assign source and target to edges.
    edges["source"] = edges["source"].fillna(edgeBoundary_nodes["id_s"])
    edges["target"] = edges["target"].fillna(edgeBoundary_nodes["id_t"])
    print(edges)
    print("t411" in edges["target"])
    print("t411" in edgeBoundary_nodes["id_t"])
    edgeBoundary_nodes = geopd.GeoDataFrame(
        pd.concat(
            [
                edgeBoundary_nodes.explode().iloc[::2].set_index("id_s", drop=True),
                edgeBoundary_nodes.explode().iloc[1::2].set_index("id_t", drop=True),
            ]
        )
    )
    print(edgeBoundary_nodes)
    print("t411" in edgeBoundary_nodes.index)
    nodes = geopd.GeoDataFrame(
        pd.concat([nodes, edgeBoundary_nodes]), geometry="geometry", crs=nodes.crs
    )
    print("t411" in nodes.index)
    nodes = nodes.loc[list(set(edges.source.tolist() + edges.target.tolist()))]
    print(nodes)
    print(edges)
    edges.to_file("/tmp/pippo.geojson")
    nodes.to_file("/tmp/pippo2.geojson")

    exit()
    edges = project_edges_to_nodes(
        edges.to_crs(PRJ_MET), nodes.to_crs(PRJ_MET), max_distance=500
    ).to_crs(PRJ_DEG)

    graph = Graph(
        edges=edges,
        region=geopd.GeoDataFrame([{"region": 0}], geometry=[enclosing_polygon], crs=PRJ_DEG),
    )
    log.info("Get nodes from edges extremes.")
    graph.nodes = nodes
    #
    # log.info("Get power plants.")

    return graph


def osm_powerlines(
    place: str | geometry.Polygon | geometry.MultiPolygon,
    substation_distance: float = 500,
    voltage_fillvalue: float | None = None,
    voltage_threshold: float = 1000,
    node_prefix: str = "",
) -> tuple[Graph, geopd.GeoDataFrame]:
    """Retrieve the power-line network.

    Parameters
    ----------
    place: str | geometry.Polygon | geometry.MultiPolygon
        Retrieve data from this place: e.g. `Padova`, `London`, `France`… or a `[Multi]Polygon`.
    voltage_fillvalue : float | None :
         (Default value = None)
    voltage_threshold : float
        (Default value = 1000)
    node_prefix :  str
        A prefix for the nodes.

    Returns
    -------
    graph : Graph
        nodes and edges
    powerplants : GeoDataFrame
        the powerplants
    """
    # Retrieving the enclosing poligon.
    enclosing_polygon = get_geolocation(place=place) if isinstance(place, str) else place
    if enclosing_polygon.area > 10:
        log.warning(f"Area of size {enclosing_polygon.area}")
    else:
        log.info(f"Area of size {enclosing_polygon.area}")
    # Retrieve lines from OpenStreetMap.
    log.info("Retrieving edges from OpenStreetMap.")
    edges = retrieve_edges(keys={"power": ["cable", "line"]}, polygon=enclosing_polygon).rename(
        columns={"source": "power_source"}
    )
    edges = edges[
        edges.columns.intersection(
            [
                "id",
                "index",
                "cables",
                "capacity",
                "frequency",
                "line",
                "geometry",
                "maxcurrent",
                "name",
                "operator",
                "power",
                "power:maximum",
                "power:used",
                "power_source",
                "source",
                "source:power",
                "submarine",
                "target",
                "type",
                "voltage",
            ]
        )
    ]
    if len(edges) > 0:
        if "voltage" in edges.columns:
            edges.voltage = _clean_voltage(edges.voltage)
        else:
            edges["voltage"] = pd.NA

        if voltage_fillvalue is None:
            edges = edges.dropna(subset="voltage")
            edges = edges[edges.voltage >= voltage_threshold]
        else:
            # Retains the lower voltage (1k-69k Volts) power-lines.
            # Assumes non reported voltage is lower than 69kV.
            # This is risky as the network is not that accurately reported.
            edges.voltage = edges.voltage.fillna(voltage_fillvalue)
            log.warning("You assume that edges with no voltage are below 69k.")
        edges = split_edges_when_touching(edges.reset_index(drop=False))

        log.info("Retrieving substations from OpenStreetMap.")
        # Retrieve substations (additional nodes).
        # These substations are used to *merge* lines that do not touch.
        if len(edges) > 0:
            substation_polygon = ops.transform(
                MET2DEG,
                edges.to_crs(PRJ_MET).buffer(2 * substation_distance, resolution=2).union_all(),
            )
            log.info("get nodes")
            substations = retrieve_nodes(keys={"power": ["substation"]}, polygon=substation_polygon)
            log.info("add substations fuzzily")
            if len(substations) > 0:
                substations = cluster_points(substations, distance=substation_distance)
                edges = add_fuzzy_nodes(substations, edges, distance=substation_distance)
    log.info("Build graph")
    graph = Graph(
        edges=edges,
        region=geopd.GeoDataFrame([{"region": 0}], geometry=[enclosing_polygon], crs=PRJ_DEG),
    )

    log.info("Get nodes from edges extremes.")
    graph = graph.nodes_from_edges(node_prefix=node_prefix)

    log.info("Get power plants.")
    powerplants = retrieve_nodes(keys={"power": ["plant"]}, polygon=enclosing_polygon)
    powerplants = powerplants[
        powerplants.columns.intersection(
            [
                "id",
                "addr_city",
                "frequency",
                "generator_capacity",
                "geometry",
                "name",
                "operator",
                "owner",
                "plant_method",
                "plant_output",
                "plant_source",
                "plant_storage",
                "plant_type",
                "power",
                "source",
                "type",
                "underground",
                "voltage",
                "voltage",
            ]
        )
    ]

    return graph, powerplants


def retrieve_nodes(
    keys: dict,
    place: str | None = None,
    polygon: geometry.Polygon | geometry.MultiPolygon | None = None,
) -> geopd.GeoDataFrame:
    """Retrieve node data from OpenStreetMap.

    Use this to download data that should be points.

    Warning: when a polygon is retrieved it is transformed to a Point through
    the `representative_point` method.

    Parameters
    ----------
    keys : dict
        The OpenStreetMap keys to use for download.
        The Union of the results are retrieved.
    place : str
        Retrieve data from this place: e.g. `Padova`, `London`, `France`…

    Returns
    -------
    nodes : GeoDataFrame
        Points

    """
    log.info("Retrieving `Point` data from OpenStreetMap.")
    data = []
    for k, vals in keys.items():
        tags = {k: vals}
        try:
            if place is not None:
                _data = ox.features_from_place(place, tags)
            elif polygon is not None:
                _data = ox.features_from_polygon(polygon, tags)
            else:
                raise ValueError("You should provide either `place` or `polygon`.")
        except (ox._errors.InsufficientResponseError, shapely.errors.EmptyPartError):
            log.warning(f"No data for {tags}. Skipping")
            pass
        else:
            if isinstance(_data, list):
                data.extend(_data)
            else:
                data.append(_data)

    # Build a `DataFrame`
    if len(data) == 0:
        return geopd.GeoDataFrame([], geometry=[], crs=PRJ_DEG)
    nodes = geopd.GeoDataFrame(pd.concat(data), geometry="geometry", crs=PRJ_DEG).droplevel(0)
    nodes.columns = [c.replace(":", "_") for c in nodes.columns]
    nodes = nodes.dropna(how="all", axis=1)
    log.info(f"Collected {len(nodes)} nodes.")
    # Transform everything to nodes
    nodes.geometry = nodes.geometry.representative_point()
    # Drop duplicates
    return nodes.loc[~nodes.index.duplicated(keep="first"), :]


def retrieve_edges(
    keys: dict,
    place: str | None = None,
    polygon: geometry.Polygon | geometry.MultiPolygon | None = None,
) -> geopd.GeoDataFrame:
    """Retrieve edge data from OpenStreetMap.

    Only `LineString` will be kept.

    Parameters
    ----------
    keys : dict
        The OpenStreetMap keys to use for download.
        The Union of the results are retrieved.
    place : str
        Retrieve data from this place: e.g. `Padova`, `London`, `France`…

    Returns
    -------
    edges : GeoDataFrame
        The edges

    """
    log.info("Retrieving `LineString` data from OpenStreetMap.")
    data = []
    for k, vals in keys.items():
        try:
            if place is not None:
                features = ox.features_from_place(place, {k: vals}).loc["way"]
            elif polygon is not None:
                features = ox.features_from_polygon(polygon, {k: vals}).loc["way"]
            else:
                raise ValueError("You should provide either `place` or `polygon`.")
        except ox._errors.InsufficientResponseError:
            log.warning("Nothing to see here")
            continue

        data.append(features)

    # Build a dataframe
    if len(data) > 0:
        edges = geopd.GeoDataFrame(pd.concat(data), geometry="geometry")
    else:
        return geopd.GeoDataFrame([], geometry=[])
    edges.geometry = edges.geometry.line_merge()
    # Keep only `LineString` objects
    edges = (
        edges[edges.geometry.geom_type.isin({"LineString", "MultiLineString"})]
        .explode()
        .reset_index(drop=False)
    )
    edges.geometry = edges.geometry.line_merge()
    # Remove duplicates
    edges = edges[~edges.index.duplicated(keep="first")]
    edges["name"] = range(len(edges))
    log.info(f"Collected {len(edges)} edges.")
    edges = edges.dropna(how="all", axis=1)

    return geopd.GeoDataFrame(edges, geometry="geometry", crs=PRJ_DEG)


# More utilities


@dataclass
class Graph:
    nodes: geopd.GeoDataFrame = field(init=False)
    edges: geopd.GeoDataFrame
    region: geopd.GeoDataFrame

    @property
    def region_shape(self):
        return self.region.union_all(method="coverage")

    def nodes_from_edges(
        self, source_col: str = "source", target_col: str = "target", node_prefix: str = ""
    ) -> Self:
        """Extract and deduplicate the nodes at the edge extremes.

        Update edges to include `source` and `target`

        Parameters
        ----------
        edges : geopd.GeoDataFrame
            The edges
        source_col: str :
             (Default value = "source")
             The name of the new column with source nodes.
        target_col: str :
             (Default value = "target")
             The name of the new column with target nodes.

        Returns
        -------
        nodes : GeoDataFrame
            All nodes
        edges : GeoDataFrame
            Same edges with columns `source_col` and `target_col` added

        """
        log.info("Extracting nodes from edges.")

        if len(self.edges) == 0:
            self.nodes = geopd.GeoDataFrame({"id": [], "geometry": []}, crs=self.edges.crs)
            return self
        node_map: dict[geometry.base.BaseGeometry, str] = {}
        source_target: list[dict[str, int | Hashable]] = []
        for id, line in self.edges.geometry.items():
            # Nodes geometry
            try:
                s, t = line.boundary.geoms
            except ValueError:
                raise
            # Source node id
            s_id = node_map.setdefault(s, node_prefix + str(len(node_map)))
            # Target node id
            t_id = node_map.setdefault(t, node_prefix + str(len(node_map)))
            source_target.append({source_col: s_id, target_col: t_id, "index": id})

        self.nodes = geopd.GeoDataFrame(
            [{"id": v, "geometry": k} for k, v in node_map.items()],
            geometry="geometry",
            crs=self.edges.crs,
        ).set_index("id", drop=True)

        self.edges = geopd.GeoDataFrame(
            pd.concat(
                [self.edges, pd.DataFrame(source_target).set_index("index", drop=True)], axis=1
            )
        )
        return self

    def index(self) -> pd.Index:
        return self.edges.index

    def filter_edges(self, keep_ids: list | np.ndarray | pd.Index) -> Graph:
        """Return a new graph with only `keep_ids` edges."""
        graph = Graph(self.edges.loc[keep_ids], region=self.region)
        graph.nodes = self.nodes.loc[
            pd.concat([graph.edges.source, graph.edges.target]).drop_duplicates()
        ]
        return graph

    def filter_nodes(self, keep_ids: list | np.ndarray | pd.Index) -> Graph:
        """Return a new graph with only the mentioned nodes."""
        graph = Graph(
            self.edges[(self.edges.source.isin(keep_ids)) & (self.edges.target.isin(keep_ids))],
            region=self.region,
        )
        graph.nodes = self.nodes.loc[keep_ids]
        return graph

    def intersect(self, other: Graph, return_same: bool | None = False) -> list:
        """Return edges in both (deduplicated).

        - For same edges: return one (if `return_same` is True)
        - For one edge covering the other: return smaller
        - For one edge present in only one graph: return it
        """
        sindex = shapely.STRtree(other.edges.geometry)
        edges = {
            id: set(sindex.query(edge.geometry, predicate="within"))
            for id, edge in self.edges.iterrows()
            if len(set(sindex.query(edge.geometry, predicate="contains"))) == 0
        }
        if return_same:
            edges |= {
                id: set(sindex.query_nearest(edge.geometry, max_distance=1e-10, all_matches=False))
                for id, edge in self.edges.iterrows()
            }

        return list(edges.keys())

    def split_edges(self, other):
        """Split edges in those intersecting the other.region and the others."""
        # Edges in self, crossing to other.region
        sindex = shapely.STRtree(self.edges.geometry)
        crossing_ids = self.edges.index[sindex.query(other.region_shape, predicate="intersects")]

        g_inner = self.filter_edges(self.index().difference(list(crossing_ids)))
        g_outer = self.filter_edges(crossing_ids)
        return g_inner, g_outer

    def merge_edges(self, other: Graph) -> Graph:
        """Merge the edges of two `Graph` and adapt nodes accordingly.

        1. find edges in self that covers edges in other
        2. find edges in other that covers edges is self
        3. remove them (identical edges should be taken only once)
        4. rename the nodes.
        """
        all_nodes = pd.concat([self.nodes, other.nodes])
        all_edges = pd.concat([self.edges, other.edges], ignore_index=True)
        all_edge_sindex = shapely.STRtree(all_edges.geometry)

        used = set()
        keep_edges = set()
        transform_edges = []
        rename_nodes = {}

        for edgeid, edge in all_edges.geometry.items():
            if edgeid in used:
                continue

            # fetch all edges contained in this one
            contains_idx = set(all_edge_sindex.query(edge, predicate="contains"))
            involved = contains_idx - {edgeid}

            if len(involved) > 0:
                involved.add(edgeid)
                used |= involved

                involved_edges = all_edges.loc[list(involved)]
                involved_nodes = all_nodes.loc[
                    pd.concat([involved_edges.source, involved_edges.target])
                ].rename(index=rename_nodes)
                involved_nodes = involved_nodes.loc[~involved_nodes.index.duplicated()]
                nodes = [sorted(n.index) for _, n in involved_nodes.groupby("geometry")]
                new_renames = {nn: n[0] for n in nodes if len(n) > 1 for nn in n[1:]}
                rename_nodes |= new_renames

                # remove newly added renames
                involved_nodes = involved_nodes.rename(index=new_renames)
                involved_nodes = involved_nodes.loc[~involved_nodes.index.duplicated()]

                # split the containing edge by the points
                new_lines = align_line_points(edge, geopd.GeoDataFrame(involved_nodes))
                for k, v in all_edges.loc[edgeid].items():
                    if k not in {"geometry", "source", "target"}:
                        new_lines[k] = v
                transform_edges.append(new_lines)
                keep_edges -= involved

            else:
                keep_edges.add(edgeid)

        # Almost overlapping nodes:
        sindex = shapely.STRtree(self.nodes.geometry)
        rename_overlapping_nodes = {
            id_node: self.nodes.index[sindex.query_nearest(node, max_distance=1e-6)]
            for id_node, node in other.nodes.geometry.items()
        }
        rename_overlapping_nodes = {
            k: v[0] for k, v in rename_overlapping_nodes.items() if len(v) > 0
        }
        print(rename_nodes)
        print(rename_overlapping_nodes)
        rename_nodes |= rename_overlapping_nodes

        g = Graph(
            edges=geopd.GeoDataFrame(
                pd.concat([all_edges.iloc[list(keep_edges)]] + transform_edges), crs=self.edges.crs
            ),
            region=geopd.GeoDataFrame(pd.concat([self.region, other.region]), crs=self.region.crs),
        )
        g.edges.source = g.edges["source"].apply(lambda x: rename_nodes.get(x, x))
        g.edges.target = g.edges["target"].apply(lambda x: rename_nodes.get(x, x))
        all_nodes = all_nodes.rename(index=rename_nodes)
        all_nodes = all_nodes[~all_nodes.index.duplicated()]

        g.nodes = geopd.GeoDataFrame(all_nodes)
        return g

    def merge(self, other: Graph) -> Graph:
        """Merge the two graphs.

        Merge edges that are crossing borders.

        Assumptions:
            - No edges from the same source are overlapping (one covers the other, otherwise both are returned).
            - The overlapping part of an edge should be close to one extreme (the one in the other region)
              Otherwise the uncovered part is lost.

        See `osm_test.py` for examples.
        """
        print(self.nodes)
        print(other.nodes)
        # Edges in self, crossing to other.region
        g_inner1, g_outer1 = self.split_edges(other)
        g_inner2, g_outer2 = other.split_edges(self)

        # nodes that need attention
        i1 = g_outer1.intersect(g_outer2, return_same=True)
        i2 = g_outer2.intersect(g_outer1, return_same=False)
        merged = geopd.GeoDataFrame(
            pd.concat(
                [g_inner1.edges, g_inner2.edges, g_outer1.edges.loc[i1], g_outer2.edges.loc[i2]]
            ),
            crs=self.edges.crs,
        )

        # Fix nodes
        region2 = g_outer2.region_shape
        wrong_region = [
            node
            for k, edge in g_outer1.edges.iterrows()
            for node in edge.loc[["source", "target"]]
            if g_outer1.nodes.loc[node].geometry.within(region2)
        ]
        sindex = shapely.STRtree(other.nodes.geometry)
        rename = {
            node: other.nodes.index[
                sindex.query_nearest(g_outer1.nodes.loc[node].geometry, max_distance=1e-10)
            ]
            for node in wrong_region
        }
        region1 = g_outer1.region_shape
        wrong_region = [
            node
            for k, edge in g_outer2.edges.iterrows()
            for node in edge.loc[["source", "target"]]
            if g_outer2.nodes.loc[node].geometry.within(region1)
        ]
        sindex = shapely.STRtree(self.nodes.geometry)
        rename |= {
            node: self.nodes.index[
                sindex.query_nearest(g_outer2.nodes.loc[node].geometry, max_distance=1e-10)
            ]
            for node in wrong_region
        }
        rename = {k: v[0] for k, v in rename.items() if len(v) > 0}

        mgraph = Graph(
            edges=merged.reset_index(drop=True),
            region=geopd.GeoDataFrame(pd.concat([self.region, other.region]), crs=self.region.crs),
        )
        mgraph.edges.index.name = "index"
        mgraph.edges["source"] = mgraph.edges["source"].map(lambda x: rename.get(x, x))
        mgraph.edges["target"] = mgraph.edges["target"].map(lambda x: rename.get(x, x))
        mgraph.nodes = geopd.GeoDataFrame(
            pd.concat(
                [
                    self.nodes.loc[self.nodes.index.difference(pd.Index(rename))],
                    other.nodes.loc[other.nodes.index.difference(pd.Index(rename))],
                ]
            ),
            crs=self.edges.crs,
        )
        mgraph.nodes.index.name = "id"
        return mgraph

    def largest_component(self):
        g = nx.from_pandas_edgelist(self.edges, source="source", target="target")
        largest_component = list(max(nx.connected_components(g), key=len))

        return self.filter_nodes(largest_component)

    def __str__(self) -> str:
        return "\n".join(map(str, [self.edges, self.nodes]))

    def __len__(self) -> int:
        return len(self.edges)

    @classmethod
    def read(cls, path: Path) -> Graph:
        edges = geopd.read_file(path, layer="edges")
        nodes = geopd.read_file(path, layer="nodes").set_index("id", drop=True)
        region = geopd.read_file(path, layer="region")

        g = Graph(edges=edges, region=region)
        g.nodes = nodes

        return g

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        path.unlink(missing_ok=True)
        self.edges.drop(columns=["index"], errors="ignore").to_file(
            path, driver="GPKG", layer="edges"
        )
        self.nodes.to_file(path, driver="GPKG", layer="nodes")
        self.region.to_file(path, driver="GPKG", layer="region")

    def plot(self, ax: axes.Axes, color: str = "r", text_args: dict | None = None):
        plotting.plot_polygon(self.region_shape, color=color, ax=ax, alpha=0.2)
        plotting.plot_line(self.edges.union_all(), color=color, ax=ax, alpha=0.3)
        if hasattr(self, "nodes"):
            plotting.plot_points(
                self.nodes.union_all(), ax=ax, color=color, alpha=0.5, markersize=10
            )

            for x, y, name in zip(self.nodes.geometry.x, self.nodes.geometry.y, self.nodes.index):
                text_args = {} if text_args is None else text_args
                ax.annotate(name, (x, y), xytext=(x, y + 1), **text_args)
        ax.grid()


def _clean_voltage(voltage: pd.Series) -> pd.Series:
    """Clean the voltage column."""

    def _convert(cell: Number | str) -> Number:
        if isinstance(cell, Number):
            return cell
        text = cell.split(";")[0]

        try:
            val = int(text)
        except ValueError:
            return np.nan

        return val

    return voltage.apply(_convert)


def project_edges_to_nodes(
    edges: geopd.GeoDataFrame, nodes: geopd.GeoDataFrame, max_distance: float
) -> geopd.GeoDataFrame:
    """Change the edges to link the new set of nodes.

    Use the shortest path to link nodes.
    """
    e_sindex = shapely.STRtree(edges.geometry)
    edgelist = e_sindex.query(edges.geometry, predicate="intersects")
    network = nx.from_edgelist([(e1, e2) for e1, e2 in edgelist.T if e1 != e2])
    print(network)

    nodes_to_edges = e_sindex.query(nodes.geometry, predicate="dwithin", distance=500)
    n2e = {}
    for e, n in nodes_to_edges.T:
        n2e.setdefault(n, []).append(e)

    print("TODO")
    exit()
    for node_i, node_j in product(range(len(nodes)), repeat=2):
        if node_i == node_j:
            continue
        # find shortest path between closest edges.
        lines = [nx.shortest_path(network, l1, l2, weight=1)]
    return edges


def split_lines_at_points(
    nodes: geopd.GeoDataFrame, edges: geopd.GeoDataFrame, max_distance: float = 0.0
) -> geopd.GeoDataFrame:
    """Split lines which fall close to the given nodes.

    Move points to the line.

    Only the closed points to each extreme are taken in consideration.
    Example: Powerlines getting close to substations.

    Parameters
    ----------
    nodes : geopd.GeoDataFrame
        The nodes dataframe.
    edges : geopd.GeoDataFrame
        The edges dataframe.
    distance : float
         (Default value = 0.0)
         The distance up to which edges should be connected to the a node

    Returns
    -------
    edges : GeoDataFrame
        The edges connected to the points within `distance`

    """
    if "name" not in nodes.columns:
        nodes["name"] = range(len(nodes))
    if "name" not in edges.columns:
        edges["name"] = range(len(edges))
    mnodes = nodes.copy().to_crs(PRJ_MET)
    medges = edges.copy().to_crs(PRJ_MET)

    # Use `sindex` for quickness
    edges_sindex = shapely.STRtree(medges.geometry)
    Cache = namedtuple(
        "Cache", field_names=["toremove", "newedges", "usednodes"], defaults=[[], [], []]
    )

    cache = Cache()
    for id, mnode in mnodes.geometry.items():
        # closest node to source
        i_edges = edges_sindex.query_nearest(mnode, max_distance=max_distance, all_matches=False)

        if len(i_edges) > 0:
            # if there is some edges within distance, it should be just one.
            ied = i_edges[0]

            # if this edge has already be used,
            # skip it. We will check it next iteration
            if ied in cache.toremove:
                continue

            old_edge = medges.iloc[ied]
            new_edge1 = old_edge.to_dict()
            new_edge2 = old_edge.to_dict()

            # split in the closest point
            node_pos = old_edge.geometry.project(mnode)
            if node_pos <= 0:
                cache.toremove.append(ied)
                new_edge1["source"] = id
                cache.newedges.append(new_edge1)
                cache.usednodes.append(id)
            elif node_pos >= old_edge.geometry.length:
                cache.toremove.append(ied)
                new_edge1["target"] = id
                cache.newedges.append(new_edge1)
                cache.usednodes.append(id)

            new_edge1["geometry"] = ops.substring(old_edge.geometry, 0, node_pos)
            new_edge1["target"] = id
            new_edge2["geometry"] = ops.substring(
                old_edge.geometry, node_pos, old_edge.geometry.length
            )
            new_edge2["source"] = id

            cache.toremove.append(ied)
            cache.newedges.extend([new_edge1, new_edge2])
            cache.usednodes.append(id)

    # Substitute with the updated geometries

    _edges = geopd.GeoDataFrame(
        pd.concat(
            [medges.drop(index=cache.toremove), pd.DataFrame(cache.newedges)], ignore_index=True
        )
    ).set_crs(PRJ_MET, allow_override=True)
    if len(cache.toremove) > 0:
        _edges = split_lines_at_points(
            nodes=nodes.drop(index=cache.usednodes), edges=_edges, max_distance=max_distance
        )
    return _edges.to_crs(PRJ_DEG)


def extend_line_to_point(
    nodes: geopd.GeoDataFrame, edges: geopd.GeoDataFrame, max_distance: float = 0.0
) -> geopd.GeoDataFrame:
    """Split lines which fall close to the given nodes.

    Only the closed points to each extreme are taken in consideration.
    Example: Powerlines getting close to substations.

    Parameters
    ----------
    nodes : geopd.GeoDataFrame
        The nodes dataframe.
    edges : geopd.GeoDataFrame
        The edges dataframe.
    distance : float
         (Default value = 0.0)
         The distance up to which edges should be connected to the a node

    Returns
    -------
    edges : GeoDataFrame
        The edges connected to the points within `distance`

    """
    if "name" not in nodes.columns:
        nodes["name"] = range(len(nodes))
    if "name" not in edges.columns:
        edges["name"] = range(len(edges))
    mnodes = nodes[["name", "geometry"]].copy().to_crs(PRJ_MET)
    medges = edges[["name", "geometry"]].copy().to_crs(PRJ_MET)

    # Use `sindex` for quickness
    nodes_sindex = shapely.STRtree(mnodes.geometry)
    update = {}
    for id, medge in medges.geometry.items():
        try:
            source, target = medge.boundary.geoms
        except ValueError:
            # Circular lines!!!!
            continue

        # closest node to source
        imes = nodes_sindex.query_nearest(source, max_distance=max_distance, all_matches=False)
        # closest node to target
        imet = nodes_sindex.query_nearest(target, max_distance=max_distance, all_matches=False)

        ids = np.concat([imes, imet])
        assert len(ids) <= 2

        # If both ends of the edge fall close to the same point,
        # remove that edge
        if len(ids) == 2 and ids[0] == ids[1]:
            continue

        if len(ids) > 0 and isinstance(medge, geometry.LineString):
            update[id] = _join_line_point_buffered(
                medge,
                geopd.GeoSeries(mnodes.geometry.iloc[ids], index=mnodes.index[ids]),
                max_distance,
            )
            continue

        update[id] = medge

    # Substitute with the updated geometries
    _update = geopd.GeoSeries(update, crs=mnodes.crs).dropna().to_crs(edges.crs)
    edges = edges.loc[_update.index]
    edges.geometry = _update.geometry
    return edges


def add_fuzzy_nodes(
    nodes: geopd.GeoDataFrame, edges: geopd.GeoDataFrame, distance: float = 0.0
) -> geopd.GeoDataFrame:
    """Join lines which bournaries fall close to the given nodes.

    Only the closed points to each extreme are taken in consideration.
    Example: Powerlines getting close to substations.

    Parameters
    ----------
    nodes : geopd.GeoDataFrame
        The nodes dataframe.
    edges : geopd.GeoDataFrame
        The edges dataframe.
    distance : float
         (Default value = 0.0)
         The distance up to which edges should be connected to the a node

    Returns
    -------
    edges : GeoDataFrame
        The edges connected to the points within `distance`

    """
    if "name" not in nodes.columns:
        nodes["name"] = range(len(nodes))
    if "name" not in edges.columns:
        edges["name"] = range(len(edges))
    mnodes = nodes[["name", "geometry"]].copy().to_crs(PRJ_MET)
    medges = edges[["name", "geometry"]].copy().to_crs(PRJ_MET)

    # Use `sindex` for quickness
    nodes_sindex = shapely.STRtree(mnodes.geometry)
    update = {}
    for id, medge in medges.geometry.items():
        try:
            source, target = medge.boundary.geoms
        except ValueError:
            # Circular lines!!!!
            continue

        # closest node to source
        imes = nodes_sindex.query_nearest(source, max_distance=distance, all_matches=False)
        # closest node to target
        imet = nodes_sindex.query_nearest(target, max_distance=distance, all_matches=False)

        ids = np.concat([imes, imet])
        assert len(ids) <= 2

        # If both ends of the edge fall close to the same point,
        # remove that edge
        if len(ids) == 2 and ids[0] == ids[1]:
            continue

        if len(ids) > 0 and isinstance(medge, geometry.LineString):
            update[id] = _join_line_point_buffered(
                medge, geopd.GeoSeries(mnodes.geometry.iloc[ids], index=mnodes.index[ids]), distance
            )
            continue

        update[id] = medge

    # Substitute with the updated geometries
    _update = geopd.GeoSeries(update, crs=mnodes.crs).dropna().to_crs(edges.crs)
    edges = edges.loc[_update.index]
    edges.geometry = _update.geometry
    return edges


def cluster_points(
    points: geopd.GeoDataFrame, distance: float = 100, column: str = "__node_clusters__"
) -> geopd.GeoDataFrame:
    """Cluster nodes that are within `distance` (meters). Based on the `DBSCAN`.

    This adds a column to the `GeoDataFrame` with the labels of each cluster.
    """
    data_in_meters = points.geometry.to_crs(PRJ_MET)
    labels = cluster.DBSCAN(eps=distance, min_samples=2).fit_predict(
        [[p.x, p.y] for p in data_in_meters]
    )
    solitons = labels == 1
    maxlabels = labels.max()
    labels[solitons] = np.arange(sum(solitons)) + (maxlabels + 1)
    points[column] = labels
    return points.dissolve(by=column, aggfunc="first").reset_index(drop=True)


def _join_line_point_buffered(
    line: geometry.LineString, points: geopd.GeoSeries, distance: float
) -> geometry.LineString:
    """Cut the linestring at `distance` and join to the point that end.

    Parameters
    ----------
    line : LineString
        The line to be splitted
    points : geopd.GeoSeries
        The point[s] to be added as extremes (may be one or two).
    distance : float
        The distance to join

    Returns
    -------
    line : LineString
        The line with the extreme[s] merged to the point[s]

    """
    _line = line

    for pid, point in points.items():
        # A circle around the point
        if not isinstance(point, geometry.Point):
            point = point.representative_point()
        buffer = point.buffer(distance).boundary
        intersection = _line.intersection(buffer)

        if not isinstance(intersection, geometry.Point):
            continue

        dist_point = _line.project(point)
        dist_intsc = _line.project(intersection)

        if dist_point < dist_intsc:
            l1 = ops.substring(_line, start_dist=dist_intsc, end_dist=_line.length)
            _line = geometry.LineString([point] + list(l1.coords))
        else:
            l1 = ops.substring(_line, start_dist=0, end_dist=dist_intsc)
            _line = geometry.LineString(list(l1.coords) + [point])

    return _line


def split_edges_when_touching(edges: geopd.GeoDataFrame) -> geopd.GeoDataFrame:
    """Joins edges at their intersection.

    If the extreme of an edge touches the another edge, the latter is splitted in that point

    Parameters
    ----------
    edges: geopd.GeoDataFrame :


    Returns
    -------

    """
    log.info("Splitting edges to connect when touching.")
    # Use the spatial index for speed
    sindex = edges.sindex

    # Check if two edges touch each other and eventually split them at the intersection
    new_edges = {}
    for edge_id1, edge1 in edges.geometry.items():
        # If edges are already chopped
        if edge_id1 in new_edges:
            continue

        # Find close-by edges (only with higher index to consider only one direction)
        possible = [edges.index[int(i)] for i in sindex.intersection(edge1.bounds)]
        # Remove self
        possible = [id for id in possible if id != edge_id1]
        # Remove already chopped
        possible = [id for id in possible if id not in new_edges]

        # Find edges that actually touch the original edge
        touches = [edge_id2 for edge_id2 in possible if edge1.touches(edges.geometry.loc[edge_id2])]

        if len(touches) == 0:
            continue

        # Split edge in shorter edges at intersections.
        edges_touch = shapely.MultiLineString(lines=edges.geometry.loc[touches].tolist())
        edge_splitted = ops.split(edge1, edges_touch)  # MultiLineString
        if len(edge_splitted.geoms) > 1:
            new_edges.setdefault(edge_id1, []).extend(edge_splitted.geoms)

        # Split the other edges on the original edge.
        for touched in touches:
            edge_splitted = ops.split(edges.geometry.loc[touched], edge1)
            if len(edge_splitted.geoms) > 1:
                new_edges.setdefault(touched, []).extend(edge_splitted.geoms)

    result = geopd.GeoDataFrame(
        pd.concat(
            [
                edges.loc[~edges.index.isin(new_edges)],
                pd.DataFrame(
                    [
                        edges.loc[id].to_dict() | {"geometry": line}
                        for id, lines in new_edges.items()
                        for line in lines
                    ]
                ),
            ],
            ignore_index=True,
        )
    ).set_crs(crs=edges.crs, allow_override=True)
    if len(new_edges) == 0:
        return result
    log.info("Performing a new deeper recursive level")
    return split_edges_when_touching(result)


def check_cache_size():
    """Compute the size of the caching folder."""
    size = 0
    for root, dirs, files in CACHE.walk():
        for fl in files:
            size += (root / fl).stat().st_size
    log.debug(f"Size of cache folder ({CACHE}): {sizeof_fmt(size)}")


def sizeof_fmt(num, suffix: str = "iB", scale: float = 1024.0):
    for unit in ("", "k", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < scale:
            return f"{num:3.1f}{unit}{suffix}"
        num /= scale
    return f"{num:.1f}Yi{suffix}"


def get_geolocation(place: str) -> geometry.Polygon | geometry.MultiPolygon:
    enclosing_polygon: geopd.GeoDataFrame = ox.geocoder.geocode_to_gdf(place)
    return enclosing_polygon.union_all()


def align_line_points(line: geometry.LineString, points: geopd.GeoDataFrame) -> geopd.GeoDataFrame:
    points["__dist__"] = [line.line_locate_point(p) for p in points.geometry]
    points = points.sort_values(by="__dist__")
    lines = geopd.GeoDataFrame(
        [
            {
                "geometry": ops.substring(line, p1["__dist__"], p2["__dist__"]),
                "source": p1name,
                "target": p2name,
            }
            for (p1name, p1), (p2name, p2) in zip(
                points.iloc[0:-1].iterrows(), points.iloc[1:].iterrows()
            )
        ]
    )
    return lines


def load_regions(buffer_size: float | None = None, test: bool | None = None) -> geopd.GeoDataFrame:
    """Load the countries (and regions) with an optional buffer.

    For now Europe.
    """
    fn = "../data/prov_test.geojson" if test else "../data/regions_europe.geojson"
    regions = geopd.read_file(fn).rename(columns={"index": "code"})[["code", "geometry"]]

    if buffer_size is None or buffer_size <= 0.0:
        return regions

    reg_buffer = buffer(regions, buf_size=buffer_size)
    reg_buffer = split_cells(reg_buffer, cell_size=buffer_size)
    regions = geopd.GeoDataFrame(pd.concat([regions, reg_buffer]))

    return regions


def split_cells(areas: geopd.GeoDataFrame, cell_size: float = 2.0) -> geopd.GeoDataFrame:
    """Split large polygons in smaller cells."""

    bounds = areas.geometry.total_bounds
    nx = int((bounds[2] - bounds[0]) / cell_size) + 1
    ny = int((bounds[3] - bounds[1]) / cell_size) + 1

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
    tiles.geometry = tiles.intersection(areas.union_all(grid_size=0.01))
    # keep only the overlapping areas
    tiles = tiles[tiles.area > 0]

    return tiles


def buffer(data: geopd.GeoDataFrame, buf_size: float = 2.0) -> geopd.GeoDataFrame:
    """Create a buffer around a `GeoDataFrame`."""
    cache = Path("../data/regions_europe_buffer.geojson")

    if cache.is_file():
        return geopd.read_file(cache)

    buf = None
    for country in tqdm.tqdm(data.geometry):
        if buf is None:
            buf = country.buffer(buf_size)
        else:
            buf = buf.union(country.buffer(buf_size), grid_size=0.01)

    if buf is not None:
        buffer = geopd.GeoDataFrame(
            [{"region": "EU"}],
            geometry=[shapely.difference(buf, data.union_all(), grid_size=0.01)],
            crs=4326,
        )

    else:
        buffer = geopd.GeoDataFrame([], geometry=[])

    # cache it
    buffer.to_file(cache)
    return buffer


check_cache_size()
