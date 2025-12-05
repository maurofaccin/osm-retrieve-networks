"""Base functions and utilities.

Base functions for retrieving and post-processing data from OpenStreetMap
to build a network from the infrastructures.
"""

from __future__ import annotations

import logging
from collections import namedtuple
from dataclasses import dataclass, field
from itertools import combinations, product
from numbers import Number
from pathlib import Path
from typing import Literal, Self

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
from networkx import exception as nxexc
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
    edges.data = edges.data[
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
        edges.split_edges_when_touching()

    # get train stations
    log.info("Getting stations from OpenStreetMap.")
    nodes = retrieve_nodes({"railway": ["station", "halt", "stop"]}, polygon=enclosing_polygon)
    nodes.data = nodes.data[
        ["geometry", "name", "public_transport", "railway", "operator", "network", "wheelchair"]
    ]
    nodes.data["id"] = [f"{node_prefix}{i}" for i in range(len(nodes))]
    nodes.data = nodes.data.set_index("id", drop=True)
    nodes.data["__keep__"] = True

    # get nodes from edges and assigne source and target columns accordingly
    edgeBoundary_nodes = edges.nodes_from_boundaries(
        prefix="__tmp_node_", source_col="source", target_col="target"
    )
    edgeBoundary_nodes.data["__keep__"] = False
    edgeBoundary_nodes.data.loc[
        ~edgeBoundary_nodes.data.geometry.intersects(enclosing_polygon), "__keep__"
    ] = True

    # split lines passing through stations.
    # edges = split_lines_at_points(
    #     nodes=nodes, edges=edges, max_distance=100, max_distance_bounds=10
    # )
    edges = edges.to_meters()
    nodes = nodes.to_meters()
    edges.add_nodes(nodes, max_distance=300, max_distance_bounds=10)

    # Join temp and real nodes.
    nodes = nodes.append(edgeBoundary_nodes.to_meters())

    # Clean from unused nodes.
    nodes.data = nodes.data.loc[list(set(edges.source.tolist() + edges.target.tolist()))]

    # Clean form self loops.
    edges = edges.remove_self_loops()
    edges = edges.drop_duplicates_edges()
    edges.data.to_file(f"/tmp/edges_{node_prefix}.geojson")
    nodes.data.to_file(f"/tmp/nodes_{node_prefix}.geojson")

    g = graph_from_shortest_path(edges, nodes, grouping_distance=500)
    edges = Edges(g[0])
    nodes = Edges(g[1])

    edges = edges.to_degree()
    nodes = nodes.to_degree()
    graph = Graph(
        edges=edges.data,
        region=geopd.GeoDataFrame([{"region": 0}], geometry=[enclosing_polygon], crs=PRJ_DEG),
    )
    log.info("Get nodes from edges extremes.")
    graph.nodes = nodes.data.representative_point()
    #
    # log.info("Get power plants.")

    return graph


def osm_railways_old(
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
    log.info("Getting stations from OpenStreetMap.")
    nodes = retrieve_nodes({"railway": ["station", "halt", "stop"]}, polygon=enclosing_polygon)
    nodes = nodes[
        [
            "geometry",
            "name",
            "public_transport",
            "railway",
            "ref",
            "operator",
            "network",
            "wheelchair",
        ]
    ]
    nodes["id"] = [f"{node_prefix}{i}" for i in range(len(nodes))]
    nodes = nodes.set_index("id", drop=True)
    nodes["__keep__"] = True

    # get nodes from edges
    edgeBoundary_nodes, source_target = nodes_from_edges(edges, prefix="__tmp_node_")
    edgeBoundary_nodes["__keep__"] = False

    # assign source and target to edges.
    edges["source"] = source_target["source"]
    edges["target"] = source_target["target"]

    # split lines passing through stations.
    edges = split_lines_at_points(
        nodes=nodes, edges=edges, max_distance=100, max_distance_bounds=10
    )

    # Join temp and real nodes.
    nodes = geopd.GeoDataFrame(
        pd.concat([nodes, edgeBoundary_nodes]), geometry="geometry", crs=nodes.crs
    )

    # Clean from unused nodes.
    nodes = nodes.loc[list(set(edges.source.tolist() + edges.target.tolist()))]

    edges.to_file("/tmp/pippo.geojson")
    nodes.to_file("/tmp/pippo2.geojson")

    graph_from_shortest_path(edges, nodes, grouping_distance=150)
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


def mode(data: list):
    """Return the mode or a random entry."""
    data = pd.Series(data).dropna()
    if len(data) == 0:
        return None

    return pd.Series(data).value_counts().sort_values().index[-1]


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
) -> Nodes:
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
        return Nodes(geopd.GeoDataFrame([], geometry=[], crs=PRJ_DEG))
    nodes = geopd.GeoDataFrame(pd.concat(data), geometry="geometry", crs=PRJ_DEG).droplevel(0)
    nodes.columns = [c.replace(":", "_") for c in nodes.columns]
    nodes = nodes.dropna(how="all", axis=1)
    log.info(f"Collected {len(nodes)} nodes.")
    # Transform everything to nodes
    nodes.geometry = nodes.geometry.representative_point()
    # Drop duplicates
    return Nodes(nodes.loc[~nodes.index.duplicated(keep="first"), :])


def retrieve_edges(
    keys: dict,
    place: str | None = None,
    polygon: geometry.Polygon | geometry.MultiPolygon | None = None,
) -> Edges:
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
        return Edges(geopd.GeoDataFrame([], geometry=[]))
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
    # edges = edges.dropna(how="all", axis=1)

    # drop eventual duplicated edges.
    edges.geometry = edges.normalize()
    edges = edges.drop_duplicates("geometry")
    edges = edges.reset_index()
    log.info(f"Collected {len(edges)} edges.")
    return Edges(geopd.GeoDataFrame(edges, geometry="geometry", crs=PRJ_DEG))


# More utilities


@dataclass
class Base:
    data: geopd.GeoDataFrame

    def __post_init__(self):
        pass

    def to_meters(self) -> Self:
        self.data = self.data.to_crs(PRJ_MET)
        return self

    def to_degree(self, inline: bool = False) -> Self:
        self.data = self.data.to_crs(PRJ_DEG)
        return self

    def append(self, other: Self) -> Self:
        assert self.crs == other.crs
        self.data = geopd.GeoDataFrame(
            pd.concat([self.data, other.data]), geometry="geometry", crs=self.crs
        )
        return self

    def drop_duplicates(self) -> Self:
        # drop eventual duplicated items.
        self.data.geometry = self.data.normalize()
        self.data = self.data.drop_duplicates("geometry")
        return self

    @property
    def crs(self):
        return self.data.crs

    @property
    def index(self) -> pd.Index:
        return self.data.index

    def __len__(self):
        return len(self.data)


class Nodes(Base):
    def __post_init__(self):
        # Utilities and cache
        self._sindex_updated_ = False
        self._sindex_ = None

    def strtree(self) -> shapely.STRtree:
        """The `STRtree` of nodes."""
        if not self._sindex_updated_ or self._sindex_ is None:
            self._sindex_ = shapely.STRtree(self.data.geometry)
        return self._sindex_

    def aggregate(self, distance: float = 0.0) -> Self:
        self.data = cluster_points(
            self.data.reset_index(),
            distance=distance,
            aggfunc={"id": list} | {c: mode for c in set(self.data.columns) - {"geometry"}},
        )
        self.data.index = [id[0] for id in self.data["id"]]
        return self


class Edges(Base):
    def __post_init__(self) -> None:
        # Utilities and cache
        self._sindex_updated_ = {"e": False, "s": False, "t": False}
        self._sindex_ = {}

    def node(self, id):
        return self.data[(self.data["source"] == id) | (self.data["target"] == id)]

    def remove_self_loops(self) -> Self:
        self.data = self.data[self.data["source"] != self.data["target"]]
        return self

    @property
    def source(self):
        return self.data["source"]

    @property
    def target(self):
        return self.data["target"]

    def add_nodes(
        self, nodes: Nodes, max_distance: float = 0.0, max_distance_bounds: float = 0.0
    ) -> None:
        """Add nodes as source and targets and eventually split edges accordingly."""

        rename = {}
        for id, node in nodes.data.geometry.items():
            # check if close to boundary
            i_source = self.strtree("s").query(
                node, predicate="dwithin", distance=max_distance_bounds
            )
            i_target = self.strtree("t").query(
                node, predicate="dwithin", distance=max_distance_bounds
            )
            for ied in i_source:
                rename[self.data.iloc[ied]["source"]] = id
            for ied in i_target:
                rename[self.data.iloc[ied]["target"]] = id

            # Closest node to source
            i_edges = self.strtree("e").query_nearest(
                node, max_distance=max_distance, all_matches=True
            )
            # Cycle over all closest edges
            for ied in i_edges:
                old_edge = self.data.iloc[ied]

                # split in the closest point
                node_pos = old_edge.geometry.project(node)
                if node_pos <= 1e-8:
                    rename[self.data.iloc[ied]["source"]] = id
                elif node_pos >= old_edge.geometry.length - 1e-8:
                    rename[self.data.iloc[ied]["target"]] = id
                else:
                    new_edge1 = old_edge.to_dict()
                    new_edge2 = old_edge.to_dict()
                    new_edge1["geometry"] = ops.substring(old_edge.geometry, 0, node_pos)
                    new_edge1["target"] = id
                    new_edge2["geometry"] = ops.substring(
                        old_edge.geometry, node_pos, old_edge.geometry.length
                    )
                    new_edge2["source"] = id

                    self.data = geopd.GeoDataFrame(
                        pd.concat(
                            [self.data.drop(index=[ied]), pd.DataFrame([new_edge1, new_edge2])],
                            ignore_index=True,
                        )
                    ).set_crs(self.data.crs, allow_override=True, inplace=True)
                    self._sindex_updated_["e"] = False
                    self._sindex_updated_["s"] = False
                    self._sindex_updated_["t"] = False

        self.data["source"] = self.data["source"].replace(rename)
        self.data["target"] = self.data["target"].replace(rename)

        self.drop_duplicates()

    def strtree(self, kind: Literal["e", "s", "t"] = "e") -> shapely.STRtree:
        """The `STRtree` of edges.

        e -> edges (whole lines)
        s -> source nodes
        t -> target nodes
        """
        if not self._sindex_updated_[kind]:
            if kind == "e":
                self._sindex_[kind] = shapely.STRtree(self.data.geometry)
            elif kind == "s":
                self._sindex_[kind] = shapely.STRtree([p.geoms[0] for p in self.data.boundary])
            elif kind == "t":
                self._sindex_[kind] = shapely.STRtree([p.geoms[1] for p in self.data.boundary])
            self._sindex_updated_[kind] = True
        return self._sindex_[kind]

    def split_edges_when_touching(self) -> Edges:
        """Joins edges at their intersection.

        If the extreme of an edge touches the another edge, the latter is splitted in that point

        Parameters
        ----------
        edges: geopd.GeoDataFrame :


        Returns
        -------

        """
        log.info("Splitting edges to connect when touching.")

        # Check if two edges touch each other and eventually split them at the intersection
        new_edges = {}
        for edge_id1, edge1 in self.data.geometry.items():
            # find touching edges.
            touches_s = set(self.strtree("s").query(edge1, predicate="intersects"))
            touches_s -= set(self.strtree("s").query(edge1.boundary, predicate="intersects"))
            touches_t = set(self.strtree("t").query(edge1, predicate="intersects"))
            touches_t -= set(self.strtree("t").query(edge1.boundary, predicate="intersects"))
            touches = touches_s | touches_t

            if len(touches) > 0:
                ll = shapely.MultiLineString(self.data.iloc[list(touches)]["geometry"].tolist())
                try:
                    new_edges[self.data.index[edge_id1]] = shapely.MultiLineString(
                        ops.split(edge1, ll)
                    )
                except ValueError:
                    # sometime they are overlapping
                    pass
                # no need to recompute strtree!

        new_edges = pd.Series(new_edges)
        self.data.loc[new_edges.index, "geometry"] = new_edges
        self.data.geometry = self.data.geometry.replace(new_edges)

        self.data = self.data.explode("geometry", ignore_index=True)

        self.drop_duplicates()

    def nodes_from_boundaries(
        self, prefix: str, source_col: str | None = None, target_col: str | None = None
    ) -> Nodes:
        """Extract nodes at the boundaries of each edge.

        Aggregate overlapping nodes.
        """
        edgeBoundary_nodes = self.data.boundary
        edgeBoundary_nodes = geopd.GeoDataFrame(
            pd.DataFrame(
                {
                    "source": [f"s{i}" for i in range(len(edgeBoundary_nodes))],
                    "target": [f"t{i}" for i in range(len(edgeBoundary_nodes))],
                    "geometry": edgeBoundary_nodes,
                }
            ),
            geometry="geometry",
            crs=self.crs,
        )

        exploded = edgeBoundary_nodes.explode()
        exploded["id"] = (
            exploded["source"].iloc[0::2].tolist() + exploded["target"].iloc[1::2].tolist()
        )
        exploded["id"] = [
            e[label] for _, e in edgeBoundary_nodes.iterrows() for label in ["source", "target"]
        ]

        # find clusters within very short distance
        clustered = cluster_points(
            geopd.GeoDataFrame(exploded[["geometry", "id"]]),
            distance=1e-5,
            aggfunc={"id": [list, "first"]},
        )
        clustered = clustered.rename(columns={("id", "list"): "idlist", ("id", "first"): "id"})
        cluster_map = {
            node: prefix + str(ic)
            for ic, cluster in enumerate(clustered["idlist"])
            for node in cluster
        }

        edgeBoundary_nodes["source"] = edgeBoundary_nodes["source"].map(cluster_map)
        edgeBoundary_nodes["target"] = edgeBoundary_nodes["target"].map(cluster_map)

        clustered["id"] = clustered["id"].map(cluster_map)

        if source_col is not None:
            self.data[source_col] = edgeBoundary_nodes["source"]
        if target_col is not None:
            self.data[target_col] = edgeBoundary_nodes["target"]

        return Nodes(clustered.drop(columns="idlist").set_index("id", drop=True))

    def drop_duplicates_edges(self) -> Self:
        # remove multiple paths between the same nodes (usually parallel paths.)
        self.data["__idx__"] = pd.Index(
            [frozenset([e["source"], e["target"]]) for _, e in self.data.iterrows()]
        )
        self.data = self.data.drop_duplicates("__idx__").drop(columns=["__idx__"])

        return self


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

        self.nodes, source_target = nodes_from_edges(self.edges, prefix=node_prefix)
        self.edges["source"] = source_target["source"]
        self.edges["target"] = source_target["target"]

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

        1. Find edges in self that covers edges in other
        2. Find edges in other that covers edges is self
        3. Remove them (identical edges should be taken only once)
        4. Rename the nodes.
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


def graph_from_shortest_path(
    edges: Edges, nodes: Nodes, tokeep: str = "__keep__", grouping_distance: float = 0.0
) -> tuple[geopd.GeoDataFrame, geopd.GeoDataFrame]:
    """From nodes build the network following shortest paths if they do not overlap."""
    points = nodes.data[nodes.data[tokeep]]
    pedges = edges.data
    pedges["weight"] = pedges.length

    graph = nx.from_pandas_edgelist(pedges[["source", "target", "weight"]], edge_attr=True)
    # Aggregate if possible
    if grouping_distance > 0.0:
        points = cluster_points(
            points.reset_index(),
            distance=grouping_distance,
            aggfunc={"id": list} | {c: mode for c in set(points.columns) - {"geometry"}},
        )
        for _, point in points.iterrows():
            graph.add_edges_from(nx.complete_graph(point["id"]).edges(), weight=1e-10)
        points.index = pd.Index([min(i) for _, i in points["id"].items()])
    else:
        points["id"] = [[idx] for idx in points.index]

    # Use `frozenset` to check for undirected edges.
    pedges.index = pd.Index(
        [frozenset([e["source"], e["target"]]) for _, e in edges.data.iterrows()]
    )
    ids = []
    for p1, p2 in combinations(points.index, 2):
        d = []
        for pp1, pp2 in product(points.loc[p1, "id"], points.loc[p2, "id"]):
            try:
                d.append(nx.shortest_path(graph, pp1, pp2, "weight"))
            except nxexc.NetworkXNoPath:
                pass

        if len(d) == 0:
            continue

        shortest_path = min(d, key=lambda x: nx.path_weight(graph, x, "weight"))

        nodes_along_path = nodes.data.loc[shortest_path[1:-1], "geometry"].union_all()
        nodes_exept_boundary = points.geometry.drop([p1, p2]).union_all()

        # check if this path pass through other points.
        if len(points.index.intersection(shortest_path[1:-1])) == 0:
            if nodes_along_path.distance(nodes_exept_boundary) < 800:
                continue
            ids += [
                {
                    "edge": frozenset([e1, e2]),
                    "__id__": len(ids),
                    "source": shortest_path[0],
                    "target": shortest_path[-1],
                }
                for e1, e2 in zip(shortest_path[:-1], shortest_path[1:])
            ]

    new_edges = pd.DataFrame(ids)
    new_edges = geopd.GeoDataFrame(
        pd.concat(
            [
                new_edges,
                pedges.drop(columns=["source", "target"]).loc[new_edges["edge"]].reset_index(),
            ],
            axis=1,
        ),
        crs=edges.crs,
    )

    merged = new_edges.dissolve("__id__")
    merged.geometry = merged.geometry.line_merge()
    merged["type"] = merged.geometry.geom_type
    merged.to_file("/tmp/merged.geojson")

    return merged, points


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
    nodes: geopd.GeoDataFrame,
    edges: geopd.GeoDataFrame,
    max_distance: float = 0.0,
    max_distance_bounds: float = 0.0,
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
    mnodes = nodes.copy().to_crs(PRJ_MET)
    medges = Edges(edges).to_meters()

    medges.add_nodes(mnodes, max_distance=max_distance, max_distance_bounds=max_distance_bounds)
    return medges.to_degree().data

    Cache = namedtuple(
        "Cache", field_names=["toremove", "newedges", "usednodes"], defaults=[[], [], []]
    )

    cache = Cache()
    to_repeat = False

    for id, mnode in mnodes.geometry.items():
        # check if close to boundary
        if max_distance_bounds > 0.0:
            i_source = source_sindex.query(mnode, predicate="within", distance=10)
            i_target = target_sindex.query(mnode, predicate="within", distance=10)

            for ied in i_source:
                if ied in cache.toremove:
                    to_repeat = True
                else:
                    medges.loc[medges.index[ied], "source"] = id
            for ied in i_target:
                if ied in cache.toremove:
                    to_repeat = True
                else:
                    medges.loc[medges.index[ied], "target"] = id

        # closest node to source
        i_edges = edges_sindex.query_nearest(mnode, max_distance=max_distance, all_matches=True)

        for ied in i_edges:
            # if this edge has already be used,
            # skip it. We will check it next iteration
            if ied in cache.toremove:
                continue

            old_edge = medges.iloc[ied]
            new_edge1 = old_edge.to_dict()
            new_edge2 = old_edge.to_dict()

            # split in the closest point
            node_pos = old_edge.geometry.project(mnode)

            if ied in cache.toremove:
                to_repeat = True
                continue
            if node_pos <= 1e-5:
                medges.loc[medges.index[ied], "source"] = id
            elif node_pos >= old_edge.geometry.length - 1e-5:
                medges.loc[medges.index[ied], "target"] = id
            else:
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
    if to_repeat > 0:
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
    points: geopd.GeoDataFrame, distance: float = 100, column: str = "__node_clusters__", **kwargs
) -> geopd.GeoDataFrame:
    """Cluster nodes that are within `distance` (meters). Based on the `DBSCAN`.

    This adds a column to the `GeoDataFrame` with the labels of each cluster.
    """
    data_in_meters = points.geometry.to_crs(PRJ_MET)
    labels = cluster.DBSCAN(eps=distance, min_samples=2).fit_predict(
        [[p.x, p.y] for p in data_in_meters]
    )
    solitons = labels == -1
    maxlabels = labels.max()
    labels[solitons] = np.arange(sum(solitons)) + (maxlabels + 1)
    points[column] = labels
    return points.dissolve(by=column, **kwargs).reset_index(drop=True)


def nodes_from_edges(
    edges: geopd.GeoDataFrame, prefix: str
) -> tuple[geopd.GeoDataFrame, pd.DataFrame]:
    """Extract nodes at the boundaries of each edge.

    Aggregate overlapping nodes.
    """
    edgeBoundary_nodes = edges.boundary
    edgeBoundary_nodes = geopd.GeoDataFrame(
        pd.DataFrame(
            {
                "source": [f"s{i}" for i in range(len(edgeBoundary_nodes))],
                "target": [f"t{i}" for i in range(len(edgeBoundary_nodes))],
                "geometry": edgeBoundary_nodes,
            }
        ),
        geometry="geometry",
        crs=edges.crs,
    )
    exploded = edgeBoundary_nodes.explode()
    exploded["id"] = exploded["source"].iloc[0::2].tolist() + exploded["target"].iloc[1::2].tolist()
    exploded["id"] = [
        e[label] for _, e in edgeBoundary_nodes.iterrows() for label in ["source", "target"]
    ]

    # find clusters within very short distance
    clustered = cluster_points(
        geopd.GeoDataFrame(exploded[["geometry", "id"]]),
        distance=1e-5,
        aggfunc={"id": [list, "first"]},
    )
    clustered = clustered.rename(columns={("id", "list"): "idlist", ("id", "first"): "id"})
    cluster_map = {
        node: prefix + str(ic) for ic, cluster in enumerate(clustered["idlist"]) for node in cluster
    }

    edgeBoundary_nodes["source"] = edgeBoundary_nodes["source"].map(cluster_map)
    edgeBoundary_nodes["target"] = edgeBoundary_nodes["target"].map(cluster_map)
    clustered["id"] = clustered["id"].map(cluster_map)

    return clustered.drop(columns="idlist").set_index("id", drop=True), edgeBoundary_nodes[
        ["source", "target"]
    ]


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


def split_edges_when_touching(edges: Edges) -> Edges:
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


def merge(paths: str[Path]) -> Graph | None:
    """Merge two or more graphs from a list of paths."""
    # Load the countries

    graph: Graph | None = None

    for pl_path in paths:
        log.info(pl_path)

        pl = Graph.read(pl_path)

        if len(pl) == 0:
            continue

        if graph is None:
            graph = Graph(edges=pl.edges.copy(), region=pl.region)
            graph.nodes = pl.nodes.copy()
        else:
            graph = graph.merge_edges(pl)

    return graph


check_cache_size()
