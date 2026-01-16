"""Base functions and utilities.

Base functions for retrieving and postprocessing data from OpenStreetMap
to build a network from the infra-structures.
"""

from __future__ import annotations

import logging
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations, product
from numbers import Number
from pathlib import Path
from typing import Hashable, Literal, Self

import geopandas as gpd
import igraph as ig
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
from sklearn.cluster import DBSCAN
from tqdm.contrib.concurrent import process_map

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
    place: geometry.Polygon | geometry.MultiPolygon,
    osm_dump_file: str | Path | tuple[str | Path, str | Path],
    node_prefix: str = "",
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
    # Retrieving the enclosing polygon.
    enclosing_polygon = get_geolocation(place=place) if isinstance(place, str) else place

    # get file location
    if isinstance(osm_dump_file, (str, Path)):
        osm_dump_file_nodes = osm_dump_file
        osm_dump_file_edges = osm_dump_file
    else:
        osm_dump_file_nodes, osm_dump_file_edges = osm_dump_file

    # Retrieve lines from `OpenStreetMap`.
    graph = retrieve_edges(
        osm_dump_file=osm_dump_file_edges,
        # keys={
        #     "railway": ["rail", "construction", "preserved", "narrow_gauge"],
        #     "route": ["train"],
        #     "construction": ["rail", "railway"],
        #     "construction:railway": ["rail", "railway"],
        # },
        polygon=enclosing_polygon,
        columns=[
            "id",
            "osm_id",
            "operator",
            "ref",
            "name",
            "electrified",
            "frequency",
            "gauge",
            "maxspeed",
            "geometry",
        ],
        node_prefix=f"__tmp_{node_prefix}",
        split_when_touching=True,
    )
    if len(graph.edges) == 0:
        return graph

    graph.nodes.data["__keep__"] = ~graph.nodes.data.geometry.intersects(enclosing_polygon)

    # get train stations
    log.info("Getting stations from OpenStreetMap.")
    nodes = retrieve_nodes(
        osm_dump_file=osm_dump_file_nodes,
        keys={"railway": ["station", "halt", "stop"]},
        polygon=enclosing_polygon,
        columns=[
            "geometry",
            "name",
            "osm_id",
            "public_transport",
            "railway",
            "operator",
            "network",
            "wheelchair",
            "uic_ref",
        ],
    )
    nodes.data["__keep__"] = True

    if len(nodes) == 0:
        return graph
    graph = graph.to_meters()
    # merge nodes within 10 meters
    graph = graph.aggregate_nodes(10)

    nodes = nodes.to_meters()

    # Add real stations and overwrite overlapping nodes.
    graph = graph.add_nodes(nodes, max_distance=300, max_distance_bounds=10)

    # Remove nodes that do not belong to any edge
    graph = graph.clean_disconnected_nodes()
    nodes = Nodes(nodes.data.loc[nodes.index.intersection(graph.nodes.index)])

    # merge stations into meta-stations
    aggregated = nodes.aggregate(distance=500).data["NODE_ID"]
    add_out_of_borders = graph.nodes.data[
        ~graph.nodes.data.geometry.within(graph.region.iloc[0].geometry)
    ].index
    add_out_of_borders = pd.Series([[x] for x in add_out_of_borders], index=add_out_of_borders)
    aggregated = pd.concat([aggregated, add_out_of_borders])

    # Add a clique between nodes/stops in the same station
    new_edges = new_cliques(nodes, aggregated)
    graph.edges = graph.edges.append(new_edges).drop_duplicated_edges()

    graph = graph_from_shortest_path(
        graph,
        metanodes=aggregated,
        avoid_distance=300,  # do not increase too much
        force_smooth=False,
    )
    return graph.to_degree()


def osm_powerlines(
    place: str | geometry.Polygon | geometry.MultiPolygon,
    substation_distance: float = 500,
    voltage_fillvalue: float | None = None,
    voltage_threshold: float = 1000,
    node_prefix: str = "",
) -> tuple[Graph, gpd.GeoDataFrame]:
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
        region=gpd.GeoDataFrame([{"region": 0}], geometry=[enclosing_polygon], crs=PRJ_DEG),
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
    osm_dump_file: str | Path,
    keys: dict | None = None,
    filename: str | None = None,
    polygon: geometry.Polygon | geometry.MultiPolygon | None = None,
    columns: list[str] | None = None,
    node_prefix: str = "NODE_",
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
    nodes = retrieve_data(
        osm_dump_file=osm_dump_file, keys=keys, polygon=polygon, columns=columns, layer="points"
    )
    if len(nodes) == 0:
        return Nodes()
    nodes.columns = [c.replace(":", "_") for c in nodes.columns]
    nodes = nodes.dropna(how="all", axis=1)
    log.info(f"Collected {len(nodes)} nodes.")
    # Transform everything to nodes
    nodes.geometry = nodes.geometry.representative_point()
    if columns is not None:
        cols = nodes.columns.intersection(columns)
        nodes = nodes[cols]
    # Drop duplicates
    nodes = nodes.loc[~nodes.index.duplicated(keep="first"), :]

    # Prepare new index
    nodes.index = pd.Index([f"{node_prefix}{i}" for i in range(len(nodes))])
    return Nodes(nodes).cleanup()


def retrieve_edges(
    osm_dump_file: str | Path,
    keys: dict | None = None,
    polygon: geometry.Polygon | geometry.MultiPolygon | None = None,
    columns: list[str] | None = None,
    node_prefix: str = "_tmp_",
    split_when_touching: bool | None = None,
) -> Graph:
    """Retrieve edge data from OpenStreetMap.

    Only `LineString` will be kept.

    Parameters
    ----------
    keys : dict
        The OpenStreetMap keys to use for download.
        The Union of the results are retrieved.
    place : str
        Retrieve data from this place: e.g. `Padova`, `London`, `France`…
    polygon: geometry.Polygon | geometry.MultiPolygon | None
        An enclosing polygon
    columns : list[str]
        The list of metadata to keep

    Returns
    -------
    edges : Edges
        The edges

    """
    log.info("Retrieving `LineString` data from OpenStreetMap.")
    data = retrieve_data(
        osm_dump_file=osm_dump_file, keys=keys, polygon=polygon, columns=columns, layer="lines"
    )

    # Build a dataframe
    if len(data) == 0:
        return Graph(
            Edges(),
            region=gpd.GeoDataFrame([{"region": node_prefix}], geometry=[polygon], crs=PRJ_DEG),
        )
    edges = Edges(data)
    edges = edges.cleanup()
    edges.data["name"] = range(len(edges))
    log.info(f"Collected {len(edges)} edges.")
    if columns is not None:
        cols = edges.data.columns.intersection(columns)
        edges.data = edges.data[cols]

    if split_when_touching:
        edges = edges.split_edges_when_touching()

    nodes = edges.nodes_from_boundaries(prefix=node_prefix)
    edges = edges.drop_duplicated_edges()
    return Graph(
        edges=edges,
        region=gpd.GeoDataFrame([{"region": node_prefix}], geometry=[polygon], crs=PRJ_DEG),
        nodes=nodes,
    )


def retrieve_data(
    osm_dump_file: str | Path,
    keys: dict | None = None,
    polygon: geometry.Polygon | geometry.MultiPolygon | None = None,
    columns: list[str] | None = None,
    layer: Literal["points", "lines"] = "points",
) -> gpd.GeoDataFrame:
    """Get the raw data."""
    # load data from osm dump file
    data = gpd.read_file(
        str(Path(osm_dump_file).expanduser()),
        layer=layer,
        # bbox=(12.5986, 47.719, 13.322, 48.1577),
        # bbox=(13, 47.5, 14, 49),
        # bbox=(13.0697, 47.8361, 13.4183, 48.0475),
        mask=polygon,
    ).explode()

    # expand tags
    additional_tags = data["other_tags"].str.extractall(r'"(.*?)"=>"(.*?)"')
    additional_tags = additional_tags.reset_index(level=0)
    additional_tags = additional_tags.pivot(values=1, columns=0, index="level_0")
    data = gpd.GeoDataFrame(
        pd.concat([data, additional_tags], axis=1), geometry="geometry", crs=PRJ_DEG
    )

    # filter rows from `keys`
    if keys is not None:
        mask = pd.Series(np.zeros(len(data), dtype=bool))
        for k, v in keys.items():
            if k in data.columns:
                mask = mask | (data[k].isin(v))
        data = data.loc[mask, :]

    return gpd.GeoDataFrame(data, geometry="geometry", crs=PRJ_DEG)


# More utilities


@dataclass
class Base:
    data: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame([]))

    def __post_init__(self):
        pass

    def iterrows(self):
        yield from self.data.iterrows()

    def to_meters(self) -> Self:
        data = self.data.to_crs(PRJ_MET)
        return type(self)(data)

    def to_degree(self, inline: bool = False) -> Self:
        data = self.data.to_crs(PRJ_DEG)
        return type(self)(data)

    def drop(self, *args, **kwargs) -> Self:
        return type(self)(self.data.drop(*args, **kwargs))

    def append(self, other: Self) -> Self:
        if len(self) > 0:
            assert self.crs == other.crs
            crs = self.crs
        else:
            crs = other.crs

        data = gpd.GeoDataFrame(pd.concat([self.data, other.data]), geometry="geometry", crs=crs)

        # deduplicate
        data = data[~data.geometry.duplicated(keep="first")]
        return type(self)(data)

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

    def __str__(self):
        return f"""{type(self)}

    """ + str(self.data)


class Nodes(Base):
    def __post_init__(self):
        # Utilities and cache
        self._sindex_updated_ = False
        self._sindex_ = None
        self.data.index.name = "NODE_ID"

    def strtree(self) -> shapely.STRtree:
        """The `STRtree` of nodes."""
        if not self._sindex_updated_ or self._sindex_ is None:
            self._sindex_ = shapely.STRtree(self.data.geometry)
        return self._sindex_

    def aggregate(self, distance: float = 0.0) -> Nodes:
        data = self.data.reset_index()
        index_name = data.columns[0]
        data = cluster_points(
            data,
            distance=distance,
            aggfunc={index_name: list} | {c: mode for c in set(self.data.columns) - {"geometry"}},
        )
        data.index = pd.Index([id[0] for id in data[index_name]])
        return Nodes(data)

    def cleanup(self) -> Nodes:
        return Nodes(self.data.drop_duplicates("geometry", keep="first"))

    def subset(self, subset: list) -> Nodes:
        nodes = self.data.loc[list(set(subset))]
        return Nodes(nodes)


class Edges(Base):
    """Edge data.

    Warning: the columns `source`, `target` and `weight` will be overwritten.
    """

    def __post_init__(self) -> None:
        # Utilities and cache
        self._sindex_updated_ = {"e": False, "s": False, "t": False}
        self._sindex_ = {}
        self._mapping_ = {}

    def get_link(self, source: Hashable, target: Hashable) -> pd.Series | None:
        """Return the link data if it exists, None otherwise,"""
        if len(self._mapping_) == 0:
            self._mapping_ = {frozenset((e.source, e.target)): ie for ie, e in self.data.iterrows()}
        edge = frozenset((source, target))

        if edge not in self._mapping_:
            return None
        return self.data.loc[self._mapping_[edge]]

    def node(self, id):
        """Return edges incident in given node."""
        return self.data[(self.source == id) | (self.target == id)]

    def remove_self_loops(self) -> Self:
        self.data = self.data[self.source != self.target]
        self._mapping_ = {}
        return self

    @property
    def source(self):
        return self.data["source"]

    @property
    def target(self):
        return self.data["target"]

    def add_nodes(
        self, nodes: Nodes, max_distance: float = 0.0, max_distance_bounds: float = 0.0
    ) -> Edges:
        """Add nodes as source and targets and eventually split edges accordingly."""
        log.info(f"Adding {len(nodes)} nodes to the edges within {max_distance} meters")

        edges = process_map(
            partial(
                __add_node_to_edges__,
                edges=self,
                distance=max_distance,
                border_distance=max_distance_bounds,
                nodeid_col=nodes.data.index.name,
            ),
            list(nodes.data.iterrows()),
            chunksize=10,
        )

        # Collect all actions to be performed `split`, `source` and `target`.
        actions = pd.DataFrame(edges).set_index("nodeid", drop=True).explode("edgeid")
        new_edges = self.data

        # Rename nodes (on the boundaries)
        rename = {
            action["rename"]: nid
            for nid, action in actions.iterrows()
            if action["rename"] is not None
        }
        new_edges["source"] = new_edges["source"].replace(rename)
        new_edges["target"] = new_edges["target"].replace(rename)

        # Split edges (not on boundaries)
        actions = actions[actions["type"] == "split"]
        # nodes that will split the edges.
        splitters = nodes.data.loc[actions.index]
        splitters["edgeid"] = actions["edgeid"]
        # change to edges
        splitters = (
            splitters.reset_index().groupby("edgeid").agg({"nodeid": list, "geometry": list})
        )

        new_edges = new_edges.loc[~new_edges.index.duplicated()]

        # add source and target
        for bound in ["source", "target"]:
            splitters[bound] = new_edges.loc[splitters.index, bound]
            splitters[bound + "_geom"] = nodes.data.loc[
                splitters[bound], "geometry"
            ].values  # add column ignoring index

        new_edges = Edges(
            gpd.GeoDataFrame(
                pd.concat(
                    [
                        split_edge_at_points(
                            new_edges.loc[edgeid], splitter, self.crs, point_names="nodeid"
                        )
                        for edgeid, splitter in splitters.iterrows()
                    ]
                ),
                crs=self.crs,
            )
        )
        new_edges = new_edges.append(self.drop(index=splitters.index))
        return new_edges

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

        If the extreme of an edge touches another edge, the latter is split in that point.

        Returns
        -------

        """
        log.info("Splitting edges to connect when touching.")

        # Check if two edges touch each other and eventually split them at the intersection
        new_data = self.data

        new_edges = process_map(
            partial(__split_edge__, self), new_data.geometry.tolist(), chunksize=10
        )

        new_data.geometry = list(new_edges)
        new_data = new_data.explode("geometry", ignore_index=True)

        return Edges(new_data).cleanup()

    def nodes_from_boundaries(self, prefix: str) -> Nodes:
        """Extract nodes at the boundaries of each edge.

        Aggregate overlapping nodes.
        """
        log.info("Get nodes from edge boundaries")
        edgeBoundary_nodes = self.data.boundary
        edgeBoundary_nodes = gpd.GeoDataFrame(
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

        exploded = edgeBoundary_nodes.explode().drop_duplicates("geometry")
        exploded["id"] = [f"{prefix}_{i}" for i in range(len(exploded))]
        log.info(f"Got {len(exploded)} additional nodes")

        log.info("Remove overlapping nodes. (Keep the first name)")
        rename = exploded.set_index("geometry", drop=True)["id"].to_dict()
        edgeBoundary_nodes["source"] = [
            rename.get(edge.geometry.geoms[0], edge["source"])
            for _, edge in edgeBoundary_nodes.iterrows()
        ]
        edgeBoundary_nodes["target"] = [
            rename.get(edge.geometry.geoms[1], edge["target"])
            for _, edge in edgeBoundary_nodes.iterrows()
        ]
        log.info(f"Still {len(exploded)} nodes")

        self.data["source"] = edgeBoundary_nodes["source"]
        self.data["target"] = edgeBoundary_nodes["target"]

        return Nodes(exploded.drop(columns=["source", "target"]).set_index("id", drop=True))

    def drop_duplicated_edges(self) -> Edges:
        # remove multiple paths between the same nodes (usually parallel paths.)
        dup_indx = pd.Index([frozenset([e.source, e.target]) for _, e in self.data.iterrows()])
        return Edges(self.data.loc[~dup_indx.duplicated()])

    def rename(self, rename_dict: dict) -> Edges:
        """Rename source and target from dict.

        Nodes in keys will be replaced with the corresponding value.
        """
        edges = self.data
        edges.source = edges.source.replace(rename_dict)
        edges.target = edges.target.replace(rename_dict)
        return Edges(edges)

    def cleanup(self) -> Edges:
        # remove MultiLineString if possible
        self.data.geometry = self.data.normalize().line_merge()

        if "source" in self.data.columns or "target" in self.data.columns:
            log.warning("Cleaning but not exploding. (`source` and `target` should be recomputed)")
            self.data = self.data.loc[self.data.geometry.geom_type == "LineString"]
        else:
            self.data = self.data.loc[
                self.data.geometry.geom_type.isin({"LineString", "MultiLineString"})
            ].explode("geometry", ignore_index=True)

        # Drop cycles
        self.data = self.data.loc[[len(p.geoms) == 2 for p in self.data.boundary]]

        self.drop_duplicates()
        return self


@dataclass
class Graph:
    edges: Edges
    region: gpd.GeoDataFrame
    nodes: Nodes = field(default_factory=lambda: Nodes())

    @property
    def region_shape(self) -> shapely.Geometry:
        """`MultiPolygon` corresponding to the union of all shapes."""
        return self.region.union_all(method="coverage")

    def nodes_from_edges(self, node_prefix: str = "") -> Graph:
        """Extract and deduplicate the nodes at the edge extremes.

        Update edges to include `source` and `target`

        Parameters
        ----------
        edges : geopd.GeoDataFrame
            The edges

        Returns
        -------
        nodes : GeoDataFrame
            All nodes
        edges : GeoDataFrame
            Same edges with columns `source_col` and `target_col` added

        """
        log.info("Extracting nodes from edges.")
        new_nodes = self.edges.nodes_from_boundaries(prefix=node_prefix)
        return Graph(edges=self.edges, region=self.region, nodes=new_nodes)

    def to_meters(self) -> Graph:
        return Graph(
            edges=self.edges.to_meters(),
            region=self.region.to_crs(PRJ_MET),
            nodes=self.nodes.to_meters(),
        )

    def to_degree(self) -> Graph:
        return Graph(
            edges=self.edges.to_degree(),
            region=self.region.to_crs(PRJ_DEG),
            nodes=self.nodes.to_degree(),
        )

    def index(self) -> pd.Index:
        return self.edges.index

    def add_nodes(
        self, nodes: Nodes, max_distance: float = 0.0, max_distance_bounds: float = 0.0
    ) -> Graph:
        """Add nodes.

        Add nodes as source and targets and eventually split edges accordingly.
        Will overwrite existing nodes if overlapping.
        """
        log.info(f"Adding {len(nodes)} nodes to the edges within {max_distance} meters")

        edges = pd.DataFrame(
            process_map(
                partial(
                    __add_node_to_edges__,
                    edges=self.edges,
                    distance=max_distance,
                    border_distance=max_distance_bounds,
                    nodeid_col=nodes.data.index.name,
                ),
                list(nodes.data.iterrows()),
                chunksize=10,
            )
        )
        if "edgeid" not in edges.columns:
            log.warning("No nodes where close to an edge (check crs)")
            return self
        # no edges close to the node
        edges = edges.dropna(subset="edgeid")

        # Collect all actions to be performed `split`, `source` and `target`.
        actions = pd.DataFrame(edges).set_index("nodeid", drop=True).explode("edgeid")
        new_edges = self.edges.data
        new_nodes = gpd.GeoDataFrame(
            pd.concat([nodes.data, self.nodes.data]), geometry="geometry", crs=nodes.crs
        )

        # Rename nodes (on the boundaries)
        rename = {
            action["rename"]: nid
            for nid, action in actions.iterrows()
            if action["rename"] is not None
        }
        new_edges["source"] = new_edges["source"].replace(rename)
        new_edges["target"] = new_edges["target"].replace(rename)
        new_nodes = new_nodes.drop(index=list(rename.keys()))

        # Split edges (not on boundaries)
        actions = actions[actions["type"] == "split"]
        # nodes that will split the edges.
        splitters = nodes.data.loc[actions.index]
        splitters["edgeid"] = actions["edgeid"]
        # change to edges
        splitters = (
            splitters.reset_index().groupby("edgeid").agg({"nodeid": list, "geometry": list})
        )

        new_edges = new_edges.loc[~new_edges.index.duplicated()]

        # add source and target
        for bound in ["source", "target"]:
            splitters[bound] = new_edges.loc[splitters.index, bound]
            splitters[bound + "_geom"] = new_nodes.loc[
                splitters[bound], "geometry"
            ].values  # add column ignoring index

        new_edges = Edges(
            gpd.GeoDataFrame(
                pd.concat(
                    [
                        split_edge_at_points(
                            new_edges.loc[edgeid], splitter, self.edges.crs, point_names="nodeid"
                        )
                        for edgeid, splitter in splitters.iterrows()
                    ]
                ),
                crs=self.edges.crs,
            )
        )
        new_edges = new_edges.append(self.edges.drop(index=splitters.index))

        # Only nodes involved in links
        new_nodes_idx = list(set(new_edges.data.source) | set(new_edges.data.target))
        return Graph(edges=new_edges, region=self.region, nodes=Nodes(new_nodes))

    def aggregate_nodes(self, distance: float) -> Graph:
        """Aggregate nodes within distance."""
        nodes = self.nodes.aggregate(distance=distance).data["NODE_ID"]
        nodes = nodes.loc[[len(x) > 1 for x in nodes]]
        rename = {k: v for v, nns in nodes.items() for k in nns if k != v}

        new_nodes = self.nodes.drop(index=list(rename.keys()))
        new_edges = self.edges.rename(rename)
        return Graph(edges=new_edges, region=self.region, nodes=new_nodes)

    def filter_edges(self, keep_ids: list | np.ndarray | pd.Index) -> Graph:
        """Return a new graph with only `keep_ids` edges."""
        graph = Graph(edges=Edges(self.edges.data.loc[keep_ids]), region=self.region)
        graph.nodes.append(
            Nodes(
                self.nodes.data.loc[
                    pd.concat([graph.edges.source, graph.edges.target]).drop_duplicates()
                ]
            )
        )
        return graph

    def filter_nodes(self, keep_ids: list | np.ndarray | pd.Index) -> Graph:
        """Return a new graph with only the mentioned nodes."""
        graph = Graph(
            edges=Edges(
                self.edges.data.loc[
                    (self.edges.source.isin(keep_ids)) & (self.edges.target.isin(keep_ids))
                ]
            ),
            region=self.region,
        )
        graph.nodes.append(Nodes(self.nodes.data.loc[keep_ids]))
        return graph

    def intersect(self, other: Graph, return_same: bool | None = False) -> list:
        """Return edges in both (deduplicated).

        - For same edges: return one (if `return_same` is True)
        - For one edge covering the other: return smaller
        - For one edge present in only one graph: return it
        """
        self.edges.data.geometry = self.edges.data.normalize()
        other.edges.data.geometry = other.edges.data.normalize()

        sindex = shapely.STRtree(other.edges.data.geometry)

        edges = {
            id: set(sindex.query(edge.geometry, predicate="within"))
            for id, edge in self.edges.data.iterrows()
            if len(set(sindex.query(edge.geometry, predicate="contains"))) == 0
        }
        if return_same:
            edges |= {
                id: set(sindex.query_nearest(edge.geometry, max_distance=1e-10, all_matches=False))
                for id, edge in self.edges.data.iterrows()
            }

        return list(edges.keys())

    def split_edges(self, other: Graph):
        """Split edges in those intersecting the other.region and the others."""
        # Edges in self, crossing to other.region
        sindex = shapely.STRtree(self.edges.data.geometry)
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
        all_nodes = pd.concat([self.nodes.data, other.nodes.data])
        all_edges = pd.concat([self.edges.data, other.edges.data], ignore_index=True)
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
                new_lines = align_line_points(edge, gpd.GeoDataFrame(involved_nodes))
                for k, v in all_edges.loc[edgeid].items():
                    if k not in {"geometry", "source", "target"}:
                        new_lines[k] = v
                transform_edges.append(new_lines)
                keep_edges -= involved

            else:
                keep_edges.add(edgeid)

        # Almost overlapping nodes:
        sindex = shapely.STRtree(self.nodes.data.geometry)
        rename_overlapping_nodes = {
            id_node: self.nodes.index[sindex.query_nearest(node, max_distance=1e-6)]
            for id_node, node in other.nodes.data.geometry.items()
        }
        rename_overlapping_nodes = {
            k: v[0] for k, v in rename_overlapping_nodes.items() if len(v) > 0
        }
        rename_nodes |= rename_overlapping_nodes

        g = Graph(
            edges=Edges(
                gpd.GeoDataFrame(
                    pd.concat([all_edges.iloc[list(keep_edges)]] + transform_edges),
                    crs=self.edges.crs,
                )
            ),
            region=gpd.GeoDataFrame(pd.concat([self.region, other.region]), crs=self.region.crs),
        )
        g.edges.data.source = g.edges.data["source"].apply(lambda x: rename_nodes.get(x, x))
        g.edges.data.target = g.edges.data["target"].apply(lambda x: rename_nodes.get(x, x))
        all_nodes = all_nodes.rename(index=rename_nodes)
        all_nodes = all_nodes[~all_nodes.index.duplicated()]

        g.nodes.append(Nodes(gpd.GeoDataFrame(all_nodes)))
        return g

    def intersecting_edges(self, other: Graph) -> dict:
        self.edges.data.geometry = self.edges.data.normalize()
        other.edges.data.geometry = other.edges.data.normalize()

        def _normalize_(data: dict) -> dict:
            assert all([len(v) <= 1 for v in data.values()])
            return {k: v for k, v in data.items() if len(v) == 1}

        otree = other.edges.strtree("e")
        edges = {}

        for idedge, edge in self.edges.data.geometry.items():
            within = otree.query(edge, predicate="within")
            contains = otree.query(edge, predicate="contains")
            equals = {e for e in within if edge.equals(other.edges.data.loc[e, "geometry"])}
            overlaps = otree.query(edge, predicate="overlaps")

            # within -= equals
            # contains -= equals

            status = {}
            if len(equals) >= 1:
                assert len(equals) == 1
                status["equals"] = list(equals)[0]
            if len(within) > 0:
                assert len(within) == 1
                status["within"] = list(within)[0]
            if len(contains) > 0:
                assert len(contains) == 1
                status["contains"] = list(contains)[0]
            if len(overlaps) > 0:
                assert len(overlaps) == 1
                status["overlaps"] = list(overlaps)[0]
            if len(status) > 0:
                edges[idedge] = status
                print(idedge, status)

        return edges

    def blend(self, other: Graph) -> Graph:
        """Blend and merge two graphs into one.

        1. Check edges that are intersecting from the two sets.
        2. Merge those edges
        3. Rename nodes
        """

        for e1, e2 in self.intersecting_edges(other).items():
            print(e1)
            print(e2)
            print()
        raise NotImplementedError
        return self

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
        merged = gpd.GeoDataFrame(
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
        sindex = shapely.STRtree(other.nodes.data.geometry)
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
        sindex = shapely.STRtree(self.nodes.data.geometry)
        rename |= {
            node: self.nodes.index[
                sindex.query_nearest(g_outer2.nodes.loc[node].geometry, max_distance=1e-10)
            ]
            for node in wrong_region
        }
        rename = {k: v[0] for k, v in rename.items() if len(v) > 0}

        mgraph = Graph(
            edges=Edges(merged.reset_index(drop=True)),
            region=gpd.GeoDataFrame(pd.concat([self.region, other.region]), crs=self.region.crs),
        )
        mgraph.edges.index.name = "index"
        mgraph.edges.data["source"] = mgraph.edges.data["source"].map(lambda x: rename.get(x, x))
        mgraph.edges.data["target"] = mgraph.edges.data["target"].map(lambda x: rename.get(x, x))
        mgraph.nodes = Nodes(
            gpd.GeoDataFrame(
                pd.concat(
                    [
                        self.nodes.data.loc[self.nodes.index.difference(pd.Index(rename))],
                        other.nodes.data.loc[other.nodes.index.difference(pd.Index(rename))],
                    ]
                ),
                crs=self.edges.crs,
            )
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
        edges = gpd.read_file(path, layer="edges")
        nodes = gpd.read_file(path, layer="nodes")  # .set_index("id", drop=True)
        region = gpd.read_file(path, layer="region")

        g = Graph(edges=Edges(edges), region=region)
        g.nodes.append(Nodes(nodes))

        return g

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        path.unlink(missing_ok=True)
        self.edges.data.drop(columns=["index"], errors="ignore").to_file(
            path, driver="GPKG", layer="edges"
        )
        self.nodes.data.to_file(path, driver="GPKG", layer="nodes")
        self.region.to_file(path, driver="GPKG", layer="region")

    def graph_ig(self):
        """Return a igraph Graph."""
        return ig.Graph.TupleList(
            self.edges.data[["source", "target", "weight"]].itertuples(index=False),
            weights=True,
            directed=False,
        )

    def shortest_paths(self, subset: pd.Index | None = None):
        """Yields the shortest paths between any two nodes."""
        graph_ig = self.graph_ig()
        nodes = np.array(graph_ig.vs["name"])

        if subset is None:
            subset = self.nodes.index
        assert len(set(subset) - set(nodes)) == 0, f"Not included {set(subset) - set(nodes)}"

        for node in tqdm.tqdm(subset, desc="SPs"):
            for path in graph_ig.get_shortest_paths(
                node, [x for x in subset if x > node], weights="weight", mode="all"
            ):
                if len(path) == 0:
                    continue
                yield nodes[path]

    def clean_disconnected_nodes(self) -> Graph:
        n = self.nodes.data
        n = n.loc[sorted(set(self.edges.data["source"]) | set(self.edges.data["target"]))]
        return Graph(edges=self.edges, region=self.region, nodes=Nodes(n))

    def plot(self, ax: axes.Axes, color: str = "r", text_args: dict | None = None):
        plotting.plot_polygon(self.region_shape, color=color, ax=ax, alpha=0.2)
        plotting.plot_line(self.edges.data.union_all(), color=color, ax=ax, alpha=0.3)
        if hasattr(self, "nodes"):
            plotting.plot_points(
                self.nodes.data.union_all(), ax=ax, color=color, alpha=0.5, markersize=10
            )

            for x, y, name in zip(
                self.nodes.data.geometry.x, self.nodes.data.geometry.y, self.nodes.index
            ):
                text_args = {} if text_args is None else text_args
                ax.annotate(name, (x, y), xytext=(x, y + 1), **text_args)
        ax.grid()


def _clean_voltage(voltage: pd.Series) -> pd.Series:
    """Clean the voltage column."""

    def _convert(cell: Number | str) -> Number | int | float:
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
    graph: Graph,
    metanodes: pd.Series | pd.Index,
    avoid_distance: float = 0.0,
    force_smooth: bool = False,
) -> Graph:
    """From nodes build the network following shortest paths if they do not overlap.

    WARNING: the column `weight` for edges will be overwritten.

    Parameters
    ----------
    graph: Graph
        the graph
    metanodes: pd.Series|pd.Index
        either the list of nodes to use to construct the shortest path graph (pd.Index) or the list of representative nodes with their corresponding set of nodes (pd.Series).
    avoid_distance: float
        path approaching other nodes to keep are discarded if they fall within this distance
    force_smooth: bool
        discard non-smooth paths
    """
    # nodes origin of links
    if isinstance(metanodes, pd.Index):
        points = graph.nodes.data.loc[metanodes]
        avoid = gpd.GeoDataFrame([])
        metanodes = pd.Series([[x] for x in metanodes], index=metanodes)
    elif isinstance(metanodes, pd.Series):
        points = graph.nodes.data.loc[metanodes.index]
        avoid = graph.nodes.data.loc[metanodes.explode()]

    edges = graph.edges
    edges.data["weight"] = edges.data.length

    iloc = {
        (e[side[0]], e[side[1]]): ie
        for ie, (_, e) in enumerate(edges.data.iterrows())
        for side in [("source", "target"), ("target", "source")]
    }

    # compute all paths
    paths = (
        __shortest_path__(
            graph,
            path,
            metanodes,
            nodes_to_avoid=avoid.index,
            avoid_within=avoid_distance,
            force_smooth=force_smooth,
        )
        for path in graph.shortest_paths(subset=points.index)
    )
    paths = (p for p in paths if p is not None)
    ids = [
        {"edge": iloc[(e1, e2)], "__id__": ipath, "source": path[0], "target": path[-1]}
        for ipath, path in enumerate(paths)
        for e1, e2 in zip(path, path[1:], strict=False)
    ]

    new_edges = pd.DataFrame(ids)
    new_edges = gpd.GeoDataFrame(
        pd.concat(
            [
                new_edges,
                edges.data.drop(columns=["source", "target"]).iloc[new_edges["edge"]].reset_index(),
            ],
            axis=1,
        ),
        crs=edges.crs,
    )

    merged = new_edges.dissolve("__id__")
    merged = merged.drop(columns=["edge", "weight", "index"], errors="ignore")
    merged.geometry = merged.geometry.line_merge()

    return Graph(
        edges=Edges(merged).drop_duplicated_edges(), region=graph.region, nodes=Nodes(points)
    )


def project_edges_to_nodes(
    edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, max_distance: float
) -> gpd.GeoDataFrame:
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
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    max_distance: float = 0.0,
    max_distance_bounds: float = 0.0,
) -> gpd.GeoDataFrame:
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

    _edges = gpd.GeoDataFrame(
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
    nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, max_distance: float = 0.0
) -> gpd.GeoDataFrame:
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
                gpd.GeoSeries(mnodes.geometry.iloc[ids], index=mnodes.index[ids]),
                max_distance,
            )
            continue

        update[id] = medge

    # Substitute with the updated geometries
    _update = gpd.GeoSeries(update, crs=mnodes.crs).dropna().to_crs(edges.crs)
    edges = edges.loc[_update.index]
    edges.geometry = _update.geometry
    return edges


def add_fuzzy_nodes(
    nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, distance: float = 0.0
) -> gpd.GeoDataFrame:
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
                medge, gpd.GeoSeries(mnodes.geometry.iloc[ids], index=mnodes.index[ids]), distance
            )
            continue

        update[id] = medge

    # Substitute with the updated geometries
    _update = gpd.GeoSeries(update, crs=mnodes.crs).dropna().to_crs(edges.crs)
    edges = edges.loc[_update.index]
    edges.geometry = _update.geometry
    return edges


def cluster_points(
    points: gpd.GeoDataFrame, distance: float = 100, column: str = "__node_clusters__", **kwargs
) -> gpd.GeoDataFrame:
    """Cluster nodes that are within `distance` (coordinate distance). Based on the `DBSCAN`.

    This adds a column to the `GeoDataFrame` with the labels of each cluster.
    """
    data_in_meters = points.geometry
    labels = DBSCAN(eps=distance, min_samples=2).fit_predict([[p.x, p.y] for p in data_in_meters])
    solitons = labels == -1
    maxlabels = labels.max()
    labels[solitons] = np.arange(sum(solitons)) + (maxlabels + 1)
    points[column] = labels
    return points.dissolve(by=column, **kwargs).reset_index(drop=True)


def nodes_from_edges(edges: gpd.GeoDataFrame, prefix: str) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Extract nodes at the boundaries of each edge.

    Aggregate overlapping nodes.
    """
    edgeBoundary_nodes = edges.boundary
    edgeBoundary_nodes = gpd.GeoDataFrame(
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
        gpd.GeoDataFrame(exploded[["geometry", "id"]]),
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
    line: geometry.LineString, points: gpd.GeoSeries, distance: float
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

    result = gpd.GeoDataFrame(
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


def get_geolocation(place: str) -> shapely.Polygon | shapely.MultiPolygon:
    enclosing_polygon: gpd.GeoDataFrame = ox.geocoder.geocode_to_gdf(place)
    poly = enclosing_polygon.union_all()
    if poly.area > 10:
        log.warning(f"Area of size {poly.area}")
    else:
        log.info(f"Area of size {poly.area}")
    return poly


def align_line_points(line: geometry.LineString, points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points["__dist__"] = [line.line_locate_point(p) for p in points.geometry]
    points = points.sort_values(by="__dist__")
    lines = gpd.GeoDataFrame(
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


def load_regions(buffer_size: float | None = None, test: bool | None = None) -> gpd.GeoDataFrame:
    """Load the countries (and regions) with an optional buffer.

    For now Europe.
    """
    fn = "../data/prov_test.geojson" if test else "../data/regions_europe.geojson"
    regions = gpd.read_file(fn).rename(columns={"index": "code"})[["code", "geometry"]]

    if buffer_size is None or buffer_size <= 0.0:
        return regions

    reg_buffer = buffer(regions, buf_size=buffer_size)
    reg_buffer = split_cells(reg_buffer, cell_size=buffer_size)
    regions = gpd.GeoDataFrame(pd.concat([regions, reg_buffer]))

    return regions


def split_cells(areas: gpd.GeoDataFrame, cell_size: float = 2.0) -> gpd.GeoDataFrame:
    """Split large polygons in smaller cells."""

    bounds = areas.geometry.total_bounds
    nx = int((bounds[2] - bounds[0]) / cell_size) + 1
    ny = int((bounds[3] - bounds[1]) / cell_size) + 1

    xes = np.linspace(bounds[0], bounds[2], nx)
    yes = np.linspace(bounds[1], bounds[3], ny)

    tiles = gpd.GeoDataFrame(
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


def buffer(data: gpd.GeoDataFrame, buf_size: float = 2.0) -> gpd.GeoDataFrame:
    """Create a buffer around a `GeoDataFrame`."""
    cache = Path("../data/regions_europe_buffer.geojson")

    if cache.is_file():
        return gpd.read_file(cache)

    buf = None
    for country in tqdm.tqdm(data.geometry):
        if buf is None:
            buf = country.buffer(buf_size)
        else:
            buf = buf.union(country.buffer(buf_size), grid_size=0.01)

    if buf is not None:
        buffer = gpd.GeoDataFrame(
            [{"region": "EU"}],
            geometry=[shapely.difference(buf, data.union_all(), grid_size=0.01)],
            crs=4326,
        )

    else:
        buffer = gpd.GeoDataFrame([], geometry=[])

    # cache it
    buffer.to_file(cache)
    return buffer


def merge(paths: list[Path]) -> Graph | None:
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


def check_smooth(l1: shapely.LineString, l2: shapely.LineString) -> bool:
    """Check if the angle between the two lines is less that 90."""
    p11, p12 = l1.boundary.geoms
    p21, p22 = l2.boundary.geoms

    if p11 == p21:
        sub_l1 = segment(l1, 0)
        sub_l2 = segment(l2, 0)
    elif p11 == p22:
        sub_l1 = segment(l1, 0)
        sub_l2 = segment(l2, -2).reverse()
    elif p12 == p21:
        sub_l1 = segment(l1, -2).reverse()
        sub_l2 = segment(l2, 0)
    elif p12 == p22:
        sub_l1 = segment(l1, -2).reverse()
        sub_l2 = segment(l2, -2).reverse()
    else:
        return False

    p1 = np.array(sub_l1.coords)
    p1 = p1[1] - p1[0]

    p2 = np.array(sub_l2.coords)
    p2 = p2[1] - p2[0]

    if np.dot(p1, p2) > 0:
        return False

    return True


def segment(line: shapely.LineString, nsegment: int = 0) -> shapely.LineString:
    """Return a segment of the given line.

    Warning: $nsegment \\in [0, N-2]$
    """
    return shapely.LineString([line.coords[nsegment], line.coords[nsegment + 1]])


def mode(data: list):
    """Return the mode or a random entry."""
    data_s = pd.Series(data).dropna()
    if len(data_s) == 0:
        return None

    return data_s.value_counts().sort_values().index[-1]


def __split_edge__(
    edges: Edges, edge: shapely.LineString
) -> shapely.LineString | shapely.MultiLineString:
    """Split edge if touched by another edge."""
    touches_s = set(edges.strtree("s").query(edge, predicate="intersects"))
    touches_s -= set(edges.strtree("s").query(edge.boundary, predicate="intersects"))
    touches_t = set(edges.strtree("t").query(edge, predicate="intersects"))
    touches_t -= set(edges.strtree("t").query(edge.boundary, predicate="intersects"))
    touches = touches_s | touches_t

    if len(touches) == 0:
        return edge

    touching_edges = shapely.MultiLineString(edges.data.iloc[list(touches)]["geometry"].tolist())

    try:
        new_edges = shapely.MultiLineString(ops.split(edge, touching_edges))
    except ValueError:
        # sometime they are partially overlapping
        return edge
    return new_edges


def __add_node_to_edges__(
    node_tuple: tuple[Hashable, pd.Series],
    edges: Edges,
    distance: float,
    border_distance: float = 0.0,
    nodeid_col: str = "id",
) -> dict:
    """Find the closest edge and return a dict."""
    nodeid, node = node_tuple
    result = {"nodeid": nodeid}

    # Check if node is close to the source of ONE edge
    edges_ids = edges.strtree("s").query_nearest(
        node.geometry, max_distance=border_distance, all_matches=False
    )

    if len(edges_ids) > 0:
        result["edgeid"] = edges.data.index[edges_ids]
        result["type"] = "source"
        result["rename"] = edges.data.iloc[edges_ids[0]].source

        return result

    # Check if node is close to the end of ONE line
    edges_ids = edges.strtree("t").query_nearest(
        node.geometry, max_distance=border_distance, all_matches=False
    )
    if len(edges_ids) > 0:
        result["edgeid"] = edges.data.index[edges_ids]
        result["type"] = "target"
        result["rename"] = edges.data.iloc[edges_ids[0]].target

        return result

    # Check if node is close to some lines
    edges_ids = edges.strtree("e").query(node.geometry, predicate="dwithin", distance=distance)
    if len(edges_ids) > 0:
        result["edgeid"] = edges.data.index[edges_ids]
        result["type"] = "split"
        result["rename"] = None

        return result

    return result


def __add_nodes__(
    edge: pd.Series,
    nodes: Nodes,
    distance: float,
    border_distance: float = 0.0,
    nodeid_col: str = "id",
) -> tuple[pd.Series | gpd.GeoDataFrame, dict]:
    """Find the nodes closer to the edge and split the latter accordingly."""
    node_ids = nodes.strtree().query(edge.geometry, predicate="dwithin", distance=distance)
    # Nodes that are close to the edge

    # a mapping to rename old nodes
    rename = {}

    if len(node_ids) == 0:
        return edge, rename

    # collect interesting nodes
    lnode = nodes.data.reset_index(drop=False).iloc[node_ids]
    # compute distance from the beginning
    lnode["__location__"] = [edge.geometry.project(p, normalized=True) for p in lnode.geometry]
    # sort nodes from the beginning of the line
    lnode = lnode.sort_values(by="__location__")

    e1, e2 = edge.geometry.boundary.geoms

    # if close to the borders, do not split
    # keep only one per side
    source = edge["source"]
    while shapely.distance(e1, lnode.iloc[0]["geometry"]) <= border_distance:
        rename[source] = lnode.iloc[0][nodeid_col]
        edge["source"] = lnode.iloc[0][nodeid_col]
        lnode = lnode.iloc[1:]

        if len(lnode) == 0:
            return edge, rename

    target = edge["target"]
    while shapely.distance(e2, lnode.iloc[-1]["geometry"]) <= border_distance:
        rename[target] = lnode.iloc[-1][nodeid_col]
        edge["target"] = lnode.iloc[-1][nodeid_col]
        lnode = lnode.iloc[:-1]

        if len(lnode) == 0:
            return edge, rename

    # collect all points at which one should split
    points = shapely.MultiPoint(
        [
            shapely.line_interpolate_point(
                edge.geometry, shapely.line_locate_point(edge.geometry, p)
            )
            for p in lnode.geometry
        ]
    )
    edge["geometry"] = shapely.MultiLineString(ops.split(edge["geometry"], points))

    # collect node's name to save in source and target
    ids = [edge["source"]] + lnode[nodeid_col].tolist() + [edge["target"]]
    splitted_edges = gpd.GeoDataFrame(pd.DataFrame(edge).T, crs=nodes.crs)
    splitted_edges = splitted_edges.explode(column="geometry")
    splitted_edges["source"] = ids[:-1]
    splitted_edges["target"] = ids[1:]

    return splitted_edges, rename


def __shortest_path_old__(
    graph: nx.Graph,
    nodes: Nodes,
    edges: Edges,
    pair: tuple,
    nodes_to_avoid: pd.Index | None = None,
    avoid_within: float = 0.0,
    force_smooth: bool = False,
) -> list | None:
    """Find the shortest path between `pair = (p1, p2)`."""
    p1, p2 = pair
    try:
        path = nx.shortest_path(graph, p1, p2, "weight")
    except nxexc.NetworkXNoPath:
        # There is no shortest path between `p1` and `p2`
        return

    # avoid all nodes except its own boundaries.
    if nodes_to_avoid is not None:
        metanode1 = nodes.data.loc[p1, "__metanode__"]
        metanode2 = nodes.data.loc[p2, "__metanode__"]
        _nodes_to_avoid = set(nodes_to_avoid.drop(metanode1 + metanode2))
        if len(_nodes_to_avoid.intersection(path)) > 0:
            return

        if avoid_within > 0.0:
            min_distance = shapely.distance(
                nodes.data.loc[path[1:-1]].geometry.union_all(),
                nodes.data.loc[list(_nodes_to_avoid)].geometry.union_all(),
            )
            if min_distance < avoid_within:
                return

    # if `force_smooth` remove non smooth paths
    if force_smooth and len(path) > 2:
        for p1, p2, p3 in zip(path, path[1:], path[2:], strict=False):
            e1 = edges.get_link(p1, p2)
            e2 = edges.get_link(p2, p3)
            if e1 is None or e2 is None:
                msg = f"No such edge. (e1: {p1}-{p1}, e2: {p2}-{p3})"
                raise ValueError(msg)
            if not check_smooth(e1.geometry, e2.geometry):
                return

    return path


def __shortest_path__(
    graph: Graph,
    path: list,
    metanodes: pd.Series,
    nodes_to_avoid: pd.Index | None = None,
    avoid_within: float = 0.0,
    force_smooth: bool = False,
) -> list | None:
    """Find the shortest path between `pair = (p1, p2)`."""
    p1, p2 = path[0], path[-1]

    # avoid all nodes except its own boundaries.
    if nodes_to_avoid is not None:
        metanode1 = metanodes[p1]
        metanode2 = metanodes[p2]
        _nodes_to_avoid = set(nodes_to_avoid.drop(metanode1 + metanode2))

        if avoid_within > 0.0:
            min_distance = shapely.distance(
                graph.nodes.data.loc[path[1:-1]].geometry.union_all(),
                graph.nodes.data.loc[list(_nodes_to_avoid)].geometry.union_all(),
            )
            if min_distance < avoid_within:
                return

    # if `force_smooth` remove non smooth paths
    if force_smooth and len(path) > 2:
        for p1, p2, p3 in zip(path, path[1:], path[2:], strict=False):
            e1 = graph.edges.get_link(p1, p2)
            e2 = graph.edges.get_link(p2, p3)
            if e1 is None or e2 is None:
                msg = f"No such edge. (e1: {p1}-{p1}, e2: {p2}-{p3})"
                raise ValueError(msg)
            if not check_smooth(e1.geometry, e2.geometry):
                return

    return path


def clique(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    full_network = gpd.GeoDataFrame(
        pd.DataFrame(
            [
                {
                    "source": pp1,
                    "target": pp2,
                    "geometry": shapely.LineString(
                        [nodes.geometry.loc[pp1], nodes.geometry.loc[pp2]]
                    ),
                }
                for pp1, pp2 in combinations(nodes.index, 2)
            ]
        ),
        crs=nodes.crs,
    )
    return full_network


def split_edge_at_points(
    edge: pd.Series, points: pd.Series, crs: str | int, point_names: str = "nodeid"
) -> gpd.GeoDataFrame:
    """Return a list of lines from edge split at the closest points to points."""
    line = edge.geometry
    splitter = points.geometry  # list of Points

    if line.project(points["source_geom"]) < line.project(points["target_geom"]):
        source, target = points["source"], points["target"]
    else:
        target, source = points["source"], points["target"]

    extremes = (
        pd.DataFrame(
            {
                "loc": [0.0] + [line.project(s, normalized=True) for s in splitter] + [1.0],
                "path": [source] + points[point_names] + [target],
                "dist": [0.0] + [line.distance(s) for s in splitter] + [0.0],
            }
        )
        .sort_values(["loc", "dist"])
        .drop_duplicates(
            "loc", keep="first"
        )  # if multiple nodes at the same place (`loc`) keep the closest (first)
    )

    local_edge = edge.copy()
    local_edge.geometry = shapely.MultiLineString(
        shapely.GeometryCollection(
            [
                ops.substring(line, p1, p2, normalized=True)
                for p1, p2 in zip(extremes["loc"], extremes.iloc[1:]["loc"], strict=False)
            ]
        )
    )
    new_edges = gpd.GeoDataFrame(local_edge.to_frame().T, crs=crs).explode()
    new_edges["source"] = extremes.path.iloc[:-1].tolist()
    new_edges["target"] = extremes.path.iloc[1:].tolist()
    return new_edges


def new_cliques(nodes: Nodes, clusters: pd.Series) -> Edges:
    new_edges = []
    for cluster in clusters:
        if len(cluster) > 1:
            new_edges.append(clique(nodes.data.loc[cluster]))

    return Edges(gpd.GeoDataFrame(pd.concat(new_edges), geometry="geometry", crs=nodes.crs))


check_cache_size()
