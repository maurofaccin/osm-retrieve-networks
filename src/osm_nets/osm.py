"""Base functions and utilities.

Base functions for retrieving and postprocessing data from OpenStreetMap
to build a network from the infra-structures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations
from numbers import Number
from pathlib import Path
from typing import Hashable, Iterable, Literal, Self

import geopandas as gpd
import igraph as ig
import numpy as np
import osmnx as ox
import pandas as pd
import pygeoops
import pyproj
import shapely
import tqdm
from scipy import sparse
from scipy.sparse import csgraph
from shapely import geometry, ops
from sklearn.cluster import DBSCAN
from tqdm.contrib.concurrent import process_map

from osm_nets import logconfig

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
    enclosing_polygon = place
    if enclosing_polygon.area > 10:
        log.warning(f"Area of size {enclosing_polygon.area}")
    else:
        log.info(f"Area of size {enclosing_polygon.area}")

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
        node_prefix=node_prefix,
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
    graph = graph.drop_disconnected_nodes()
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
    if len(new_edges) > 0:
        graph.edges = graph.edges.append(new_edges).drop_duplicated_edges()

    graph.edges.data = graph.edges.data.reset_index(drop=True)
    graph = graph_from_shortest_path(
        graph,
        metanodes=aggregated,
        avoid_distance=300,  # do not increase too much
        force_smooth=True,
    )
    return graph.to_degree()


def osm_roads(
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
    """
    # Retrieving the enclosing polygon.
    enclosing_polygon = place
    if enclosing_polygon.area > 10:
        log.warning(f"Area of size {enclosing_polygon.area}")
    else:
        log.info(f"Area of size {enclosing_polygon.area}")

    # get file location
    if isinstance(osm_dump_file, (str, Path)):
        osm_dump_file_nodes = osm_dump_file
        osm_dump_file_edges = osm_dump_file
    else:
        osm_dump_file_nodes, osm_dump_file_edges = osm_dump_file

    # Retrieve lines from `OpenStreetMap`.
    graph = retrieve_edges(
        osm_dump_file=osm_dump_file_edges,
        polygon=enclosing_polygon,
        node_prefix=node_prefix,
        split_when_touching=True,
        layer="lines",
    )
    if len(graph.edges) == 0:
        return graph

    graph = graph.to_meters()
    # merge nodes within 10 meters
    graph = graph.aggregate_nodes(200)
    graph.edges = graph.edges.drop_duplicated_edges().remove_self_loops()
    graph = graph.complete_lines(
        complete_to_boundary=True, complete_multilines=True
    ).merge_chained_edges()

    graph.edges.data = graph.edges.data.reset_index(drop=True)
    return graph.to_degree()


def osm_powerlines(
    place: geometry.Polygon | geometry.MultiPolygon,
    osm_dump_file: str | Path | tuple[str | Path, str | Path],
    node_prefix: str = "",
    substation_distance: float = 500,
    voltage_fillvalue: float | None = None,
    voltage_threshold: float = 1000,
) -> tuple[Graph, Nodes]:
    """Retrieve the power-line network.

    Parameters
    ----------
    place: geometry.Polygon | geometry.MultiPolygon
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
    enclosing_polygon = place
    if enclosing_polygon.area > 10:
        log.warning(f"Area of size {enclosing_polygon.area}")
    else:
        log.info(f"Area of size {enclosing_polygon.area}")

    # get file location
    if isinstance(osm_dump_file, (str, Path)):
        osm_dump_file_nodes = osm_dump_file
        osm_dump_file_edges = osm_dump_file
    else:
        osm_dump_file_nodes, osm_dump_file_edges = osm_dump_file

    # Retrieve lines from OpenStreetMap.
    graph = retrieve_edges(
        osm_dump_file=osm_dump_file_edges,
        keys={"power": ["cable", "line"]},
        polygon=enclosing_polygon,
        columns=[
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
        ],
        node_prefix=f"{node_prefix}_",
        split_when_touching=True,
    )
    graph.edges = graph.edges.rename(columns={"source": "power_source"})

    if len(graph.edges) == 0:
        return graph, gpd.GeoDataFrame([], crs=PRJ_DEG)

    if "voltage" in graph.edges.columns:
        graph.edges.data["voltage"] = _clean_voltage(graph.edges.data["voltage"])
    else:
        graph.edges.data["voltage"] = pd.NA

    if voltage_fillvalue is None:
        graph.edges = graph.edges.dropna(subset="voltage")
        graph.edges.data = graph.edges.data[graph.edges.data["voltage"] >= voltage_threshold]
    else:
        # Retains the lower voltage (1k-69k Volts) power-lines.
        # Assumes non reported voltage is lower than 69kV.
        # This is risky as the network is not that accurately reported.
        graph.edges.data["voltage"] = graph.edges.data["voltage"].fillna(voltage_fillvalue)
        log.warning("You assume that edges with no voltage are below 69k.")

    log.info("Retrieving substations from OpenStreetMap.")
    # Retrieve substations (additional nodes).
    # These substations are used to *merge* lines that do not touch.
    # Look for substations within the following polygon
    substation_polygon = ops.transform(
        MET2DEG,
        graph.edges.to_meters().data.buffer(2 * substation_distance, resolution=2).union_all(),
    )
    log.info("get nodes")
    substations = retrieve_nodes(
        osm_dump_file=osm_dump_file_nodes,
        node_prefix="substation_{node_prefix}_",
        polygon=substation_polygon,
        keys={"power": ["substation"]},
    )

    log.info("add substations fuzzily")
    if len(substations) > 0:
        substations = cluster_points(substations.data, distance=substation_distance)
        graph = graph.add_nodes(
            Nodes(substations),
            max_distance=substation_distance,
            max_distance_bounds=substation_distance,
        )
    log.info("Get nodes from edges extremes.")

    log.info("Get power plants.")
    powerplants = retrieve_nodes(
        osm_dump_file_nodes,
        keys={"power": ["plant"]},
        node_prefix=f"powerplant_{node_prefix}_",
        polygon=enclosing_polygon,
        columns=[
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
        ],
    )

    return graph.to_degree(), powerplants.to_degree()


def retrieve_nodes(
    osm_dump_file: str | Path,
    keys: dict | None = None,
    filename: str | None = None,
    polygon: geometry.Polygon | geometry.MultiPolygon | None = None,
    columns: list[str] | None = None,
    node_prefix: str = "NODE_",
    layer: str = "points",
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
        osm_dump_file=osm_dump_file, keys=keys, polygon=polygon, columns=columns, layer=layer
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
    layer: str = "ways",
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
        osm_dump_file=osm_dump_file, keys=keys, polygon=polygon, columns=columns, layer=layer
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
    layer: str = "points",
) -> gpd.GeoDataFrame:
    """Get the raw data."""
    # load data from osm dump file
    print(layer)
    data = gpd.read_file(str(Path(osm_dump_file).expanduser()), layer=layer, mask=polygon).explode()

    # expand tags
    if "other_tags" in data.columns:
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
        if len(self.data) == 0:
            return self
        data = self.data.to_crs(PRJ_MET)
        return type(self)(data)

    def to_degree(self, inline: bool = False) -> Self:
        if len(self.data) == 0:
            return self
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

    def subsumple(
        self,
        predicate: Literal["within", "intersects"],
        shape: shapely.Geometry,
        invert: bool = False,
    ) -> gpd.GeoDataFrame:
        """Sub-sample objects based on `shape`.

        Parameters
        ==========
        predicate : str
            the predicate to check
        shape : shapely.Geometry
            the geometry to test against
        invert: bool
            whether to invert the match or not

        Return
        ======
        Output : GeoDataFrame
            The subset of objects
        """
        if predicate == "within":
            allowed = self.data.geometry.within(shape)
        elif predicate == "intersects":
            allowed = self.data.geometry.intersects(shape)

        if invert:
            return self.data.loc[~allowed]
        return self.data.loc[allowed]

    @property
    def crs(self):
        return self.data.crs

    @property
    def index(self) -> pd.Index:
        return self.data.index

    @property
    def columns(self) -> pd.Index:
        return self.data.columns

    def rename(self, columns: dict | None = None, index: dict | None = None) -> Self:
        return type(self)(
            gpd.GeoDataFrame(self.data.rename(index=index, columns=columns), crs=self.crs)
        )

    def dropna(self, **kwargs) -> Self:
        return type(self)(gpd.GeoDataFrame(self.data.dropna(**kwargs), crs=self.crs))

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
        if self.data.index.name != "NODE_ID":
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

    @classmethod
    def concat(cls, gdfs: Iterable[Self], geometry="geometry", crs=PRJ_DEG) -> Self:
        return cls(concat([g.data for g in gdfs], geometry=geometry, crs=crs, ignore_index=False))


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

    def nodes(self, ids: Iterable) -> pd.DataFrame:
        """Return edges incident in given node."""
        return self.data[(self.source.isin(ids)) | (self.target.isin(ids))]

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
            desc="Adding nodes",
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
            partial(__split_edge__, self),
            new_data.geometry.tolist(),
            chunksize=10,
            desc="Split nodes",
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

    def rename_nodes(self, rename_dict: dict) -> Edges:
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

    def merge_edges(self, edge_ids: Iterable[Iterable[Hashable]]) -> Edges:
        """Merge listed edges."""
        edges = self.data
        edges["__label__"] = np.zeros(len(edges))
        for icls, cls in enumerate(edge_ids):
            edges.loc[list(cls), "__label__"] = icls + 1

        edges = (
            gpd.GeoDataFrame(
                pd.concat(
                    [edges[edges["__label__"] == 0]]
                    + [_merge_edges_(e) for label, e in edges.groupby("__label__") if label > 0]
                )
            )
            .set_crs(self.data.crs, allow_override=True)
            .reset_index()
            .drop(columns="__label__")
        )
        return Edges(edges)

    def complete_lines(
        self,
        nodes: Nodes,
        complete_to_boundary: bool | None = None,
        complete_multilines: bool | None = None,
    ) -> Edges:
        new_geometries = []

        new_geometries = process_map(
            partial(
                _complete_join_multilinestring_,
                complete_multilines=complete_multilines,
                complete_to_boundary=complete_to_boundary,
            ),
            [
                (
                    edge.geometry,
                    nodes.data.loc[edge.loc["source"], "geometry"],
                    nodes.data.loc[edge.loc["target"], "geometry"],
                )
                for _, edge in self.data.iterrows()
            ],
            chunksize=len(self.data) // 100 + 1,
            desc="Joining MultiLineStrings",
        )

        edges = self.data
        edges["geometry"] = new_geometries
        return Edges(edges)

    @classmethod
    def concat(
        cls, gdfs: Iterable[Self], geometry="geometry", crs=PRJ_DEG, ignore_index: bool = True
    ) -> Self:
        return cls(
            concat([g.data for g in gdfs], geometry=geometry, crs=crs, ignore_index=ignore_index)
        )


@dataclass
class Graph:
    edges: Edges
    region: gpd.GeoDataFrame
    nodes: Nodes = field(default_factory=lambda: Nodes())

    @property
    def region_shape(self) -> shapely.Geometry:
        """`MultiPolygon` corresponding to the union of all shapes."""
        return self.region.union_all(method="unary")

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
        """Index of the edges (`self.edges.index`)"""
        return self.edges.index

    def add_nodes(
        self, nodes: Nodes, max_distance: float = 0.0, max_distance_bounds: float = 0.0
    ) -> Graph:
        """Add nodes.

        Add nodes as source and targets and eventually split edges accordingly.
        Will overwrite existing nodes if overlapping.

        Parameters:
        ----------
        nodes : Nodes
            There are the nodes to be added
        max_distance : float
            If nodes are within this distance from an edge, this edge is split and the node becomes the source or target.
        max_distance_bounds : float
            If the node is within this distance from an edge border is substitute the formed source or target node.
        Returns
        -------
        new_graph : Graph
            The new graph with additional nodes.
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
                desc="Add nodes (find edges)",
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
            action["rename"]: nid for nid, action in actions.iterrows() if action["type"] != "split"
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

        if len(splitters) > 0:
            new_edges = Edges(
                gpd.GeoDataFrame(
                    pd.concat(
                        [
                            split_edge_at_points(
                                new_edges.loc[edgeid],
                                splitter,
                                self.edges.crs,
                                point_names="nodeid",
                            )
                            for edgeid, splitter in splitters.iterrows()
                        ]
                    ),
                    crs=self.edges.crs,
                )
            )
            new_edges = new_edges.append(self.edges.drop(index=splitters.index))
        else:
            new_edges = self.edges

        # Only nodes involved in links
        new_nodes_idx = list(set(new_edges.data.source) | set(new_edges.data.target))
        return Graph(edges=new_edges, region=self.region, nodes=Nodes(new_nodes))

    def aggregate_nodes(self, distance: float) -> Graph:
        """Aggregate nodes within distance."""
        nodes = self.nodes.aggregate(distance=distance).data["NODE_ID"]
        nodes = nodes.loc[[len(x) > 1 for x in nodes]]
        rename = {k: v for v, nns in nodes.items() for k in nns if k != v}

        new_nodes = self.nodes.drop(index=list(rename.keys()))
        new_edges = self.edges.rename_nodes(rename)
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
        return Graph(
            edges=Edges(
                self.edges.data.loc[
                    (self.edges.source.isin(keep_ids)) & (self.edges.target.isin(keep_ids))
                ]
            ),
            region=self.region,
            nodes=Nodes(self.nodes.data.loc[keep_ids]),
        )

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

    # def split_edges(self, other: Graph):
    #     """Split edges in those intersecting the other.region and the others."""
    #     # Edges in self, crossing to other.region
    #     sindex = shapely.STRtree(self.edges.data.geometry)
    #     crossing_ids = self.edges.index[sindex.query(other.region_shape, predicate="intersects")]
    #
    #     g_inner = self.filter_edges(self.index().difference(list(crossing_ids)))
    #     g_outer = self.filter_edges(crossing_ids)
    #     return g_inner, g_outer

    def merge(self, other: Graph, tol: float = 50) -> Graph:
        """Blend and merge two graphs into one.

        1. Check edges that are intersecting from the two sets.
        2. Merge those edges
        3. Rename nodes
        """
        region1 = self.region_shape
        region2 = other.region_shape.difference(region1)

        # nodes that are inside their regions
        nodes_allowed_1 = self.nodes.subsumple("within", region1, invert=False)
        nodes_allowed_2 = other.nodes.subsumple("within", region2, invert=False)
        nodes_suspicious_1 = self.nodes.subsumple("within", region2)
        nodes_suspicious_2 = other.nodes.subsumple("within", region1)

        e1 = self.edges.subsumple("intersects", region1.boundary)
        e2 = other.edges.subsumple("intersects", region2.boundary)
        tree = shapely.STRtree(e2.geometry)

        g = []

        new = {"lines": [], "old1": [], "old2": []}
        for id1, edge1 in tqdm.tqdm(e1.iterrows(), total=len(e1), leave=False):
            nearest = e2.iloc[
                tree.query_nearest(
                    edge1.geometry, max_distance=tol, exclusive=True, all_matches=True
                )
            ]

            # for all edges of other coming close enough
            # check if a link is possible.
            for id2, edge2 in nearest.iterrows():
                log = "ITA.18_1_14" in [edge1.target, edge2.target, edge1.source, edge2.source]
                inner = []
                if edge1["source"] in nodes_allowed_1.index:
                    inner.append(edge1["source"])
                if edge1["target"] in nodes_allowed_1.index:
                    inner.append(edge1["target"])
                if edge2["source"] in nodes_allowed_2.index:
                    inner.append(edge2["source"])
                if edge2["target"] in nodes_allowed_2.index:
                    inner.append(edge2["target"])

                if len(inner) == 2:
                    if log:
                        g.append(
                            (
                                edge1.geometry,
                                edge2.geometry,
                                mmerge_lines(edge1.geometry, edge1.geometry),
                            )
                        )
                    metadata1 = edge1.dropna().to_dict()
                    metadata2 = edge2.dropna().to_dict()
                    metadata1.update(metadata2)
                    # metadata1["geometry"] = shapely.union_all([l1, l2])
                    metadata1["geometry"] = mmerge_lines(edge1.geometry, edge2.geometry)
                    metadata1["source"] = min(inner)
                    metadata1["target"] = max(inner)
                    new["lines"].append(metadata1)
                new["old1"].append(id1)
                new["old2"].append(id2)

        # edges to be removed because they are substituted by the merged ones
        old1_edges = pd.Index(new["old1"])
        oe = self.edges.nodes(nodes_suspicious_1.index)
        old1_edges = old1_edges.union(oe.index.tolist())
        old2_edges = pd.Index(new["old2"])
        oe = other.edges.nodes(nodes_suspicious_2.index)
        old2_edges = old2_edges.union(oe.index.tolist())

        if len(g) > 0:
            print("writing")
            gpd.GeoSeries([x[0] for x in g], crs=PRJ_MET).to_crs(PRJ_DEG).to_file("xxx_e1.gpkg")
            gpd.GeoSeries([x[1] for x in g], crs=PRJ_MET).to_crs(PRJ_DEG).to_file("xxx_e2.gpkg")
            gpd.GeoSeries([x[2] for x in g], crs=PRJ_MET).to_crs(PRJ_DEG).to_file("xxx_merged.gpkg")

        new_edges = Edges.concat(
            [
                self.edges.drop(index=old1_edges, errors="ignore"),
                other.edges.drop(index=old2_edges, errors="ignore"),
            ],
            crs=self.edges.crs,
        )
        if len(new["lines"]) > 0:
            new_edges = Edges.concat(
                [
                    new_edges,
                    Edges(
                        gpd.GeoDataFrame(
                            new["lines"], geometry="geometry", crs=self.edges.crs
                        ).drop_duplicates(subset=["source", "target"])
                    ),
                ],
                crs=self.edges.crs,
            )
        new_graph = Graph(
            new_edges,
            region=concat([self.region, other.region], crs=self.region.crs),
            nodes=Nodes.concat([self.nodes, other.nodes], crs=self.nodes.crs),
        )
        return new_graph.drop_disconnected_nodes()

    def __str__(self) -> str:
        return "\n".join(map(str, [self.edges, self.nodes]))

    def __len__(self) -> int:
        return len(self.edges)

    @classmethod
    def read(cls, path: Path, node_index: str | None = None) -> Graph:
        edges = gpd.read_file(path, layer="edges").set_crs(PRJ_DEG, allow_override=True)
        nodes = gpd.read_file(path, layer="nodes").set_crs(PRJ_DEG, allow_override=True)
        if node_index is not None:
            nodes = nodes.set_index(node_index, drop=True)
        region = gpd.read_file(path, layer="region").set_crs(PRJ_DEG, allow_override=True)

        g = Graph(edges=Edges(edges), region=region, nodes=Nodes(nodes))
        g.nodes.append(Nodes(nodes))

        return g

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        path.unlink(missing_ok=True)
        self.edges.data.drop(columns=["index"], errors="ignore").to_file(
            path, driver="GPKG", layer="edges"
        )
        if self.nodes.index.name in self.nodes.data.columns:
            self.nodes.data.to_file(path, driver="GPKG", layer="nodes", index=False)
        else:
            self.nodes.data.to_file(path, driver="GPKG", layer="nodes", index=True)
        self.region.to_file(path, driver="GPKG", layer="region")

    def graph_ig(self):
        """Return a igraph Graph."""
        if "weight" in self.edges.data.columns:
            return ig.Graph.TupleList(
                self.edges.data[["source", "target", "weight"]].itertuples(index=False),
                weights=True,
                directed=False,
            )
        return ig.Graph.TupleList(
            self.edges.data[["source", "target"]].itertuples(index=False),
            weights=True,
            directed=False,
        )

    @property
    def connected_components(self) -> list[set]:
        try:
            return self.cc
        except AttributeError:
            pass
        graph_ig = self.graph_ig()
        nodes = np.array(graph_ig.vs["name"])
        self.cc = [set(nodes[cc]) for cc in graph_ig.connected_components(mode="strong")]
        return self.cc

    def largest_component(self) -> Graph:
        largest_component = list(max(self.connected_components, key=len))
        return self.filter_nodes(largest_component)

    def shortest_paths(self, subset: pd.Index | None = None):
        """Yields the shortest paths between any two nodes."""
        graph_ig = self.graph_ig()
        nodes = np.array(graph_ig.vs["name"])

        if subset is None:
            subset = self.nodes.index

        assert len(set(subset) - set(nodes)) == 0, f"Not included {set(subset) - set(nodes)}"

        if False:
            # parallel processing requires a lot of memory
            for paths in process_map(
                partial(__yield_shortest_paths__, graph=graph_ig, subset=subset),
                self.connected_components,
                chunksize=10,
            ):
                yield from [nodes[path] for path in paths]
        else:
            for component in tqdm.tqdm(self.connected_components, desc="Short paths"):
                for path in __yield_shortest_paths__(component, graph_ig, subset):
                    yield nodes[path]

    def drop_disconnected_nodes(self) -> Graph:
        n = self.nodes.data
        n = n.loc[sorted(set(self.edges.source) | set(self.edges.target))]
        return Graph(edges=self.edges, region=self.region, nodes=Nodes(n))

    def complete_lines(
        self, complete_to_boundary: bool | None = None, complete_multilines: bool | None = None
    ) -> Graph:
        """Complete links such that they touch the source and target nodes."""
        return Graph(
            self.edges.complete_lines(
                self.nodes,
                complete_to_boundary=complete_to_boundary,
                complete_multilines=complete_multilines,
            ),
            region=self.region,
            nodes=self.nodes,
        )

    def merge_chained_edges(self) -> Graph:
        """Join chained edges adsorbing nodes with degree = 2."""
        deg = self.degree()
        # get only nodes with degree two.
        deg = deg[deg == 2]

        edges = self.edges.data[
            (self.edges.source.isin(deg.index) | self.edges.target.isin(deg.index))
        ]
        linked_edges = {}
        for eid, edge in edges.reset_index().iterrows():
            if edge.source in deg.index:
                linked_edges.setdefault(edge.source, []).append(eid)
            if edge.target in deg.index:
                linked_edges.setdefault(edge.target, []).append(eid)

        gr = ig.Graph(edges=list(linked_edges.values()))
        new_edges = self.edges.merge_edges([edges.index[cc] for cc in gr.connected_components()])
        return Graph(new_edges, region=self.region, nodes=self.nodes).drop_disconnected_nodes()

    def degree(self) -> pd.Series:
        """Degree of the nodes."""
        n1 = self.edges.source.value_counts()
        n2 = self.edges.target.value_counts()

        return n1.add(n2, fill_value=0.0)


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
    # do not avoid nodes outside the region of interest
    # This is because those nodes have not been aggregated and can fall close to the station.
    avoid = avoid[avoid.geometry.within(graph.region.iloc[0].geometry)]

    edges = graph.edges
    edges.data["weight"] = edges.data.length

    iloc = {
        (e[side[0]], e[side[1]]): ie
        for ie, (_, e) in enumerate(edges.data.iterrows())
        for side in [("source", "target"), ("target", "source")]
    }

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
        # cannot decide
        return True

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
        _nodes_to_avoid = set(nodes_to_avoid.drop(metanode1 + metanode2, errors="ignore"))

        if avoid_within > 0.0:
            # remove this path if any of the nodes to avoid fall within distance from the path
            min_distance = shapely.distance(
                graph.nodes.data.loc[path[1:-1]].geometry.union_all(),
                graph.nodes.data.loc[list(_nodes_to_avoid)].geometry.union_all(),
            )
            if min_distance < avoid_within:
                return
        else:
            # remove only if the nodes to avoid are in the path
            if len(_nodes_to_avoid & set(path)) > 0:
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
    new_edges = [clique(nodes.data.loc[cluster]) for cluster in clusters if len(cluster) > 1]

    if len(new_edges) > 0:
        return Edges(gpd.GeoDataFrame(pd.concat(new_edges), geometry="geometry", crs=nodes.crs))
    return Edges()


def concat(
    gdfs: Iterable[gpd.GeoDataFrame], geometry="geometry", crs=PRJ_DEG, ignore_index: bool = True
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=ignore_index, axis="index"), geometry=geometry, crs=crs
    )


def merge_lines(
    l1: shapely.LineString, l2: shapely.LineString, force_joint: bool = False
) -> shapely.LineString | shapely.MultiLineString:
    """Merge lines two lines discarding the overlapping part.

    force_join imply, add the segment that join separated lines.
    """
    # Find the nearest points between l1 and l2
    point_on_l1, point_on_l2 = ops.nearest_points(l1, l2)

    l1_p1_pos = l1.line_locate_point(point_on_l1)
    p1 = shapely.Point(l1.coords[0])
    p2 = shapely.Point(l1.coords[-1])
    l1_p2_pos = (
        l1.line_locate_point(p1) if l2.distance(p1) > l2.distance(p2) else l1.line_locate_point(p2)
    )

    l2_p1_pos = l2.line_locate_point(point_on_l2)
    p1 = shapely.Point(l2.coords[0])
    p2 = shapely.Point(l2.coords[-1])
    l2_p2_pos = (
        l2.line_locate_point(p1) if l1.distance(p1) > l1.distance(p2) else l2.line_locate_point(p2)
    )

    lines = []
    newline = ops.substring(l1, l1_p1_pos, l1_p2_pos)
    if isinstance(newline, shapely.LineString):
        lines.append(newline)
    if force_joint and point_on_l1 != point_on_l2:
        lines.append(
            shapely.LineString(
                [l1.line_interpolate_point(l1_p1_pos), l2.line_interpolate_point(l2_p1_pos)]
            )
        )
    newline = ops.substring(l2, l2_p2_pos, l2_p1_pos)
    if isinstance(newline, shapely.LineString):
        lines.append(newline)

    return ops.linemerge(lines)


def mmerge_lines(
    l1: shapely.LineString | shapely.MultiLineString,
    l2: shapely.LineString | shapely.MultiLineString,
) -> shapely.LineString:
    """Merge [Multi]lines."""
    lines = list(l1.geoms) if isinstance(l1, shapely.MultiLineString) else [l1]
    lines += list(l2.geoms) if isinstance(l2, shapely.MultiLineString) else [l2]

    lines = shapely.MultiLineString(lines)

    merged = pygeoops.centerline(lines.buffer(50, cap_style=2), min_branch_length=-2, extend=True)
    if isinstance(merged, (shapely.LineString, shapely.MultiLineString)):
        return merged
    return lines


def __yield_shortest_paths__(
    component: Iterable, graph: ig.Graph, subset: pd.Index
) -> list[pd.Index]:
    nodes = subset.intersection(component)
    paths = []
    if len(nodes) < 2:
        return paths

    for node in nodes:
        for path in graph.get_shortest_paths(
            node, to=[x for x in nodes if x > node], weights="weight", mode="all"
        ):
            if len(path) == 0 or path is None:
                continue
            paths.append(path)

    return paths


def _complete_line_(
    line: shapely.LineString | shapely.MultiLineString, p1: shapely.Point, p2: shapely.Point
) -> shapely.LineString | shapely.MultiLineString:
    closest1 = min(line.boundary.geoms, key=lambda x: shapely.distance(p1, x))
    closest2 = min(line.boundary.geoms, key=lambda x: shapely.distance(p2, x))

    segment1 = shapely.LineString([p1, closest1])
    segment2 = shapely.LineString([p2, closest2])

    if isinstance(line, shapely.LineString):
        return ops.linemerge([segment1, line, segment2]).simplify(1e-10)

    return ops.linemerge(list(line.geoms) + [segment1, segment2]).simplify(1e-10)


def _join_multilinestring_(
    multiline: shapely.MultiLineString | shapely.LineString,
) -> shapely.LineString:
    """Join MultiLineStrings adding missing segments."""
    if isinstance(multiline, shapely.LineString):
        return multiline

    # construct the graph of lines
    boundaries = [(iline, line.boundary) for iline, line in enumerate(multiline.geoms)]

    if len(boundaries) == 2:
        # if only two lines, that's easy
        tomerge = [(1.0, 0, 1)]
    else:
        # compute the minimum spanning tree
        dist = np.array(
            [(shapely.distance(x[1], y[1]), x[0], y[0]) for x, y in combinations(boundaries, 2)]
        )
        sparse_graph = sparse.coo_matrix(
            (dist[:, 0], (dist[:, 1], dist[:, 2])), shape=(len(boundaries), len(boundaries))
        )
        mstree = csgraph.minimum_spanning_tree(sparse_graph + sparse_graph.T)

        # assume the minimum spanning tree best represent the line
        tomerge = sorted(zip(mstree.data, *mstree.nonzero()))

    lines = list(multiline.geoms)
    segments = []
    used = set()
    for w, i, j in tomerge:
        line1 = lines[i]
        line2 = lines[j]
        closest1 = min(line1.boundary.geoms, key=lambda x: shapely.distance(line2.boundary, x))
        if closest1 in used:
            # this is to force the topology to be a line
            # avoiding branches
            closest1 = max(line1.boundary.geoms, key=lambda x: shapely.distance(line2.boundary, x))
        used.add(closest1)

        closest2 = min(line2.boundary.geoms, key=lambda x: shapely.distance(line1.boundary, x))
        if closest2 in used:
            closest2 = max(line2.boundary.geoms, key=lambda x: shapely.distance(line1.boundary, x))
        used.add(closest2)

        segment = shapely.LineString([closest1, closest2])
        segments.append(segment)

    return ops.linemerge(lines + segments).simplify(1e-10)


def _complete_join_multilinestring_(
    objects: tuple[shapely.LineString | shapely.MultiLineString, shapely.Point, shapely.Point],
    complete_multilines: bool,
    complete_to_boundary: bool,
) -> shapely.LineString | shapely.MultiLineString:
    new_edge, p1, p2 = objects

    if complete_multilines:
        new_edge = _complete_line_(new_edge, p1, p2)

    if complete_to_boundary:
        new_edge = _join_multilinestring_(new_edge)

    return new_edge


def _merge_edges_(to_merge: pd.DataFrame) -> pd.DataFrame:
    # merge all geometries together
    if len(to_merge) <= 1 or isinstance(to_merge, pd.Series):
        return to_merge

    geom_list = [
        [line] if isinstance(line, shapely.LineString) else list(line.geoms)
        for line in to_merge.geometry
    ]
    geom_list = [g for geom in geom_list for g in geom]

    try:
        new_edge_geom = ops.linemerge(geom_list)
    except NotImplementedError:
        print(to_merge)
        raise

    dd = pd.concat([to_merge["source"], to_merge["target"]]).value_counts()
    dd = dd[dd == 1]
    try:
        assert len(dd) == 2
    except AssertionError:
        return to_merge
    n1, n2 = dd.index
    new_edge = to_merge.aggregate(mode)
    new_edge["source"] = n1
    new_edge["target"] = n2
    new_edge["geometry"] = new_edge_geom

    return new_edge.to_frame().T


check_cache_size()
