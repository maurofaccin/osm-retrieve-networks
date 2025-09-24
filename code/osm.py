"""Base functions and utilities.

Base functions for retrieving and post-processing data from OpenStreetMap
to build a network from the infrastructures.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Self

import geopandas as geopd
import logconfig
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import shapely
from matplotlib import axes
from shapely import geometry, ops, plotting

CACHE = Path("~/.cache").expanduser() / "osm_retrieve_networks"
CACHE.mkdir(parents=True, exist_ok=True)

DATA = Path("../data")
DATA.mkdir(parents=True, exist_ok=True)
PRJ_DEG = pyproj.CRS.from_epsg(4326)
PRJ_MET = pyproj.CRS.from_epsg(3857)
DEG2MET = pyproj.Transformer.from_crs(PRJ_DEG, PRJ_MET, always_xy=True).transform
MET2DEG = pyproj.Transformer.from_crs(PRJ_MET, PRJ_DEG, always_xy=True).transform


ox.settings.cache_folder = CACHE
logconfig.setup_logging("DEBUG")
log = logging.getLogger(__name__)


def osm_powerlines(
    place: str,
    substation_distance: float = 500,
    voltage_fillvalue: float | None = None,
    voltage_threshold: float = 1000,
) -> tuple[Graph, geopd.GeoDataFrame]:
    """Retrieve the powerline network.

    Parameters
    ----------
    place: str
        Retrieve data from this place: e.g. `Padova`, `London`, `France`…
    voltage_fillvalue : float | None :
         (Default value = None)
    voltage_threshold : float
        (Default value = 1000)

    Returns
    -------
    nodes : GeoDataFrame
        the nodes
    edges : GeoDataFrame
        the edges
    powerplants : GeoDataFrame
        the powerplants


    """
    # Retrieving the enclosing poligon.
    enclosing_polygon = get_geolocation(place=place)
    # Retrieve lines from OpenStreetMap.
    log.info("Retrieving edges from OpenStreetMap.")
    edges = retrieve_edges(keys={"power": ["cable", "line"]}, polygon=enclosing_polygon).rename(
        columns={"source": "power_source"}
    )
    edges.voltage = _clean_voltage(edges.voltage)
    if voltage_fillvalue is None:
        edges = edges.dropna(subset="voltage")
        edges = edges[edges.voltage >= voltage_threshold]
    else:
        # Retains the lower voltage (1k-69k Volts) power-lines.
        # Assumes non reported voltage is lower than 69kV.
        # This is risky as the network is not that accurately reported.
        edges.voltage = edges.voltage.fillna(voltage_fillvalue)
        log.warning("You assume that edges with no voltage are below 69k.")
    edges = join_edges_at_points(edges.reset_index(drop=False))

    log.info("Retrieving substations from OpenStreetMap.")
    # Retrieve substations (additional nodes).
    # These substations are used to *merge* lines that do not touch.
    substation_polygon = ops.transform(
        MET2DEG, edges.to_crs(PRJ_MET).buffer(2 * substation_distance, resolution=2).union_all()
    )
    log.info("get nodes")
    substations = retrieve_nodes(keys={"power": ["substation"]}, polygon=substation_polygon)
    log.info("add substations fuzzily")
    edges[edges["id"] == 1342197763].to_file("ciccio.geojson")
    substations.to_file("ciccio_pasticcio.geojson")
    edges = add_fuzzy_nodes(substations, edges, distance=substation_distance)
    log.info("Build graph")
    graph = Graph(edges=edges, region=enclosing_polygon)

    log.info("Get nodes from edges extremes.")
    graph = graph.nodes_from_edges(node_prefix=place + "_")

    log.info("Get power plants.")
    powerplants = retrieve_nodes(keys={"power": ["plant"]}, polygon=enclosing_polygon)

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
        try:
            if place is not None:
                _data = ox.features_from_place(place, {k: vals})
            elif polygon is not None:
                _data = ox.features_from_polygon(polygon, {k: vals})
            else:
                raise ValueError("You should provide either `place` or `polygon`.")
        except ox._errors.InsufficientResponseError:
            log.warning(f"No data for {dict(k=vals)}. Skipping")
            pass
        else:
            if isinstance(_data, list):
                data.extend(_data)
            else:
                data.append(_data)

    # Build a `DataFrame`
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
        if place is not None:
            features = ox.features_from_place(place, {k: vals}).loc["way"]
        elif polygon is not None:
            features = ox.features_from_polygon(polygon, {k: vals}).loc["way"]
        else:
            raise ValueError("You should provide either `place` or `polygon`.")

        data.append(features)

    # Build a dataframe
    edges = geopd.GeoDataFrame(pd.concat(data), geometry="geometry")
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
    region: geometry.Polygon | geometry.MultiPolygon

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
        graph = Graph(self.edges.loc[keep_ids], region=self.region)
        graph.nodes = self.nodes.loc[
            pd.concat([graph.edges.source, graph.edges.target]).drop_duplicates()
        ]
        return graph

    def intersect(self, other: Graph, return_same: bool | None = False) -> list:
        """Return edges in both (deduplicated).

        - For same edges: return one (if `return_same` is True)
        - For one edge covering the other: return smaller
        - For one edge present in only one graph: return it
        """
        sindex = shapely.strtree.STRtree(other.edges.geometry)
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
        sindex = shapely.strtree.STRtree(self.edges.geometry)
        crossing_ids = self.edges.index[sindex.query(other.region.boundary, predicate="intersects")]

        g_inner = self.filter_edges(self.index().difference(list(crossing_ids)))
        g_outer = self.filter_edges(crossing_ids)
        return g_inner, g_outer

    def merge(self, other: Graph) -> Graph:
        """Merge the two graphs.

        Merge edges that are crossing borders.

        Assumptions:
            - No edges from the same source are overlapping (one covers the other, otherwise both are returned)
            - The overlapping part of an edge should be close to one extreme (the one in the other region)
              Otherwise the uncovered part is lost.

        See `osm_test.py` for examples.
        """
        # Edges in self, crossing to other.region
        g_inner1, g_outer1 = self.split_edges(other)
        g_inner2, g_outer2 = other.split_edges(self)

        # nodes that need attention
        i1 = g_outer1.intersect(g_outer2, return_same=True)
        i2 = g_outer2.intersect(g_outer1, return_same=False)
        merged = geopd.GeoDataFrame(
            pd.concat([
                g_inner1.edges,
                g_inner2.edges,
                g_outer1.edges.loc[i1],
                g_outer2.edges.loc[i2],
            ]),
            crs=self.edges.crs,
        )

        # Fix nodes
        wrong_region = [
            node
            for k, edge in g_outer1.edges.iterrows()
            for node in edge.loc[["source", "target"]]
            if g_outer1.nodes.loc[node].geometry.within(g_outer2.region)
        ]
        sindex = shapely.STRtree(other.nodes.geometry)
        rename = {
            node: other.nodes.index[
                sindex.query_nearest(g_outer1.nodes.loc[node].geometry, max_distance=1e-10)
            ]
            for node in wrong_region
        }
        wrong_region = [
            node
            for k, edge in g_outer2.edges.iterrows()
            for node in edge.loc[["source", "target"]]
            if g_outer2.nodes.loc[node].geometry.within(g_outer1.region)
        ]
        sindex = shapely.STRtree(self.nodes.geometry)
        rename |= {
            node: self.nodes.index[
                sindex.query_nearest(g_outer2.nodes.loc[node].geometry, max_distance=1e-10)
            ]
            for node in wrong_region
        }
        rename = {k: v[0] for k, v in rename.items() if len(v) > 0}

        mgraph = Graph(edges=merged, region=shapely.union(self.region, other.region))
        mgraph.edges["source"] = mgraph.edges["source"].map(lambda x: rename.get(x, x))
        mgraph.edges["target"] = mgraph.edges["target"].map(lambda x: rename.get(x, x))
        mgraph.nodes = geopd.GeoDataFrame(
            pd.concat([
                self.nodes.loc[self.nodes.index.difference(pd.Index(rename))],
                other.nodes.loc[other.nodes.index.difference(pd.Index(rename))],
            ]),
            crs=self.edges.crs,
        )
        return mgraph

    def __str__(self) -> str:
        return "\n".join(map(str, [self.edges, self.nodes]))

    @classmethod
    def load(cls, basedir: Path, filename_fmt: str) -> Graph:
        edges = geopd.read_file(basedir / filename_fmt.format("edges")).set_index(
            "index", drop=True
        )
        nodes = geopd.read_file(basedir / filename_fmt.format("nodes")).set_index("id", drop=True)
        with (basedir / filename_fmt.format("region")).open("r") as fout:
            region = shapely.from_geojson(fout.read())

        graph = Graph(edges=edges, region=region)
        graph.nodes = nodes
        return graph

    def write(self, basedir: Path, filename_fmt: str) -> None:
        basedir.mkdir(parents=True, exist_ok=True)
        self.edges.to_file(basedir / filename_fmt.format("edges"))
        self.nodes.to_file(basedir / filename_fmt.format("nodes"))
        with (basedir / filename_fmt.format("region")).open("w") as fout:
            fout.write(shapely.to_geojson(self.region))

    def plot(self, ax: axes.Axes, color: str = "r", text_args: dict | None = None):
        plotting.plot_polygon(self.region, color=color, ax=ax, alpha=0.2)
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


def merge_points_within_distance(points: geopd.GeoDataFrame, distance: float) -> geopd.GeoDataFrame:
    """Merge (centroid) nodes that falls within `distance`.

    `distance` must be in the same coordinate system.

    Parameters
    ----------
    points : geopd.GeoDataFrame
        A dataframe of points.
    distance : float
        The threshold distance (same units of the points projection)

    Returns
    -------
    points : GeoDataFrame
        A dataframe with close points merged.

    """
    log.info("Merging nodes within {distance} distance.")
    sindex = points.sindex
    points["label"] = [
        min(sindex.query(point, predicate="dwithin", distance=distance))
        for point in points.geometry
    ]
    points = points.reset_index(drop=False)
    return geopd.GeoDataFrame(
        points.groupby(by="label")
        .aggregate(
            func={
                "name": "first",
                "geometry": lambda x: geometry.MultiPoint(x).centroid,
                "id": "first",
            }
        )
        .set_index("id", drop=True),
        crs=points.crs,
    )


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
    mnodes = nodes[["name", "geometry"]].copy().to_crs(PRJ_MET)
    medges = edges[["name", "geometry"]].copy().to_crs(PRJ_MET)

    # Merge all points that are closer than `distance`
    mnodes = merge_points_within_distance(mnodes, distance)

    # Use `sindex` for quickness
    nodes_sindex = shapely.strtree.STRtree(mnodes.geometry)
    update = {}
    for id, medge in medges.geometry.items():
        try:
            source, targert = medge.boundary.geoms
        except ValueError:
            # Circular lines!!!!
            continue

        imes = nodes_sindex.query_nearest(source, max_distance=distance, all_matches=False)
        imet = nodes_sindex.query_nearest(targert, max_distance=distance, all_matches=False)

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
                distance,
            )
            continue

        update[id] = medge
        continue

    # Substitute with the updated geometries
    _update = geopd.GeoSeries(update, crs=mnodes.crs).dropna().to_crs(edges.crs)
    edges = edges.loc[_update.index]
    edges.geometry = _update.geometry
    return edges


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


def join_edges_at_points(edges: geopd.GeoDataFrame) -> geopd.GeoDataFrame:
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
        possible = [edges.index[i] for i in sindex.intersection(edge1.bounds)]
        # Remove self
        possible = [id for id in possible if id != edge_id1]
        # Remove already chopped
        possible = [id for id in possible if id not in new_edges]

        # Find edges that actually touch the original edge
        touches = [edge_id2 for edge_id2 in possible if edge1.touches(edges.geometry.loc[edge_id2])]

        if len(touches) == 0:
            continue

        # Split edge in shorter edges at intersections.
        edges_touch = geometry.MultiLineString(lines=edges.geometry.loc[touches].tolist())
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
                pd.DataFrame([
                    edges.loc[id].to_dict() | {"geometry": line}
                    for id, lines in new_edges.items()
                    for line in lines
                ]),
            ],
            ignore_index=True,
        ),
    ).set_crs(crs=edges.crs, allow_override=True)
    if len(new_edges) == 0:
        return result
    log.info("Performing a new deeper recursive level")
    return join_edges_at_points(result)


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


check_cache_size()
