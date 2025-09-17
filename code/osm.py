"""Base functions and utilities.

Base functions for retrieving and post-processing data from OpenStreetMap
to build a network from the infrastructures.
"""

from pathlib import Path
import numpy as np
from collections.abc import Hashable
from numbers import Number
import logging
import logconfig


import geopandas as geopd
import networkx as nx
import osmnx as ox

import pandas as pd
import shapely
from shapely import ops
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, base

CACHE = Path("~/.cache").expanduser() / "osm_retrieve_networks"
CACHE.mkdir(parents=True, exist_ok=True)

DATA = Path("../data")
DATA.mkdir(parents=True, exist_ok=True)

ox.settings.cache_folder = CACHE
logconfig.setup_logging("DEBUG")
log = logging.getLogger(__name__)


def osm_powerlines(
    place: str, fillvoltage: float | None = None
) -> tuple[geopd.GeoDataFrame, geopd.GeoDataFrame]:
    """Retrieve the powerline network.

    Parameters
    ----------
    place: str :
        Retrieve data from this place: e.g. `Padova`, `London`, `France`…

    fillvoltage: float | None :
         (Default value = None)

    Returns
    -------


    """
    # Retrieve lines from OpenStreetMap.
    log.info("Retrieving edges from OpenStreetMap.")
    edges = retrieve_edges(keys={"power": ["cable", "line"]}, place=place).rename(
        columns={"source": "power_source"}
    )
    edges.voltage = _clean_voltage(edges.voltage)
    if fillvoltage is None:
        edges = edges.dropna(subset="voltage")
        edges = edges[edges.voltage > 69000]
    else:
        # Retains the lower voltage (1k-69k Volts) power-lines.
        # Assumes non reported voltage is lower than 69kV.
        # This is risky as the network is not that accurately reported.
        edges.voltage = edges.voltage.fillna(fillvoltage)
        log.warning("You assume that edges with no voltage are below 69k.")
    edges = join_edges_at_points(edges)

    log.info("Retrieving substations from OpenStreetMap.")
    # Retrieve substations (additional nodes).
    # These substations are used to *merge* lines that do not touch.
    substations = retrieve_nodes(keys={"power": ["substation"]}, place=place)

    log.info("Get nodes from edges extremes.")
    edges = add_fuzzy_nodes(substations, edges, distance=500)

    return nodes_from_lines(edges)


def retrieve_nodes(keys: dict, place: str) -> geopd.GeoDataFrame:
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
            _data = ox.features_from_place(place, {k: vals})
        except ox._errors.InsufficientResponseError:
            log.warning(f"No data for {dict(k=vals)}. Skipping")
            pass
        else:
            if isinstance(_data, list):
                data.extend(_data)
            else:
                data.append(_data)

    # Build a `DataFrame`
    nodes = geopd.GeoDataFrame(pd.concat(data), geometry="geometry", crs=4326).droplevel(0)
    log.info(f"Collected {len(nodes)} nodes.")
    # Transform everything to nodes
    nodes.geometry = nodes.geometry.representative_point()
    # Drop duplicates
    return nodes.loc[~nodes.index.duplicated(keep="first"), :]


def retrieve_edges(keys: dict, place: str) -> geopd.GeoDataFrame:
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
        data.append(ox.features_from_place(place, {k: vals}).loc["way"])

    # Build a dataframe
    edges = geopd.GeoDataFrame(pd.concat(data), geometry="geometry")
    edges.geometry = edges.geometry.line_merge()
    # Keep only `LineString` objects
    edges = edges[edges.geometry.geom_type == "LineString"]
    # Remove duplicates
    edges = edges[~edges.index.duplicated(keep="first")]
    edges["name"] = range(len(edges))
    log.info(f"Collected {len(edges)} edges.")

    return geopd.GeoDataFrame(edges, geometry="geometry", crs=4326)


# More utilities


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
                "geometry": lambda x: MultiPoint(x).centroid,
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
    mnodes = nodes[["name", "geometry"]].copy().to_crs(3857)
    medges = edges[["name", "geometry"]].copy().to_crs(3857)

    # Merge all points that are closer than `distance`
    mnodes = merge_points_within_distance(mnodes, distance)

    # Use `sindex` for quickness
    sindex = shapely.strtree.STRtree(mnodes.geometry)
    update = {}
    for id, medge in medges.geometry.items():
        try:
            source, targert = medge.boundary.geoms
        except ValueError:
            # Circular lines!!!!
            continue

        imes = sindex.query_nearest(source, max_distance=distance, all_matches=False)
        imet = sindex.query_nearest(targert, max_distance=distance, all_matches=False)

        ids = np.concat([imes, imet])
        assert len(ids) <= 2

        # If both ends of the edge fall close to the same point,
        # remove that edge
        if len(ids) == 2 and ids[0] == ids[1]:
            continue

        if len(ids) > 0 and isinstance(medge, LineString):
            mns = mnodes.geometry.iloc[ids]
            update[id] = _join_line_point_buffered(
                medge,
                geopd.GeoSeries(mns, index=mnodes.index[ids]),
                distance,
            )
            continue

        update[id] = medge
        continue

    update = geopd.GeoSeries(update, crs=mnodes.crs).dropna().to_crs(edges.crs)
    edges = edges.loc[update.index]
    edges.geometry = update.geometry
    return edges


def _join_line_point_buffered(
    line: LineString, points: geopd.GeoSeries, distance: float
) -> LineString:
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
        buffer = point.buffer(distance).boundary
        intersection = _line.intersection(buffer)

        if not isinstance(intersection, Point):
            continue

        l1 = max(
            ops.split(_line, buffer).geoms,
            key=lambda x: shapely.distance(x, point),
        )
        l2 = LineString([intersection, point])
        _line = shapely.line_merge(MultiLineString([l1, l2]))
    return _line


def nodes_from_lines(
    edges: geopd.GeoDataFrame, source_col: str = "source", target_col: str = "target"
) -> tuple[geopd.GeoDataFrame, geopd.GeoDataFrame]:
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
    node_map: dict[base.BaseGeometry, int] = {}
    source_target: list[dict[str, int | Hashable]] = []
    for id, line in edges.geometry.items():
        s, t = line.boundary.geoms
        s_id = node_map.setdefault(s, len(node_map))
        t_id = node_map.setdefault(t, len(node_map))
        source_target.append({source_col: s_id, target_col: t_id, "index": id})

    nodes = geopd.GeoDataFrame(
        [{"id": v, "geometry": k} for k, v in node_map.items()],
        geometry="geometry",
        crs=edges.crs,
    ).set_index("id", drop=True)

    edges = geopd.GeoDataFrame(
        pd.concat([edges, pd.DataFrame(source_target).set_index("index", drop=True)], axis=1)
    )
    return nodes, edges


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
        edges_touch = MultiLineString(lines=edges.geometry.loc[touches].tolist())
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


def main() -> None:
    """Do the main."""
    location = "Italy"
    nodes, edges = osm_powerlines(location, fillvoltage=None)
    nodes.to_file(Path("test_nodes.geojson"))
    edges.sort_values(by="voltage").to_file(Path("test_edges.geojson"))

    city = ox.geocoder.geocode_to_gdf(location)
    city.to_file("test_boundaries.geojson")

    g = nx.from_pandas_edgelist(edges, source="source", target="target")
    print(len(list(nx.connected_components(g))))

    # nx.draw_networkx(g)
    # plt.show()


if __name__ == "__main__":
    main()

check_cache_size()
