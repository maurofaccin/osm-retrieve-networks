from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import tqdm


def load_regions(
    buffer_size: float | None = None, grow_regions: float | None = None, test: bool | None = None
) -> gpd.GeoDataFrame:
    """Load the countries (and regions) with an optional buffer.

    For now Europe.
    """
    fn = "../data/prov_test.geojson" if test else "../data/regions_europe.geojson"
    keep = ["code", "geometry"] if test else ["code", "geometry", "NAME_0", "GID_0"]
    regions = gpd.read_file(fn).rename(columns={"index": "code"})[keep]
    full = regions.union_all()

    if grow_regions is not None:
        regions.geometry = [
            shapely.union(
                g, shapely.difference(shapely.buffer(g, distance=grow_regions, quad_segs=4), full)
            )
            for g in tqdm.tqdm(regions.geometry, desc="Grow cells")
        ]

    if buffer_size is None or buffer_size <= 0.0:
        return regions

    reg_buffer = buffer(regions, buf_size=buffer_size, cell_size=buffer_size)
    regions = gpd.GeoDataFrame(pd.concat([regions, reg_buffer]))

    return regions


def buffer(
    data: gpd.GeoDataFrame, buf_size: float = 2.0, cell_size: float | None = None
) -> gpd.GeoDataFrame:
    """Create a buffer around a `GeoDataFrame`."""
    cache = Path(f"../data/regions_europe_buffer_{buf_size:5.4f}.geojson")
    print(cache)

    if cache.is_file() and False:
        buffer = gpd.read_file(cache)
        return buffer

    # Create one buffer per time to use less Memory.
    buf = shapely.Polygon()
    for country in tqdm.tqdm(data.geometry):
        buf = buf.union(country.buffer(buf_size), grid_size=0.01)
    buf = buf.difference(data.union_all(method="coverage"))
    buf = buf.buffer(0)  # fix eventual non valid polygons

    if cell_size is None:
        cells = gpd.GeoSeries([buf], index=["BUF"], crs=data.crs)
    else:
        cells = split_cells(buf, cell_size=cell_size)

    buffer = gpd.GeoDataFrame({"geometry": cells}, geometry="geometry", crs=4326)
    buffer["code"] = [f"BUFFER_{i:04d}" for i in range(len(buffer))]
    buffer["GID_0"] = "XXX"
    buffer["NAME_0"] = "Buffer"

    # Cache it
    buffer.to_file(cache)
    return buffer


def split_cells(areas: shapely.MultiPolygon, cell_size: float = 2.0) -> gpd.GeoSeries:
    """Split large polygons in smaller cells."""

    bounds = areas.bounds
    nx = int((bounds[2] - bounds[0]) / cell_size) + 1
    ny = int((bounds[3] - bounds[1]) / cell_size) + 1

    xes = np.linspace(bounds[0], bounds[2], nx)
    yes = np.linspace(bounds[1], bounds[3], ny)

    tiles = gpd.GeoSeries(
        [
            shapely.box(x1, y1, x2, y2)
            for x1, x2 in zip(xes, xes[1:], strict=False)
            for y1, y2 in zip(yes, yes[1:], strict=False)
        ]
    )
    tiles = tiles[tiles.intersects(areas)]
    tiles = tiles.normalize().intersection(areas)

    return tiles
