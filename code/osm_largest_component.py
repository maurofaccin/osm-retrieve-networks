"""Loag a Graph and get only the largest component."""

from pathlib import Path

import geopandas as gpd
import osm
import shapely
import pandas as pd

datapath = Path("../data/graphs/")

graph = osm.Graph.read(datapath / "graph_all_graph.gpkg")
print(graph)
lc = graph.largest_component()
print(lc)
lc.write(datapath / "graph_largestComponent_graph.gpkg")

# Merge all regions and overwrite the union
regions = gpd.GeoDataFrame(
    pd.concat(
        [
            gpd.read_file(region_path, layer="region").rename(
                index={0: region_path.name.split("_")[1]}
            )
            for region_path in datapath.glob("*_[A-Z]*.gpkg")
        ]
    ),
    crs=osm.PRJ_DEG,
)
regions.to_file(datapath / "graph_largestComponent_region.geojson")
