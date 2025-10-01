"""Loag a Graph and get only the largest component."""

from pathlib import Path

import geopandas as gpd
import osm
import shapely

datapath = Path("../data/graphs/")

graph = osm.Graph.read(datapath / "graph_all_graph.gpkg")
print(graph)
lc = graph.largest_component()
print(lc)
lc.write(datapath / "graph_largestComponent_graph.gpkg")

# Merge all regions and overwrite the union
regions = gpd.GeoDataFrame(
    [
        {
            "name": region_path.name.split("_")[1],
            "geometry": shapely.from_geojson(region_path.read_text()),
        }
        for region_path in datapath.glob("*region*.geojson")
        if region_path.name.split("_")[1][0] not in {"a", "l"}
    ],
    crs=osm.PRJ_DEG,
)
regions.to_file(datapath / "graph_largestComponent_region.geojson")
