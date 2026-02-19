"""Extract data from `.osm.pbf` files.

some code from
https://github.com/do-me/fast-osm-extraction
"""

from collections import Counter
from pathlib import Path
from typing import Literal

import duckdb
import geopandas as gpd
import pandas as pd


def build_query(
    pbf_file: Path,
    kind: Literal["railway", "power_distribution"] = "railway",
    geom: Literal["points", "lines"] = "lines",
) -> str:
    """Build the query to perform on the database."""
    print(pbf_file)
    if kind == "railway":
        if geom == "lines":
            tags = {
                "railway": ["rail", "construction", "preserved", "narrow_gauge"],
                "construction:railway": ["rail"],
                "route": ["train"],
                "construction": ["rail", "railway"],
                "gauge": True,
                "electrified": True,
            }
        else:
            tags = {"railway": ["station", "halt", "stop"]}
    elif kind == "power_distribution":
        tags = {
            "transformer": ["distribution"],
            "substation": ["minor_distribution"],
            "power": ["substation"],
        }
    else:
        raise NotImplementedError()

    built_query = " \n    OR ".join(
        [
            f"tags['{k}'] IN ({', '.join([f"'{x}'" for x in v])})"
            if isinstance(v, list)
            else f"tags['{k}'] IS NOT NULL"
            for k, v in tags.items()
        ]
    )

    if geom == "lines":
        return f"""
        PRAGMA enable_optimizer;
        WITH
            -- Step 1: Filter ways based on tags first. This is the most selective filter.
            -- The optimizer will push this down to the OSM reader.
            filtered_ways AS (
                SELECT id, tags, refs
                FROM st_readOSM('{pbf_file}')
                WHERE
                    kind = 'way'
                    AND ({built_query})
                    -- It's good practice to also exclude proposed/planned features if not wanted
                    AND (tags['construction'] IS NULL OR tags['construction'] NOT IN ('proposed', 'planned'))
                    AND array_length(refs) >= 2
            ),

            -- Step 2: Get only the nodes that are referenced by our filtered ways.
            -- This is more direct than creating a separate CTE for IDs.
            nodes AS (
                SELECT id, ST_Point(lon, lat) AS geom
                FROM st_readOSM('{pbf_file}')
                WHERE
                    kind = 'node'
                    AND id IN (SELECT UNNEST(refs) FROM filtered_ways)
            ),

            -- Step 3: Unnest way-node relationships for ordering
            way_nodes AS (
                SELECT
                    w.id AS way_id,
                    w.tags,
                    UNNEST(w.refs) AS node_id,
                    UNNEST(generate_series(1, array_length(w.refs))) AS node_order
                FROM filtered_ways w
            )

        -- Step 4: Final geometry construction
        SELECT
            wn.way_id AS id,
            wn.tags,
            ST_AsWKB(ST_MakeLine(LIST(n.geom ORDER BY wn.node_order))) AS geom_wkb
        FROM way_nodes wn
        INNER JOIN nodes n ON wn.node_id = n.id
        GROUP BY wn.way_id, wn.tags;
        """

    return f"""
    PRAGMA enable_optimizer;
    WITH
        -- Step 1: Filter nodes based on tags.
        -- This is more direct than creating a separate CTE for IDs.
        nodes AS (
            SELECT id, tags, ST_Point(lon, lat) AS geom
            FROM st_readOSM('{pbf_file}')
            WHERE
                kind = 'node'
                AND ({built_query})
                AND (tags['construction'] IS NULL OR tags['construction'] NOT IN ('proposed', 'planned'))
        )

        -- Step 3: Unnest way-node relationships for ordering
        -- way_nodes AS (
        --     SELECT
        --         w.id AS way_id,
        --         w.tags,
        --         UNNEST(w.refs) AS node_id,
        --         UNNEST(generate_series(1, array_length(w.refs))) AS node_order
        --     FROM filtered_ways w
        -- )

    -- Step 4: Final geometry construction
    SELECT
        n.id AS id,
        n.tags,
        ST_AsWKB(n.geom) AS geom_wkb
    FROM nodes n
    GROUP BY n.id, n.tags, n.geom;
    """


def dict_eval(data: str | dict, cols: set[str]):
    """Convert to dictionary only tags within `cols`."""
    d = eval(data) if isinstance(data, str) else data
    ks = cols.intersection(d.keys())
    ko = set(d.keys()) - cols
    new_tags = {k: d[k] for k in ks}
    if len(ko) > 0:
        new_tags["other_tags"] = str({k: d[k] for k in ko})
    return new_tags


def togdf(query: str, explode: int = 50):
    with duckdb.connect(database=":memory:") as con:
        # Optimize DuckDB for M3 Max performance
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")

        # Performance tuning for M3 Max
        con.execute("SET threads=16;")  # M3 Max with 16 perf. cores
        con.execute("SET memory_limit='24GB';")  # Dont even need all that 128Gb RAM!
        con.execute("SET max_memory='24GB';")
        con.execute("SET temp_directory='./tmp';")
        con.execute("SET enable_progress_bar=true;")

        # Parallel processing optimizations
        con.execute("SET preserve_insertion_order=false;")  # Allow reordering for performance

        print("Running optimized geometry query...")
        df = con.sql(query).to_df()

    print(f"Query completed! Retrieved {len(df)} items.")

    gdf = gpd.GeoDataFrame(
        df.drop(columns=["geom_wkb"]),
        geometry=gpd.GeoSeries.from_wkb(df["geom_wkb"].apply(bytes)),
        crs="EPSG:4326",
    )

    # Explode tags
    if isinstance(gdf["tags"].iloc[0], dict):
        tag_counter = Counter([k for v in gdf["tags"] for k in v.keys()])
    else:
        tag_counter = Counter([k for v in gdf["tags"] for k in eval(v).keys()])
    cols = {x[0] for x in tag_counter.most_common(explode)}

    data_tags = pd.DataFrame([dict_eval(t, cols) for t in gdf["tags"]], index=gdf.index)
    gdf = gpd.GeoDataFrame(
        pd.concat([gdf.drop(columns="tags"), data_tags], axis=1), geometry="geometry", crs=gdf.crs
    ).dropna(axis="columns", how="all")
    return gdf


pbf_file = Path("~/curro/working_data/osm_sources/europe-latest.osm.pbf").expanduser()
# pbf_file = Path("~/Downloads/albania-260210.osm.pbf").expanduser()


gdf = pd.concat(
    [
        togdf(build_query(pbf_file, kind="power_distribution", geom="points")),
        togdf(build_query(pbf_file, kind="power_distribution", geom="lines")),
    ],
    axis=0,
)
gdf.geometry = gdf.geometry.representative_point()
gdf.to_file("power_distribution.gpkg")
exit()

gdf = togdf(build_query(pbf_file, kind="railway", geom="points"))
gdf.to_file(Path("EU_railways.gpkg"), driver="GPKG", layer="points", mode="w")
gdf = togdf(build_query(pbf_file, kind="railway", geom="lines"))
gdf.to_file(Path("EU_railways.gpkg"), driver="GPKG", layer="ways", mode="w")
