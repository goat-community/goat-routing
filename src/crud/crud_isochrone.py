import time
import uuid
from typing import Any

import polars as pl

from src.core.config import settings
from src.core.isochrone import compute_isochrone
from src.core.jsoline import generate_jsolines
from src.db.db import Database

# from src.db import models
# from src.db.session import legacy_engine
from src.schemas.isochrone import IIsochroneActiveMobility, TravelTimeCostActiveMobility

segment_schema = {
    "id": pl.Int64,
    "length_m": pl.Float64,
    "length_3857": pl.Float64,
    "class_": pl.Utf8,
    "impedance_slope": pl.Float64,
    "impedance_slope_reverse": pl.Float64,
    "impedance_surface": pl.Float32,
    "coordinates_3857": pl.Utf8,
    "source": pl.Int64,
    "target": pl.Int64,
    "tags": pl.Utf8,
    "h3_3": pl.Int32,
    "h3_6": pl.Int32,
}

valid_walking_classes = [
    "secondary",
    "tertiary",
    "residential",
    "livingStreet",
    "trunk",
    "unclassified",
    "parkingAisle",
    "driveway",
    "pedestrian",
    "footway",
    "steps",
    "track",
    "bridleway",
    "unknown",
]

valid_bicycle_classes = [
    "secondary",
    "tertiary",
    "residential",
    "livingStreet",
    "trunk",
    "unclassified",
    "parkingAisle",
    "driveway",
    "pedestrian",
    "track",
    "cycleway",
    "bridleway",
    "unknown",
]


class FetchRoutingNetwork:
    def fetch(self):
        """Fetch routing network (processed segments) and load into memory."""

        start_time = time.time()

        # Get network H3 cells
        h3_3_grid = []
        db = Database(settings.POSTGRES_DATABASE_URI)
        try:
            sql_get_h3_3_grid = f"""
                WITH region AS (
                    SELECT geom from {settings.NETWORK_REGION_TABLE}
                )
                SELECT g.h3_short FROM region r,
                LATERAL temporal.fill_polygon_h3_3(r.geom) g;
            """
            for h3_index in db.select(sql_get_h3_3_grid):
                if h3_index[0] in [8077]:
                    h3_3_grid.append(h3_index[0])
        except Exception as e:
            print(e)
        finally:
            db.conn.close()

        # Load segments into polars data frames
        segments_df = {}
        df_size = 0.0
        try:
            for h3_index in h3_3_grid:
                print(f"Loading network for H3 cell {h3_index}.")

                segments_df[h3_index] = pl.read_database_uri(
                    query=f"""
                        SELECT
                            id, length_m, length_3857,
                            class_, impedance_slope, impedance_slope_reverse,
                            impedance_surface, CAST(coordinates_3857 AS TEXT) AS coordinates_3857,
                            source, target, CAST(tags AS TEXT) AS tags, h3_3, h3_6
                        FROM basic.segment
                        WHERE h3_3 = {h3_index}
                    """,
                    uri=settings.POSTGRES_DATABASE_URI,
                    schema_overrides=segment_schema,
                )
                segments_df[h3_index] = segments_df[h3_index].with_columns(
                    pl.col("coordinates_3857").str.json_extract()
                )
                df_size += segments_df[h3_index].estimated_size("gb")
        except Exception as e:
            print(e)

        print(f"Network load time: {round((time.time() - start_time) / 60, 1)} min")
        print(f"Network in-memory size: {round(df_size, 1)} GB")

        return segments_df


class CRUDIsochrone:
    def read_network(
        self, db, routing_network: dict, obj_in: IIsochroneActiveMobility
    ) -> Any:
        """Read relevant sub-network for isochrone calculation from polars dataframe."""

        # Create input table for isochrone origin points
        input_table, num_points = self.create_input_table(db, obj_in)

        # Get valid segment classes based on transport mode
        valid_segment_classes = (
            valid_walking_classes
            if obj_in.routing_type == "walking"
            else valid_bicycle_classes
        )

        # Compute buffer distance for identifying relevant H3_6 cells
        if type(obj_in.travel_cost) == TravelTimeCostActiveMobility:
            buffer_dist = obj_in.travel_cost.max_traveltime * (
                (obj_in.travel_cost.speed * 1000) / 60
            )
        else:
            buffer_dist = obj_in.travel_cost.max_distance

        # Identify H3_3 & H3_6 cells relevant to this isochrone calculation
        h3_3_cells = set()
        h3_6_cells = set()

        sql_get_relevant_cells = f"""
            WITH point AS (
                SELECT geom FROM temporal.\"{input_table}\" LIMIT {num_points}
            ),
            num_cells AS (
                SELECT generate_series(0, CASE WHEN value < 1.0 THEN 0 ELSE ROUND(value)::int END) AS value
                FROM (SELECT ({buffer_dist} / (h3_get_hexagon_edge_length_avg(6, 'm') * 2)) AS value) sub
            ),
            cells AS (
                SELECT DISTINCT h3_grid_ring_unsafe(h3_lat_lng_to_cell(point.geom::point, 6), sub.value) AS h3_index
                FROM point,
                LATERAL (SELECT * FROM num_cells) sub
            )
            SELECT to_short_h3_3(h3_cell_to_parent(h3_index, 3)::bigint) AS h3_3, ARRAY_AGG(to_short_h3_6(h3_index::bigint)) AS h3_6
            FROM cells
            GROUP BY h3_3;
        """
        for h3_3_cell in db.select(sql_get_relevant_cells):
            h3_3_cells.add(h3_3_cell[0])
            for h3_6_cell in h3_3_cell[1]:
                h3_6_cells.add(h3_6_cell)

        # Get relevant segments & connectors
        sub_network = pl.DataFrame()
        for h3_3 in h3_3_cells:
            sub_df = routing_network[h3_3].filter(
                pl.col("h3_6").is_in(h3_6_cells)
                & pl.col("class_").is_in(valid_segment_classes)
            )
            if sub_network.width > 0:
                sub_network.extend(sub_df)
            else:
                sub_network = sub_df

        # Create necessary artifical segments and add them to our sub network
        origin_point_connectors = []
        segments_to_discard = []
        sql_get_artificial_segments = f"""
            SELECT
                point_id,
                old_id,
                id, length_m, length_3857, class_, impedance_slope,
                impedance_slope_reverse, impedance_surface,
                CAST(coordinates_3857 AS TEXT) AS coordinates_3857,
                source, target, tags, h3_3, h3_6
            FROM temporal.get_artificial_segments(
                '{input_table}',
                {num_points},
                '{",".join(valid_segment_classes)}'
            );
        """
        for a_seg in db.select(sql_get_artificial_segments):
            if a_seg[0] is not None:
                origin_point_connectors.append(a_seg[10])
                segments_to_discard.append(a_seg[1])

            new_df = pl.DataFrame(
                [
                    {
                        "id": a_seg[2],
                        "length_m": a_seg[3],
                        "length_3857": a_seg[4],
                        "class_": a_seg[5],
                        "impedance_slope": a_seg[6],
                        "impedance_slope_reverse": a_seg[7],
                        "impedance_surface": a_seg[8],
                        "coordinates_3857": a_seg[9],
                        "source": a_seg[10],
                        "target": a_seg[11],
                        "tags": a_seg[12],
                        "h3_3": a_seg[13],
                        "h3_6": a_seg[14],
                    }
                ],
                schema_overrides=segment_schema,
            )
            new_df = new_df.with_columns(pl.col("coordinates_3857").str.json_extract())
            sub_network.extend(new_df)

        # Remove segments which are now replaced by artificial segments
        sub_network = sub_network.filter(~pl.col("id").is_in(segments_to_discard))

        # TODO: We need to read the scenario network dynamically from DB if a scenario is selected.

        # Replace all NULL values in the impedance columns with 0
        sub_network = sub_network.with_columns(pl.col("impedance_slope").fill_null(0))
        sub_network = sub_network.with_columns(
            pl.col("impedance_slope_reverse").fill_null(0)
        )
        sub_network = sub_network.with_columns(pl.col("impedance_surface").fill_null(0))

        # Compute cost for each segment
        if type(obj_in.travel_cost) == TravelTimeCostActiveMobility:
            # If producing a travel time cost based isochrone, compute segment cost accordingly
            sub_network = self.compute_segment_cost(
                sub_network,
                obj_in.routing_type,
                obj_in.travel_cost.speed / 3.6,
            )
        else:
            # If producing a distance cost based isochrone, use the segment length as cost
            sub_network = sub_network.with_columns(
                pl.col("length_m").alias("cost"),
                pl.col("length_m").alias("reverse_cost"),
            )

        # Select columns required for computing isochrone and convert to dictionary of numpy arrays
        sub_network = {
            "id": sub_network.get_column("id").to_numpy().copy(),
            "source": sub_network.get_column("source").to_numpy().copy(),
            "target": sub_network.get_column("target").to_numpy().copy(),
            "cost": sub_network.get_column("cost").to_numpy().copy(),
            "reverse_cost": sub_network.get_column("reverse_cost").to_numpy().copy(),
            "length": sub_network.get_column("length_3857").to_numpy().copy(),
            "geom": sub_network.get_column("coordinates_3857").to_numpy().copy(),
        }

        return sub_network, origin_point_connectors

    def create_input_table(self, db, obj_in: IIsochroneActiveMobility):
        """Create the input table for the isochrone calculation."""

        # Generate random table name
        table_name = str(uuid.uuid4()).replace("-", "_")

        # Create temporary table for storing the isochrone input
        db.perform(
            f"""
                CREATE TABLE temporal.\"{table_name}\" (
                    id serial PRIMARY KEY,
                    geom geometry(Point, 4326)
                );
            """
        )

        # Insert the isochrone input into the temporary table
        for i in range(len(obj_in.starting_points.latitude)):
            latitude = obj_in.starting_points.latitude[i]
            longitude = obj_in.starting_points.longitude[i]
            db.perform(
                f"""
                    INSERT INTO temporal.\"{table_name}\" (geom)
                    VALUES (ST_SetSRID(ST_MakePoint({latitude}, {longitude}), 4326));
                """
            )

        return table_name, len(obj_in.starting_points.latitude)

    def compute_segment_cost(self, sub_network, mode, speed):
        """Compute the cost of a segment based on the mode, speed, impedance, etc."""

        if mode == "walking":
            return sub_network.with_columns(
                (pl.col("length_m") / speed).alias("cost"),
                (pl.col("length_m") / speed).alias("reverse_cost"),
            )
        elif mode == "bicycle":
            return sub_network.with_columns(
                pl.when(pl.col("class_") != "pedestrian")
                .then(
                    (
                        pl.col("length_m")
                        * (1 + pl.col("impedance_slope") + pl.col("impedance_surface"))
                    )
                    / speed
                )
                .otherwise(
                    pl.col("length_m") / speed
                )  # This calculation is invoked when the segment class requires cyclists to walk their bicycle
                .alias("cost"),
                pl.when(pl.col("class_") != "pedestrian")
                .then(
                    (
                        pl.col("length_m")
                        * (
                            1
                            + pl.col("impedance_slope_reverse")
                            + pl.col("impedance_surface")
                        )
                    )
                    / speed
                )
                .otherwise(
                    pl.col("length_m") / speed
                )  # This calculation is invoked when the segment class requires cyclists to walk their bicycle
                .alias("reverse_cost"),
            )
        elif mode == "pedelec":
            return sub_network.with_columns(
                pl.when(pl.col("class_") != "pedestrian")
                .then((pl.col("length_m") * (1 + pl.col("impedance_surface"))) / speed)
                .otherwise(
                    pl.col("length_m") / speed
                )  # This calculation is invoked when the segment class requires cyclists to walk their pedelec
                .alias("cost"),
                pl.when(pl.col("class_") != "pedestrian")
                .then((pl.col("length_m") * (1 + pl.col("impedance_surface"))) / speed)
                .otherwise(
                    pl.col("length_m") / speed
                )  # This calculation is invoked when the segment class requires cyclists to walk their pedelec
                .alias("reverse_cost"),
            )
        else:
            return None

    def save_result(self, db, obj_in: IIsochroneActiveMobility, shapes, network, grid):
        """Save the result of the isochrone computation to the database."""

        if obj_in.isochrone_type == "polygon":
            # Save isochrone geometry data (shapes)
            shapes = shapes["incremental"]
            insert_string = ""
            for i in shapes.index:
                geom = shapes["geometry"][i]
                minute = shapes["minute"][i]
                insert_string += f"('{obj_in.layer_id}', ST_SetSRID(ST_GeomFromText('{geom}'), 4326), {minute}),"
            insert_string = f"""
                INSERT INTO user_data.{obj_in.result_table} (layer_id, geom, integer_attr1)
                VALUES {insert_string.rstrip(",")};
            """
            db.perform(insert_string)
        elif obj_in.isochrone_type == "network":
            # Save isochrone network data
            batch_size = 1000
            insert_string = ""
            for i in range(0, len(network["features"])):
                coordinates = network["features"][i]["geometry"]["coordinates"]
                cost = network["features"][i]["properties"]["cost"]
                points_string = ""
                for pair in coordinates:
                    points_string += f"ST_MakePoint({pair[0]}, {pair[1]}),"
                insert_string += f"""(
                    '{obj_in.layer_id}',
                    ST_Transform(ST_SetSRID(ST_MakeLine(ARRAY[{points_string.rstrip(',')}]), 3857), 4326),
                    {cost}
                ),"""
                if i % batch_size == 0 or i == (len(network["features"]) - 1):
                    insert_string = f"""
                        INSERT INTO user_data.{obj_in.result_table} (layer_id, geom, float_attr1)
                        VALUES {insert_string.rstrip(",")};
                    """
                    db.perform(insert_string)
                    insert_string = ""
        else:
            # Save isochrone grid data
            pass

    def run(self, routing_network: dict, obj_in: IIsochroneActiveMobility):
        """Compute isochrones for the given request parameters."""

        total_start = time.time()

        # Read & process routing network to extract relevant sub-network
        start_time = time.time()
        db = Database(settings.POSTGRES_DATABASE_URI)
        sub_routing_network = None
        origin_connector_ids = None
        try:
            sub_routing_network, origin_connector_ids = self.read_network(
                db,
                routing_network,
                obj_in,
            )
        except Exception as e:
            print(e)
            raise e  # TODO Return error status/message instead
        finally:
            db.close()
        print(f"Network read time: {round(time.time() - start_time, 2)} sec")

        # Compute isochrone utilizing processed sub-network
        start_time = time.time()
        isochrone_grid = None
        isochrone_network = None
        isochrone_shapes = None
        try:
            is_travel_time_isochrone = (
                type(obj_in.travel_cost) == TravelTimeCostActiveMobility
            )

            isochrone_grid, isochrone_network = compute_isochrone(
                edge_network_input=sub_routing_network,
                start_vertices=origin_connector_ids,
                travel_time=obj_in.travel_cost.max_traveltime
                if is_travel_time_isochrone
                else obj_in.travel_cost.max_distance,
                speed=obj_in.travel_cost.speed / 3.6
                if is_travel_time_isochrone
                else None,
                zoom=12,
                use_distance=(not is_travel_time_isochrone),
            )
            print("Computed isochrone grid & network.")

            if obj_in.isochrone_type == "polygon":
                isochrone_shapes = generate_jsolines(
                    grid=isochrone_grid,
                    travel_time=obj_in.travel_cost.max_traveltime
                    if is_travel_time_isochrone
                    else obj_in.travel_cost.max_distance,
                    percentile=5,
                    step=obj_in.travel_cost.traveltime_step
                    if is_travel_time_isochrone
                    else obj_in.travel_cost.distance_step,  # TODO Fix shape for distance cost based isochrones
                )
                print("Computed isochrone shapes.")
        except Exception as e:
            print(e)
            raise e  # TODO Return error status/message instead
        print(f"Isochrone computation time: {round(time.time() - start_time, 2)} sec")

        # Write output of isochrone computation to database
        start_time = time.time()
        db = Database(settings.POSTGRES_DATABASE_URI)
        try:
            self.save_result(
                db, obj_in, isochrone_shapes, isochrone_network, isochrone_grid
            )
        except Exception as e:
            print(e)
            raise e  # TODO Return error status/message instead
        finally:
            db.close()
        print(f"Result save time: {round(time.time() - start_time, 2)} sec")

        print(f"Total time: {round(time.time() - total_start, 2)} sec")
