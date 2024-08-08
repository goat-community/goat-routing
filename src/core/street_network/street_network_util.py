import time
from uuid import UUID

import polars as pl
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.street_network.street_network_cache import StreetNetworkCache
from src.schemas.catchment_area import CONNECTOR_DATA_SCHEMA, SEGMENT_DATA_SCHEMA


class StreetNetworkUtil:
    def __init__(self, db_connection: AsyncSession):
        self.db_connection = db_connection

    async def _get_layer_and_user_id(self, layer_project_id: int):
        """Get the layer ID and user ID of the specified layer project ID."""

        layer_id: UUID = None
        user_id: UUID = None

        try:
            # Get the associated layer ID
            result = await self.db_connection.execute(
                text(
                    f"""SELECT layer_id
                FROM {settings.CUSTOMER_SCHEMA}.layer_project
                WHERE id = {layer_project_id};"""
                )
            )
            layer_id = UUID(str(result.fetchone()[0]))

            # Get the user ID of the layer
            result = await self.db_connection.execute(
                text(
                    f"""SELECT user_id
                FROM {settings.CUSTOMER_SCHEMA}.layer
                WHERE id = '{layer_id}';"""
                )
            )
            user_id = UUID(str(result.fetchone()[0]))
        except Exception:
            raise ValueError(
                f"Could not fetch layer and user ID for layer project ID {layer_project_id}."
            )

        return layer_id, user_id

    async def _get_street_network_tables(
        self,
        street_network_edge_layer_project_id: int,
        street_network_node_layer_project_id: int,
    ):
        """Get table names and layer IDs of the edge and node tables."""

        edge_table: str = None
        edge_layer_id: UUID = None
        node_table: str = None
        node_layer_id: UUID = None

        # Get edge table name if a layer project ID is specified
        if street_network_edge_layer_project_id:
            try:
                # Get the edge layer ID and associated user ID
                edge_layer_id, user_id = await self._get_layer_and_user_id(
                    street_network_edge_layer_project_id
                )

                # Produce the edge table name
                edge_table = f"{settings.USER_DATA_SCHEMA}.street_network_line_{str(user_id).replace('-', '')}"
            except Exception:
                raise ValueError(
                    f"Could not fetch edge table name for layer project ID {street_network_edge_layer_project_id}."
                )

        # Get node table name if a layer project ID is specified
        if street_network_node_layer_project_id:
            try:
                # Get the node layer ID and associated user ID
                node_layer_id, user_id = await self._get_layer_and_user_id(
                    street_network_node_layer_project_id
                )

                # Produce the node table name
                node_table = f"{settings.USER_DATA_SCHEMA}.street_network_point_{str(user_id).replace('-', '')}"
            except Exception:
                raise ValueError(
                    f"Could not fetch node table name for layer project ID {street_network_node_layer_project_id}."
                )

        return edge_table, edge_layer_id, node_table, node_layer_id

    async def _get_street_network_region_h3_3_cells(self, region_geofence_table: str):
        """Get list of H3_3 cells covering the street network region."""

        h3_3_cells = []
        try:
            sql_fetch_h3_3_cells = f"""
                WITH region AS (
                    SELECT ST_Union(geom) AS geom FROM {region_geofence_table}
                )
                SELECT g.h3_short FROM region r,
                LATERAL basic.fill_polygon_h3_3(r.geom) g;
            """
            result = (
                await self.db_connection.execute(text(sql_fetch_h3_3_cells))
            ).fetchall()

            for h3_short in result:
                h3_3_cells.append(h3_short[0])
        except Exception:
            raise ValueError(
                f"Could not fetch H3_3 grid for street network geofence {settings.NETWORK_REGION_TABLE}."
            )

        return h3_3_cells

    async def fetch(
        self,
        edge_layer_project_id: int,
        node_layer_project_id: int,
        region_geofence_table: str,
    ):
        """Fetch street network from specified layer and load into Polars dataframes."""

        # Street network is stored as a dictionary of Polars dataframes, with the H3_3 index as the key
        street_network_edge: dict = {}
        street_network_node: dict = {}

        start_time = time.time()
        street_network_size: float = 0.0

        # Get H3_3 cells covering the street network region
        street_network_region_h3_3_cells = (
            await self._get_street_network_region_h3_3_cells(region_geofence_table)
        )

        # Get table names and layer IDs of the edge and node tables
        (
            street_network_edge_table,
            street_network_edge_layer_id,
            street_network_node_table,
            street_network_node_layer_id,
        ) = await self._get_street_network_tables(
            edge_layer_project_id, node_layer_project_id
        )

        # Initialize cache
        street_network_cache = StreetNetworkCache()

        try:
            for h3_short in street_network_region_h3_3_cells:
                if edge_layer_project_id is not None:
                    if street_network_cache.edge_cache_exists(
                        edge_layer_project_id, h3_short
                    ):
                        # Read edge data from cache
                        edge_df = street_network_cache.read_edge_cache(
                            edge_layer_project_id, h3_short
                        )
                    else:
                        if settings.DEBUG_MODE:
                            print(
                                f"Fetching street network edge data for H3_3 cell {h3_short}"
                            )

                        # Read edge data from database
                        edge_df = pl.read_database_uri(
                            query=f"""
                                SELECT
                                    edge_id AS id, length_m, length_3857, class_, impedance_slope, impedance_slope_reverse,
                                    impedance_surface, CAST(coordinates_3857 AS TEXT) AS coordinates_3857, maxspeed_forward,
                                    maxspeed_backward, source, target, h3_3, h3_6
                                FROM {street_network_edge_table}
                                WHERE h3_3 = {h3_short}
                                AND layer_id = '{str(street_network_edge_layer_id)}'
                            """,
                            uri=settings.POSTGRES_DATABASE_URI,
                            schema_overrides=SEGMENT_DATA_SCHEMA,
                        )
                        edge_df = edge_df.with_columns(
                            pl.col("coordinates_3857").str.json_extract()
                        )

                        # Write edge data into cache
                        street_network_cache.write_edge_cache(
                            edge_layer_project_id, h3_short, edge_df
                        )
                    # Update street network edge dictionary and memory usage
                    street_network_edge[h3_short] = edge_df
                    street_network_size += edge_df.estimated_size("gb")

                if node_layer_project_id is not None:
                    if street_network_cache.node_cache_exists(
                        node_layer_project_id, h3_short
                    ):
                        # Read node data from cache
                        node_df = street_network_cache.read_node_cache(
                            node_layer_project_id, h3_short
                        )
                    else:
                        if settings.DEBUG_MODE:
                            print(
                                f"Fetching street network node data for H3_3 cell {h3_short}"
                            )

                        # Read node data from database
                        node_df = pl.read_database_uri(
                            query=f"""
                                SELECT node_id AS id, h3_3, h3_6
                                FROM {street_network_node_table}
                                WHERE h3_3 = {h3_short}
                                AND layer_id = '{str(street_network_node_layer_id)}'
                            """,
                            uri=settings.POSTGRES_DATABASE_URI,
                            schema_overrides=CONNECTOR_DATA_SCHEMA,
                        )

                        # Write node data into cache
                        street_network_cache.write_node_cache(
                            node_layer_project_id, h3_short, node_df
                        )

                    # Update street network node dictionary and memory usage
                    street_network_node[h3_short] = node_df
                    street_network_size += node_df.estimated_size("gb")
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch street network data from database, error: {e}"
            )

        # Raise error if a edge layer project ID is specified but no edge data is fetched
        if edge_layer_project_id is not None and len(street_network_edge) == 0:
            raise RuntimeError(
                f"Failed to fetch street network edge data for layer project ID {edge_layer_project_id}."
            )

        # Raise error if a node layer project ID is specified but no node data is fetched
        if node_layer_project_id is not None and len(street_network_node) == 0:
            raise RuntimeError(
                f"Failed to fetch street network node data for layer project ID {node_layer_project_id}."
            )

        end_time = time.time()

        if settings.DEBUG_MODE:
            print(
                f"Street network load time: {round((end_time - start_time) / 60, 1)} min"
            )
            print(f"Street network in-memory size: {round(street_network_size, 1)} GB")

        return street_network_edge, street_network_node
