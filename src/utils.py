import json
import math
import os

import numpy as np
from numba import njit


@njit(cache=True)
def z_scale(z):
    """
    2^z represents the tile number. Scale that by the number of pixels in each tile.
    """
    PIXELS_PER_TILE = 256
    return 2**z * PIXELS_PER_TILE


def longitude_to_pixel(longitude, zoom):
    return ((longitude + 180) / 360) * z_scale(zoom)


def latitude_to_pixel(latitude, zoom):
    lat_rad = (latitude * math.pi) / 180
    return (
        (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2
    ) * z_scale(zoom)


@njit(cache=True)
def web_mercator_x_to_pixel_x(x, zoom):
    return (x + (40075016.68557849 / 2.0)) / (40075016.68557849 / (z_scale(zoom)))


@njit(cache=True)
def web_mercator_y_to_pixel_y(y, zoom):
    return (y - (40075016.68557849 / 2.0)) / (40075016.68557849 / (-1 * z_scale(zoom)))


def coordinate_to_pixel(
    input, zoom, return_dict=True, round_int=False, web_mercator=False
):
    """
    Convert coordinate to pixel coordinate
    """
    if web_mercator:
        x = web_mercator_x_to_pixel_x(input[0], zoom)
        y = web_mercator_y_to_pixel_y(input[1], zoom)
    else:
        x = longitude_to_pixel(input[0], zoom)
        y = latitude_to_pixel(input[1], zoom)
    if round_int:
        x = round(x)
        y = round(y)
    if return_dict:
        return {"x": x, "y": y}
    else:
        return [x, y]


def compute_r5_surface(grid: dict, percentile: int) -> np.array:
    """
    Compute single value surface from the grid
    """
    if (
        grid["data"] is None
        or grid["width"] is None
        or grid["height"] is None
        or grid["depth"] is None
    ):
        return None
    travel_time_percentiles = [5, 25, 50, 75, 95]
    percentile_index = travel_time_percentiles.index(percentile)

    if grid["depth"] == 1:
        # if only one percentile is requested, return the grid as is
        surface = grid["data"]
    else:
        grid_percentiles = np.reshape(grid["data"], (grid["depth"], -1))
        surface = grid_percentiles[percentile_index]

    return surface.astype(np.uint16)


@njit(cache=True)
def coordinate_from_pixel(input, zoom, round_int=False, web_mercator=False):
    """
    Convert pixel coordinate to longitude and latitude
    """
    if web_mercator:
        x = pixel_x_to_web_mercator_x(input[0], zoom)
        y = pixel_y_to_web_mercator_y(input[1], zoom)
    else:
        x = pixel_to_longitude(input[0], zoom)
        y = pixel_to_latitude(input[1], zoom)
    if round_int:
        x = round(x)
        y = round(y)

    return [x, y]


@njit(cache=True)
def pixel_x_to_web_mercator_x(x, zoom):
    return x * (40075016.68557849 / (z_scale(zoom))) - (40075016.68557849 / 2.0)


@njit(cache=True)
def pixel_y_to_web_mercator_y(y, zoom):
    return y * (40075016.68557849 / (-1 * z_scale(zoom))) + (40075016.68557849 / 2.0)


@njit(cache=True)
def pixel_to_longitude(pixel_x, zoom):
    """
    Convert pixel x coordinate to longitude
    """
    return (pixel_x / z_scale(zoom)) * 360 - 180


@njit(cache=True)
def pixel_to_latitude(pixel_y, zoom):
    """
    Convert pixel y coordinate to latitude
    """
    lat_rad = math.atan(math.sinh(math.pi * (1 - (2 * pixel_y) / z_scale(zoom))))
    return lat_rad * 180 / math.pi


def decode_r5_grid(grid_data_buffer: bytes) -> dict:
    """
    Decode R5 grid data
    """
    CURRENT_VERSION = 0
    HEADER_ENTRIES = 7
    HEADER_LENGTH = 9  # type + entries
    TIMES_GRID_TYPE = "ACCESSGR"

    # -- PARSE HEADER
    ## - get header type
    header = {}
    header_data = np.frombuffer(grid_data_buffer, count=8, dtype=np.byte)
    header_type = "".join(map(chr, header_data))
    if header_type != TIMES_GRID_TYPE:
        raise ValueError("Invalid grid type")
    ## - get header data
    header_raw = np.frombuffer(
        grid_data_buffer, count=HEADER_ENTRIES, offset=8, dtype=np.int32
    )
    version = header_raw[0]
    if version != CURRENT_VERSION:
        raise ValueError("Invalid grid version")
    header["zoom"] = header_raw[1]
    header["west"] = header_raw[2]
    header["north"] = header_raw[3]
    header["width"] = header_raw[4]
    header["height"] = header_raw[5]
    header["depth"] = header_raw[6]
    header["version"] = version

    # -- PARSE DATA --
    gridSize = header["width"] * header["height"]
    # - skip the header
    data = np.frombuffer(
        grid_data_buffer,
        offset=HEADER_LENGTH * 4,
        count=gridSize * header["depth"],
        dtype=np.int32,
    )
    # - reshape the data
    data = data.reshape(header["depth"], gridSize)
    reshaped_data = np.array([], dtype=np.int32)
    for i in range(header["depth"]):
        reshaped_data = np.append(reshaped_data, data[i].cumsum())
    data = reshaped_data
    # - decode metadata
    raw_metadata = np.frombuffer(
        grid_data_buffer,
        offset=(HEADER_LENGTH + header["width"] * header["height"] * header["depth"])
        * 4,
        dtype=np.int8,
    )
    metadata = json.loads(raw_metadata.tostring())

    return header | metadata | {"data": data, "errors": [], "warnings": []}


def make_dir(dir_path: str):
    """Creates a new directory if it doesn't already exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def format_value_null_sql(value) -> str:
    if value is None:
        return "NULL"
    else:
        return f"'{value}'"
