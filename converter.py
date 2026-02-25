"""
KangahGPS — Coordinate conversion engine.
Supports War Office (EPSG:2136), WGS 84 (EPSG:4326), and UTM zones.
Supports Molodensky (3-param) and Bursa-Wolf (7-param) transformation methods.
"""

import math
from pyproj import Transformer, CRS


# ── Coordinate Reference Systems ─────────────────────────────────────────────

CRS_OPTIONS = {
    "War Office / Ghana National Grid (EPSG:2136)": {
        "epsg": "EPSG:2136",
        "fields": ("Easting (Gold Coast ft)", "Northing (Gold Coast ft)"),
        "is_latlon": False,
        "is_utm": False,
    },
    "WGS 84 — Lat/Lon (EPSG:4326)": {
        "epsg": "EPSG:4326",
        "fields": ("Latitude", "Longitude"),
        "is_latlon": True,
        "is_utm": False,
    },
    "UTM (auto-detect zone)": {
        "epsg": None,  # determined at runtime
        "fields": ("Easting (m)", "Northing (m)"),
        "is_latlon": False,
        "is_utm": True,
    },
}


# ── Transformation Methods ────────────────────────────────────────────────────
# The War Office ellipsoid parameters used in the proj string:
#   a = 6378300, rf = 296
#
# Molodensky 3-parameter (EPSG:6896, ~6 m accuracy, default for Ghana onshore):
#   dX=-170, dY=33, dZ=326
#
# Bursa-Wolf 7-parameter (derived from oil industry / research studies):
#   dX=-171.16, dY=17.29, dZ=325.21, rX=0, rY=0, rZ=0.814, dS=-0.38
#   (Position Vector convention, EPSG:8571/15495)

TRANSFORM_METHODS = {
    "Molodensky (3-param, ~6 m)": {
        "key": "molodensky",
        "towgs84": "-170,33,326,0,0,0,0",
        "description": "3 translation parameters (EPSG:6896). ~6 m accuracy. Default for onshore Ghana.",
    },
    "Bursa-Wolf (7-param, ~1 m)": {
        "key": "bursa_wolf",
        "towgs84": "-171.16,17.29,325.21,0,0,0.814,-0.38",
        "description": "7 parameters with rotation & scale (EPSG:8571). ~1 m accuracy.",
    },
}

TRANSFORM_NAMES = list(TRANSFORM_METHODS.keys())


def _build_waroffice_crs(towgs84: str) -> CRS:
    """Build a War Office geographic CRS with the given towgs84 parameters."""
    proj_str = (
        "+proj=longlat +a=6378300 +rf=296 "
        f"+towgs84={towgs84} +no_defs +type=crs"
    )
    return CRS.from_proj4(proj_str)


def _build_waroffice_projected_crs(towgs84: str) -> CRS:
    """Build the Ghana National Grid projected CRS with the given towgs84."""
    proj_str = (
        "+proj=tmerc +lat_0=4.66666666666667 +lon_0=-1 +k=0.99975 "
        "+x_0=274319.739163358 +y_0=0 "
        f"+a=6378300 +rf=296 +towgs84={towgs84} "
        "+to_meter=0.304799710181509 +no_defs +type=crs"
    )
    return CRS.from_proj4(proj_str)


def utm_zone_from_lon(lon: float) -> int:
    """Return the UTM zone number for a given longitude."""
    return int((lon + 180) / 6) + 1


def utm_epsg(zone: int, northern: bool = True) -> str:
    """Return the EPSG code string for a UTM zone."""
    base = 32600 if northern else 32700
    return f"EPSG:{base + zone}"


def get_epsg(crs_name: str, *, lon: float | None = None, zone: int | None = None) -> str:
    """
    Resolve the EPSG code for a CRS name.
    For UTM, either provide `lon` (auto-detect) or `zone` (manual).
    """
    info = CRS_OPTIONS[crs_name]
    if not info["is_utm"]:
        return info["epsg"]

    if zone is not None:
        return utm_epsg(zone)
    if lon is not None:
        return utm_epsg(utm_zone_from_lon(lon))
    # Default for Ghana
    return "EPSG:32630"


def _resolve_crs(epsg_str: str, transform_method: str | None = None):
    """
    Resolve a CRS object. If epsg_str is the War Office CRS and a
    transform_method is given, build a custom CRS with the appropriate
    towgs84 parameters. Otherwise use the EPSG code directly.
    """
    if transform_method and epsg_str == "EPSG:2136":
        method_info = TRANSFORM_METHODS[transform_method]
        return _build_waroffice_projected_crs(method_info["towgs84"])
    return CRS.from_user_input(epsg_str)


def convert_single(
    src_epsg: str,
    tgt_epsg: str,
    val1: float,
    val2: float,
    src_is_latlon: bool,
    tgt_is_latlon: bool,
    transform_method: str | None = None,
) -> tuple[float, float]:
    """
    Convert a single coordinate pair.
    val1/val2 are in user order: (lat, lon) for latlon, (easting, northing) otherwise.
    Returns (out1, out2) in the same convention.

    transform_method: one of TRANSFORM_NAMES, or None for pyproj default.
    """
    src_crs = _resolve_crs(src_epsg, transform_method)
    tgt_crs = _resolve_crs(tgt_epsg, transform_method)
    transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

    if src_is_latlon:
        x_in, y_in = val2, val1  # lon, lat
    else:
        x_in, y_in = val1, val2  # easting, northing

    x_out, y_out = transformer.transform(x_in, y_in)

    if tgt_is_latlon:
        return y_out, x_out  # lat, lon
    return x_out, y_out


def convert_batch(
    src_epsg: str,
    tgt_epsg: str,
    coords: list[tuple[float, float]],
    src_is_latlon: bool,
    tgt_is_latlon: bool,
    transform_method: str | None = None,
) -> list[tuple[float, float]]:
    """Convert a list of coordinate pairs."""
    src_crs = _resolve_crs(src_epsg, transform_method)
    tgt_crs = _resolve_crs(tgt_epsg, transform_method)
    transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
    results = []
    for val1, val2 in coords:
        if src_is_latlon:
            x_in, y_in = val2, val1
        else:
            x_in, y_in = val1, val2

        x_out, y_out = transformer.transform(x_in, y_in)

        if tgt_is_latlon:
            results.append((y_out, x_out))
        else:
            results.append((x_out, y_out))
    return results
