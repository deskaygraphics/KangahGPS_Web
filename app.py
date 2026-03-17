#!/usr/bin/env python3
"""
KangahGPS Web — Advanced Ghana Coordinate Converter
=====================================================
Features:
  1. Single-point conversion with auto UTM zone detection
  2. Batch conversion via CSV upload/download
  3. Map preview for WGS 84 output
"""

import io
import json
import zipfile
import tempfile
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import Draw, MeasureControl, LocateControl
from streamlit_folium import st_folium
from converter import (
    CRS_OPTIONS,
    TRANSFORM_METHODS,
    TRANSFORM_NAMES,
    convert_single,
    convert_batch,
    get_epsg,
    utm_zone_from_lon,
)


# ── Export helpers ────────────────────────────────────────────────────────────


def _make_gdf(df: pd.DataFrame, lat_col: str, lon_col: str) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame from a DataFrame with lat/lon columns."""
    geometry = [Point(lon, lat) for lat, lon in zip(df[lat_col], df[lon_col])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def _to_kml(gdf: gpd.GeoDataFrame, name_prefix: str = "Point") -> str:
    """Convert a GeoDataFrame of points to KML string."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        f"  <name>KangahGPS Export</name>",
    ]
    for i, row in gdf.iterrows():
        pt = row.geometry
        lines.append("  <Placemark>")
        lines.append(f"    <name>{name_prefix} {i+1}</name>")
        lines.append(f"    <description>Lat: {pt.y:.8f}, Lon: {pt.x:.8f}</description>")
        lines.append("    <Point>")
        lines.append(f"      <coordinates>{pt.x},{pt.y},0</coordinates>")
        lines.append("    </Point>")
        lines.append("  </Placemark>")
    lines.append("</Document>")
    lines.append("</kml>")
    return "\n".join(lines)


def _to_shapefile_zip(gdf: gpd.GeoDataFrame) -> bytes:
    """Write GeoDataFrame to a zipped shapefile in memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = f"{tmpdir}/kangahgps_export.shp"
        gdf.to_file(shp_path, driver="ESRI Shapefile")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            import os

            for fname in os.listdir(tmpdir):
                zf.write(f"{tmpdir}/{fname}", fname)
        return buf.getvalue()


def show_export_buttons(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    key_prefix: str,
):
    """Show CSV, KML, and Shapefile download buttons."""
    st.markdown("##### 📥 Export")
    gdf = _make_gdf(df, lat_col, lon_col)

    ecol1, ecol2, ecol3 = st.columns(3)
    with ecol1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 CSV",
            data=csv_bytes,
            file_name="kangahgps_export.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_csv",
        )
    with ecol2:
        kml_str = _to_kml(gdf)
        st.download_button(
            "🌐 KML",
            data=kml_str.encode("utf-8"),
            file_name="kangahgps_export.kml",
            mime="application/vnd.google-earth.kml+xml",
            use_container_width=True,
            key=f"{key_prefix}_kml",
        )
    with ecol3:
        shp_zip = _to_shapefile_zip(gdf)
        st.download_button(
            "🗺️ Shapefile (.zip)",
            data=shp_zip,
            file_name="kangahgps_export_shp.zip",
            mime="application/zip",
            use_container_width=True,
            key=f"{key_prefix}_shp",
        )


# ── Distance helper ──────────────────────────────────────────────────────────

from math import radians, cos, sin, asin, sqrt, atan2, degrees


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float):
    """Return (meters, feet) between two WGS-84 points."""
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    )
    m = R * 2 * asin(sqrt(a))
    return m, m * 3.28084


def _polygon_order(connections):
    """Return ordered list of point indices forming a closed polygon, or None."""
    if len(connections) < 3:
        return None
    adj = {}
    for c in connections:
        i, j = c["from"], c["to"]
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    nodes = list(adj)
    if not all(len(adj[n]) == 2 for n in nodes):
        return None
    start = nodes[0]
    path, prev, cur = [start], None, start
    for _ in range(len(nodes)):
        a, b = adj[cur]
        nxt = b if a == prev else a
        if nxt == start:
            break
        if nxt in path[1:]:
            return None
        path.append(nxt)
        prev, cur = cur, nxt
    return path if len(path) == len(nodes) and len(path) >= 3 else None


def _polygon_area(pts):
    """Return (m², ft², acres, hectares) for a list of {lat, lon} dicts."""
    from pyproj import Geod
    from shapely.geometry import Polygon

    geod = Geod(ellps="WGS84")
    coords = [(p["lon"], p["lat"]) for p in pts]
    area_m2 = abs(geod.geometry_area_perimeter(Polygon(coords))[0])
    return area_m2, area_m2 * 10.7639, area_m2 / 4_046.856, area_m2 / 10_000


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return bearing in degrees (0=N, 90=E, 180=S, 270=W) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    return (degrees(atan2(x, y)) + 360) % 360


def _cardinal(brg: float) -> str:
    """Convert a bearing in degrees to a 16-point compass label (e.g. 'NNE')."""
    dirs = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    return dirs[round(brg / 22.5) % 16]


def _relative_direction(bearing: float, heading: float) -> str:
    """Return a human-readable turn direction given device heading and bearing to target."""
    diff = (bearing - heading + 360) % 360
    if diff < 22.5 or diff >= 337.5:
        return "straight ahead"
    elif diff < 67.5:
        return "bear right"
    elif diff < 112.5:
        return "turn right"
    elif diff < 157.5:
        return "sharp right"
    elif diff < 202.5:
        return "behind you — turn around"
    elif diff < 247.5:
        return "sharp left"
    elif diff < 292.5:
        return "turn left"
    else:
        return "bear left"


# ── Basemaps ───────────────────────────────────────────────────────────────

BASEMAPS = {
    "🏙️ Light (CartoDB)": {"tiles": "CartoDB positron", "attr": None},
    "🌑 Dark (CartoDB)": {"tiles": "CartoDB dark_matter", "attr": None},
    "🗺️ Streets (OSM)": {"tiles": "OpenStreetMap", "attr": None},
    "🛰️ Satellite (Esri)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Source: Esri, USGS, AeroGRID, IGN & GIS Community",
    },
    "🏔️ Topo (Esri)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China, and the GIS Community",
    },
    "🌍 NatGeo (Esri)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC",
    },
    "🛣️ World Street (Esri)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China, Esri Thailand, TomTom, 2012",
    },
}


def _make_map(location: list, zoom: int) -> folium.Map:
    """Create a folium map with all basemaps switchable via the layer control."""
    m = folium.Map(location=location, zoom_start=zoom, tiles=None)
    first = True
    for name, bm in BASEMAPS.items():
        tl_kwargs = {"name": name, "show": first}
        if bm["attr"]:
            tl_kwargs["attr"] = bm["attr"]
        folium.TileLayer(bm["tiles"], **tl_kwargs).add_to(m)
        first = False
    folium.LayerControl(position="bottomleft", collapsed=True).add_to(m)
    return m


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="KangahGPS — Ghana Coordinate Converter",
    page_icon="🌍",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <h1 style='text-align: center;'>🌍 KangahGPS</h1>
    <p style='text-align: center; color: grey;'>
        Ghana Coordinate Converter &mdash; War Office &bull; WGS 84 &bull; UTM
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar: Transformation Method ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Datum Transformation Method**")
    st.caption(
        "Choose how War Office coordinates are transformed to/from WGS 84. "
        "This only affects conversions involving the War Office CRS."
    )
    tfm_name = st.radio(
        "Method",
        TRANSFORM_NAMES,
        index=0,
        key="tfm_method",
    )
    tfm_info = TRANSFORM_METHODS[tfm_name]
    st.info(tfm_info["description"])
    st.markdown(f"`towgs84={tfm_info['towgs84']}`")


# ── Tabs ─────────────────────────────────────────────────────────────────────────────

tab_single, tab_batch, tab_draw, tab_about = st.tabs(
    ["📍 Single Conversion", "📄 Batch Conversion", "✏️ Draw & Plot", "ℹ️ About"]
)

CRS_NAMES = list(CRS_OPTIONS.keys())


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Single Conversion
# ═══════════════════════════════════════════════════════════════════════════════

with tab_single:
    col_src, col_tgt = st.columns(2)

    with col_src:
        st.subheader("Source")
        src_name = st.selectbox("Source CRS", CRS_NAMES, index=0, key="src_crs")
        src_info = CRS_OPTIONS[src_name]
        f1, f2 = src_info["fields"]

        # UTM zone selector when source is UTM
        src_utm_zone = None
        if src_info["is_utm"]:
            src_utm_zone = st.number_input(
                "UTM Zone", min_value=1, max_value=60, value=30, step=1, key="src_zone"
            )

        val1 = st.number_input(
            f1, format="%.8f" if src_info["is_latlon"] else "%.3f", key="v1"
        )
        val2 = st.number_input(
            f2, format="%.8f" if src_info["is_latlon"] else "%.3f", key="v2"
        )

    with col_tgt:
        st.subheader("Target")
        default_tgt = 1 if src_name == CRS_NAMES[0] else 0
        tgt_name = st.selectbox(
            "Target CRS", CRS_NAMES, index=default_tgt, key="tgt_crs"
        )
        tgt_info = CRS_OPTIONS[tgt_name]

        # UTM zone selector when target is UTM
        tgt_utm_zone = None
        if tgt_info["is_utm"]:
            auto_zone = st.checkbox("Auto-detect UTM zone", value=True, key="auto_zone")
            if not auto_zone:
                tgt_utm_zone = st.number_input(
                    "UTM Zone",
                    min_value=1,
                    max_value=60,
                    value=30,
                    step=1,
                    key="tgt_zone",
                )

    st.markdown("")
    convert_clicked = st.button("🔄  Convert", use_container_width=True, type="primary")

    if convert_clicked:
        if src_name == tgt_name:
            st.warning("Source and target CRS are the same.")
        else:
            try:
                # Resolve source EPSG
                if src_info["is_utm"]:
                    src_epsg = get_epsg(src_name, zone=src_utm_zone)
                else:
                    src_epsg = get_epsg(src_name)

                # Resolve target EPSG
                if tgt_info["is_utm"]:
                    if tgt_utm_zone is not None:
                        tgt_epsg = get_epsg(tgt_name, zone=tgt_utm_zone)
                    else:
                        if src_info["is_latlon"]:
                            auto_lon = val2
                        else:
                            tmp_lat, tmp_lon = convert_single(
                                src_epsg,
                                "EPSG:4326",
                                val1,
                                val2,
                                src_info["is_latlon"],
                                True,
                                tfm_name,
                            )
                            auto_lon = tmp_lon
                        detected_zone = utm_zone_from_lon(auto_lon)
                        tgt_epsg = get_epsg(tgt_name, zone=detected_zone)
                else:
                    tgt_epsg = get_epsg(tgt_name)

                out1, out2 = convert_single(
                    src_epsg,
                    tgt_epsg,
                    val1,
                    val2,
                    src_info["is_latlon"],
                    tgt_info["is_latlon"],
                    tfm_name,
                )

                # Store results in session state so they persist
                st.session_state.single_result = {
                    "out1": out1,
                    "out2": out2,
                    "tf1": tgt_info["fields"][0],
                    "tf2": tgt_info["fields"][1],
                    "is_latlon": tgt_info["is_latlon"],
                    "src_is_latlon": src_info["is_latlon"],
                    "src_val1": val1,
                    "src_val2": val2,
                    "zone": (
                        (tgt_utm_zone or detected_zone) if tgt_info["is_utm"] else None
                    ),
                }

            except Exception as e:
                st.session_state.pop("single_result", None)
                st.error(f"Conversion error: {e}")

    # ── Display persisted results ───────────────────────────────────────────────
    if "single_result" in st.session_state:
        r = st.session_state.single_result
        out1, out2 = r["out1"], r["out2"]
        fmt = ".8f" if r["is_latlon"] else ".3f"

        st.success("Conversion successful!")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric(r["tf1"], f"{out1:{fmt}}")
        res_col2.metric(r["tf2"], f"{out2:{fmt}}")

        if r["zone"]:
            st.info(f"UTM Zone: **{r['zone']}N**  (EPSG:{32600 + r['zone']})")

        # Map preview
        if r["is_latlon"]:
            st.markdown("##### 🗺️ Map Preview")
            preview = _make_map([out1, out2], 10)
            folium.Marker(
                [out1, out2],
                popup=f"Lat: {out1:.6f}<br>Lon: {out2:.6f}",
                icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
            ).add_to(preview)
            st_folium(preview, width=700, height=400, key="single_map")
        elif r["src_is_latlon"]:
            st.markdown("##### 🗺️ Map Preview (source location)")
            preview = _make_map([r["src_val1"], r["src_val2"]], 10)
            folium.Marker(
                [r["src_val1"], r["src_val2"]],
                popup=f"Lat: {r['src_val1']:.6f}<br>Lon: {r['src_val2']:.6f}",
                icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
            ).add_to(preview)
            st_folium(preview, width=700, height=400, key="single_map_src")

        # Send to Draw
        _send_lat = (
            out1 if r["is_latlon"] else (r["src_val1"] if r["src_is_latlon"] else None)
        )
        _send_lon = (
            out2 if r["is_latlon"] else (r["src_val2"] if r["src_is_latlon"] else None)
        )
        if _send_lat is not None:
            if st.button(
                "📌 Send to Draw", key="single_send_draw", use_container_width=True
            ):
                if "draw_points" not in st.session_state:
                    st.session_state.draw_points = []
                st.session_state.draw_points.append(
                    {"lat": _send_lat, "lon": _send_lon}
                )
                st.success(
                    f"Point sent to Draw & Plot tab! ({_send_lat:.6f}, {_send_lon:.6f})"
                )

        # Export single point
        single_df = pd.DataFrame(
            {
                r["tf1"]: [out1],
                r["tf2"]: [out2],
            }
        )
        if r["is_latlon"]:
            show_export_buttons(single_df, r["tf1"], r["tf2"], "single")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Batch Conversion (CSV)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.subheader("Batch Conversion")
    st.markdown(
        "Upload a **CSV or Excel** file (.csv, .xlsx, .xls) with coordinate columns. "
        "You can include an **ID** column (optional) plus "
        "**Easting/Longitude/X** and **Northing/Latitude/Y**."
    )

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        bsrc_name = st.selectbox("Source CRS", CRS_NAMES, index=0, key="bsrc")
        bsrc_info = CRS_OPTIONS[bsrc_name]
        bsrc_zone = None
        if bsrc_info["is_utm"]:
            bsrc_zone = st.number_input("Source UTM Zone", 1, 60, 30, key="bsrc_zone")
    with bcol2:
        btgt_name = st.selectbox("Target CRS", CRS_NAMES, index=1, key="btgt")
        btgt_info = CRS_OPTIONS[btgt_name]
        btgt_zone = None
        if btgt_info["is_utm"]:
            b_auto = st.checkbox("Auto-detect UTM zone", value=True, key="b_auto_zone")
            if not b_auto:
                btgt_zone = st.number_input(
                    "Target UTM Zone", 1, 60, 30, key="btgt_zone"
                )

    uploaded = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="csv_upload",
        help="Accepted formats: .csv, .xlsx, .xls",
    )

    if uploaded is not None:
        try:
            _fname = uploaded.name.lower()
            if _fname.endswith(".xlsx") or _fname.endswith(".xls"):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)
            st.markdown(f"**{len(df)} rows** loaded. Preview:")
            st.dataframe(df.head(10), use_container_width=True)

            col_names = df.columns.tolist()
            if len(col_names) < 2:
                st.error("File must have at least 2 columns.")
            else:
                # Column mapping
                st.markdown("##### Map Columns")
                mc1, mc2, mc3 = st.columns(3)
                bf1, bf2 = bsrc_info["fields"]

                with mc1:
                    id_col = st.selectbox(
                        "ID column (optional)",
                        ["— None —"] + col_names,
                        index=0,
                        key="id_col",
                    )
                with mc2:
                    # Try to auto-detect the first coord column
                    default_x = 1 if id_col != "— None —" and len(col_names) > 1 else 0
                    x_col = st.selectbox(
                        f"{bf1} column",
                        col_names,
                        index=min(default_x, len(col_names) - 1),
                        key="x_col",
                    )
                with mc3:
                    default_y = (
                        col_names.index(x_col) + 1
                        if col_names.index(x_col) + 1 < len(col_names)
                        else 0
                    )
                    remaining = [c for c in col_names if c != x_col]
                    y_col = st.selectbox(
                        f"{bf2} column",
                        remaining,
                        index=0,
                        key="y_col",
                    )

                if st.button(
                    "🔄  Convert All", use_container_width=True, key="batch_btn"
                ):
                    coords = list(zip(df[x_col], df[y_col]))

                    # Resolve EPSGs
                    if bsrc_info["is_utm"]:
                        bsrc_epsg = get_epsg(bsrc_name, zone=bsrc_zone)
                    else:
                        bsrc_epsg = get_epsg(bsrc_name)

                    if btgt_info["is_utm"] and btgt_zone is None:
                        first_v1, first_v2 = coords[0]
                        if bsrc_info["is_latlon"]:
                            auto_lon = first_v2
                        else:
                            _, auto_lon = convert_single(
                                bsrc_epsg,
                                "EPSG:4326",
                                first_v1,
                                first_v2,
                                bsrc_info["is_latlon"],
                                True,
                                tfm_name,
                            )
                        btgt_epsg = get_epsg(
                            btgt_name, zone=utm_zone_from_lon(auto_lon)
                        )
                    elif btgt_info["is_utm"]:
                        btgt_epsg = get_epsg(btgt_name, zone=btgt_zone)
                    else:
                        btgt_epsg = get_epsg(btgt_name)

                    results = convert_batch(
                        bsrc_epsg,
                        btgt_epsg,
                        coords,
                        bsrc_info["is_latlon"],
                        btgt_info["is_latlon"],
                        tfm_name,
                    )

                    tf1, tf2 = btgt_info["fields"]

                    # Build output dataframe with ID if present
                    out_data = {}
                    if id_col != "— None —":
                        out_data["ID"] = df[id_col].values
                    out_data[x_col] = df[x_col].values
                    out_data[y_col] = df[y_col].values
                    out_data[f"{tf1} (converted)"] = [r[0] for r in results]
                    out_data[f"{tf2} (converted)"] = [r[1] for r in results]
                    out_df = pd.DataFrame(out_data)

                    # Store in session state
                    st.session_state.batch_result = {
                        "out_df": out_df,
                        "results": results,
                        "is_latlon": btgt_info["is_latlon"],
                        "count": len(results),
                        "has_id": id_col != "— None —",
                    }

        except Exception as e:
            st.session_state.pop("batch_result", None)
            st.error(f"Error processing CSV: {e}")

    # ── Display persisted batch results ─────────────────────────────────────────
    if "batch_result" in st.session_state:
        br = st.session_state.batch_result
        st.success(f"Converted {br['count']} points!")
        st.dataframe(br["out_df"], use_container_width=True)

        if br["is_latlon"]:
            st.markdown("##### 🗺️ Map Preview")
            lats = [r[0] for r in br["results"]]
            lons = [r[1] for r in br["results"]]
            batch_map = _make_map([sum(lats) / len(lats), sum(lons) / len(lons)], 7)
            ids = br["out_df"]["ID"].values if br["has_id"] else None
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                label = f"ID: {ids[i]}" if ids is not None else f"Point {i+1}"
                folium.Marker(
                    [lat, lon],
                    popup=f"{label}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
                    tooltip=label,
                    icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
                ).add_to(batch_map)
            st_folium(batch_map, width=700, height=400, key="batch_map")

        # Export buttons
        if br["is_latlon"]:
            out_df = br["out_df"]
            # Find the converted lat/lon column names
            conv_cols = [c for c in out_df.columns if "(converted)" in c]
            if len(conv_cols) >= 2:
                show_export_buttons(out_df, conv_cols[0], conv_cols[1], "batch")
            if st.button(
                "📌 Send All to Draw", key="batch_send_draw", use_container_width=True
            ):
                if "draw_points" not in st.session_state:
                    st.session_state.draw_points = []
                for lat, lon in br["results"]:
                    st.session_state.draw_points.append({"lat": lat, "lon": lon})
                st.success(f"Sent {len(br['results'])} points to Draw & Plot tab!")
        else:
            csv_bytes = br["out_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Download Results (CSV)",
                data=csv_bytes,
                file_name="kangahgps_converted.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Draw & Plot
# ═══════════════════════════════════════════════════════════════════════════════

with tab_draw:
    st.subheader("Draw & Plot")
    st.markdown(
        "Plot points on the map, draw lines and polygons to connect them. "
        "You can also **add points manually** below the map."
    )

    # ── Manual point entry ────────────────────────────────────────────────
    st.markdown("##### Add Points")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        draw_src = st.selectbox("Input CRS", CRS_NAMES, index=0, key="draw_src")
        draw_src_info = CRS_OPTIONS[draw_src]
        draw_zone = None
        if draw_src_info["is_utm"]:
            draw_zone = st.number_input("UTM Zone", 1, 60, 30, key="draw_zone")
    with dcol2:
        df1, df2 = draw_src_info["fields"]
        fmt = "%.8f" if draw_src_info["is_latlon"] else "%.3f"
        pt_v1 = st.number_input(df1, format=fmt, key="pt_v1")
        pt_v2 = st.number_input(df2, format=fmt, key="pt_v2")

    if st.button("➕ Add Point", key="add_pt"):
        # Convert to WGS84 for plotting
        if draw_src_info["is_latlon"]:
            lat, lon = pt_v1, pt_v2
        else:
            src_epsg = get_epsg(draw_src, zone=draw_zone)
            lat, lon = convert_single(
                src_epsg,
                "EPSG:4326",
                pt_v1,
                pt_v2,
                draw_src_info["is_latlon"],
                True,
                tfm_name,
            )
        if "draw_points" not in st.session_state:
            st.session_state.draw_points = []
        st.session_state.draw_points.append({"lat": lat, "lon": lon})
        st.success(f"Added point: {lat:.6f}, {lon:.6f}")

    # Clear points
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear All Points", key="clear_pts"):
            st.session_state.draw_points = []
            st.session_state.connections = []
            st.session_state.connect_first = None
            st.session_state.connect_last = None
            st.session_state.last_processed_click = None
            st.rerun()
    with c2:
        connect = st.toggle(
            "Connect points as polyline", value=False, key="connect_pts"
        )

    points = st.session_state.get("draw_points", [])

    if connect:
        _cl_ui = st.session_state.get("connect_last")
        _cf_ui = st.session_state.get("connect_first")
        if len(points) < 2:
            st.info("ℹ️ Add at least 2 points to start connecting.")
        elif _cl_ui is None:
            st.info("🖱️ Click a point on the map to start the chain.")
        elif not st.session_state.get("connections"):
            st.info(
                f"📍 **Point {_cl_ui + 1}** selected — click the next point to connect."
            )
        else:
            st.info(
                f"🔗 Continuing from **Point {_cl_ui + 1}** — click next point, or click **Point {_cf_ui + 1}** to close the polygon."
            )
        if st.session_state.get("connections"):
            if st.button(
                "🗑️ Clear Connections", key="clear_conns", use_container_width=True
            ):
                st.session_state.connections = []
                st.session_state.connect_first = None
                st.session_state.connect_last = None
                st.session_state.last_processed_click = None
                st.rerun()

    # ── Build folium map ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 🗺️ Interactive Map")
    st.caption(
        "Use the toolbar on the left to draw markers, lines, polygons, or rectangles. "
        "Switch basemaps from the **layers panel on the right side** of the map."
    )

    # Initial map center/zoom (the component key keeps zoom stable across reruns)
    if points:
        center_lat = sum(p["lat"] for p in points) / len(points)
        center_lon = sum(p["lon"] for p in points) / len(points)
        zoom = 10
    else:
        center_lat, center_lon = 7.95, -1.03  # Ghana center
        zoom = 7

    m = _make_map([center_lat, center_lon], zoom)

    # Add draw tools
    Draw(
        draw_options={
            "polyline": {"shapeOptions": {"color": "#e74c3c", "weight": 3}},
            "polygon": {"shapeOptions": {"color": "#3498db", "weight": 2}},
            "rectangle": {"shapeOptions": {"color": "#2ecc71", "weight": 2}},
            "circle": False,
            "circlemarker": False,
            "marker": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    # Add measure tool
    MeasureControl(
        primary_length_unit="feet",
        secondary_length_unit="meters",
        primary_area_unit="sqmeters",
        secondary_area_unit="hectares",
    ).add_to(m)

    # GPS live location tracking (uses browser/device GPS)
    LocateControl(
        auto_start=False,
        flyTo=True,
        keepCurrentZoomLevel=False,
        strings={"title": "Track my location"},
        locateOptions={
            "enableHighAccuracy": True,
            "maxAge": 0,
            "timeout": 10000,
            "watch": True,
        },
    ).add_to(m)

    # Plot user points: blue = chain tip, green = start point (click to close)
    _connect_first = st.session_state.get("connect_first")
    _connect_last = st.session_state.get("connect_last")
    for i, pt in enumerate(points):
        if connect and i == _connect_last:
            color = "blue"
        elif connect and i == _connect_first and _connect_first != _connect_last:
            color = "green"
        else:
            color = "red"
        folium.Marker(
            location=[pt["lat"], pt["lon"]],
            popup=f"Point {i+1}<br>Lat: {pt['lat']:.6f}<br>Lon: {pt['lon']:.6f}",
            tooltip=f"Point {i+1}",
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(m)

    # Draw connections with distance labels + polygon fill if closed
    _conns = st.session_state.get("connections", [])
    _poly_order = _polygon_order(_conns) if connect else None
    if connect:
        # Filled polygon if connections form a closed loop
        if _poly_order:
            poly_coords = [[points[i]["lat"], points[i]["lon"]] for i in _poly_order]
            folium.Polygon(
                locations=poly_coords,
                color="#3498db",
                weight=2,
                fill=True,
                fill_color="#3498db",
                fill_opacity=0.15,
                dash_array="6 4",
                tooltip="Closed polygon",
            ).add_to(m)
        for conn in _conns:
            i, j = conn["from"], conn["to"]
            if i < len(points) and j < len(points):
                p1, p2 = points[i], points[j]
                folium.PolyLine(
                    locations=[[p1["lat"], p1["lon"]], [p2["lat"], p2["lon"]]],
                    color="#e74c3c",
                    weight=3,
                    opacity=0.8,
                ).add_to(m)
                mid_lat = (p1["lat"] + p2["lat"]) / 2
                mid_lon = (p1["lon"] + p2["lon"]) / 2
                label = f"{conn['feet']:.1f} ft  |  {conn['meters']:.1f} m"
                folium.Marker(
                    location=[mid_lat, mid_lon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:11px;color:#c0392b;font-weight:bold;white-space:nowrap;text-shadow:1px 1px 2px white,-1px -1px 2px white;">{label}</div>',
                        icon_size=(180, 22),
                        icon_anchor=(90, 11),
                    ),
                ).add_to(m)

    # Render map — key keeps the Leaflet instance alive across reruns (preserves zoom/pan)
    map_data = st_folium(
        m,
        width=700,
        height=500,
        key="draw_map",
        returned_objects=["all_drawings", "last_object_clicked"],
    )

    # ── Process map clicks for point connection ──────────────────────────
    if connect and map_data and len(points) >= 2:
        clicked = map_data.get("last_object_clicked")
        last_processed = st.session_state.get("last_processed_click")
        if clicked and clicked != last_processed:
            clat = clicked.get("lat")
            clng = clicked.get("lng")
            if clat is not None and clng is not None:
                closest_idx = min(
                    range(len(points)),
                    key=lambda k: (points[k]["lat"] - clat) ** 2
                    + (points[k]["lon"] - clng) ** 2,
                )
                min_d = (points[closest_idx]["lat"] - clat) ** 2 + (
                    points[closest_idx]["lon"] - clng
                ) ** 2
                if min_d < 1e-6:
                    st.session_state.last_processed_click = clicked
                    _cf = st.session_state.get("connect_first")
                    _cl = st.session_state.get("connect_last")
                    if _cl is None:
                        # First click — start the chain
                        st.session_state.connect_first = closest_idx
                        st.session_state.connect_last = closest_idx
                    elif closest_idx == _cl:
                        pass  # clicked the same point, ignore
                    else:
                        # Draw line from chain tip to the new point
                        p1, p2 = points[_cl], points[closest_idx]
                        meters, feet = _haversine(
                            p1["lat"], p1["lon"], p2["lat"], p2["lon"]
                        )
                        if "connections" not in st.session_state:
                            st.session_state.connections = []
                        pairs = [
                            (c["from"], c["to"]) for c in st.session_state.connections
                        ]
                        if (_cl, closest_idx) not in pairs and (
                            closest_idx,
                            _cl,
                        ) not in pairs:
                            st.session_state.connections.append(
                                {
                                    "from": _cl,
                                    "to": closest_idx,
                                    "meters": meters,
                                    "feet": feet,
                                }
                            )
                        if closest_idx == _cf:
                            # Clicked the start point — polygon closed!
                            st.session_state.connect_first = None
                            st.session_state.connect_last = None
                        else:
                            st.session_state.connect_last = closest_idx
                    st.rerun()

    # ── Auto-import markers
    if map_data and map_data.get("all_drawings"):
        _new_pts = False
        for _drawing in map_data["all_drawings"]:
            if _drawing.get("geometry", {}).get("type") == "Point":
                _dlon, _dlat = _drawing["geometry"]["coordinates"]
                _existing = st.session_state.get("draw_points", [])
                if not any(
                    abs(p["lat"] - _dlat) < 1e-8 and abs(p["lon"] - _dlon) < 1e-8
                    for p in _existing
                ):
                    st.session_state.setdefault("draw_points", []).append(
                        {"lat": _dlat, "lon": _dlon}
                    )
                    _new_pts = True
        if _new_pts:
            st.rerun()

    # ── Polygon area ──────────────────────────────────────────────────────
    if connect and _poly_order and len(_poly_order) >= 3:
        _poly_pts = [points[i] for i in _poly_order]
        _m2, _ft2, _acres, _ha = _polygon_area(_poly_pts)
        st.markdown("##### 📐 Polygon Area")
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("m²", f"{_m2:,.2f}")
        ac2.metric("ft²", f"{_ft2:,.2f}")
        ac3.metric("Acres", f"{_acres:.4f}")
        ac4.metric("Hectares", f"{_ha:.4f}")

    # ── Connections list with individual delete ─────────────────────────
    if connect and st.session_state.get("connections"):
        st.markdown("##### 🔗 Connections")
        for _cidx in range(len(st.session_state.connections) - 1, -1, -1):
            _conn = st.session_state.connections[_cidx]
            _ci, _cj = _conn["from"], _conn["to"]
            _cl1, _cl2 = st.columns([8, 1])
            with _cl1:
                st.write(
                    f"Point {_ci+1} → Point {_cj+1}: **{_conn['feet']:.1f} ft** | **{_conn['meters']:.1f} m**"
                )
            with _cl2:
                if st.button("✕", key=f"del_conn_{_cidx}", help="Delete"):
                    st.session_state.connections.pop(_cidx)
                    st.session_state.connect_first = None
                    st.session_state.connect_last = None
                    st.rerun()

    # ── Show drawn features as GeoJSON ────────────────────────────────────
    if map_data and map_data.get("all_drawings"):
        drawings = map_data["all_drawings"]
        if drawings:
            st.markdown("##### Drawn Features")
            geojson = {
                "type": "FeatureCollection",
                "features": drawings,
            }
            geojson_str = json.dumps(geojson, indent=2)
            st.json(geojson)

            st.download_button(
                "⬇️ Download GeoJSON",
                data=geojson_str,
                file_name="kangahgps_drawings.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )

    # ── Points table with individual delete ────────────────────────────
    if points:
        st.markdown("##### Points Table")
        for _pidx in range(len(points)):
            _pt = points[_pidx]
            _pp1, _pp2, _pp3 = st.columns([1, 7, 1])
            with _pp1:
                st.write(f"**{_pidx + 1}**")
            with _pp2:
                st.write(f"Lat: {_pt['lat']:.6f}  |  Lon: {_pt['lon']:.6f}")
            with _pp3:
                if st.button("✕", key=f"del_pt_{_pidx}", help="Delete point"):
                    st.session_state.draw_points.pop(_pidx)
                    # Remove connections involving this point and remap the rest
                    _new_conns = []
                    for _c in st.session_state.get("connections", []):
                        if _c["from"] == _pidx or _c["to"] == _pidx:
                            continue
                        _new_conns.append(
                            {
                                **_c,
                                "from": _c["from"] - (1 if _c["from"] > _pidx else 0),
                                "to": _c["to"] - (1 if _c["to"] > _pidx else 0),
                            }
                        )
                    st.session_state.connections = _new_conns
                    st.session_state.connect_first = None
                    st.session_state.connect_last = None
                    st.rerun()

        csv_bytes = pd.DataFrame(points).to_csv(index=True).encode("utf-8")
        st.download_button(
            "⬇️ Download Points (CSV)",
            data=csv_bytes,
            file_name="kangahgps_points.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_pts",
        )

    # ── Navigate to Boundary Points ──────────────────────────────────────────
    st.markdown("---")
    with st.expander("🧭 Navigate to Boundary Points", expanded=False):
        if not points:
            st.info("Add boundary points above, then use navigation here.")
        else:
            nav_src = st.radio(
                "Location source",
                ["📍 Browser GPS", "⌨️ Enter manually"],
                horizontal=True,
                key="nav_source",
            )
            my_loc = None

            if nav_src == "📍 Browser GPS":
                st.caption(
                    "Enable GPS to fetch your live position from the browser/device. "
                    "Allow location access when prompted."
                )
                gps_on = st.toggle("🛰️ Enable GPS", key="nav_gps_toggle")
                if gps_on:
                    try:
                        from streamlit_js_eval import get_geolocation  # noqa: PLC0415

                        raw = get_geolocation()
                        if raw and raw.get("coords"):
                            _c = raw["coords"]
                            my_loc = {
                                "lat": _c["latitude"],
                                "lon": _c["longitude"],
                                "accuracy": _c.get("accuracy"),
                                "heading": _c.get("heading"),
                            }
                            st.session_state.nav_location = my_loc
                    except ImportError:
                        st.warning(
                            "📦 `streamlit-js-eval` not installed. "
                            "Run `pip install streamlit-js-eval` or switch to manual entry."
                        )
            else:
                nm1, nm2 = st.columns(2)
                with nm1:
                    man_lat = st.number_input(
                        "My Latitude",
                        format="%.8f",
                        key="nav_man_lat",
                        help="Your current latitude in WGS84",
                    )
                with nm2:
                    man_lon = st.number_input(
                        "My Longitude",
                        format="%.8f",
                        key="nav_man_lon",
                        help="Your current longitude in WGS84",
                    )
                if st.button(
                    "📍 Use these coordinates",
                    key="nav_use_manual",
                    use_container_width=True,
                ):
                    st.session_state.nav_location = {
                        "lat": man_lat,
                        "lon": man_lon,
                        "accuracy": None,
                        "heading": None,
                    }

            # Fall back to last known location stored in session
            if my_loc is None:
                my_loc = st.session_state.get("nav_location")

            if my_loc is None:
                st.info(
                    "Enable GPS or enter your coordinates to see navigation guidance."
                )
            else:
                # ── Accuracy badge ────────────────────────────────────────────
                _acc = my_loc.get("accuracy")
                _hdg = my_loc.get("heading")
                if _acc is not None:
                    if _acc < 5:
                        _albl, _acol = f"±{_acc:.1f} m — Excellent", "green"
                    elif _acc < 15:
                        _albl, _acol = f"±{_acc:.1f} m — Good", "green"
                    elif _acc < 50:
                        _albl, _acol = f"±{_acc:.1f} m — Fair", "orange"
                    else:
                        _albl, _acol = f"±{_acc:.1f} m — Poor", "red"
                    st.markdown(
                        f"**GPS Accuracy:** :{_acol}[{_albl}] &nbsp;&nbsp; "
                        f"**Position:** `{my_loc['lat']:.6f}, {my_loc['lon']:.6f}`"
                    )
                else:
                    st.markdown(
                        f"**Position:** `{my_loc['lat']:.6f}, {my_loc['lon']:.6f}`"
                    )
                if _hdg is not None:
                    st.caption(f"Device heading: {_hdg:.0f}° (direction of travel)")
                else:
                    st.caption(
                        "Heading not available — left/right guidance requires the "
                        "device to be moving."
                    )

                # ── Distance + direction to each point ────────────────────────
                _nav = []
                for _ni, _npt in enumerate(points):
                    _dm, _df = _haversine(
                        my_loc["lat"], my_loc["lon"], _npt["lat"], _npt["lon"]
                    )
                    _brg = _bearing(
                        my_loc["lat"], my_loc["lon"], _npt["lat"], _npt["lon"]
                    )
                    _card = _cardinal(_brg)
                    _rel = _relative_direction(_brg, _hdg) if _hdg is not None else None
                    _nav.append(
                        {
                            "idx": _ni,
                            "dist_m": _dm,
                            "dist_ft": _df,
                            "bearing": _brg,
                            "cardinal": _card,
                            "rel": _rel,
                        }
                    )
                _nav.sort(key=lambda x: x["dist_m"])  # nearest first

                st.markdown("##### Boundary Points — nearest first")
                for _nr in _nav:
                    _dm, _df = _nr["dist_m"], _nr["dist_ft"]
                    _icon = "🟢" if _dm < 3 else ("🟡" if _dm < 25 else "🔴")
                    _dir_txt = f"Head **{_nr['cardinal']}** ({_nr['bearing']:.0f}°)"
                    if _nr["rel"]:
                        _dir_txt += f" — **{_nr['rel']}**"
                    _arrived = "  ✅ **You're here!**" if _dm < 3 else ""
                    st.markdown(
                        f"{_icon} **Point {_nr['idx']+1}** &nbsp;—&nbsp; "
                        f"`{_dm:.1f} m` / `{_df:.1f} ft` &nbsp; · &nbsp; {_dir_txt}{_arrived}"
                    )

                # ── Navigation mini-map ───────────────────────────────────────
                st.markdown("---")
                st.markdown("**Navigation Map**")
                _nmap = _make_map([my_loc["lat"], my_loc["lon"]], 16)
                # Current position
                folium.Marker(
                    [my_loc["lat"], my_loc["lon"]],
                    popup=(
                        f"📍 My Location<br>Acc: ±{_acc:.1f} m"
                        if _acc
                        else "📍 My Location"
                    ),
                    tooltip="📍 My Location",
                    icon=folium.Icon(color="blue", icon="user", prefix="fa"),
                ).add_to(_nmap)
                if _acc:
                    folium.Circle(
                        [my_loc["lat"], my_loc["lon"]],
                        radius=_acc,
                        color="#3498db",
                        fill=True,
                        fill_opacity=0.12,
                        weight=2,
                        dash_array="4 4",
                        tooltip=f"GPS accuracy ±{_acc:.1f} m",
                    ).add_to(_nmap)
                # Boundary point markers + dashed guide lines
                for _nr in _nav:
                    _i = _nr["idx"]
                    _npt = points[_i]
                    folium.Marker(
                        [_npt["lat"], _npt["lon"]],
                        popup=(
                            f"<b>Point {_i+1}</b><br>"
                            f"{_nr['dist_m']:.1f} m / {_nr['dist_ft']:.1f} ft<br>"
                            f"Head {_nr['cardinal']} ({_nr['bearing']:.0f}°)"
                            + (f"<br>→ {_nr['rel']}" if _nr["rel"] else "")
                        ),
                        tooltip=f"Point {_i+1}: {_nr['dist_m']:.1f} m | {_nr['cardinal']}",
                        icon=folium.Icon(color="red", icon="info-sign"),
                    ).add_to(_nmap)
                    folium.PolyLine(
                        [[my_loc["lat"], my_loc["lon"]], [_npt["lat"], _npt["lon"]]],
                        color="#e74c3c",
                        weight=2,
                        opacity=0.75,
                        dash_array="6 5",
                        tooltip=f"{_nr['dist_m']:.1f} m | {_nr['cardinal']}",
                    ).add_to(_nmap)
                    _mlat = (my_loc["lat"] + _npt["lat"]) / 2
                    _mlon = (my_loc["lon"] + _npt["lon"]) / 2
                    folium.Marker(
                        [_mlat, _mlon],
                        icon=folium.DivIcon(
                            html=(
                                f'<div style="font-size:10px;color:#c0392b;font-weight:bold;'
                                f"white-space:nowrap;text-shadow:1px 1px 2px white,"
                                f'-1px -1px 2px white;">'
                                f'P{_i+1}: {_nr["dist_m"]:.0f} m</div>'
                            ),
                            icon_size=(120, 18),
                            icon_anchor=(60, 9),
                        ),
                    ).add_to(_nmap)
                st_folium(_nmap, width=700, height=450, key="nav_map")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — About
# ═══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.subheader("About KangahGPS")
    st.markdown(
        """
        **KangahGPS** is a coordinate conversion tool built for Ghana.

        #### Supported Coordinate Systems
        - **War Office / Ghana National Grid** (EPSG:2136)
          - Datum: Accra · Ellipsoid: War Office
          - Unit: Gold Coast foot · Projection: Transverse Mercator
        - **WGS 84** (EPSG:4326)
          - Global latitude/longitude
        - **UTM** (auto-detect zone)
          - Ghana spans UTM zones **30N** (west of 0°) and **31N** (east of 0°)
          - Auto-detection picks the correct zone from your coordinates

        #### WGS 84 ↔ War Office Transformation
        Helmert 3-parameter: **dX = −170, dY = 33, dZ = 326**

        #### How to Use
        1. **Single Conversion** — enter one coordinate pair and convert
        2. **Batch Conversion** — upload a CSV with two columns and convert all rows at once
           - Download the results as a new CSV
           - Preview points on the map (when output is WGS 84)
        3. **Draw & Plot** — add points in any CRS, plot them on a map, and:
           - Connect points as polylines
           - Draw polygons, lines, rectangles freehand
           - Measure distances and areas
           - Export drawings as GeoJSON

        ---
        Built with [Streamlit](https://streamlit.io) and [pyproj](https://pyproj4.github.io/pyproj/).
        """
    )

    st.markdown("---")
    st.subheader("👨‍💻 Developer")
    st.markdown(
        """
        **Desmond Kangah**

        Surveyor & Geospatial Engineer

        - 🎓 B.Sc. Geomatic Engineering — **University of Mines and Technology (UMaT), Ghana** · *First Class Honours*
        - 👨‍🏫 Supervised by **Dr. Yao Yevenyo Ziggah** (Geodesist, UMaT)
        - 🇺🇸 Currently pursuing a **Ph.D. in Civil Engineering** in the United States
          under the supervision of **Dr. Ahmed Abdalla**

        **Research Areas**
        - InSAR (Interferometric Synthetic Aperture Radar)
        - Geodesy
        - Remote Sensing
        - GIS (Geographic Information Systems)
        - Photogrammetry
        """
    )
