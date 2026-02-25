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
from folium.plugins import Draw, MeasureControl
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
        '<Document>',
        f'  <name>KangahGPS Export</name>',
    ]
    for i, row in gdf.iterrows():
        pt = row.geometry
        lines.append('  <Placemark>')
        lines.append(f'    <name>{name_prefix} {i+1}</name>')
        lines.append(f'    <description>Lat: {pt.y:.8f}, Lon: {pt.x:.8f}</description>')
        lines.append('    <Point>')
        lines.append(f'      <coordinates>{pt.x},{pt.y},0</coordinates>')
        lines.append('    </Point>')
        lines.append('  </Placemark>')
    lines.append('</Document>')
    lines.append('</kml>')
    return '\n'.join(lines)


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
    df: pd.DataFrame, lat_col: str, lon_col: str, key_prefix: str,
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
    ["📍 Single Conversion", "📄 Batch Conversion (CSV)", "✏️ Draw & Plot", "ℹ️ About"]
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

        val1 = st.number_input(f1, format="%.8f" if src_info["is_latlon"] else "%.3f", key="v1")
        val2 = st.number_input(f2, format="%.8f" if src_info["is_latlon"] else "%.3f", key="v2")

    with col_tgt:
        st.subheader("Target")
        default_tgt = 1 if src_name == CRS_NAMES[0] else 0
        tgt_name = st.selectbox("Target CRS", CRS_NAMES, index=default_tgt, key="tgt_crs")
        tgt_info = CRS_OPTIONS[tgt_name]

        # UTM zone selector when target is UTM
        tgt_utm_zone = None
        if tgt_info["is_utm"]:
            auto_zone = st.checkbox("Auto-detect UTM zone", value=True, key="auto_zone")
            if not auto_zone:
                tgt_utm_zone = st.number_input(
                    "UTM Zone", min_value=1, max_value=60, value=30, step=1, key="tgt_zone"
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
                                src_epsg, "EPSG:4326", val1, val2,
                                src_info["is_latlon"], True, tfm_name,
                            )
                            auto_lon = tmp_lon
                        detected_zone = utm_zone_from_lon(auto_lon)
                        tgt_epsg = get_epsg(tgt_name, zone=detected_zone)
                else:
                    tgt_epsg = get_epsg(tgt_name)

                out1, out2 = convert_single(
                    src_epsg, tgt_epsg, val1, val2,
                    src_info["is_latlon"], tgt_info["is_latlon"], tfm_name,
                )

                # Store results in session state so they persist
                st.session_state.single_result = {
                    "out1": out1, "out2": out2,
                    "tf1": tgt_info["fields"][0], "tf2": tgt_info["fields"][1],
                    "is_latlon": tgt_info["is_latlon"],
                    "src_is_latlon": src_info["is_latlon"],
                    "src_val1": val1, "src_val2": val2,
                    "zone": (tgt_utm_zone or detected_zone) if tgt_info["is_utm"] else None,
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
            preview = folium.Map(
                location=[out1, out2], zoom_start=10,
                tiles="CartoDB positron",
            )
            folium.Marker(
                [out1, out2],
                popup=f"Lat: {out1:.6f}<br>Lon: {out2:.6f}",
                icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
            ).add_to(preview)
            st_folium(preview, width=700, height=400, key="single_map")
        elif r["src_is_latlon"]:
            st.markdown("##### 🗺️ Map Preview (source location)")
            preview = folium.Map(
                location=[r["src_val1"], r["src_val2"]], zoom_start=10,
                tiles="CartoDB positron",
            )
            folium.Marker(
                [r["src_val1"], r["src_val2"]],
                popup=f"Lat: {r['src_val1']:.6f}<br>Lon: {r['src_val2']:.6f}",
                icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
            ).add_to(preview)
            st_folium(preview, width=700, height=400, key="single_map_src")

        # Export single point
        single_df = pd.DataFrame({
            r["tf1"]: [out1], r["tf2"]: [out2],
        })
        if r["is_latlon"]:
            show_export_buttons(single_df, r["tf1"], r["tf2"], "single")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Batch Conversion (CSV)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.subheader("Batch Conversion")
    st.markdown(
        "Upload a CSV with columns for coordinates. "
        "You can include an **ID** column (optional) plus "
        "**Easting/Longitude/X** and **Northing/Latitude/Y**."
    )

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        bsrc_name = st.selectbox("Source CRS", CRS_NAMES, index=0, key="bsrc")
        bsrc_info = CRS_OPTIONS[bsrc_name]
        bsrc_zone = None
        if bsrc_info["is_utm"]:
            bsrc_zone = st.number_input(
                "Source UTM Zone", 1, 60, 30, key="bsrc_zone"
            )
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

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.markdown(f"**{len(df)} rows** loaded. Preview:")
            st.dataframe(df.head(10), use_container_width=True)

            col_names = df.columns.tolist()
            if len(col_names) < 2:
                st.error("CSV must have at least 2 columns.")
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
                    default_y = col_names.index(x_col) + 1 if col_names.index(x_col) + 1 < len(col_names) else 0
                    remaining = [c for c in col_names if c != x_col]
                    y_col = st.selectbox(
                        f"{bf2} column",
                        remaining,
                        index=0,
                        key="y_col",
                    )

                if st.button("🔄  Convert All", use_container_width=True, key="batch_btn"):
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
                                bsrc_epsg, "EPSG:4326", first_v1, first_v2,
                                bsrc_info["is_latlon"], True, tfm_name,
                            )
                        btgt_epsg = get_epsg(btgt_name, zone=utm_zone_from_lon(auto_lon))
                    elif btgt_info["is_utm"]:
                        btgt_epsg = get_epsg(btgt_name, zone=btgt_zone)
                    else:
                        btgt_epsg = get_epsg(btgt_name)

                    results = convert_batch(
                        bsrc_epsg, btgt_epsg, coords,
                        bsrc_info["is_latlon"], btgt_info["is_latlon"], tfm_name,
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
            batch_map = folium.Map(
                location=[sum(lats)/len(lats), sum(lons)/len(lons)],
                zoom_start=7, tiles="CartoDB positron",
            )
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
                src_epsg, "EPSG:4326", pt_v1, pt_v2,
                draw_src_info["is_latlon"], True, tfm_name,
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
            st.rerun()
    with c2:
        connect = st.checkbox("Connect points as polyline", value=True, key="connect_pts")

    # ── Build folium map ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 🗺️ Interactive Map")
    st.caption("Use the toolbar on the left to draw markers, lines, polygons, or rectangles.")

    points = st.session_state.get("draw_points", [])

    # Center map on Ghana or on the points
    if points:
        center_lat = sum(p["lat"] for p in points) / len(points)
        center_lon = sum(p["lon"] for p in points) / len(points)
        zoom = 10
    else:
        center_lat, center_lon = 7.95, -1.03  # Ghana center
        zoom = 7

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="CartoDB positron")

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
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="sqmeters",
        secondary_area_unit="hectares",
    ).add_to(m)

    # Plot user points
    for i, pt in enumerate(points):
        folium.Marker(
            location=[pt["lat"], pt["lon"]],
            popup=f"Point {i+1}<br>Lat: {pt['lat']:.6f}<br>Lon: {pt['lon']:.6f}",
            tooltip=f"Point {i+1}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    # Connect points as polyline
    if connect and len(points) >= 2:
        line_coords = [[p["lat"], p["lon"]] for p in points]
        folium.PolyLine(
            locations=line_coords,
            color="#e74c3c",
            weight=3,
            opacity=0.8,
            tooltip="Connected points",
        ).add_to(m)

    # Render map
    map_data = st_folium(m, width=700, height=500, returned_objects=["all_drawings"])

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

    # ── Points table ──────────────────────────────────────────────────────
    if points:
        st.markdown("##### Points Table")
        pts_df = pd.DataFrame(points)
        pts_df.index = range(1, len(pts_df) + 1)
        pts_df.index.name = "#"
        st.dataframe(pts_df, use_container_width=True)

        csv_bytes = pts_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            "⬇️ Download Points (CSV)",
            data=csv_bytes,
            file_name="kangahgps_points.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_pts",
        )


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
