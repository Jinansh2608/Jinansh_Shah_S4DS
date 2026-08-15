"""
map3d.py
--------
The 3D Crime Intelligence Map - the signature visual feature of the project.

The idea is simple, even though the code has a few more moving parts than the
rest of the EDA charts:

1. Chicago is cut into a grid of small rectangular cells using latitude and
   longitude (this is NOT a real GIS operation, just simple binning with
   pandas `cut` - easy to explain in a viva).
2. Crimes are aggregated inside every cell (count, arrests, arrest rate,
   dominant crime type, peak hour, dominant district).
3. Every cell becomes one 3D bar ("tower") in a Plotly `Mesh3d` figure.
   Tower height = number of crimes in that cell, so the whole figure looks
   like a city skyline - taller towers mean more reported crime.

Plotly has no built-in "3D bar chart" trace, so each tower is built from a
rectangular box (8 corner points + 12 triangles) and all the towers are
combined into a single Mesh3d trace, which keeps the figure fast even with a
few hundred towers.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Chicago city limits (same values used for data-quality checks elsewhere)
LAT_RANGE = (41.6, 42.1)
LON_RANGE = (-87.95, -87.5)

# Grid resolution presets: number of bins along each axis.
RESOLUTIONS = {"Low Resolution": 12, "Medium Resolution": 20, "High Resolution": 32}

# How many individual crime types get their own color in "Crime Type" mode;
# everything else is grouped into "Other".
TOP_CRIME_TYPES_FOR_COLOR = 8

# Colors used for the "Crime Volume" and "Arrest Rate" continuous modes.
VOLUME_COLORSCALE = [[0.0, "#DCE8F5"], [0.5, "#5B8FC7"], [1.0, "#0B3D74"]]
ARREST_COLORSCALE = [[0.0, "#2E5A88"], [0.5, "#B7B7B7"], [1.0, "#C0504D"]]

CRIME_TYPE_PALETTE = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
DISTRICT_PALETTE = ["#2E5A88", "#C0504D"]


# ---------------------------------------------------------------------------
# 1. Grid aggregation
# ---------------------------------------------------------------------------
def aggregate_grid(df, bins=20):
    """
    Aggregate crime records into a lat/lon grid.

    Parameters
    ----------
    df : DataFrame with (at least) Latitude, Longitude, ArrestFlag columns.
    bins : number of grid cells along each axis.

    Returns
    -------
    A DataFrame with one row per non-empty grid cell:
    lat_center, lon_center, lat_min, lat_max, lon_min, lon_max,
    crime_count, arrest_count, arrest_rate, dominant_crime, peak_hour,
    dominant_district (only the columns that can actually be computed from
    the available data are included).
    """
    if df.empty or "Latitude" not in df.columns or "Longitude" not in df.columns:
        return pd.DataFrame()

    points = df.dropna(subset=["Latitude", "Longitude"]).copy()
    points = points[
        points["Latitude"].between(*LAT_RANGE) & points["Longitude"].between(*LON_RANGE)
    ]
    if points.empty:
        return pd.DataFrame()

    lat_edges = np.linspace(points["Latitude"].min(), points["Latitude"].max(), bins + 1)
    lon_edges = np.linspace(points["Longitude"].min(), points["Longitude"].max(), bins + 1)
    # Guard against a degenerate (single-point) range
    lat_edges = np.unique(lat_edges)
    lon_edges = np.unique(lon_edges)
    if len(lat_edges) < 2 or len(lon_edges) < 2:
        return pd.DataFrame()

    points["lat_bin"] = pd.cut(points["Latitude"], bins=lat_edges, include_lowest=True)
    points["lon_bin"] = pd.cut(points["Longitude"], bins=lon_edges, include_lowest=True)

    grouped = points.groupby(["lat_bin", "lon_bin"], observed=True)

    rows = []
    for (lat_bin, lon_bin), cell in grouped:
        if len(cell) == 0:
            continue
        row = {
            "lat_min": lat_bin.left,
            "lat_max": lat_bin.right,
            "lon_min": lon_bin.left,
            "lon_max": lon_bin.right,
            "lat_center": (lat_bin.left + lat_bin.right) / 2,
            "lon_center": (lon_bin.left + lon_bin.right) / 2,
            "crime_count": len(cell),
        }
        if "ArrestFlag" in cell.columns:
            row["arrest_count"] = int(cell["ArrestFlag"].sum())
            row["arrest_rate"] = float(cell["ArrestFlag"].mean() * 100)
        if "Primary Type" in cell.columns:
            row["dominant_crime"] = cell["Primary Type"].mode().iloc[0]
        if "Hour" in cell.columns:
            row["peak_hour"] = int(cell["Hour"].mode().iloc[0])
        if "DayName" in cell.columns:
            row["peak_day"] = cell["DayName"].mode().iloc[0]
        if "District" in cell.columns:
            row["dominant_district"] = cell["District"].mode().iloc[0]
        rows.append(row)

    grid = pd.DataFrame(rows)
    return grid.reset_index(drop=True)


def crime_type_color_groups(grid_df, top_n=TOP_CRIME_TYPES_FOR_COLOR):
    """Map every cell's dominant crime type to one of the top-N types + 'Other'."""
    if "dominant_crime" not in grid_df.columns:
        return grid_df.assign(color_group="Unknown"), {"Unknown": "#888888"}

    top_types = grid_df["dominant_crime"].value_counts().head(top_n).index.tolist()
    color_group = grid_df["dominant_crime"].where(grid_df["dominant_crime"].isin(top_types), "Other")

    categories = top_types + (["Other"] if (color_group == "Other").any() else [])
    color_map = {cat: CRIME_TYPE_PALETTE[i % len(CRIME_TYPE_PALETTE)] for i, cat in enumerate(categories)}
    color_map["Other"] = "#B0B0B0"

    return grid_df.assign(color_group=color_group), color_map


# ---------------------------------------------------------------------------
# 2. Building one 3D box (tower) as triangles for Mesh3d
# ---------------------------------------------------------------------------
def _cuboid(x0, x1, y0, y1, z0, z1):
    """
    Return the 8 corner points and 12 triangle-index triples of a rectangular
    box, so many boxes can be merged into a single Mesh3d trace.
    """
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]

    # 6 faces x 2 triangles = 12 triangles, referencing the 8 points above (0-7)
    i = [0, 0, 4, 4, 0, 0, 3, 3, 0, 1, 1, 2]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 2, 5, 6]
    k = [2, 3, 6, 7, 5, 4, 6, 7, 7, 6, 6, 7]

    return xs, ys, zs, i, j, k


def _build_mesh(grid_df, colors, hover_texts, gap_ratio=0.15, z_column="crime_count"):
    """
    Merge one box per grid row into the arrays Mesh3d needs.

    `colors` is a list of one color per row (repeated 12x internally for the
    12 triangular faces of each box). `gap_ratio` shrinks every box slightly
    so neighbouring towers don't touch, which keeps the skyline readable.
    """
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    face_colors = []
    vertex_hover = []

    offset = 0
    for row_idx, (_, row) in enumerate(grid_df.iterrows()):
        lat_span = row["lat_max"] - row["lat_min"]
        lon_span = row["lon_max"] - row["lon_min"]
        pad_lat = lat_span * gap_ratio / 2
        pad_lon = lon_span * gap_ratio / 2

        x0, x1 = row["lon_min"] + pad_lon, row["lon_max"] - pad_lon
        y0, y1 = row["lat_min"] + pad_lat, row["lat_max"] - pad_lat
        z0, z1 = 0, max(row[z_column], 0.01)  # avoid a zero-height (invisible) box

        xs, ys, zs, i, j, k = _cuboid(x0, x1, y0, y1, z0, z1)
        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)
        all_i.extend([v + offset for v in i])
        all_j.extend([v + offset for v in j])
        all_k.extend([v + offset for v in k])
        face_colors.extend([colors[row_idx]] * 12)
        vertex_hover.extend([hover_texts[row_idx]] * 8)
        offset += 8

    return dict(
        x=all_x, y=all_y, z=all_z, i=all_i, j=all_j, k=all_k,
        facecolor=face_colors, hovertext=vertex_hover,
    )


# ---------------------------------------------------------------------------
# 3. Hover text
# ---------------------------------------------------------------------------
def build_hover_text(row):
    """Build the multi-line hover string for one grid cell, skipping missing fields."""
    lines = []
    if "dominant_district" in row and pd.notna(row["dominant_district"]):
        lines.append(f"District: {row['dominant_district']}")
    lines.append(f"Crime Count: {int(row['crime_count']):,}")
    if "arrest_count" in row:
        lines.append(f"Arrest Count: {int(row['arrest_count']):,}")
    if "arrest_rate" in row:
        lines.append(f"Arrest Rate: {row['arrest_rate']:.2f}%")
    if "dominant_crime" in row and pd.notna(row["dominant_crime"]):
        lines.append(f"Top Crime: {row['dominant_crime'].title()}")
    if "peak_hour" in row and pd.notna(row["peak_hour"]):
        lines.append(f"Peak Hour: {int(row['peak_hour']):02d}:00")
    if "peak_day" in row and pd.notna(row["peak_day"]):
        lines.append(f"Peak Day: {row['peak_day']}")
    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# 4. Figure builders
# ---------------------------------------------------------------------------
def build_skyline_figure(grid_df, color_by="Crime Volume", title="3D Crime Skyline",
                          custom_color_map=None, custom_color_column=None):
    """
    Build the 3D skyline Mesh3d figure.

    color_by: "Crime Volume" | "Arrest Rate" | "Crime Type" | "Custom"
      - "Custom" expects `custom_color_map` (value -> hex color) and
        `custom_color_column` (the grid_df column holding that value); used
        for the district-comparison view.
    """
    if grid_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No grid cells to display for the current filters",
                           showarrow=False, font=dict(size=14))
        fig.update_layout(height=550)
        return fig, None

    hover_texts = [build_hover_text(row) for _, row in grid_df.iterrows()]

    colorbar_title = None
    if color_by == "Crime Volume":
        vmin, vmax = grid_df["crime_count"].min(), grid_df["crime_count"].max()
        norm = (grid_df["crime_count"] - vmin) / max(vmax - vmin, 1)
        colors = [_sample_colorscale(VOLUME_COLORSCALE, v) for v in norm]
        colorbar_title = "Crime Count"
    elif color_by == "Arrest Rate" and "arrest_rate" in grid_df.columns:
        vmin, vmax = 0, 100
        norm = (grid_df["arrest_rate"] - vmin) / max(vmax - vmin, 1)
        colors = [_sample_colorscale(ARREST_COLORSCALE, v) for v in norm]
        colorbar_title = "Arrest Rate (%)"
    elif color_by == "Crime Type" and "dominant_crime" in grid_df.columns:
        colored_df, color_map = crime_type_color_groups(grid_df)
        colors = colored_df["color_group"].map(color_map).tolist()
        grid_df = colored_df
    elif color_by == "Custom" and custom_color_column is not None:
        colors = grid_df[custom_color_column].map(custom_color_map).tolist()
    else:
        colors = [VOLUME_COLORSCALE[-1][1]] * len(grid_df)

    mesh_data = _build_mesh(grid_df, colors, hover_texts)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=mesh_data["x"], y=mesh_data["y"], z=mesh_data["z"],
                i=mesh_data["i"], j=mesh_data["j"], k=mesh_data["k"],
                facecolor=mesh_data["facecolor"],
                hovertext=mesh_data["hovertext"],
                hoverinfo="text",
                flatshading=True,
                lighting=dict(ambient=0.6, diffuse=0.6, specular=0.2, roughness=0.6),
                lightposition=dict(x=100, y=200, z=300),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Crime Count",
            aspectmode="manual",
            aspectratio=dict(x=1.3, y=1.3, z=0.7),
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # A legend for categorical modes (crime type / district) is added manually,
    # since a single Mesh3d trace cannot show a normal Plotly legend.
    legend_map = None
    if color_by == "Crime Type" and "dominant_crime" in grid_df.columns:
        _, legend_map = crime_type_color_groups(grid_df)
    elif color_by == "Custom" and custom_color_map is not None:
        legend_map = custom_color_map

    return fig, legend_map


def _sample_colorscale(colorscale, t):
    """Linearly interpolate a hex color from a [ [0,color],[0.5,color],[1,color] ] colorscale."""
    t = min(max(t, 0.0), 1.0)
    stops = colorscale
    for idx in range(len(stops) - 1):
        t0, c0 = stops[idx]
        t1, c1 = stops[idx + 1]
        if t0 <= t <= t1:
            local_t = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            return _interp_hex(c0, c1, local_t)
    return stops[-1][1]


def _interp_hex(c0, c1, t):
    c0 = c0.lstrip("#")
    c1 = c1.lstrip("#")
    r = round(int(c0[0:2], 16) + (int(c1[0:2], 16) - int(c0[0:2], 16)) * t)
    g = round(int(c0[2:4], 16) + (int(c1[2:4], 16) - int(c0[2:4], 16)) * t)
    b = round(int(c0[4:6], 16) + (int(c1[4:6], 16) - int(c0[4:6], 16)) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# 5. Hotspot analysis helpers
# ---------------------------------------------------------------------------
def hotspot_label(row):
    """Readable label for the hotspot-selection dropdown."""
    district = f" - District {row['dominant_district']}" if "dominant_district" in row and pd.notna(row["dominant_district"]) else ""
    return f"({row['lat_center']:.3f}, {row['lon_center']:.3f}){district} - {int(row['crime_count']):,} crimes"


def hotspot_table(grid_df, top_n=15):
    """Top grid cells by crime count, used to populate the hotspot dropdown."""
    if grid_df.empty:
        return grid_df
    table = grid_df.sort_values("crime_count", ascending=False).head(top_n).reset_index(drop=True)
    table["label"] = table.apply(hotspot_label, axis=1)
    return table


# ---------------------------------------------------------------------------
# 6. District comparison
# ---------------------------------------------------------------------------
def district_summary(df, district):
    """Simple summary stats for one district, used in the comparison panel."""
    subset = df[df["District"] == district]
    if subset.empty:
        return None

    summary = {
        "district": district,
        "crime_count": len(subset),
        "arrest_count": int(subset["ArrestFlag"].sum()),
        "arrest_rate": float(subset["ArrestFlag"].mean() * 100),
    }
    if "Primary Type" in subset.columns:
        summary["top_crime"] = subset["Primary Type"].mode().iloc[0]
    if "Hour" in subset.columns:
        summary["peak_hour"] = int(subset["Hour"].mode().iloc[0])
    return summary
