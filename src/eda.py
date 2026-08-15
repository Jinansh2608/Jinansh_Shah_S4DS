"""
eda.py
------
All the charts used in the Exploratory Data Analysis pages.

Every function takes a dataframe and returns a Plotly figure, so the Streamlit
app only has to call st.plotly_chart(...). Keeping the plots here means the app
file stays short and readable.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .preprocessing import DAY_NAMES, MONTH_NAMES

# One colour scheme for the whole project so the app looks consistent
COLOR_MAIN = "#2E5A88"
COLOR_ACCENT = "#C0504D"
COLOR_SEQUENCE = ["#2E5A88", "#C0504D"]


def _empty_figure(message="No data available for the current filters"):
    """Fallback figure so the app never crashes on an empty selection."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=300)
    return fig


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
def plot_missing_values(missing_report):
    """Horizontal bar chart of the columns that actually have missing values."""
    data = missing_report[missing_report["Missing Count"] > 0]
    if data.empty:
        return _empty_figure("No missing values in the dataset")

    fig = px.bar(
        data.sort_values("Missing %"),
        x="Missing %",
        y="Column",
        orientation="h",
        text="Missing Count",
        title="Missing Values by Column (%)",
        color_discrete_sequence=[COLOR_MAIN],
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(height=max(300, 40 * len(data)), xaxis_title="Missing (%)", yaxis_title="")
    return fig


# ---------------------------------------------------------------------------
# Univariate: crime
# ---------------------------------------------------------------------------
def plot_top_crime_types(df, top_n=10):
    """Top N most frequent crime categories."""
    if df.empty or "Primary Type" not in df.columns:
        return _empty_figure()

    counts = df["Primary Type"].value_counts().head(top_n).sort_values()
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        text=counts.values,
        title=f"Top {top_n} Crime Types",
        color_discrete_sequence=[COLOR_MAIN],
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(xaxis_title="Number of Crimes", yaxis_title="", height=450)
    return fig


def plot_arrest_distribution(df):
    """Bar chart of Arrest vs No Arrest, labelled with counts and percentages."""
    if df.empty:
        return _empty_figure()

    counts = df["ArrestFlag"].value_counts().sort_index()
    labels = ["No Arrest", "Arrest"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    total = sum(values) or 1
    text = [f"{v:,} ({v / total * 100:.1f}%)" for v in values]

    fig = px.bar(
        x=labels, y=values, text=text,
        title="Arrest Distribution",
        color=labels,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_title="Number of Crimes",
                      showlegend=False, height=400)
    return fig


def plot_top_descriptions(df, top_n=10):
    """Most frequent detailed offence descriptions."""
    if df.empty or "Description" not in df.columns:
        return _empty_figure()

    counts = df["Description"].value_counts().head(top_n).sort_values()
    fig = px.bar(
        x=counts.values, y=counts.index, orientation="h", text=counts.values,
        title=f"Top {top_n} Crime Descriptions",
        color_discrete_sequence=[COLOR_MAIN],
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(xaxis_title="Number of Crimes", yaxis_title="", height=450)
    return fig


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------
def plot_crimes_by_year(df):
    if df.empty or "Year" not in df.columns:
        return _empty_figure()

    counts = df["Year"].value_counts().sort_index()
    fig = px.line(x=counts.index, y=counts.values, markers=True,
                  title="Crimes by Year", color_discrete_sequence=[COLOR_MAIN])
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Crimes", height=400)
    fig.update_xaxes(dtick=1)
    return fig


def plot_crimes_by_month(df):
    if df.empty or "Month" not in df.columns:
        return _empty_figure()

    counts = df["Month"].value_counts().sort_index()
    fig = px.bar(x=[MONTH_NAMES[m - 1] for m in counts.index], y=counts.values,
                 title="Crimes by Month", color_discrete_sequence=[COLOR_MAIN])
    fig.update_layout(xaxis_title="Month", yaxis_title="Number of Crimes", height=400)
    return fig


def plot_crimes_by_hour(df):
    if df.empty or "Hour" not in df.columns:
        return _empty_figure()

    counts = df["Hour"].value_counts().sort_index()
    fig = px.bar(x=counts.index, y=counts.values,
                 title="Crimes by Hour of Day", color_discrete_sequence=[COLOR_MAIN])
    fig.update_layout(xaxis_title="Hour (0 = midnight)", yaxis_title="Number of Crimes", height=400)
    fig.update_xaxes(dtick=1)
    return fig


def plot_crimes_by_dayofweek(df):
    if df.empty or "DayOfWeek" not in df.columns:
        return _empty_figure()

    counts = df["DayOfWeek"].value_counts().sort_index()
    fig = px.bar(x=[DAY_NAMES[d] for d in counts.index], y=counts.values,
                 title="Crimes by Day of Week", color_discrete_sequence=[COLOR_MAIN])
    fig.update_layout(xaxis_title="Day", yaxis_title="Number of Crimes", height=400)
    return fig


# ---------------------------------------------------------------------------
# Location analysis
# ---------------------------------------------------------------------------
def plot_top_districts(df, top_n=10):
    if df.empty or "District" not in df.columns:
        return _empty_figure()

    counts = df["District"].value_counts().head(top_n).sort_values()
    fig = px.bar(x=counts.values, y=counts.index.astype(str), orientation="h",
                 text=counts.values, title=f"Top {top_n} Districts by Crime Count",
                 color_discrete_sequence=[COLOR_MAIN])
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(xaxis_title="Number of Crimes", yaxis_title="District", height=450)
    return fig


def plot_top_locations(df, top_n=10):
    if df.empty or "Location Description" not in df.columns:
        return _empty_figure()

    counts = df["Location Description"].value_counts().head(top_n).sort_values()
    fig = px.bar(x=counts.values, y=counts.index, orientation="h", text=counts.values,
                 title=f"Top {top_n} Crime Locations",
                 color_discrete_sequence=[COLOR_MAIN])
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(xaxis_title="Number of Crimes", yaxis_title="", height=450)
    return fig


def plot_arrest_rate_by_district(df):
    if df.empty or "District" not in df.columns:
        return _empty_figure()

    rates = (df.groupby("District")["ArrestFlag"].mean() * 100).sort_values(ascending=False)
    fig = px.bar(x=rates.index.astype(str), y=rates.values,
                 title="Arrest Rate by District (%)",
                 color=rates.values, color_continuous_scale="Blues")
    fig.update_layout(xaxis_title="District", yaxis_title="Arrest Rate (%)",
                      height=420, coloraxis_showscale=False)
    fig.update_xaxes(type="category")
    return fig


def plot_crime_map(df, max_points=5000, random_state=42):
    """Simple scatter map of crime locations, coloured by arrest outcome."""
    if df.empty or "Latitude" not in df.columns or "Longitude" not in df.columns:
        return _empty_figure("Latitude/Longitude are not available")

    points = df.dropna(subset=["Latitude", "Longitude"])
    if points.empty:
        return _empty_figure("No valid coordinates for the current filters")

    # Plotting 100k points would freeze the browser, so we draw a sample.
    if len(points) > max_points:
        points = points.sample(max_points, random_state=random_state)

    points = points.assign(Outcome=np.where(points["ArrestFlag"] == 1, "Arrest", "No Arrest"))
    fig = px.scatter_map(
        points, lat="Latitude", lon="Longitude", color="Outcome",
        color_discrete_map={"Arrest": COLOR_ACCENT, "No Arrest": COLOR_MAIN},
        hover_data={"Primary Type": True, "Latitude": False, "Longitude": False},
        zoom=9, height=550, opacity=0.5,
        title=f"Crime Locations (sample of {len(points):,} records)",
    )
    fig.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ---------------------------------------------------------------------------
# Bivariate analysis (feature vs Arrest)
# ---------------------------------------------------------------------------
def arrest_rate_table(df, column, min_count=50):
    """
    Arrest rate for every value of `column`.
    Groups with fewer than `min_count` records are ignored because their
    percentage would not be reliable.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[column, "Crimes", "Arrests", "Arrest Rate (%)"])

    grouped = df.groupby(column)["ArrestFlag"].agg(["count", "sum", "mean"]).reset_index()
    grouped.columns = [column, "Crimes", "Arrests", "Arrest Rate (%)"]
    grouped["Arrest Rate (%)"] = (grouped["Arrest Rate (%)"] * 100).round(2)
    grouped["Arrests"] = grouped["Arrests"].astype(int)
    grouped = grouped[grouped["Crimes"] >= min_count]
    return grouped.sort_values("Arrest Rate (%)", ascending=False).reset_index(drop=True)


def plot_arrest_rate_by_crime_type(df, top_n=15, min_count=50):
    table = arrest_rate_table(df, "Primary Type", min_count=min_count).head(top_n)
    if table.empty:
        return _empty_figure()

    table = table.sort_values("Arrest Rate (%)")
    fig = px.bar(table, x="Arrest Rate (%)", y="Primary Type", orientation="h",
                 text="Arrest Rate (%)", hover_data=["Crimes", "Arrests"],
                 title=f"Arrest Rate by Crime Type (top {len(table)})",
                 color="Arrest Rate (%)", color_continuous_scale="Blues")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_title="", height=500, coloraxis_showscale=False)
    return fig


def plot_arrest_rate_by_hour(df):
    if df.empty or "Hour" not in df.columns:
        return _empty_figure()

    rates = (df.groupby("Hour")["ArrestFlag"].mean() * 100)
    fig = px.line(x=rates.index, y=rates.values, markers=True,
                  title="Arrest Rate by Hour of Day (%)",
                  color_discrete_sequence=[COLOR_ACCENT])
    fig.update_layout(xaxis_title="Hour", yaxis_title="Arrest Rate (%)", height=400)
    fig.update_xaxes(dtick=1)
    return fig


def plot_arrest_rate_by_dayofweek(df):
    if df.empty or "DayOfWeek" not in df.columns:
        return _empty_figure()

    rates = (df.groupby("DayOfWeek")["ArrestFlag"].mean() * 100).sort_index()
    fig = px.bar(x=[DAY_NAMES[d] for d in rates.index], y=rates.values,
                 title="Arrest Rate by Day of Week (%)",
                 color_discrete_sequence=[COLOR_ACCENT])
    fig.update_layout(xaxis_title="Day", yaxis_title="Arrest Rate (%)", height=400)
    return fig


def plot_arrest_rate_by_month(df):
    if df.empty or "Month" not in df.columns:
        return _empty_figure()

    rates = (df.groupby("Month")["ArrestFlag"].mean() * 100).sort_index()
    fig = px.line(x=[MONTH_NAMES[m - 1] for m in rates.index], y=rates.values, markers=True,
                  title="Arrest Rate by Month (%)", color_discrete_sequence=[COLOR_ACCENT])
    fig.update_layout(xaxis_title="Month", yaxis_title="Arrest Rate (%)", height=400)
    return fig


def plot_crime_vs_arrest_counts(df, top_n=10):
    """Grouped bar chart: arrests vs non-arrests for the most common crime types."""
    if df.empty or "Primary Type" not in df.columns:
        return _empty_figure()

    top_types = df["Primary Type"].value_counts().head(top_n).index
    subset = df[df["Primary Type"].isin(top_types)]
    grouped = (subset.groupby(["Primary Type", "ArrestFlag"]).size()
               .reset_index(name="Count"))
    grouped["Outcome"] = np.where(grouped["ArrestFlag"] == 1, "Arrest", "No Arrest")

    fig = px.bar(grouped, x="Primary Type", y="Count", color="Outcome",
                 barmode="group", title=f"Arrest vs No Arrest for Top {top_n} Crime Types",
                 color_discrete_map={"Arrest": COLOR_ACCENT, "No Arrest": COLOR_MAIN})
    fig.update_layout(xaxis_title="", yaxis_title="Number of Crimes", height=480)
    fig.update_xaxes(tickangle=-40)
    return fig


# ---------------------------------------------------------------------------
# Multivariate analysis
# ---------------------------------------------------------------------------
def plot_hour_day_heatmap(df):
    """Crime count heatmap: rows = day of week, columns = hour."""
    if df.empty or "Hour" not in df.columns or "DayOfWeek" not in df.columns:
        return _empty_figure()

    matrix = (df.pivot_table(index="DayOfWeek", columns="Hour",
                             values="ArrestFlag", aggfunc="size")
              .reindex(index=range(7), columns=range(24)).fillna(0))

    fig = px.imshow(matrix.values, x=[str(h) for h in range(24)], y=DAY_NAMES,
                    color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="Hour of Day", y="Day of Week", color="Crimes"),
                    title="Crime Count Heatmap: Day of Week x Hour")
    fig.update_layout(height=420)
    return fig


def plot_crimetype_district_heatmap(df, top_types=10, top_districts=12):
    """Arrest rate heatmap for the busiest crime types and districts."""
    if df.empty or "Primary Type" not in df.columns or "District" not in df.columns:
        return _empty_figure()

    types = df["Primary Type"].value_counts().head(top_types).index
    districts = df["District"].value_counts().head(top_districts).index
    subset = df[df["Primary Type"].isin(types) & df["District"].isin(districts)]
    if subset.empty:
        return _empty_figure()

    matrix = subset.pivot_table(index="Primary Type", columns="District",
                                values="ArrestFlag", aggfunc="mean") * 100
    matrix = matrix.reindex(index=types, columns=sorted(districts))

    fig = px.imshow(matrix.values, x=[str(d) for d in matrix.columns], y=list(matrix.index),
                    color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="District", y="Crime Type", color="Arrest %"),
                    title="Arrest Rate (%) by Crime Type and District")
    fig.update_layout(height=520)
    return fig


def plot_crimetype_hour_heatmap(df, top_types=10):
    """When during the day is each crime type reported?"""
    if df.empty or "Primary Type" not in df.columns or "Hour" not in df.columns:
        return _empty_figure()

    types = df["Primary Type"].value_counts().head(top_types).index
    subset = df[df["Primary Type"].isin(types)]
    matrix = (subset.pivot_table(index="Primary Type", columns="Hour",
                                 values="ArrestFlag", aggfunc="size")
              .reindex(index=types, columns=range(24)).fillna(0))

    fig = px.imshow(matrix.values, x=[str(h) for h in range(24)], y=list(matrix.index),
                    color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="Hour of Day", y="Crime Type", color="Crimes"),
                    title="Crime Count Heatmap: Crime Type x Hour")
    fig.update_layout(height=520)
    return fig


# ---------------------------------------------------------------------------
# Automatic insights (all numbers come from the data, nothing is hard-coded)
# ---------------------------------------------------------------------------
def generate_insights(df):
    """Build a list of short findings calculated from the current dataframe."""
    insights = []
    if df.empty:
        return ["No data available for the current filters."]

    total = len(df)
    arrest_rate = df["ArrestFlag"].mean() * 100
    insights.append(
        f"Out of {total:,} reported crimes, {df['ArrestFlag'].sum():,} led to an arrest "
        f"({arrest_rate:.1f}%), so the target variable is imbalanced towards 'No Arrest'."
    )

    if "Primary Type" in df.columns:
        top_type = df["Primary Type"].value_counts()
        share = top_type.iloc[0] / total * 100
        insights.append(
            f"'{top_type.index[0].title()}' is the most frequently reported crime "
            f"({top_type.iloc[0]:,} records, {share:.1f}% of all crimes), followed by "
            f"'{top_type.index[1].title()}' and '{top_type.index[2].title()}'."
        )

        rates = arrest_rate_table(df, "Primary Type", min_count=50)
        if not rates.empty:
            best, worst = rates.iloc[0], rates.iloc[-1]
            insights.append(
                f"Arrest rates differ enormously between crime types: "
                f"'{best['Primary Type'].title()}' reaches {best['Arrest Rate (%)']:.1f}% "
                f"while '{worst['Primary Type'].title()}' is only {worst['Arrest Rate (%)']:.1f}%. "
                "Crime type is therefore the strongest predictor available."
            )

    if "Hour" in df.columns:
        by_hour = df["Hour"].value_counts()
        peak, quiet = by_hour.idxmax(), by_hour.idxmin()
        insights.append(
            f"Crime is not spread evenly across the day: the busiest hour is "
            f"{peak:02d}:00 ({by_hour.max():,} crimes) and the quietest is "
            f"{quiet:02d}:00 ({by_hour.min():,} crimes)."
        )

    if "DayName" in df.columns:
        by_day = df["DayName"].value_counts()
        insights.append(
            f"{by_day.idxmax()} records the most crimes ({by_day.max():,}) and "
            f"{by_day.idxmin()} the fewest ({by_day.min():,})."
        )

    if "District" in df.columns:
        by_district = df["District"].value_counts()
        district_rates = arrest_rate_table(df, "District", min_count=50)
        insights.append(
            f"District {by_district.idxmax()} reports the highest crime volume "
            f"({by_district.max():,} records), which is {by_district.max() / by_district.min():.1f}x "
            f"more than the quietest district ({by_district.idxmin()})."
        )
        if not district_rates.empty:
            # .iloc[0] on a mixed-dtype row upcasts every value to a common dtype
            # (e.g. int District -> float), so District is cast back to int explicitly.
            top_district = int(district_rates.iloc[0]["District"])
            bottom_district = int(district_rates.iloc[-1]["District"])
            insights.append(
                f"Arrest rates also vary by area: district {top_district} "
                f"arrests in {district_rates.iloc[0]['Arrest Rate (%)']:.1f}% of cases versus "
                f"{district_rates.iloc[-1]['Arrest Rate (%)']:.1f}% in district "
                f"{bottom_district}."
            )

    if "Location Description" in df.columns:
        top_loc = df["Location Description"].value_counts()
        insights.append(
            f"The most common crime location is '{top_loc.index[0].title()}' "
            f"({top_loc.iloc[0]:,} records, {top_loc.iloc[0] / total * 100:.1f}%)."
        )

    if "Year" in df.columns and df["Year"].nunique() > 1:
        by_year = df["Year"].value_counts().sort_index()
        change = (by_year.iloc[-1] - by_year.iloc[0]) / by_year.iloc[0] * 100
        direction = "increased" if change > 0 else "decreased"
        insights.append(
            f"Between {by_year.index[0]} and {by_year.index[-1]} the number of reported "
            f"crimes {direction} by {abs(change):.1f}%."
        )

    if "Domestic" in df.columns:
        dom = df.groupby(df["Domestic"].astype(str))["ArrestFlag"].mean() * 100
        if len(dom) > 1:
            insights.append(
                f"Domestic-related incidents have an arrest rate of {dom.get('True', float('nan')):.1f}% "
                f"compared with {dom.get('False', float('nan')):.1f}% for non-domestic incidents."
            )

    return insights
