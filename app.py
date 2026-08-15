"""
app.py
------
Chicago Crime Analysis & Arrest Prediction - Streamlit application.

This is the entry point of the project. It only handles the UI (sidebar
navigation, filters, layout) and calls into the src/ package for the actual
data loading, cleaning, EDA charts and machine learning.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import eda, map3d
from src.data_loader import (
    TARGET_COLUMN,
    basic_overview,
    duplicate_report,
    load_raw_data,
    missing_value_report,
    split_column_types,
)
from src.model import (
    comparison_long_table,
    cv_results_table,
    predict_single,
    select_best_model,
    test_results_table,
    train_and_evaluate,
)
from src.preprocessing import (
    CANDIDATE_CATEGORICAL,
    CANDIDATE_NUMERICAL,
    DAY_NAMES,
    EXCLUDED_COLUMNS,
    MONTH_NAMES,
    prepare_dataframe,
    select_features,
)

st.set_page_config(
    page_title="Chicago Crime Analysis & Arrest Prediction",
    page_icon="🚔",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar - navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Chicago Crime Analysis")
PAGES = [
    "1. Overview",
    "2. Data Quality",
    "3. Crime Analysis",
    "4. Time Analysis",
    "5. Location Analysis",
    "6. Relationship Analysis",
    "7. Machine Learning",
    "8. Model Comparison",
    "9. Insights",
    "10. 3D Crime Map",
]
page = st.sidebar.radio("Go to", PAGES)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Settings")

# ---------------------------------------------------------------------------
# Load & prepare data (cached so the app stays fast when switching pages)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def get_raw_data(sample_size):
    return load_raw_data(sample_size=sample_size)


@st.cache_data(show_spinner="Cleaning data and building features...")
def get_clean_data(sample_size):
    raw_df, info = get_raw_data(sample_size)
    clean_df, log = prepare_dataframe(raw_df)
    return raw_df, clean_df, log, info


# The full file has ~120k rows, which is fast enough to use in full by default.
# A sample option is kept so the app stays smooth on slower machines.
raw_preview, _preview_info = get_raw_data(None)
FULL_ROWS = _preview_info["total_rows_in_file"]

use_sample = st.sidebar.checkbox("Use a sample instead of the full dataset", value=False)
if use_sample:
    sample_size = st.sidebar.slider(
        "Sample size", min_value=5_000, max_value=min(FULL_ROWS, 100_000),
        value=min(30_000, FULL_ROWS), step=5_000,
    )
else:
    sample_size = None

raw_df, clean_df, cleaning_log, load_info = get_clean_data(sample_size)

st.sidebar.caption(
    f"Using **{load_info['rows_loaded']:,}** of **{load_info['total_rows_in_file']:,}** rows "
    f"({'sample' if load_info['sampled'] else 'full dataset'})."
)

# ---------------------------------------------------------------------------
# Sidebar - interactive filters (applied to the EDA pages)
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filters (EDA pages)")

years = sorted(clean_df["Year"].dropna().unique().tolist()) if "Year" in clean_df.columns else []
crime_types = sorted(clean_df["Primary Type"].dropna().unique().tolist()) if "Primary Type" in clean_df.columns else []
districts = sorted(clean_df["District"].dropna().unique().tolist()) if "District" in clean_df.columns else []

selected_years = st.sidebar.multiselect("Year", years, default=[])
selected_types = st.sidebar.multiselect("Crime Type", crime_types, default=[])
selected_districts = st.sidebar.multiselect("District", districts, default=[])
arrest_filter = st.sidebar.selectbox("Arrest", ["All", "Arrest Only", "No Arrest Only"])

filtered_df = clean_df.copy()
if selected_years:
    filtered_df = filtered_df[filtered_df["Year"].isin(selected_years)]
if selected_types:
    filtered_df = filtered_df[filtered_df["Primary Type"].isin(selected_types)]
if selected_districts:
    filtered_df = filtered_df[filtered_df["District"].isin(selected_districts)]
if arrest_filter == "Arrest Only":
    filtered_df = filtered_df[filtered_df["ArrestFlag"] == 1]
elif arrest_filter == "No Arrest Only":
    filtered_df = filtered_df[filtered_df["ArrestFlag"] == 0]

st.sidebar.caption(f"Filtered rows: **{len(filtered_df):,}**")


# ===========================================================================
# PAGE 1 - OVERVIEW
# ===========================================================================
if page == "1. Overview":
    st.title("🚔 Chicago Crime Analysis & Arrest Prediction")
    st.markdown(
        """
        This project explores reported crimes in Chicago and builds a simple
        Machine Learning model that predicts **whether a reported crime will
        result in an arrest**. It was built for an ML1 (introductory Machine
        Learning) college assignment, so every step uses basic, easy-to-explain
        techniques - no deep learning, no advanced ensembles.
        """
    )

    overview = basic_overview(clean_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{overview['rows']:,}")
    c2.metric("Total Features", f"{overview['columns']}")
    c3.metric("Number of Crime Types", f"{overview.get('crime_types', 0)}")
    c4.metric("Arrest Rate", f"{overview.get('arrest_rate', 0):.1f}%")

    c5, c6 = st.columns(2)
    c5.metric("Missing Values (raw file)", f"{raw_df.isna().sum().sum():,}")
    c6.metric("Duplicate Rows (raw file)", f"{raw_df.duplicated().sum():,}")

    st.markdown("---")
    st.subheader("Dataset Shape")
    st.write(f"**Rows:** {raw_df.shape[0]:,}  |  **Columns:** {raw_df.shape[1]}")

    st.subheader("First 5 Rows")
    st.dataframe(raw_df.head(), use_container_width=True)

    st.subheader("Last 5 Rows")
    st.dataframe(raw_df.tail(), use_container_width=True)

    st.subheader("Data Types")
    dtype_df = pd.DataFrame({"Column": raw_df.columns, "Data Type": raw_df.dtypes.astype(str)})
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.subheader("Numerical Summary Statistics")
    numerical_cols, categorical_cols = split_column_types(raw_df)
    st.dataframe(raw_df[numerical_cols].describe().T, use_container_width=True)

    st.subheader("Categorical Column Information")
    cat_info = pd.DataFrame(
        {
            "Column": categorical_cols,
            "Unique Values": [raw_df[c].nunique() for c in categorical_cols],
            "Most Frequent": [str(raw_df[c].mode().iloc[0]) if not raw_df[c].mode().empty else "-" for c in categorical_cols],
        }
    )
    st.dataframe(cat_info, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 2 - DATA QUALITY
# ===========================================================================
elif page == "2. Data Quality":
    st.title("Data Quality Analysis")

    st.subheader("Missing Values")
    report = missing_value_report(raw_df)
    st.dataframe(report, use_container_width=True, hide_index=True)
    st.plotly_chart(eda.plot_missing_values(report), use_container_width=True)

    st.markdown("---")
    st.subheader("Duplicate Records")
    dup = duplicate_report(raw_df)
    c1, c2 = st.columns(2)
    c1.metric("Duplicate Rows", f"{dup['duplicate_rows']:,}")
    if "duplicate_ids" in dup:
        c2.metric("Duplicate IDs", f"{dup['duplicate_ids']:,}")

    st.markdown("---")
    st.subheader("Invalid / Inconsistent Values Checked")
    checks = []
    if "Latitude" in raw_df.columns:
        bad_lat = (~raw_df["Latitude"].between(41.6, 42.1)).sum()
        checks.append({"Check": "Latitude outside Chicago city limits", "Count": int(bad_lat)})
    if "Longitude" in raw_df.columns:
        bad_lon = (~raw_df["Longitude"].between(-87.95, -87.5)).sum()
        checks.append({"Check": "Longitude outside Chicago city limits", "Count": int(bad_lon)})
    if "Date" in raw_df.columns:
        unparsed = pd.to_datetime(raw_df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce").isna().sum()
        checks.append({"Check": "Unparsable Date values", "Count": int(unparsed)})
    for col in ["Case Number", "Description", "Location Description", "Block"]:
        if col in raw_df.columns:
            blanks = raw_df[col].astype(str).str.strip().eq("").sum()
            checks.append({"Check": f"Empty strings in '{col}'", "Count": int(blanks)})
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Cleaning Steps Applied")
    st.caption(
        "The dataset is cleaned conservatively: nothing is deleted unless clearly justified. "
        "Below is exactly what was done to produce the data used across the rest of this app."
    )
    for step in cleaning_log:
        st.markdown(f"- {step}")


# ===========================================================================
# PAGE 3 - CRIME ANALYSIS (univariate)
# ===========================================================================
elif page == "3. Crime Analysis":
    st.title("Crime Analysis (Univariate)")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(eda.plot_top_crime_types(filtered_df), use_container_width=True)
    with c2:
        st.plotly_chart(eda.plot_arrest_distribution(filtered_df), use_container_width=True)

    st.plotly_chart(eda.plot_top_descriptions(filtered_df), use_container_width=True)


# ===========================================================================
# PAGE 4 - TIME ANALYSIS
# ===========================================================================
elif page == "4. Time Analysis":
    st.title("Temporal Analysis")

    st.plotly_chart(eda.plot_crimes_by_year(filtered_df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(eda.plot_crimes_by_month(filtered_df), use_container_width=True)
    with c2:
        st.plotly_chart(eda.plot_crimes_by_dayofweek(filtered_df), use_container_width=True)

    st.plotly_chart(eda.plot_crimes_by_hour(filtered_df), use_container_width=True)


# ===========================================================================
# PAGE 5 - LOCATION ANALYSIS
# ===========================================================================
elif page == "5. Location Analysis":
    st.title("Location Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(eda.plot_top_districts(filtered_df), use_container_width=True)
    with c2:
        st.plotly_chart(eda.plot_top_locations(filtered_df), use_container_width=True)

    st.plotly_chart(eda.plot_arrest_rate_by_district(filtered_df), use_container_width=True)

    st.subheader("Crime Map")
    st.caption("A random sample of points is shown so the map stays responsive.")
    st.plotly_chart(eda.plot_crime_map(filtered_df), use_container_width=True)


# ===========================================================================
# PAGE 6 - RELATIONSHIP ANALYSIS (bivariate + multivariate)
# ===========================================================================
elif page == "6. Relationship Analysis":
    st.title("Relationship Analysis")

    st.header("Bivariate Analysis")
    st.plotly_chart(eda.plot_arrest_rate_by_crime_type(filtered_df), use_container_width=True)
    st.plotly_chart(eda.plot_crime_vs_arrest_counts(filtered_df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(eda.plot_arrest_rate_by_hour(filtered_df), use_container_width=True)
    with c2:
        st.plotly_chart(eda.plot_arrest_rate_by_dayofweek(filtered_df), use_container_width=True)
    st.plotly_chart(eda.plot_arrest_rate_by_month(filtered_df), use_container_width=True)

    st.markdown("---")
    st.header("Multivariate Analysis")
    st.plotly_chart(eda.plot_hour_day_heatmap(filtered_df), use_container_width=True)
    st.plotly_chart(eda.plot_crimetype_district_heatmap(filtered_df), use_container_width=True)
    st.plotly_chart(eda.plot_crimetype_hour_heatmap(filtered_df), use_container_width=True)


# ===========================================================================
# PAGE 7 - MACHINE LEARNING
# ===========================================================================
elif page == "7. Machine Learning":
    st.title("Machine Learning: Arrest Prediction")

    st.markdown(
        f"""
        **Target:** `{TARGET_COLUMN}` (converted to `1 = Arrest`, `0 = No Arrest`)
        **Problem type:** Binary classification
        """
    )

    X, y, categorical, numerical = select_features(clean_df)

    st.subheader("Features Used")
    c1, c2 = st.columns(2)
    c1.markdown("**Categorical features**")
    c1.markdown("\n".join(f"- {c}" for c in categorical))
    c2.markdown("**Numerical features**")
    c2.markdown("\n".join(f"- {c}" for c in numerical))

    with st.expander("Why other columns were excluded"):
        for col, reason in EXCLUDED_COLUMNS.items():
            if col in raw_df.columns:
                st.markdown(f"- **{col}**: {reason}")

    st.subheader("Models")
    st.markdown("- Logistic Regression (`max_iter=1000`)\n- Decision Tree (`max_depth=10`)")

    # Train once per dataset/sample-size combination and cache the result,
    # since training on the full ~120k rows takes a few seconds.
    @st.cache_resource(show_spinner="Training models (preprocessing + Logistic Regression + Decision Tree)...")
    def get_training_results(sample_size):
        X, y, categorical, numerical = select_features(clean_df)
        return train_and_evaluate(X, y, categorical, numerical), categorical, numerical

    training, categorical, numerical = get_training_results(sample_size)
    results = training["results"]
    split_info = training["split_info"]

    st.markdown("---")
    st.subheader("Train / Test Split")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Rows", f"{split_info['train_rows']:,}")
    c2.metric("Test Rows", f"{split_info['test_rows']:,}")
    c3.metric("Train Arrest Rate", f"{split_info['train_arrest_rate']:.1f}%")
    c4.metric("Test Arrest Rate", f"{split_info['test_arrest_rate']:.1f}%")
    st.caption("80% train / 20% test, stratified on Arrest so both sets keep the same class balance.")

    st.markdown("---")
    st.subheader("5-Fold Cross-Validation Results (training set)")
    st.dataframe(cv_results_table(results), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Test Set Evaluation")
    st.dataframe(test_results_table(results), use_container_width=True, hide_index=True)
    st.caption(
        "Because roughly two-thirds of crimes do NOT end in an arrest, accuracy alone can be "
        "misleading. Precision tells us how many predicted arrests were correct, and recall tells "
        "us how many real arrests the model actually found - both matter here, which is why the "
        "F1-score (their balance) is used to pick the best model."
    )

    st.markdown("---")
    st.subheader("Confusion Matrices")
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        test = res["test"]
        with col:
            st.markdown(f"**{name}**")
            cm_df = pd.DataFrame(
                [[test["tn"], test["fp"]], [test["fn"], test["tp"]]],
                index=["Actual: No Arrest", "Actual: Arrest"],
                columns=["Predicted: No Arrest", "Predicted: Arrest"],
            )
            st.dataframe(cm_df, use_container_width=True)
            st.caption(
                f"True Negative: {test['tn']:,}  |  False Positive: {test['fp']:,}  \n"
                f"False Negative: {test['fn']:,}  |  True Positive: {test['tp']:,}"
            )
    st.info(
        "**True Negative**: correctly predicted no arrest. **False Positive**: predicted arrest but "
        "there was none. **False Negative**: predicted no arrest but one occurred. "
        "**True Positive**: correctly predicted an arrest."
    )

    st.markdown("---")
    st.subheader("Best Model")
    best_name = training["best_model_name"]
    best_test = results[best_name]["test"]
    st.success(f"**Best Model: {best_name}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 Score", f"{best_test['f1']:.4f}")
    c2.metric("Accuracy", f"{best_test['accuracy']:.4f}")
    c3.metric("Precision", f"{best_test['precision']:.4f}")
    c4.metric("Recall", f"{best_test['recall']:.4f}")
    st.markdown(training["explanation"])

    # Save the training result in session_state so the Prediction Demo can use it
    st.session_state["training"] = training
    st.session_state["feature_columns"] = categorical + numerical
    st.session_state["categorical_features"] = categorical
    st.session_state["numerical_features"] = numerical

    # ---------------------------------------------------------------
    # Prediction demo
    # ---------------------------------------------------------------
    st.markdown("---")
    st.header("Prediction Demo")
    st.caption("Enter details of a hypothetical reported crime and predict whether it would result in an arrest.")

    best_pipeline = training["best_pipeline"]
    form_cols = st.columns(3)
    user_input = {}

    with form_cols[0]:
        if "Primary Type" in categorical:
            user_input["Primary Type"] = st.selectbox("Crime Type", sorted(clean_df["Primary Type"].unique()))
        if "Description" in categorical:
            user_input["Description"] = st.selectbox(
                "Description", sorted(clean_df["Description"].unique())[:200]
            )
        if "Location Description" in categorical:
            user_input["Location Description"] = st.selectbox(
                "Location Description", sorted(clean_df["Location Description"].unique())
            )
        if "Domestic" in categorical:
            user_input["Domestic"] = st.selectbox("Domestic Incident", ["False", "True"])

    with form_cols[1]:
        if "District" in categorical:
            user_input["District"] = st.selectbox("District", sorted(clean_df["District"].astype(str).unique()))
        if "Ward" in categorical:
            user_input["Ward"] = st.selectbox("Ward", sorted(clean_df["Ward"].astype(str).unique()))
        if "Community Area" in categorical:
            user_input["Community Area"] = st.selectbox(
                "Community Area", sorted(clean_df["Community Area"].astype(str).unique())
            )
        if "Beat" in categorical:
            user_input["Beat"] = st.selectbox("Beat", sorted(clean_df["Beat"].astype(str).unique()))

    with form_cols[2]:
        if "Hour" in numerical:
            user_input["Hour"] = st.slider("Hour of Day", 0, 23, 12)
        if "Month" in numerical:
            month_label = st.selectbox("Month", MONTH_NAMES)
            user_input["Month"] = MONTH_NAMES.index(month_label) + 1
        if "DayOfWeek" in numerical:
            day_label = st.selectbox("Day of Week", DAY_NAMES)
            user_input["DayOfWeek"] = DAY_NAMES.index(day_label)
        if "Year" in numerical:
            user_input["Year"] = st.selectbox("Year", sorted(clean_df["Year"].unique(), reverse=True))

    if st.button("Predict Arrest", type="primary"):
        prediction, probability = predict_single(
            best_pipeline, user_input, categorical + numerical
        )
        result_text = "Yes - Arrest Likely" if prediction == 1 else "No - Arrest Unlikely"
        if prediction == 1:
            st.success(f"**Predicted Arrest: {result_text}**")
        else:
            st.warning(f"**Predicted Arrest: {result_text}**")
        if probability is not None:
            st.metric("Probability of Arrest", f"{probability * 100:.1f}%")


# ===========================================================================
# PAGE 8 - MODEL COMPARISON
# ===========================================================================
elif page == "8. Model Comparison":
    st.title("Model Performance Comparison")

    if "training" not in st.session_state:
        st.info("Please open the **7. Machine Learning** page first so the models are trained.")
    else:
        training = st.session_state["training"]
        results = training["results"]

        long_table = comparison_long_table(results)
        fig = px.bar(
            long_table, x="Metric", y="Score", color="Model", barmode="group",
            title="Model Performance Comparison (Test Set)",
            color_discrete_sequence=eda.COLOR_SEQUENCE,
            text="Score",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 1], height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cross-Validation vs Test Results")
        c1, c2 = st.columns(2)
        c1.markdown("**Cross-Validation (training set)**")
        c1.dataframe(cv_results_table(results), use_container_width=True, hide_index=True)
        c2.markdown("**Test Set**")
        c2.dataframe(test_results_table(results), use_container_width=True, hide_index=True)

        best_name = training["best_model_name"]
        st.success(f"**Best Model: {best_name}**")
        st.markdown(training["explanation"])


# ===========================================================================
# PAGE 9 - INSIGHTS
# ===========================================================================
elif page == "9. Insights":
    st.title("EDA Insights")
    st.caption("All findings below are calculated directly from the currently filtered dataset.")

    insights = eda.generate_insights(filtered_df)
    for point in insights:
        st.markdown(f"- {point}")

    if "training" in st.session_state:
        st.markdown("---")
        st.subheader("Machine Learning Summary")
        training = st.session_state["training"]
        st.markdown(training["explanation"])


# ===========================================================================
# PAGE 10 - 3D CRIME INTELLIGENCE MAP (signature feature)
# ===========================================================================
elif page == "10. 3D Crime Map":
    st.title("🌆 Chicago Crime Intelligence")
    st.caption("Historical Crime Analytics — 3D Crime Skyline")

    # The whole page is meaningless without coordinates, so it fails gracefully
    # with a clear message instead of crashing if they are missing.
    if "Latitude" not in clean_df.columns or "Longitude" not in clean_df.columns:
        st.warning(
            "Latitude/Longitude columns are not available in this dataset, so the "
            "3D Crime Map cannot be built. All other pages still work normally."
        )
        st.stop()

    MAP3D_KEYS = [
        "map3d_year", "map3d_crime_type", "map3d_district", "map3d_arrest",
        "map3d_resolution", "map3d_color_by", "map3d_view",
    ]

    def _reset_map_filters():
        for k in MAP3D_KEYS:
            st.session_state.pop(k, None)

    # --- Controls (local to this page, separate from the sidebar filters) ---
    st.markdown("#### Controls")
    c1, c2, c3, c4, c5 = st.columns(5)
    map_years = ["All Years"] + sorted(clean_df["Year"].dropna().unique().tolist())
    map_types = ["All Crime Types"] + sorted(clean_df["Primary Type"].dropna().unique().tolist())
    map_districts = ["All Districts"] + sorted(clean_df["District"].dropna().unique().tolist())

    sel_year = c1.selectbox("Year", map_years, key="map3d_year")
    sel_type = c2.selectbox("Crime Type", map_types, key="map3d_crime_type")
    sel_district = c3.selectbox("District", map_districts, key="map3d_district")
    sel_arrest = c4.selectbox("Arrest Status", ["All", "Arrest Only", "No Arrest Only"], key="map3d_arrest")
    sel_resolution = c5.selectbox("Grid Resolution", list(map3d.RESOLUTIONS.keys()),
                                  index=1, key="map3d_resolution")

    c6, c7, c8 = st.columns([2, 2, 1])
    sel_color_by = c6.radio("Color By", ["Crime Volume", "Arrest Rate", "Crime Type"],
                             horizontal=True, key="map3d_color_by")
    sel_view = c7.radio("Map View", ["3D", "2D"], horizontal=True, key="map3d_view")
    c8.write("")
    c8.button("Reset Filters", on_click=_reset_map_filters, use_container_width=True)

    # --- Apply the local filters ---
    map_df = clean_df
    if sel_year != "All Years":
        map_df = map_df[map_df["Year"] == sel_year]
    if sel_type != "All Crime Types":
        map_df = map_df[map_df["Primary Type"] == sel_type]
    if sel_district != "All Districts":
        map_df = map_df[map_df["District"] == sel_district]
    if sel_arrest == "Arrest Only":
        map_df = map_df[map_df["ArrestFlag"] == 1]
    elif sel_arrest == "No Arrest Only":
        map_df = map_df[map_df["ArrestFlag"] == 0]

    # --- KPI row ---
    st.markdown("#### ")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Crimes", f"{len(map_df):,}")
    k2.metric("Arrests", f"{int(map_df['ArrestFlag'].sum()):,}" if len(map_df) else "0")
    k3.metric("Arrest Rate", f"{map_df['ArrestFlag'].mean() * 100:.1f}%" if len(map_df) else "0%")
    if len(map_df) and "District" in map_df.columns:
        top_area = map_df["District"].value_counts().idxmax()
        k4.metric("Top District", f"District {top_area}")
    else:
        k4.metric("Top District", "-")

    st.caption(
        "This visualization represents historical reported crime activity in the dataset. "
        "It does not represent current crime risk or personal safety."
    )

    if map_df.empty:
        st.info("No records match the current filters. Try widening your selection.")
        st.stop()

    # --- Build the grid (cached: expensive for the full dataset) ---
    GRID_COLUMNS = [c for c in
                    ["Latitude", "Longitude", "ArrestFlag", "Primary Type", "Hour", "DayName", "District"]
                    if c in map_df.columns]

    # A leading underscore tells st.cache_data to skip hashing the dataframe
    # itself (pandas 3's string dtype is not hashable by Streamlit's hasher);
    # `cache_key` is passed instead so the cache still invalidates correctly
    # whenever the filters/sample-size actually change the underlying data.
    @st.cache_data(show_spinner="Aggregating crimes into a geographic grid...")
    def get_grid(_data, bins, cache_key):
        return map3d.aggregate_grid(_data, bins=bins)

    bins = map3d.RESOLUTIONS[sel_resolution]
    grid_cache_key = (sample_size, sel_year, sel_type, sel_district, sel_arrest, bins)
    grid_df = get_grid(map_df[GRID_COLUMNS], bins, grid_cache_key)

    if grid_df.empty:
        st.info("No valid coordinates available for the current filters.")
        st.stop()

    # --- Main map (3D skyline or 2D scatter) ---
    st.markdown("---")
    if sel_view == "3D":
        fig, legend_map = map3d.build_skyline_figure(
            grid_df, color_by=sel_color_by,
            title=f"3D Crime Skyline — colored by {sel_color_by}",
        )
        st.plotly_chart(fig, use_container_width=True)

        if legend_map:
            st.caption(f"Legend — {sel_color_by}")
            swatch_cols = st.columns(min(len(legend_map), 6) or 1)
            for i, (label, color) in enumerate(legend_map.items()):
                with swatch_cols[i % len(swatch_cols)]:
                    st.markdown(
                        f"<span style='display:inline-block;width:12px;height:12px;"
                        f"background:{color};margin-right:6px;border-radius:2px'></span>{label}",
                        unsafe_allow_html=True,
                    )
    else:
        st.plotly_chart(eda.plot_crime_map(map_df), use_container_width=True)

    st.info(
        "**How to read this map** — Height = crime volume in that grid cell (3D view only). "
        f"Color = {sel_color_by.lower()}. Location = geographic position (latitude/longitude), "
        f"grouped into a {bins}x{bins} grid. Hover over a tower for details."
    )

    # --- Hotspot analysis (dropdown-based, reliable fallback for 3D click-selection) ---
    st.markdown("---")
    st.subheader("Hotspot Analysis")
    st.caption("3D click-selection is unreliable inside Streamlit, so hotspots are selected from a dropdown instead.")

    hotspots = map3d.hotspot_table(grid_df, top_n=15)
    chosen_label = st.selectbox("Select Hotspot", hotspots["label"].tolist())
    chosen_row = hotspots[hotspots["label"] == chosen_label].iloc[0]

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Crime Count", f"{int(chosen_row['crime_count']):,}")
    if "arrest_rate" in chosen_row:
        h2.metric("Arrest Rate", f"{chosen_row['arrest_rate']:.1f}%")
    if "dominant_crime" in chosen_row and pd.notna(chosen_row["dominant_crime"]):
        h3.metric("Most Common Crime", chosen_row["dominant_crime"].title())
    if "dominant_district" in chosen_row and pd.notna(chosen_row["dominant_district"]):
        h4.metric("District", f"{int(chosen_row['dominant_district'])}")
    h5, h6 = st.columns(2)
    if "peak_hour" in chosen_row and pd.notna(chosen_row["peak_hour"]):
        h5.metric("Peak Hour", f"{int(chosen_row['peak_hour']):02d}:00")
    if "peak_day" in chosen_row and pd.notna(chosen_row["peak_day"]):
        h6.metric("Peak Day", chosen_row["peak_day"])

    # --- District comparison ---
    st.markdown("---")
    st.subheader("District Comparison")
    if "District" in map_df.columns and map_df["District"].nunique() >= 2:
        available_districts = sorted(map_df["District"].dropna().unique().tolist())
        dc1, dc2 = st.columns(2)
        district_a = dc1.selectbox("District A", available_districts, index=0, key="map3d_district_a")
        district_b = dc2.selectbox(
            "District B", available_districts,
            index=1 if len(available_districts) > 1 else 0, key="map3d_district_b",
        )

        summary_a = map3d.district_summary(map_df, district_a)
        summary_b = map3d.district_summary(map_df, district_b)

        col_a, col_b = st.columns(2)
        for col, summary, label in [(col_a, summary_a, "District A"), (col_b, summary_b, "District B")]:
            with col:
                st.markdown(f"**{label} — District {summary['district']}**")
                st.metric("Crime Count", f"{summary['crime_count']:,}")
                st.metric("Arrest Rate", f"{summary['arrest_rate']:.1f}%")
                if "top_crime" in summary:
                    st.write(f"Most Common Crime: **{summary['top_crime'].title()}**")
                if "peak_hour" in summary:
                    st.write(f"Peak Hour: **{summary['peak_hour']:02d}:00**")

        if district_a != district_b:
            compare_df = map_df[map_df["District"].isin([district_a, district_b])]
            compare_grid = map3d.aggregate_grid(compare_df[GRID_COLUMNS], bins=bins)
            if not compare_grid.empty and "dominant_district" in compare_grid.columns:
                color_map = {district_a: map3d.DISTRICT_PALETTE[0], district_b: map3d.DISTRICT_PALETTE[1]}
                compare_fig, _ = map3d.build_skyline_figure(
                    compare_grid, color_by="Custom",
                    custom_color_map=color_map, custom_color_column="dominant_district",
                    title=f"District {district_a} (blue) vs District {district_b} (red)",
                )
                st.plotly_chart(compare_fig, use_container_width=True)
        else:
            st.caption("Choose two different districts to compare.")
    else:
        st.caption("Not enough districts in the current filters to compare.")

    # --- Optional: crime over time (no animation, just a year/month picker) ---
    st.markdown("---")
    with st.expander("Crime Over Time (optional 3D time view)"):
        st.caption(
            "Pick a year and month to see how the geographic crime pattern looked at that time. "
            "This rebuilds the grid for the selected period only (no animation, kept simple for performance)."
        )
        tc1, tc2 = st.columns(2)
        time_years = sorted(clean_df["Year"].dropna().unique().tolist())
        time_year = tc1.selectbox("Year", time_years, index=len(time_years) - 1, key="map3d_time_year")
        time_month = tc2.selectbox("Month", ["All Months"] + MONTH_NAMES, key="map3d_time_month")

        time_df = clean_df[clean_df["Year"] == time_year]
        if time_month != "All Months":
            time_df = time_df[time_df["Month"] == MONTH_NAMES.index(time_month) + 1]

        if time_df.empty:
            st.info("No records for that period.")
        else:
            time_grid = map3d.aggregate_grid(time_df[GRID_COLUMNS], bins=bins)
            time_fig, _ = map3d.build_skyline_figure(
                time_grid, color_by="Crime Volume",
                title=f"Crime Skyline — {time_month} {time_year}",
            )
            st.plotly_chart(time_fig, use_container_width=True)
