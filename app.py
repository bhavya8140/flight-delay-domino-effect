
# ================================================================
# PHASE 7: STREAMLIT DASHBOARD
# AIRCRAFT DELAY DOMINO EFFECT ANALYSIS
#
# 1. Operations Overview
# 2. Domino Explorer
# 3. Intervention Finder
# 4. What-if Simulator
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ================================================================
# 1. STREAMLIT CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Aircraft Delay Domino Effect",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 1.5rem;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e6eaf0;
}
div[data-testid="stMetricValue"] {
    font-size: 1.7rem;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# 2. PROJECT PATHS
# ================================================================

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(PROJECT_PATH, "data", "processed")
PHASE6_PATH = os.path.join(PROCESSED_PATH, "phase6")


# ================================================================
# 3. DATA LOADING
# ================================================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(path, low_memory=False)


def load_project_file(filename, folder=PROCESSED_PATH):

    path = os.path.join(folder, filename)

    if os.path.exists(path):
        return load_csv(path)

    return pd.DataFrame()


# Phase 3
phase3_connections = load_project_file(
    "aircraft_connections.csv"
)

# Phase 4
phase4_data = load_project_file(
    "phase4_analysis_data.csv"
)

# Phase 5
phase5_data = load_project_file(
    "phase5_intervention_analysis.csv"
)

# Phase 6
phase6_selected_rotation = load_project_file(
    "phase6_selected_rotation.csv",
    PHASE6_PATH
)

phase6_scenario_results = load_project_file(
    "phase6_scenario_results.csv",
    PHASE6_PATH
)

phase6_comparison = load_project_file(
    "phase6_scenario_comparison.csv",
    PHASE6_PATH
)


# ================================================================
# 4. DATA VALIDATION
# ================================================================

if phase4_data.empty and phase3_connections.empty:

    st.error(
        "No Phase 3 or Phase 4 connection dataset found. "
        "Check the files inside data/processed."
    )
    st.stop()


# Phase 4 is preferred for overview analysis.
# Phase 3 is used as a fallback.

df = (
    phase4_data.copy()
    if not phase4_data.empty
    else phase3_connections.copy()
)


numeric_columns = [
    "PREV_ARR_DELAY",
    "NEXT_DEP_DELAY",
    "NEXT_ARR_DELAY",
    "ACTUAL_TURNAROUND_MIN",
    "SCHEDULED_TURNAROUND_BUFFER_MIN",
    "BUFFER_CONSUMED_MIN",
    "OPERATIONAL_RISK_SCORE",
    "INTERVENTION_PRIORITY_SCORE",
    "DOWNSTREAM_DELAY_MINUTES"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ================================================================
# 5. SIDEBAR AND NAVIGATION
# ================================================================

st.sidebar.title("✈️ Aircraft Delay Domino Effect")

st.sidebar.caption(
    "Flight Operations and Delay Propagation Dashboard"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate Dashboard",
    [
        "Operations Overview",
        "Domino Explorer",
        "Intervention Finder",
        "What-if Simulator"
    ]
)


# ================================================================
# 6. COMMON FILTERS
# ================================================================

st.sidebar.subheader("Dashboard Filters")

delay_threshold = st.sidebar.slider(
    "Significant delay threshold (minutes)",
    min_value=5,
    max_value=60,
    value=15,
    step=5
)


# Recalculate flags using the selected threshold.

if "PREV_ARR_DELAY" in df.columns:

    df["PREVIOUS_DELAYED"] = (
        df["PREV_ARR_DELAY"] >= delay_threshold
    )

if "NEXT_DEP_DELAY" in df.columns:

    df["NEXT_DELAYED"] = (
        df["NEXT_DEP_DELAY"] >= delay_threshold
    )

if (
    "PREVIOUS_DELAYED" in df.columns
    and "NEXT_DELAYED" in df.columns
):

    df["POTENTIAL_PROPAGATION"] = (
        df["PREVIOUS_DELAYED"]
        & df["NEXT_DELAYED"]
    )


# Airline filter

if "PREV_CARRIER" in df.columns:

    carriers = sorted(
        df["PREV_CARRIER"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_carriers = st.sidebar.multiselect(
        "Select airline",
        carriers,
        default=carriers
    )

    filtered_df = df[
        df["PREV_CARRIER"].astype(str).isin(
            selected_carriers
        )
    ].copy()

else:

    filtered_df = df.copy()


# ================================================================
# 7. HELPER FUNCTIONS
# ================================================================

def safe_mean(series):

    return (
        series.mean()
        if series.notna().any()
        else np.nan
    )


def format_number(value, suffix=""):

    if pd.isna(value):
        return "N/A"

    return f"{value:,.1f}{suffix}"


def get_value(row, column, default=np.nan):

    if column in row.index:
        return row[column]

    return default


def create_flight_label(row, prefix):

    flight_id = get_value(
        row,
        prefix + "_FLIGHT_ID",
        "Unknown"
    )

    origin = get_value(
        row,
        prefix + "_ORIGIN",
        "?"
    )

    destination = get_value(
        row,
        prefix + "_DEST",
        "?"
    )

    return f"{flight_id}: {origin} → {destination}"


def filter_by_carrier(data):

    if (
        "PREV_CARRIER" in data.columns
        and selected_carriers
    ):

        return data[
            data["PREV_CARRIER"].astype(str).isin(
                selected_carriers
            )
        ].copy()

    if "PREV_CARRIER" in data.columns:

        return data.iloc[0:0].copy()

    return data.copy()


# ================================================================
# 8. MAIN HEADER
# ================================================================

st.title("✈️ Aircraft Delay Domino Effect")

st.markdown(
    "An interactive analytical platform for understanding "
    "flight delays, aircraft rotations, delay propagation "
    "and operational intervention opportunities."
)

st.divider()


# ================================================================
# PAGE 1: OPERATIONS OVERVIEW
# ================================================================

if page == "Operations Overview":

    st.header("Operations Overview")

    total_connections = len(filtered_df)

    delayed_previous = (
        filtered_df["PREVIOUS_DELAYED"].sum()
        if "PREVIOUS_DELAYED" in filtered_df.columns
        else 0
    )

    delayed_next = (
        filtered_df["NEXT_DELAYED"].sum()
        if "NEXT_DELAYED" in filtered_df.columns
        else 0
    )

    propagated = (
        filtered_df["POTENTIAL_PROPAGATION"].sum()
        if "POTENTIAL_PROPAGATION" in filtered_df.columns
        else 0
    )

    propagation_rate = (
        propagated / delayed_previous * 100
        if delayed_previous > 0
        else 0
    )

    unique_aircraft = (
        filtered_df["PREV_TAIL_NUM"].nunique()
        if "PREV_TAIL_NUM" in filtered_df.columns
        else 0
    )

    avg_next_delay = safe_mean(
        filtered_df["NEXT_DEP_DELAY"]
    )

    # KPI cards

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Aircraft Connections",
        f"{total_connections:,}"
    )

    c2.metric(
        "Unique Aircraft",
        f"{unique_aircraft:,}"
    )

    c3.metric(
        "Potential Propagation Events",
        f"{propagated:,}"
    )

    c4.metric(
        "Propagation Rate",
        format_number(propagation_rate, "%")
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Delayed Previous Flights",
        f"{delayed_previous:,}"
    )

    c6.metric(
        "Delayed Next Flights",
        f"{delayed_next:,}"
    )

    c7.metric(
        "Average Next Departure Delay",
        format_number(avg_next_delay, " min")
    )

    st.caption(
        "Propagation rate = potential propagation events / "
        "connections with a delayed previous flight."
    )

    st.divider()

    # Delay distributions

    st.subheader("Delay Distribution")

    left, right = st.columns(2)

    with left:

        fig = px.histogram(
            filtered_df,
            x="PREV_ARR_DELAY",
            nbins=40,
            title="Previous Flight Arrival Delay",
            labels={
                "PREV_ARR_DELAY": "Arrival Delay (Minutes)"
            }
        )

        fig.add_vline(
            x=delay_threshold,
            line_dash="dash"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        fig = px.histogram(
            filtered_df,
            x="NEXT_DEP_DELAY",
            nbins=40,
            title="Next Flight Departure Delay",
            labels={
                "NEXT_DEP_DELAY": "Departure Delay (Minutes)"
            }
        )

        fig.add_vline(
            x=delay_threshold,
            line_dash="dash"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Delay relationship

    st.subheader(
        "Previous Arrival vs Next Departure Delay"
    )

    hover_columns = [
        col for col in [
            "PREV_TAIL_NUM",
            "PREV_CARRIER",
            "PREV_FLIGHT_ID",
            "NEXT_FLIGHT_ID"
        ]
        if col in filtered_df.columns
    ]

    fig = px.scatter(
        filtered_df,
        x="PREV_ARR_DELAY",
        y="NEXT_DEP_DELAY",
        color="POTENTIAL_PROPAGATION",
        hover_data=hover_columns,
        title="Aircraft Connection Delay Relationship",
        labels={
            "PREV_ARR_DELAY": "Previous Arrival Delay (min)",
            "NEXT_DEP_DELAY": "Next Departure Delay (min)",
            "POTENTIAL_PROPAGATION": "Potential Propagation"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Airline comparison

    st.subheader("Airline Delay Comparison")

    if "PREV_CARRIER" in filtered_df.columns:

        airline_summary = (
            filtered_df
            .groupby("PREV_CARRIER")
            .agg(
                Connections=("NEXT_DEP_DELAY", "size"),
                Previous_Arrival_Delay=(
                    "PREV_ARR_DELAY", "mean"
                ),
                Next_Departure_Delay=(
                    "NEXT_DEP_DELAY", "mean"
                ),
                Propagation_Events=(
                    "POTENTIAL_PROPAGATION", "sum"
                )
            )
            .reset_index()
        )

        airline_summary["Propagation_Rate"] = (
            airline_summary["Propagation_Events"]
            /
            airline_summary["Connections"].replace(0, np.nan)
        ) * 100

        fig = px.bar(
            airline_summary,
            x="PREV_CARRIER",
            y="Propagation_Rate",
            title="Potential Propagation Rate by Airline",
            hover_data=["Connections"],
            labels={
                "PREV_CARRIER": "Airline",
                "Propagation_Rate": "Propagation Rate (%)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            airline_summary,
            use_container_width=True,
            hide_index=True
        )

    # Turnaround analysis

    if "ACTUAL_TURNAROUND_MIN" in filtered_df.columns:

        st.subheader("Aircraft Turnaround Analysis")

        fig = px.scatter(
            filtered_df,
            x="ACTUAL_TURNAROUND_MIN",
            y="NEXT_DEP_DELAY",
            color="POTENTIAL_PROPAGATION",
            title="Actual Turnaround vs Next Flight Delay",
            labels={
                "ACTUAL_TURNAROUND_MIN": "Actual Turnaround (Minutes)",
                "NEXT_DEP_DELAY": "Next Departure Delay (Minutes)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ================================================================
# PAGE 2: DOMINO EXPLORER
# ================================================================

elif page == "Domino Explorer":

    st.header("Domino Explorer")

    st.markdown(
        "Explore consecutive flights operated by the same "
        "aircraft and visualize delay progression."
    )

    if phase3_connections.empty:

        st.warning("Phase 3 connection dataset is unavailable.")

    else:

        domino_df = filter_by_carrier(
            phase3_connections
        )

        domino_df = domino_df.copy()

        for col in [
            "PREV_ARR_DELAY",
            "NEXT_DEP_DELAY",
            "NEXT_ARR_DELAY",
            "ACTUAL_TURNAROUND_MIN"
        ]:

            if col in domino_df.columns:

                domino_df[col] = pd.to_numeric(
                    domino_df[col],
                    errors="coerce"
                )

        if "PREV_TAIL_NUM" not in domino_df.columns:

            st.warning("Aircraft registration column is missing.")

        else:

            aircraft_options = sorted(
                domino_df["PREV_TAIL_NUM"]
                .dropna()
                .astype(str)
                .unique()
            )

            selected_tail = st.selectbox(
                "Select Aircraft Registration",
                aircraft_options
            ) if aircraft_options else None

            if selected_tail is None:

                st.warning("No aircraft registrations available.")

            else:

                aircraft_df = domino_df[
                    domino_df["PREV_TAIL_NUM"].astype(str)
                    == str(selected_tail)
                ].copy()

                aircraft_df["PREV_DEP_UTC"] = pd.to_datetime(
                    aircraft_df["PREV_DEP_UTC"],
                    errors="coerce",
                    utc=True
                )

                aircraft_df = aircraft_df.sort_values(
                    "PREV_DEP_UTC"
                ).reset_index(drop=True)

                # Recalculate propagation flags

                aircraft_df["PROPAGATION_EVENT"] = (
                    (aircraft_df["PREV_ARR_DELAY"] >= delay_threshold)
                    &
                    (aircraft_df["NEXT_DEP_DELAY"] >= delay_threshold)
                )

                st.subheader(
                    f"Aircraft Rotation: {selected_tail}"
                )

                a, b, c = st.columns(3)

                a.metric(
                    "Connections",
                    len(aircraft_df)
                )

                b.metric(
                    "Potential Propagation Events",
                    int(aircraft_df["PROPAGATION_EVENT"].sum())
                )

                c.metric(
                    "Average Previous Arrival Delay",
                    format_number(
                        safe_mean(aircraft_df["PREV_ARR_DELAY"]),
                        " min"
                    )
                )

                # Flight chain

                st.subheader("Aircraft Flight Chain")

                chain = []

                for _, row in aircraft_df.iterrows():

                    chain.append({
                        "Previous Flight": create_flight_label(
                            row, "PREV"
                        ),
                        "Next Flight": create_flight_label(
                            row, "NEXT"
                        ),
                        "Previous Arrival Delay": row["PREV_ARR_DELAY"],
                        "Next Departure Delay": row["NEXT_DEP_DELAY"],
                        "Actual Turnaround": row["ACTUAL_TURNAROUND_MIN"],
                        "Potential Propagation": row["PROPAGATION_EVENT"]
                    })

                chain_df = pd.DataFrame(chain)

                st.dataframe(
                    chain_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Delay progression

                st.subheader("Delay Progression")

                if not aircraft_df.empty:

                    flight_labels = []
                    arrival_delays = []

                    for _, row in aircraft_df.iterrows():

                        previous_id = row["PREV_FLIGHT_ID"]
                        next_id = row["NEXT_FLIGHT_ID"]

                        if (
                            not flight_labels
                            or flight_labels[-1] != str(previous_id)
                        ):

                            flight_labels.append(str(previous_id))
                            arrival_delays.append(
                                row["PREV_ARR_DELAY"]
                            )

                        flight_labels.append(str(next_id))
                        arrival_delays.append(
                            row["NEXT_ARR_DELAY"]
                        )

                    progression_df = pd.DataFrame({
                        "Flight": flight_labels,
                        "Arrival Delay": arrival_delays
                    })

                    fig = px.line(
                        progression_df,
                        x="Flight",
                        y="Arrival Delay",
                        markers=True,
                        title="Arrival Delay Across Aircraft Rotation"
                    )

                    fig.add_hline(
                        y=delay_threshold,
                        line_dash="dash",
                        annotation_text="Delay threshold"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                st.subheader("Detailed Connection Records")

                st.dataframe(
                    aircraft_df,
                    use_container_width=True,
                    hide_index=True
                )


# ================================================================
# PAGE 3: INTERVENTION FINDER
# ================================================================

elif page == "Intervention Finder":

    st.header("Intervention Finder")

    st.markdown(
        "Identify vulnerable aircraft connections and rank "
        "intervention opportunities using the actual Phase 5 analysis."
    )

    if phase5_data.empty:

        st.warning(
            "Phase 5 intervention dataset is unavailable."
        )

    else:

        intervention_df = filter_by_carrier(
            phase5_data
        )

        required_columns = [
            "INTERVENTION_PRIORITY_SCORE",
            "INTERVENTION_PRIORITY_RANK",
            "INTERVENTION_CATEGORY"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in intervention_df.columns
        ]

        if missing_columns:

            st.error(
                f"Missing Phase 5 columns: {missing_columns}"
            )

        else:

            # Use the actual Phase 5 results.
            intervention_df = intervention_df.copy()

            intervention_df = intervention_df.sort_values(
                "INTERVENTION_PRIORITY_RANK",
                ascending=True
            )

            # KPI cards

            category_counts = (
                intervention_df["INTERVENTION_CATEGORY"]
                .value_counts()
            )

            priority_count = category_counts.get(
                "Priority Review", 0
            )

            review_count = category_counts.get(
                "Review", 0
            )

            monitor_count = category_counts.get(
                "Monitor", 0
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Priority Review",
                f"{priority_count:,}"
            )

            c2.metric(
                "Review",
                f"{review_count:,}"
            )

            c3.metric(
                "Monitor",
                f"{monitor_count:,}"
            )

            st.divider()

            # Priority distribution

            st.subheader("Intervention Priority Distribution")

            priority_summary = (
                intervention_df["INTERVENTION_CATEGORY"]
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Connections")
            )

            fig = px.pie(
                priority_summary,
                names="Category",
                values="Connections",
                hole=0.45,
                title="Actual Phase 5 Intervention Categories"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Vulnerability distribution

            if "VULNERABILITY_GROUP" in intervention_df.columns:

                st.subheader("Connection Vulnerability")

                vulnerability_summary = (
                    intervention_df["VULNERABILITY_GROUP"]
                    .value_counts()
                    .rename_axis("Vulnerability")
                    .reset_index(name="Connections")
                )

                fig = px.bar(
                    vulnerability_summary,
                    x="Vulnerability",
                    y="Connections",
                    title="Aircraft Connection Vulnerability"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # Top intervention opportunities

            st.subheader("Top 20 Intervention Opportunities")

            top_columns = [
                col for col in [
                    "INTERVENTION_PRIORITY_RANK",
                    "PREV_FLIGHT_ID",
                    "NEXT_FLIGHT_ID",
                    "PREV_TAIL_NUM",
                    "PREV_CARRIER",
                    "PREV_ORIGIN",
                    "PREV_DEST",
                    "PREV_ARR_DELAY",
                    "NEXT_DEP_DELAY",
                    "VULNERABILITY_GROUP",
                    "OPERATIONAL_RISK_PERCENT",
                    "INTERVENTION_PRIORITY_SCORE",
                    "INTERVENTION_CATEGORY",
                    "DOWNSTREAM_CHAIN_LENGTH",
                    "DOWNSTREAM_DELAYED_FLIGHTS",
                    "DOWNSTREAM_DELAY_MINUTES",
                    "SUGGESTED_INTERVENTION"
                ]
                if col in intervention_df.columns
            ]

            st.dataframe(
                intervention_df[top_columns].head(20),
                use_container_width=True,
                hide_index=True
            )

            # Category filter

            selected_category = st.selectbox(
                "Filter by intervention category",
                [
                    "All",
                    "Priority Review",
                    "Review",
                    "Monitor"
                ]
            )

            if selected_category == "All":

                category_df = intervention_df.copy()

            else:

                category_df = intervention_df[
                    intervention_df["INTERVENTION_CATEGORY"]
                    == selected_category
                ].copy()

            st.subheader("Filtered Intervention Opportunities")

            st.dataframe(
                category_df[top_columns],
                use_container_width=True,
                hide_index=True
            )

            # Download

            csv = intervention_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Phase 5 Intervention Ranking",
                data=csv,
                file_name="phase5_intervention_ranking.csv",
                mime="text/csv"
            )


# ================================================================
# PAGE 4: WHAT-IF SIMULATOR
# ================================================================

elif page == "What-if Simulator":

    st.header("What-if Simulator")

    st.markdown(
        "Explore the results of the Phase 6 delay-reduction "
        "and turnaround-buffer scenarios."
    )

    st.info(
        "This dashboard displays the saved Phase 6 simulation. "
        "The simulation is a simplified heuristic model, "
        "not an operational dispatch prediction."
    )

    if phase6_comparison.empty or phase6_scenario_results.empty:

        st.warning(
            "Phase 6 simulation outputs were not found. "
            "Check the data/processed/phase6 folder."
        )

    else:

        comparison_df = phase6_comparison.copy()
        results_df = phase6_scenario_results.copy()

        # Scenario comparison

        st.subheader("Scenario Comparison")

        comparison_df["SCENARIO"] = (
            comparison_df["SCENARIO"].astype(str)
        )

        fig = px.bar(
            comparison_df,
            x="SCENARIO",
            y="Total_Departure_Delay_Minutes",
            color="SCENARIO",
            title="Total Simulated Departure Delay by Scenario",
            labels={
                "SCENARIO": "Scenario",
                "Total_Departure_Delay_Minutes": "Total Departure Delay (Minutes)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # Scenario selection

        st.subheader("Explore Individual Scenario")

        scenarios = results_df["SCENARIO"].dropna().unique().tolist()

        selected_scenario = st.selectbox(
            "Select scenario",
            scenarios
        )

        selected_results = results_df[
            results_df["SCENARIO"] == selected_scenario
        ].copy()

        if "FLIGHT_NUMBER" in selected_results.columns:

            selected_results = selected_results.sort_values(
                "FLIGHT_NUMBER"
            )

        # Scenario metrics

        selected_summary = comparison_df[
            comparison_df["SCENARIO"] == selected_scenario
        ]

        if not selected_summary.empty:

            summary = selected_summary.iloc[0]

            a, b, c = st.columns(3)

            a.metric(
                "Total Departure Delay",
                format_number(
                    summary["Total_Departure_Delay_Minutes"],
                    " min"
                )
            )

            b.metric(
                "Average Departure Delay",
                format_number(
                    summary["Average_Departure_Delay_Minutes"],
                    " min"
                )
            )

            c.metric(
                "Delay Minutes Reduced",
                format_number(
                    summary["Delay_Minutes_Reduced"],
                    " min"
                )
            )

        # Flight-level results

        st.subheader("Flight-by-Flight Simulation Results")

        fig = go.Figure()

        if "SIMULATED_DEPARTURE_DELAY" in selected_results.columns:

            fig.add_trace(
                go.Scatter(
                    x=selected_results["FLIGHT_NUMBER"],
                    y=selected_results["SIMULATED_DEPARTURE_DELAY"],
                    mode="lines+markers",
                    name="Simulated Departure Delay"
                )
            )

        if "SIMULATED_ARRIVAL_DELAY" in selected_results.columns:

            fig.add_trace(
                go.Scatter(
                    x=selected_results["FLIGHT_NUMBER"],
                    y=selected_results["SIMULATED_ARRIVAL_DELAY"],
                    mode="lines+markers",
                    name="Simulated Arrival Delay"
                )
            )

        fig.update_layout(
            title=f"Delay Progression: {selected_scenario}",
            xaxis_title="Flight Sequence",
            yaxis_title="Delay (Minutes)",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            selected_results,
            use_container_width=True,
            hide_index=True
        )

        # Download

        simulation_csv = selected_results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Selected Scenario",
            data=simulation_csv,
            file_name="phase6_scenario_results.csv",
            mime="text/csv"
        )

        # Selected rotation context

        if not phase6_selected_rotation.empty:

            with st.expander("View Selected Aircraft Rotation"):

                st.dataframe(
                    phase6_selected_rotation,
                    use_container_width=True,
                    hide_index=True
                )


# ================================================================
# 9. FOOTER
# ================================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        Aircraft Delay Domino Effect Analysis |
        Streamlit Interactive Dashboard |
        Academic Data Science Project
    </div>
    """,
    unsafe_allow_html=True
)
