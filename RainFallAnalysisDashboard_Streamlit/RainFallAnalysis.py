import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime

#------------------------------------
#              Load Data
#------------------------------------
# For grader submission
df = pd.read_csv("RainDaily_Tabular.csv")

# Clean columns and convert types
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["rain"] = pd.to_numeric(df["rain"], errors="coerce")

#------------------------------------
#              Sidebar
#------------------------------------
st.sidebar.header("Sidebar")

# Default start and end dates (as specified)
start_dt = st.sidebar.date_input("Start Date", datetime(2017, 8, 1).date())
end_dt   = st.sidebar.date_input("End Date", datetime(2017, 8, 31).date())

# Province selection
provinces = sorted(df["province"].dropna().astype(str).unique().tolist())
selected_provinces = st.sidebar.multiselect(
    "Select Province",
    provinces,
    default=["Bangkok"] if "Bangkok" in provinces else []
)

#------------------------------------
#           Data Filtering
#------------------------------------
if start_dt > end_dt:
    st.error("⚠️ Start Date must be on or before End Date.")
    st.stop()

mask_date = (df["date"].dt.date >= start_dt) & (df["date"].dt.date <= end_dt)
mask_prov = df["province"].astype(str).isin(selected_provinces) if selected_provinces else True
filtered = df.loc[mask_date & mask_prov].copy()

#------------------------------------
#              Main Area
#------------------------------------

# 1️⃣  Horizontal Bar Chart
st.subheader("Average of rain by province")
bar_src = (
    filtered.groupby("province", as_index=False)["rain"]
            .mean()
            .rename(columns={"rain": "avg_rain"})
)
if bar_src.empty:
    st.info("No data to display for the selected filters.")
else:
    bar = (
        alt.Chart(bar_src)
        .mark_bar()
        .encode(
            y=alt.Y("province:N", sort="-x", title="Province"),
            x=alt.X("avg_rain:Q", title="Average rain"),
            tooltip=[
                alt.Tooltip("province:N", title="Province"),
                alt.Tooltip("avg_rain:Q", format=".2f", title="Avg Rain (mm)")
            ]
        )
        .properties(height=max(250, 22 * len(bar_src)))
    )
    st.altair_chart(bar, use_container_width=True)

# 2️⃣  Multi-line Chart (Rain by date colored by province)
st.subheader("Average of rain by date")
line_src = (
    filtered.groupby(["date", "province"], as_index=False)["rain"]
            .mean()
            .rename(columns={"rain": "avg_rain"})
)
if line_src.empty:
    st.info("No data to display for the selected filters.")
else:
    line = (
        alt.Chart(line_src)
        .mark_line(point=False)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("avg_rain:Q", title="Average of rain"),
            color=alt.Color("province:N", title="Province"),
            tooltip=[
                alt.Tooltip("date:T", title="date"),
                alt.Tooltip("province:N", title="Province"),
                alt.Tooltip("avg_rain:Q", title="Avg Rain (mm)", format=".2f")
            ]
        )
        .interactive()
    )
    st.altair_chart(line, use_container_width=True)

# 3️⃣  Map
import pydeck as pdk
import numpy as np

st.subheader("Rain by latitude and longitude")

# Keep only rows in selected provinces (filtered already includes date range)
if selected_provinces:
    df_map = filtered[filtered["province"].astype(str).isin(selected_provinces)].copy()
else:
    df_map = filtered.copy()

# Require lat/lon + rain
needed = {"latitude", "longitude", "rain"}
if not needed.issubset(df_map.columns):
    st.info("Latitude/Longitude or rain columns not found.")
else:
    df_map = df_map[["latitude", "longitude", "rain", "province"] + ([col for col in ["Name"] if col in df_map.columns])].dropna(subset=["latitude","longitude","rain"])
    if df_map.empty:
        st.info("No data to display for the selected provinces.")
    else:
        # Normalize rain for color (blue→red) and radius
        r = pd.to_numeric(df_map["rain"], errors="coerce").fillna(0).clip(lower=0)
        r_min, r_max = float(r.min()), float(r.max())
        span = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
        r_norm = (r - r_min) / span

        # Color ramp: low rain = blue [0, 120, 255], high rain = red [255, 60, 0]
        df_map["color"] = r_norm.apply(lambda t: [
            int(0 + t*(255-0)),       # R
            int(120 + t*(60-120)),    # G
            int(255 + t*(0-255)),     # B
            180                        # A (opacity)
        ])

        # Scale radius with rain (meters)
        df_map["radius"] = (r_norm * 9000) + 3000  # tweak to taste

        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            opacity=0.7,
        )

        # Center view on selected points
        view_state = pdk.ViewState(
            latitude=float(df_map["latitude"].mean()),
            longitude=float(df_map["longitude"].mean()),
            zoom=5,
            pitch=0,
            bearing=0,
        )

        # Tooltip shows province and rain (and station name if present)
        tooltip_fields = [
            {"text": "<b>Province:</b> {province}"},
            {"text": "<b>Rain:</b> {rain} mm"},
        ]
        if "Name" in df_map.columns:
            tooltip_fields.insert(1, {"text": "<b>Station:</b> {Name}"})

        tooltip = {"html": "<br/>".join([f["text"] for f in tooltip_fields]),
                   "style": {"color": "white"}}

        st.pydeck_chart(pdk.Deck(layers=[scatter], initial_view_state=view_state, tooltip=tooltip))


# 4️⃣  Summary
st.subheader("Summary")
start = start_dt.strftime("%Y-%m-%d")
end = end_dt.strftime("%Y-%m-%d")

st.write("Start Date: ", start)
st.write("End Date: ", end)

p_list = filtered["province"].dropna().astype(str).unique().tolist()
province_text = ", ".join(p_list)

days_in_range = (end_dt - start_dt).days + 1
total_rain = float(filtered["rain"].sum(skipna=True)) if not filtered.empty else 0.0
total_prov = len(p_list)
avg_rain = total_rain / days_in_range if days_in_range > 0 else 0
max_rain = filtered["rain"].max(skipna=True) if not filtered.empty else 0
min_rain = filtered["rain"].min(skipna=True) if not filtered.empty else 0

st.write("Provinces: ", province_text or "—")
st.write("Total Days: ", str(days_in_range))
st.write("Total Provinces: ", str(total_prov))
st.write("Total Rain: ", str(round(total_rain, 2)))
st.write("Average Rain: ", str(round(avg_rain, 2)))
st.write("Max Rain: ", str(round(max_rain, 2)))
st.write("Min Rain: ", str(round(min_rain, 2)))
