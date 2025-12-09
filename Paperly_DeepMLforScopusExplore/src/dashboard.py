import streamlit as st
import pandas as pd
import altair as alt
import joblib
import io
import plotly.express as px
from collections import Counter

st.title("Paperly")
st.header("Scopus Analysis")

if st.button("to search"):
    st.sidebar.switch_page("pages/search.py")

# --- Overall Dataframe ---
df = joblib.load("src/df_for_dashboard.pkl")
desc = pd.DataFrame({
    "dtype": df.dtypes,
    "missing": df.isna().sum(),
    "unique": df.nunique()
})
st.subheader("Original Data Summary")
st.dataframe(desc)

# --- Missing Values ---
st.subheader("Data Quality Summary")
st.write("Missing Values per Column")
missing_df = df.isna().sum().reset_index()
missing_df.columns = ["Column", "Missing Count"]
st.dataframe(missing_df)

# --- Duplicates ---
st.write("Duplicate Records")

dup_titles = df.duplicated(subset="title").sum()
dup_doi = df.duplicated(subset="doi").sum()

dup_df = pd.DataFrame({
    "Type": ["Duplicate Titles", "Duplicate DOIs"],
    "Count": [dup_titles, dup_doi]
})

st.table(dup_df)

# Fix DataFrame
df_clean = df.drop_duplicates(subset="doi", keep="first")
df_clean = df_clean.drop_duplicates(subset="title", keep="first")
df_clean = df_clean.reset_index(drop=True)

data_clean = pd.DataFrame({
    "status": ["unclean", "cleaned"],
    "count": [len(df), len(df_clean)]
})

# --- Data indight  ---
st.header("Data Insight")
# --- Number of paper per year ---
st.subheader("Number of Papers per Year")

papers_per_year = df["year"].value_counts().sort_index()

fig = px.bar(
    papers_per_year,
    labels={'value': 'Count', 'index': 'Year'},
    text='value'
)

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Number of Papers",
    width=700,
    height=450
)

st.plotly_chart(fig, use_container_width=True)


# --- Number of paper per year ---
st.subheader("Top 15 Publication Sources")

top_journals = df["source_title"].value_counts().head(15)


fig = px.bar(
    top_journals.sort_values(),
    orientation="h",
    labels={'value': 'Number of Papers', 'index': 'Journal / Source'},
    text='value'
)

fig.update_layout(
    xaxis_title="Number of Papers",
    yaxis_title="Journal / Source",
    width=800,
    height=600
)

st.plotly_chart(fig, use_container_width=True)


#--- Top 20 Keywords ---
st.subheader("Top 20 Keywords")

kw_counter = Counter()

for kws in df["keywords"].dropna():
    for k in kws.split(";"):
        k = k.strip().lower()
        if k:
            kw_counter[k] += 1

top_kw = kw_counter.most_common(20)
kw_df = pd.DataFrame(top_kw, columns=["Keyword", "Count"])

fig = px.bar(
    kw_df.sort_values("Count"),
    x="Count",
    y="Keyword",
    orientation="h",
    title="Top 20 Keywords",
    text="Count",
)

fig.update_layout(
    width=800,
    height=600,
    xaxis_title="Count",
    yaxis_title="Keyword"
)

st.plotly_chart(fig, use_container_width=True)


#--- Before & After Data Cleaning ---
st.header("Before & After Data Cleaning")

chart = (
    alt.Chart(data_clean)
    .mark_bar(cornerRadius=6)
    .encode(
        x=alt.X("count:Q", title="Number of Records"),
        y=alt.Y("status:N", sort="-x", title="Data Status"),
        color=alt.Color("status:N", scale=alt.Scale(scheme="tableau10")),
        tooltip=["status", "count"]
    )
    .properties(
        width=500,
        height=200,
    )
)
st.altair_chart(chart, use_container_width=True)



