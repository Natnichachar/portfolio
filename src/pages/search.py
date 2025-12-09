import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import plotly.express as px
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import HTML
import re
import umap.umap_ as umap


# Correct project root: move up TWO levels from pages/ folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # /projectDS/src/pages
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # /projectDS

# Correct folder paths
models_dir = os.path.join(ROOT_DIR, "models")
stat_dir   = os.path.join(ROOT_DIR, "stat")
data_dir   = os.path.join(ROOT_DIR, "data")

print("ROOT_DIR =", ROOT_DIR)
print("models_dir =", models_dir)
print("stat_dir =", stat_dir)
print("data_dir =", data_dir)

@st.cache_resource
def load_models():
    #pathModel = r"D:\vscode\projectDS\models"
    encoder = SentenceTransformer(models_dir)
    kmeans_model = joblib.load(os.path.join(models_dir, "kmeans_model.pkl"))
    X_norm = joblib.load(os.path.join(models_dir, "X_norm"))
    umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=10,  # 10–20 is good
    metric="cosine",
    random_state = 0
).fit(X_norm)
    cluster_stats = joblib.load(os.path.join(stat_dir, "cluster_stats.pkl"))
    cluster_labels = joblib.load(os.path.join(stat_dir, "cluster_labels.pkl"))
    umap_vis = umap.UMAP(
    n_neighbors=15,
    min_dist=0.0,          
    n_components=2,
    metric="cosine",
    random_state=42, 
    force_approximation_algorithm=True
).fit(X_norm)
    X_vis = umap_vis.fit_transform(X_norm)
    metadata = pd.read_csv(os.path.join(data_dir, "scopus_2018_2023_cleanfinal.csv"))
    metadata2 = pd.read_csv(os.path.join(data_dir, "scopus_2018_2023_cleanVer2.csv"))
    labels = kmeans_model.labels_
    embeddings_path = os.path.join(models_dir, "specter2_embeddings.npy")
    embeddings = np.load(embeddings_path)
    return encoder,kmeans_model, X_norm, umap_model, cluster_stats, cluster_labels, umap_vis, X_vis, metadata, metadata2, labels, embeddings


encoder,kmeans_model, X_norm, umap_model, cluster_stats, cluster_labels, umap_vis, X_vis, metadata, metadata2, labels, embeddings = load_models()

def position_idea(idea_text: str):
    ideaCode = encoder.encode([idea_text])
    ideaCodeNorm = normalize(ideaCode)
    ideaUmap = umap_model.transform(ideaCodeNorm)
    clusterID = int(kmeans_model.predict(ideaUmap)[0])
    stats = cluster_stats[clusterID]
    centroid = stats["centroid"]
    dists_cluster = stats["dists"]
    mean_dist = stats["mean_dist"]
    std_dist = stats["std_dist"]
    dist = float(np.linalg.norm(ideaUmap[0] - centroid))
    novelty_z = (dist - mean_dist) / std_dist
    percentile = float((dists_cluster < dist).mean())
    clusterLabel = ""
    if clusterID == 0:
        clusterLabel = cluster_labels[0]
    elif clusterID == 1:
        clusterLabel = cluster_labels[1]
    elif clusterID == 2:
        clusterLabel = cluster_labels[2]
    elif clusterID == 3:
        clusterLabel = cluster_labels[3]
    elif clusterID == 4:
        clusterLabel = cluster_labels[4]
    
    return {
        "Cluster id": clusterID,
        "Cluster Label": clusterLabel,
        "Idea_Umap": ideaUmap[0],
        "distance_to_centroid": dist,
        "novelty_z": float(novelty_z),
        "novelty_percentile": percentile,
    }

def scholar_search(query, k=10, min_match_words=1, return_df=False):
    #Split query into words
    words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
    if not words:
        if return_df:
            return pd.DataFrame()
        else:
            return HTML("<b>No valid query words.</b>")
    
    #Make sure we have lowercased text
    if "text_lower" not in metadata2.columns:
        metadata2["text_lower"] = metadata2["text"].fillna("").str.lower().str.strip()
    
    texts = metadata2["text_lower"].fillna("")
    
    #Filter: keep rows that contain at least min_match_words from query
    mask = []
    for t in texts:
        count = 0
        for w in words:
            if w in t:
                count += 1
        mask.append(count >= min_match_words)
    
    mask = np.array(mask)
    candidate_idx = np.where(mask)[0]
    
    if len(candidate_idx) == 0:
        if return_df:
            return pd.DataFrame()
        else:
            return HTML("<b>No papers contain your query words.</b>")
    
    #Rank candidates with SPECTER2
    q_emb = encoder.encode([query])
    sims_all = cosine_similarity(q_emb, embeddings)[0]
    sims = sims_all[candidate_idx]
    
    top_local = sims.argsort()[::-1][:k]
    top_idx = candidate_idx[top_local]
    base_cols = ["year", "title", "source_title", "citations", "keywords", "paper_link"]
    cols = [c for c in base_cols if c in metadata2.columns]

    result = metadata2.iloc[top_idx][cols].copy()
    result["similarity"] = sims[top_local].round(3)
    
    if return_df:
        return result.reset_index(drop=True)

#Clickable links
    def make_clickable(url):
        if isinstance(url, str) and url.strip():
            return f'<a href="{url}" target="_blank">Open</a>'
        return ""

    if "paper_link" in result.columns:
        result["paper_link"] = result["paper_link"].apply(make_clickable)

    return HTML(result.to_html(escape=False))


# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------
#umap_model.fit(X_norm)
#umap_vis.fit(X_norm)
#X_vis = umap_vis.fit_transform(X_norm)
st.header("Paperly")
tab1, tab2 = st.tabs(["Search","Idea Positioning"])
with tab1:
    st.write("Search for semantically similar papers using SPECTER2 embeddings.")

    query = st.text_input("🔍 Enter your query (keywords / question):", key="search_query")

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results", min_value=5, max_value=50, value=10, step=5)
    with col2:
        min_match = st.slider("Minimum matched words in text", min_value=1, max_value=5, value=1, step=1)
    if st.button("Search similar papers"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            df_result = scholar_search(query, k=top_k, min_match_words=min_match, return_df=True)

            if df_result.empty:
                st.info("No papers matched your query with the current filters.")
            else:
                df_show = df_result.copy()

                # ทำ HTML link จากคอลัมน์ paper_link
                def make_clickable(url):
                    if isinstance(url, str) and url.strip():
                        return f'<a href="{url}" target="_blank">Open</a>'
                    return ""

                if "paper_link" in df_show.columns:
                    df_show["Link"] = df_show["paper_link"].apply(make_clickable)
                else:
                    df_show["Link"] = ""

                cols_order = [
                    c for c in ["year", "title", "source_title", "citations", "keywords", "similarity", "Link"]
                    if c in df_show.columns
                ]
                df_show = df_show[cols_order]

                st.write(f"Found **{len(df_show)}** results.")
                st.markdown(
                    df_show.to_html(escape=False, index=False),
                    unsafe_allow_html=True
                )
with tab2:
    st.header("World Map Research")
    color_map = {
        0: "#F1C40F",  # yellow
        1: "#9B59B6",  # purple
        2: "#3498DB",  # blue
        3: "#2ECC71",  # green
        4: "#E67E22",  # orange
    }
    df_overall = pd.DataFrame({
            "x": X_vis[:, 0],
            "y": X_vis[:, 1],
            "cluster": labels,
            "title": metadata["title"],
            "year": metadata["year"],
    })
    fig1 = px.scatter(
        df_overall,
        x="x",
        y="y",
        color="cluster",
        color_discrete_map=color_map,
        opacity=0.3,
        hover_data=["title", "year", "cluster"]
    )
    fig1.update_layout(
            width=900,
            height=650,
            title="Global Research Map (UMAP 2D)",
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            legend_title="Cluster",
        )
    st.plotly_chart(fig1)
    st.title("🧠 Idea Positioning Engine")
    st.write("Enter an idea to discover its research cluster & novelty score.")

    idea_text = st.text_area("Your idea:", height=150)

    if st.button("Analyze Idea"):
        if len(idea_text.strip()) == 0:
            st.warning("Please enter an idea first!")
        else:
            result = position_idea(idea_text)

            st.subheader("📌 Result")
            st.write(f"**Cluster ID:** {result['Cluster id']}")
            st.write(f"**Cluster Label:** {result['Cluster Label']}")
            st.write(f"**Distance to Centroid:** {result['distance_to_centroid']:.4f}")
            st.write(f"**Novelty (Z-score):** {result['novelty_z']:.2f}")
            st.write(f"**Novelty Percentile:** {result['novelty_percentile']:.2f}%")

            # novelty indicator
            if result["novelty_percentile"] > 80:
                st.success("🔥 Highly Novel Idea (Top 20%)")
            elif result["novelty_percentile"] > 50:
                st.info("✨ Moderately Novel Idea")
            else:
                st.warning("🟡 Quite similar to existing studies in this cluster.")


    # --------------------------------------------------------------------
    # Optional: Visual Position on UMAP 2D
    # --------------------------------------------------------------------


    st.markdown("---")
    st.subheader("📍 Idea Position on Global Research Map")

    if idea_text.strip():
        # 1) Encode idea → normalize → project to 2D UMAP
        idea_emb = encoder.encode([idea_text])
        idea_norm = normalize(idea_emb)
        idea_vis = umap_vis.transform(idea_norm)  # shape (1, 2)
        idea_x, idea_y = idea_vis[0]

        st.write("Your idea (red X) is plotted on top of the global research map.")
        st.write(idea_x, idea_y)

        # 2) เตรียม DataFrame สำหรับ plot
        df_plot = pd.DataFrame({
            "x": X_vis[:, 0],
            "y": X_vis[:, 1],
            "cluster": labels,
            "title": metadata["title"],
            "year": metadata["year"],
        })

        # 3) สร้าง base scatter plot
        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            color="cluster",
            opacity=0.3,
            hover_data={
                "title": True,
                "year": True,
                "cluster": True,
                "x": False,
                "y": False,
            },
        )



        # 5) แต่ง layout
        fig.update_layout(
            width=900,
            height=650,
            title="Global Research Map (UMAP 2D)",
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            legend_title="Cluster"
        )

        # 4) เพิ่มจุดของไอเดีย (สีแดง)
        fig.add_scatter(
            x=[idea_x],
            y=[idea_y],
            mode="markers",
            marker=dict(
            color="red",
            size=40,                 
            symbol="x",
            line=dict(width=3, color="black"),  
        ),
            name="Your Idea",
            hovertext=["🔴 Your Idea"],
            hoverinfo="text",
        )

        # 6) แสดงบน Streamlit
        st.plotly_chart(fig, use_container_width=True)