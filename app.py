import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# Modules Imports
from modules.data_loader import load_and_inspect_data
from modules.preprocessing import clean_data
from modules.outlier_detection import detect_zscore_outliers, detect_iqr_outliers
from modules.knn import get_knn_recommendations, predict_job_level
from modules.kmeans import run_kmeans  
from modules.decision_tree import build_tree, predict
from modules.decision_tree_features import extract_features
from modules.graph_search import bfs, dfs, a_star, CAREER_GRAPH, calculate_path_cost
from modules.live_api import fetch_jsearch_jobs, map_live_to_local_format # Naya Import

@st.cache_data(show_spinner="Analyzing job skills...")
def calculate_skill_demand(descriptions):
    skills = [
        'Python', 'SQL', 'Excel',
        'Pandas', 'Machine Learning', 'Statistics'
    ]

    descriptions = descriptions.fillna('').astype(str)

    counts = [
        int(descriptions.str.contains(
            rf'\b{skill}\b',
            case=False,
            regex=True,
            na=False
        ).sum())
        for skill in skills
    ]

    total = len(descriptions)

    demands = [
        round(count / total * 100, 1) if total else 0
        for count in counts
    ]

    top_skill = (
        skills[counts.index(max(counts))]
        if any(counts) else "N/A"
    )

    return skills, demands, top_skill

# Page configuration
st.set_page_config(page_title="CareerLens", page_icon="📊", layout="wide")

# --- CUSTOM CSS FOR UI (FORCE LIGHT MODE) ---
st.markdown("""
<style>
    /* Global Background & Fonts */
    .stApp {
        background-color: #f8f9fa;
    }
    html, body, [class*="css"] {
        color: #111827 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #002b22; /* Dark Green */
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] h1 {
        color: #00a884 !important;
        font-size: 24px;
    }
    
    /* White Cards */
    .custom-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
    }
    .custom-card h4 {
        margin: 0;
        font-size: 14px;
        color: #4b5563 !important;
        font-weight: 600;
    }
    .custom-card h2 {
        margin: 5px 0 0 0;
        font-size: 28px;
        color: #00796b !important;
        font-weight: 700;
    }
    
    /* Chart/Table Boxes */
    .chart-box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    .chart-box h4 {
        color: #374151 !important;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Improve widget label and helper-text contrast */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] label p,
    [data-testid="stCaptionContainer"] p {
        color: #374151 !important;
        opacity: 1 !important;
    }

    [data-testid="stWidgetLabel"] {
        color: #374151 !important;
        opacity: 1 !important;
    }

    /* FIX STREAMLIT WIDGETS (INPUTS, DROPDOWNS, BUTTONS) */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: white !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
    }
    input {
        color: #111827 !important;
    }
    .stButton>button {
        background-color: #00796b !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #004d40 !important;
    }
    
    /* Table Styling */
    [data-testid="stDataFrame"] th {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar
with st.sidebar:
    st.title("CareerLens")
    st.markdown("<p style='color: #a0a0a0; margin-top:-15px; font-size:14px;'>Job Market Intelligence</p>", unsafe_allow_html=True)
    
    # Naya "Live Job Feed" option add kiya gaya hai
    menu = st.radio("Navigation", 
        ["Market Overview", "Salary & Outliers", "KNN Recommendations", "K-Means Clustering", "Decision Tree", "Career Path Finder", "Live Job Feed"],
        label_visibility="collapsed")
    
    st.markdown("---")
    if st.button("Load Dataset"):
        with st.spinner("Loading..."):
            df = load_and_inspect_data("data/job_postings.csv")
            if isinstance(df, str): 
                st.error(df)
            else:
                st.session_state.data = clean_data(df)
                st.success("Data Loaded!")

st.title(menu)

if st.session_state.data is None and menu != "Live Job Feed":
    st.info("👈 Please click 'Load Dataset' in the sidebar to start.")
else:
    # Agar Live Job Feed select nahi hai tabhi data load karein
    if menu != "Live Job Feed":
        df = st.session_state.data

    # ==========================================
    # 1. MARKET OVERVIEW
    # ==========================================
    if menu == "Market Overview":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Dataset summary, top skills and job titles</p><br>", unsafe_allow_html=True)
        
                # Calculate skill demand using the full dataset
        skills, demands, top_skill = calculate_skill_demand(
            df['description']
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f'<div class="custom-card"><h4>Jobs analysed</h4><h2>{df.shape[0]:,}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(
                f'<div class="custom-card"><h4>Top tracked skill</h4>'
                f'<h2>{top_skill}</h2></div>',
                unsafe_allow_html=True
            )
        with col3: st.markdown(f'<div class="custom-card"><h4>Median salary</h4><h2>${df["salary"].median():,.0f}</h2></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="custom-card"><h4>Job categories</h4><h2>{df["job_title"].nunique()}</h2></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns([1.5, 1])
        with col_chart1:
            st.markdown('<div class="chart-box"><h4>Skill demand (% of postings)</h4>', unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            fig1.patch.set_facecolor('white')
            ax1.barh(skills[::-1], demands[::-1], color='#00796b', height=0.6)
            ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False); ax1.spines['left'].set_visible(False); ax1.spines['bottom'].set_visible(False)
            for i, v in enumerate(demands[::-1]): ax1.text(v + 2, i, f"{v}%", va='center', color='#374151', fontsize=10)
            ax1.set_xticks([]); ax1.tick_params(axis='y', colors='#374151', labelsize=11)
            st.pyplot(fig1)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_chart2:
            st.markdown('<div class="chart-box"><h4>Top job titles</h4>', unsafe_allow_html=True)
            top_titles = df['job_title'].value_counts().head(5).reset_index()
            top_titles.columns = ['Title', 'Postings']
            st.dataframe(top_titles, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 2. SALARY & OUTLIERS
    # ==========================================
    elif menu == "Salary & Outliers":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Z-score and IQR outlier detection side by side</p><br>", unsafe_allow_html=True)
        
        col_input1, col_input2, col_input3, col_input4 = st.columns(4)
        with col_input1: st.text_input("Salary field", "salary", disabled=True)
        with col_input2: st.text_input("Method", "Z-score + IQR", disabled=True)
        with col_input3: z_thresh = st.number_input(
    "Z threshold", min_value=0.5, value=3.0, step=0.5
)
        with col_input4: st.text_input("Currency / unit", "USD / year", disabled=True)
        
        if st.button("Run Analysis"):
            outliers_z, count_z = detect_zscore_outliers(df, threshold=z_thresh)
            outliers_iqr, stats_iqr = detect_iqr_outliers(df)
            col_metric1, col_metric2, col_metric3 = st.columns(3)

            with col_metric1:
                st.metric("Valid Salary Records", df["salary"].notna().sum())

            with col_metric2:
                st.metric("Z-score Outliers", count_z)

            with col_metric3:
                st.metric("IQR Outliers", len(outliers_iqr))

            # Salary distribution and IQR outlier boundaries
            valid_salaries = pd.to_numeric(
                df["salary"], errors="coerce"
            )
            valid_salaries = valid_salaries[
                np.isfinite(valid_salaries) & (valid_salaries > 0)
            ]

            if not valid_salaries.empty and stats_iqr:
                st.subheader("Salary Distribution & IQR Boundaries")

                fig, ax = plt.subplots(figsize=(10, 4))
                # Limit the chart range to improve readability.
                # Outlier calculations still use all valid salaries.
                display_limit = valid_salaries.quantile(0.99)
                display_salaries = valid_salaries[
                    valid_salaries <= display_limit
                ]

                ax.hist(
                    display_salaries,
                    bins=50,
                    color="#00796b",
                    alpha=0.75,
                    edgecolor="white"
                )
                ax.set_xlim(0, display_limit)

                ax.axvline(
                    stats_iqr["Lower Fence"],
                    color="#d97706",
                    linestyle="--",
                    label="IQR Lower Fence"
                )
                ax.axvline(
                    stats_iqr["Upper Fence"],
                    color="#be123c",
                    linestyle="--",
                    label="IQR Upper Fence"
                )

                ax.set_xlabel("Annual Salary (USD)")
                ax.set_ylabel("Number of Job Postings")
                ax.legend()
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                st.pyplot(fig)
                plt.close(fig)



    # ==========================================
    # 3. KNN RECOMMENDATIONS
    # ==========================================
    elif menu == "KNN Recommendations":
        st.caption("Find job postings with salaries closest to your target.")

        col1, col2 = st.columns(2)

        with col1:
            target_salary = st.number_input(
                "Target Annual Salary (USD)",
                min_value=1,
                value=80000,
                step=5000
            )

        with col2:
            k_neighbors = st.number_input(
                "Number of Neighbors (K)",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )

        if st.button("Find Recommendations"):
            nearest_jobs = get_knn_recommendations(
                df, target_salary, k=int(k_neighbors)
            )

            if nearest_jobs.empty:
                st.warning("No matching jobs with valid salaries were found.")
            else:
                predicted_level = predict_job_level(nearest_jobs)

                st.success(
                    f"Found {len(nearest_jobs)} nearest job postings."
                )

                if predicted_level is not None:
                    st.metric("Predicted Job Level", predicted_level)
                else:
                    st.info("Job level unavailable for these recommendations.")

                display_columns = [
                    col for col in
                    ["job_title", "company", "salary", "job_level", "distance"]
                    if col in nearest_jobs.columns
                ]

                st.dataframe(
                    nearest_jobs[display_columns],
                    hide_index=True,
                    use_container_width=True
                )

    # ==========================================
    # 4. K-MEANS CLUSTERING
    # ==========================================
    elif menu == "K-Means Clustering":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Group postings by selected numeric features</p><br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.text_input("Feature X", "salary", disabled=True)
        with col2: st.text_input("Feature Y", "job_index", disabled=True)
        with col3: k_clusters = st.number_input("K (clusters)", min_value=2, max_value=8, value=3)
        with col4: st.text_input("Max iterations", "100", disabled=True)
        
        apply_iqr = st.checkbox(
            "Exclude salary outliers using IQR",
            value=True
        )

        if st.button("Run K-Means"):
            clustering_df = df.copy()

            if apply_iqr:
                salaries = pd.to_numeric(
                    clustering_df["salary"], errors="coerce"
                )
                valid = salaries[
                    np.isfinite(salaries) & (salaries > 0)
                ]

                if not valid.empty:
                    q1 = valid.quantile(0.25)
                    q3 = valid.quantile(0.75)
                    iqr = q3 - q1

                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr

                    clustering_df = clustering_df[
                        (salaries >= lower) & (salaries <= upper)
                    ].copy()

            clustered_df, centroids = run_kmeans(
                clustering_df, column="salary", k=k_clusters
            )

            st.caption(
                f"Clustered {len(clustered_df):,} postings · "
                f"IQR filtering {'enabled' if apply_iqr else 'disabled'}"
            )
            
            col_chart1, col_chart2 = st.columns([1.5, 1])
            with col_chart1:
                st.subheader("Cluster plot")
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('white')
                colors = ['#00796b', '#d97706', '#4f46e5', '#be123c', '#10b981']
                
                for i in range(k_clusters):
                    cluster_data = clustered_df[clustered_df['cluster'] == i]
                    ax.scatter(cluster_data.index, cluster_data['salary'], c=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6)
                
                for i, centroid in enumerate(centroids):
                    ax.scatter(len(clustered_df) + 100, centroid[0], c=colors[i % len(colors)], marker='X', s=200, edgecolors='black')
                
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.set_xlabel("Job Index", color='#374151')
                ax.set_ylabel("Annual Salary (USD)", color='#374151')
                ax.legend()
                st.pyplot(fig)
                
            with col_chart2:
                st.subheader("Centroids and sizes")
                summary = clustered_df.groupby('cluster')['salary'].agg(['count', 'mean']).reset_index()
                summary.columns = ['Cluster', 'Size', 'Avg Salary']
                summary['Avg Salary'] = summary['Avg Salary'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(summary, hide_index=True, use_container_width=True)

    # ==========================================
    # ==========================================
    # 5. DECISION TREE
    # ==========================================
    elif menu == "Decision Tree":
        st.caption("Predict job seniority using salary and job-title features")

        col1, col2 = st.columns(2)

        with col1:
            job_title = st.text_input(
                "Job title",
                value="Senior Python Developer"
            )

        with col2:
            test_salary = st.number_input(
                "Annual salary (USD)",
                min_value=1,
                max_value=1000000,
                value=95000,
                step=5000
            )

        depth = st.slider("Maximum tree depth", 1, 10, 3)

        if st.button("Train & Predict"):
            training_df = df.copy()

            training_df["salary"] = pd.to_numeric(
                training_df["salary"], errors="coerce"
            )

            training_df = training_df[
                training_df["job_level"].notna()
                & np.isfinite(training_df["salary"])
                & (training_df["salary"] > 0)
            ].copy()

            if len(training_df) < 2:
                st.error("Not enough valid training records.")
            else:
                q1 = training_df["salary"].quantile(0.25)
                q3 = training_df["salary"].quantile(0.75)
                iqr = q3 - q1

                training_df = training_df[
                    training_df["salary"].between(
                        q1 - 1.5 * iqr,
                        q3 + 1.5 * iqr
                    )
                ].copy()

                features = extract_features(training_df)
                X = features.to_numpy()
                y = training_df["job_level"].to_numpy()

                with st.spinner("Training Decision Tree..."):
                    tree = build_tree(X, y, max_depth=depth)

                sample = pd.DataFrame({
                    "job_title": [job_title],
                    "salary": [test_salary]
                })

                sample_features = extract_features(sample)
                sample_features = sample_features[features.columns]
                sample_values = sample_features.iloc[0].to_numpy()

                prediction = predict(tree, sample_values)

                st.success(f"Predicted job level: {prediction}")
                st.caption(
                    f"Trained on {len(training_df):,} records "
                    f"using {X.shape[1]} features"
                )

                col_tree, col_pred = st.columns([2, 1])

                with col_tree:
                    st.subheader("Decision path")

                    current_node = tree

                    while current_node.value is None:
                        feature_idx = current_node.feature
                        feature_name = features.columns[feature_idx]
                        threshold = current_node.threshold
                        value = sample_values[feature_idx]

                        if value <= threshold:
                            st.write(
                                f"{feature_name}: {value:,.0f} "
                                f"≤ {threshold:,.0f} → Left"
                            )
                            current_node = current_node.left
                        else:
                            st.write(
                                f"{feature_name}: {value:,.0f} "
                                f"> {threshold:,.0f} → Right"
                            )
                            current_node = current_node.right

                    st.write(f"Leaf prediction: **{current_node.value}**")

                with col_pred:
                    st.subheader("Prediction")
                    st.metric("Job level", prediction)

    # ==========================================
    # 6. CAREER PATH FINDER
    # ==========================================
    elif menu == "Career Path Finder":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Compare BFS, DFS and A* on the same start and goal</p><br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        all_nodes = list(CAREER_GRAPH.keys())
        with col1: start_node = st.selectbox("Current skills", all_nodes, index=0)
        with col2: goal_node = st.selectbox("Target role", all_nodes, index=len(all_nodes)-1)
        with col3: st.text_input("Algorithm", "Compare BFS, DFS, A*", disabled=True)
        with col4: st.text_input("Cost unit", "Learning hours", disabled=True)
        
        if st.button("Find Path"):
            st.markdown('<div class="chart-box"><h4>Career graph (highlighted: A* route)</h4>', unsafe_allow_html=True)
            
            G = nx.DiGraph()
            for node, edges in CAREER_GRAPH.items():
                for neighbor, weight in edges.items():
                    G.add_edge(node, neighbor, weight=weight)
                    
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('white')
            
            pos = nx.spring_layout(G, k=1.5, seed=42)
            
            nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='#00796b', ax=ax)
            node_labels = {
                "Machine Learning": "Machine\nLearning",
                "Data Scientist": "Data\nScientist",
                "Visualization": "Visualization",
            }

            labels = {node: node_labels.get(node, node) for node in G.nodes()}

            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_color='white',
                font_size=8,
                font_weight='bold',
                ax=ax
            )
            a_path, a_cost = a_star(CAREER_GRAPH, start_node, goal_node)

            highlighted_edges = set(zip(a_path, a_path[1:])) if a_path else set()

            edge_colors = [
                '#ef4444' if edge in highlighted_edges else '#9ca3af'
                for edge in G.edges()
            ]

            edge_widths = [
                4 if edge in highlighted_edges else 2
                for edge in G.edges()
            ]

            nx.draw_networkx_edges(
                G, pos,
                arrowstyle='->',
                arrowsize=20,
                edge_color=edge_colors,
                width=edge_widths,
                ax=ax
            )
            
            ax.axis('off')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-box"><h4>Algorithm comparison</h4>', unsafe_allow_html=True)
            bfs_path, bfs_cost = bfs(CAREER_GRAPH, start_node, goal_node)
            dfs_path, dfs_cost = dfs(CAREER_GRAPH, start_node, goal_node)
            a_path, a_cost = a_star(CAREER_GRAPH, start_node, goal_node)
            
            algo_df = pd.DataFrame({
                "Algorithm": ["BFS", "DFS", "A*"],
                "Path": [" → ".join(bfs_path) if bfs_path else "None", " → ".join(dfs_path) if dfs_path else "None", " → ".join(a_path) if a_path else "None"],
                "Edges": [len(bfs_path)-1 if bfs_path else 0, len(dfs_path)-1 if dfs_path else 0, len(a_path)-1 if a_path else 0],
                "Cost": [f"{calculate_path_cost(CAREER_GRAPH, path)*40} h" if path else "N/A" for path in (bfs_path, dfs_path, a_path)]
            })
            st.dataframe(algo_df, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 7. LIVE JOB FEED (Naya Page)
    # ==========================================
    elif menu == "Live Job Feed":
        st.header("🔴 Live Job Market Feed")
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Fetch real-time jobs directly from LinkedIn/Indeed using JSearch API.</p><br>", unsafe_allow_html=True)
        
        # API Inputs
        col1, col2, col3 = st.columns(3)
        with col1: 
            # Security ke liye type="password" use kiya hai. Apni key yahan daalein.
            rapidapi_key = st.text_input("RapidAPI Key", type="password")
        with col2: 
            query = st.text_input("Job Title", "Data Scientist")
        with col3: 
            location = st.text_input("Location", "US")
            
        if st.button("Fetch Live Jobs"):
            if rapidapi_key:
                with st.spinner("Fetching live jobs from API... (Thoda time lag sakta hai)"):
                    raw_df = fetch_jsearch_jobs(rapidapi_key, query, location)
                    
                    if raw_df is not None:
                        st.success(f"Successfully fetched {len(raw_df)} live jobs!")
                        
                        # 1. Raw API Data
                        st.markdown('<div class="chart-box"><h4>Raw API Data (Sample)</h4>', unsafe_allow_html=True)
                        st.dataframe(raw_df[['job_title', 'employer_name', 'job_city', 'job_min_salary', 'job_max_salary']].head(10), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 2. Data Cleaning
                        mapped_df = map_live_to_local_format(raw_df)
                        
                        if not mapped_df.empty:
                            st.markdown('<div class="chart-box"><h4>Cleaned Data (Ready for ML Models)</h4>', unsafe_allow_html=True)
                            st.dataframe(mapped_df.head(10), use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download Button
                            csv = mapped_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Live Data as CSV",
                                data=csv,
                                file_name=f"live_{query.replace(' ', '_')}.csv",
                                mime="text/csv",
                            )
                            
                            # Note for User
                            st.info("💡 Note: Aap is live data ko CSV mein download karke apne 'data' folder mein daal sakte hain taake KNN aur Decision Tree is par kaam kar sakein.")
                        else:
                            st.warning("Live jobs mili, lekin unmein salary ki information nahi thi (ya data clean nahi ho paya).")
                    else:
                        st.error("Data fetch nahi ho paya. Please check your API key limit or query.")
            else:
                st.warning("Please enter your RapidAPI Key.")
