import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 🎵 Enhanced Amazon Music Clustering Dashboard
# ===========================

# Page configuration
st.set_page_config(
    page_title="🎧 Amazon Music Clustering Explorer",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Amazon Music Clustering Dashboard\nBuilt with Streamlit and Machine Learning"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .cluster-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# 📂 Data Loading with Caching
# ===========================

@st.cache_data
def load_data():
    """Load and preprocess the music clustering data"""
    try:
        df = pd.read_csv('amazon_music_clusters_all_methods.csv')
        
        # Get the actual columns available in the dataset
        available_numeric_cols = ['danceability', 'energy', 'valence', 'tempo']
        
        # Ensure numeric columns are properly typed
        for col in available_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add dummy columns for missing features to maintain compatibility
        missing_features = {
            'popularity_songs': 50,  # Default popularity score
            'duration_ms': 200000,   # Default 3:20 duration
            'loudness': -10.0,       # Default loudness
            'speechiness': 0.1,      # Default speechiness
            'acousticness': 0.5,     # Default acousticness
            'instrumentalness': 0.0, # Default instrumentalness
            'liveness': 0.2,         # Default liveness
            'popularity_artists': 40, # Default artist popularity
            'followers': 10000       # Default followers
        }
        
        for col, default_val in missing_features.items():
            if col not in df.columns:
                df[col] = default_val
        
        return df
    except FileNotFoundError:
        st.error("❌ File 'amazon_music_clusters_all_methods.csv' not found. Please ensure it's in the same folder as this script.")
        st.stop()

# Load data
df = load_data()

# ===========================
# 🧠 Enhanced Cluster Label Mapping
# ===========================

CLUSTER_LABELS_KMEANS = {
    0: "🎵 Acoustic & Instrumental",
    1: "🔥 High Energy & Dance",
    2: "🎤 Rap & Spoken Word",
    3: "😊 Happy Pop & Upbeat"
}

CLUSTER_LABELS_DBSCAN = {
    -1: "⚠️ Noise / Outlier",
    0: "🎵 Acoustic & Instrumental",
    1: "🔥 High Energy & Dance", 
    2: "🎤 Rap & Spoken Word"
}

CLUSTER_LABELS_HC = {
    0: "🎵 Acoustic & Instrumental",
    1: "🔥 High Energy & Dance",
    2: "🎤 Rap & Spoken Word",
    3: "😊 Happy Pop & Upbeat"
}

# Apply labels
df['cluster_kmeans_label'] = df['cluster'].map(CLUSTER_LABELS_KMEANS).fillna('Unknown')
df['cluster_dbscan_label'] = df['cluster_dbscan'].map(CLUSTER_LABELS_DBSCAN).fillna('Unknown')
df['cluster_hc_label'] = df['cluster_hc'].map(CLUSTER_LABELS_HC).fillna('Unknown')

# ===========================
# 🎨 Main Dashboard Header
# ===========================

st.markdown('<h1 class="main-header">🎧 Amazon Music Clustering Explorer</h1>', unsafe_allow_html=True)

# ===========================
# 🧭 Enhanced Sidebar
# ===========================

st.sidebar.markdown("## 🎛️ Control Panel")

# Clustering method selector
method = st.sidebar.selectbox(
    "🔬 Clustering Method",
    ["K-Means", "DBSCAN", "Hierarchical"],
    help="Choose the clustering algorithm to explore"
)

# Map selection to column and labels
if method == "K-Means":
    cluster_col = 'cluster'
    label_col = 'cluster_kmeans_label'
    labels_dict = CLUSTER_LABELS_KMEANS
elif method == "DBSCAN":
    cluster_col = 'cluster_dbscan'
    label_col = 'cluster_dbscan_label'
    labels_dict = CLUSTER_LABELS_DBSCAN
else:  # Hierarchical
    cluster_col = 'cluster_hc'
    label_col = 'cluster_hc_label'
    labels_dict = CLUSTER_LABELS_HC

# Advanced filters
st.sidebar.markdown("### 🎯 Advanced Filters")

# Genre filter
available_genres = df['genres'].dropna().unique()
selected_genres = st.sidebar.multiselect(
    "🎭 Select Genres",
    options=available_genres,
    default=available_genres[:5] if len(available_genres) > 5 else available_genres,
    help="Filter songs by genre"
)

# Audio feature filters - only for available features
st.sidebar.markdown("#### 🎵 Audio Features")
danceability_range = st.sidebar.slider(
    "Danceability", 0.0, 1.0, (0.0, 1.0), 0.01,
    help="How suitable a track is for dancing"
)
energy_range = st.sidebar.slider(
    "Energy", 0.0, 1.0, (0.0, 1.0), 0.01,
    help="Perceptual measure of intensity and power"
)
valence_range = st.sidebar.slider(
    "Valence", 0.0, 1.0, (0.0, 1.0), 0.01,
    help="Musical positivity conveyed by a track"
)
tempo_range = st.sidebar.slider(
    "Tempo (BPM)", 0.0, 250.0, (0.0, 250.0), 1.0,
    help="Overall estimated tempo in beats per minute"
)

# Note: Popularity and duration filters removed since these columns don't exist in the dataset

# ===========================
# 🔍 Data Filtering
# ===========================

# Apply filters
df_filtered = df.copy()

# Genre filter
if selected_genres:
    df_filtered = df_filtered[df_filtered['genres'].isin(selected_genres)]

# Audio feature filters - only for available features
df_filtered = df_filtered[
    (df_filtered['danceability'] >= danceability_range[0]) &
    (df_filtered['danceability'] <= danceability_range[1]) &
    (df_filtered['energy'] >= energy_range[0]) &
    (df_filtered['energy'] <= energy_range[1]) &
    (df_filtered['valence'] >= valence_range[0]) &
    (df_filtered['valence'] <= valence_range[1]) &
    (df_filtered['tempo'] >= tempo_range[0]) &
    (df_filtered['tempo'] <= tempo_range[1])
]

# ===========================
# 📊 Enhanced Summary Statistics
# ===========================

st.markdown("## 📈 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Songs", 
        f"{len(df):,}",
        delta=f"{len(df_filtered):,} filtered"
    )

with col2:
    st.metric(
        "Active Clusters", 
        df[cluster_col].nunique(),
        delta=f"{df_filtered[cluster_col].nunique()} after filtering"
    )

with col3:
    avg_energy = df_filtered['energy'].mean()
    st.metric(
        "Avg Energy", 
        f"{avg_energy:.2f}",
        delta=f"{avg_energy - df['energy'].mean():.2f} vs total"
    )

with col4:
    unique_artists = df_filtered['name_artists'].nunique()
    st.metric(
        "Unique Artists", 
        f"{unique_artists:,}",
        delta=f"{unique_artists - df['name_artists'].nunique():,} vs total"
    )

# ===========================
# 📊 Interactive Cluster Distribution
# ===========================

st.markdown("## 🎯 Cluster Distribution Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    # Interactive bar chart
    cluster_counts = df_filtered[label_col].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    fig = px.bar(
        cluster_counts, 
        x='Cluster', 
        y='Count',
        color='Count',
        color_continuous_scale='Viridis',
        title=f"Distribution of Songs by {method} Clusters",
        text='Count'
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Cluster statistics
    st.markdown("### 📊 Cluster Statistics")
    
    for cluster_id, label in labels_dict.items():
        if cluster_id in df_filtered[cluster_col].values:
            cluster_data = df_filtered[df_filtered[cluster_col] == cluster_id]
            count = len(cluster_data)
            percentage = (count / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
            
            avg_energy = cluster_data['energy'].mean()
            st.markdown(f"""
            <div class="cluster-card">
                <h4>{label}</h4>
                <p><strong>Songs:</strong> {count:,}</p>
                <p><strong>Percentage:</strong> {percentage:.1f}%</p>
                <p><strong>Avg Energy:</strong> {avg_energy:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

# ===========================
# 🎵 Interactive Audio Feature Analysis
# ===========================

st.markdown("## 🎵 Audio Feature Analysis")

# Feature selection for analysis - only use available features
audio_features = ['danceability', 'energy', 'valence', 'tempo']

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["📊 Feature Distribution", "🎯 Cluster Comparison", "🔍 Correlation Matrix", "📈 PCA Visualization"])

with tab1:
    # Feature distribution by cluster
    selected_features = st.multiselect(
        "Select features to analyze",
        audio_features,
        default=['danceability', 'energy', 'valence', 'tempo']
    )
    
    if selected_features:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=selected_features[:4],
            specs=[[{"type": "box"}, {"type": "box"}],
                   [{"type": "box"}, {"type": "box"}]]
        )
        
        for i, feature in enumerate(selected_features[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            for cluster_id in sorted(df_filtered[cluster_col].dropna().unique()):
                cluster_data = df_filtered[df_filtered[cluster_col] == cluster_id]
                fig.add_trace(
                    go.Box(
                        y=cluster_data[feature],
                        name=f"{labels_dict.get(cluster_id, f'Cluster {cluster_id}')}",
                        boxpoints='outliers'
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Radar chart for cluster comparison
    if len(df_filtered) > 0:
        cluster_means = df_filtered.groupby(cluster_col)[audio_features].mean()
        
        fig = go.Figure()
        
        for cluster_id in cluster_means.index:
            values = cluster_means.loc[cluster_id].values.tolist()
            values += values[:1]  # Complete the circle
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=audio_features + [audio_features[0]],
                fill='toself',
                name=labels_dict.get(cluster_id, f'Cluster {cluster_id}')
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Cluster Feature Comparison (Radar Chart)"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Correlation heatmap
    correlation_data = df_filtered[audio_features].corr()
    
    fig = px.imshow(
        correlation_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Audio Features Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # PCA visualization
    if len(df_filtered) > 0:
        # Prepare data for PCA
        pca_features = df_filtered[audio_features].dropna()
        
        if len(pca_features) > 0:
            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(StandardScaler().fit_transform(pca_features))
            
            # Create PCA dataframe
            pca_df = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'Cluster': df_filtered.loc[pca_features.index, cluster_col].values,
                'Cluster_Label': df_filtered.loc[pca_features.index, label_col].values
            })
            
            # Plot PCA
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster_Label',
                title=f"PCA Visualization of {method} Clusters",
                hover_data=['Cluster']
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)


# ===========================
# 🔍 Advanced Search & Recommendations
# ===========================

st.markdown("## 🔍 Advanced Search & Recommendations")

search_tab1, search_tab2, search_tab3 = st.tabs(["🎵 Song Search", "🎤 Artist Analysis", "🎯 Similar Songs"])

with search_tab1:
    # Song search
    search_query = st.text_input("🔍 Search for songs", placeholder="Enter song name or artist...")
    
    if search_query:
        # Search in song names and artists
        mask = (df_filtered['name_song'].str.contains(search_query, case=False, na=False) |
                df_filtered['name_artists'].str.contains(search_query, case=False, na=False))
        
        search_results = df_filtered[mask]
        
        if len(search_results) > 0:
            st.markdown(f"### Found {len(search_results)} songs")
            
            # Display results
            display_cols = ['name_song', 'name_artists', 'genres', 
                           'danceability', 'energy', 'valence', 'tempo', label_col]
            
            st.dataframe(
                search_results[display_cols].head(20),
                use_container_width=True
            )
        else:
            st.info("No songs found matching your search")

with search_tab2:
    # Artist analysis
    artist_query = st.text_input("🎤 Search for artist", placeholder="Enter artist name...")
    
    if artist_query:
        artist_mask = df_filtered['name_artists'].str.contains(artist_query, case=False, na=False)
        artist_songs = df_filtered[artist_mask]
        
        if len(artist_songs) > 0:
            # Artist statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Songs", len(artist_songs))
            with col2:
                st.metric("Avg Energy", f"{artist_songs['energy'].mean():.2f}")
            with col3:
                st.metric("Top Genre", artist_songs['genres'].mode().iloc[0] if len(artist_songs) > 0 else "N/A")
            
            # Cluster distribution for artist
            cluster_dist = artist_songs[label_col].value_counts()
            
            fig = px.pie(
                values=cluster_dist.values,
                names=cluster_dist.index,
                title=f"Cluster Distribution for {artist_query}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top songs by energy
            st.markdown("### 🏆 Top Songs by Energy")
            top_songs = artist_songs.nlargest(10, 'energy')
            st.dataframe(
                top_songs[['name_song', 'energy', 'danceability', 'valence', 'tempo', label_col]],
                use_container_width=True
            )

with search_tab3:
    # Similar songs recommendation
    st.markdown("### 🎯 Find Similar Songs")
    
    # Select a reference song
    reference_song = st.selectbox(
        "Choose a reference song",
        options=df_filtered['name_song'].unique(),
        help="Select a song to find similar ones"
    )
    
    if reference_song:
        # Get reference song data
        ref_song_data = df_filtered[df_filtered['name_song'] == reference_song].iloc[0]
        
        # Calculate similarities using available features
        features = ['danceability', 'energy', 'valence', 'tempo']
        ref_features = ref_song_data[features].values.reshape(1, -1)
        
        # Calculate similarities with all other songs
        all_features = df_filtered[features].values
        similarities = cosine_similarity(ref_features, all_features)[0]
        
        # Get top similar songs (excluding the reference song itself)
        similar_indices = np.argsort(-similarities)[1:11]  # Top 10 excluding self
        similar_songs = df_filtered.iloc[similar_indices].copy()
        similar_songs['Similarity'] = similarities[similar_indices]
        
        st.markdown(f"### 🎵 Songs similar to '{reference_song}' by {ref_song_data['name_artists']}")
        
        # Display similar songs
        display_cols = ['name_song', 'name_artists', 'genres', 'danceability', 'energy', 'valence', 'tempo', 'Similarity']
        st.dataframe(
            similar_songs[display_cols].style.format({
                'Similarity': '{:.3f}',
                'danceability': '{:.2f}',
                'energy': '{:.2f}',
                'valence': '{:.2f}',
                'tempo': '{:.1f}'
            }).background_gradient(subset=['Similarity'], cmap='YlOrRd'),
            use_container_width=True
        )

# ===========================
# 📊 Export & Download Options
# ===========================

st.markdown("## 📥 Export & Download")

col1, col2, col3 = st.columns(3)

with col1:
    # CSV Export
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📊 Download Filtered Data (CSV)",
        csv_data,
        "filtered_music_data.csv",
        "text/csv",
        help="Download the filtered dataset as CSV"
    )

with col2:
    # JSON Export
    json_data = df_filtered.to_json(orient='records', indent=2).encode('utf-8')
    st.download_button(
        "📋 Download as JSON",
        json_data,
        "filtered_music_data.json",
        "application/json",
        help="Download the filtered dataset as JSON"
    )

with col3:
    # Summary Report
    if st.button("📈 Generate Summary Report"):
        # Create summary statistics
        summary = {
            "Total Songs": len(df_filtered),
            "Clusters": df_filtered[cluster_col].nunique(),
            "Avg Popularity": df_filtered['popularity_songs'].mean(),
            "Top Genre": df_filtered['genres'].mode().iloc[0] if len(df_filtered) > 0 else "N/A",
            "Date Generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.json(summary)

# ===========================
# 🎉 Footer & About
# ===========================

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 💡 About This Dashboard
    
    This enhanced Amazon Music Clustering Dashboard provides:
    - **Multiple Clustering Methods**: K-Means, DBSCAN, and Hierarchical clustering
    - **Interactive Visualizations**: Powered by Plotly for better exploration
    - **Smart Recommendations**: Find similar songs and generate playlists
    - **Advanced Filtering**: Filter by audio features, genres, and popularity
    - **Real-time Analysis**: Dynamic updates based on your selections
    
    Built with ❤️ using Streamlit, Plotly, and Scikit-learn.
    """)

with col2:
    st.markdown("""
    ### 🚀 Quick Stats
    
    - **Dataset Size**: 95,837 songs
    - **Features**: 10 audio characteristics
    - **Clusters**: 3-4 per method
    - **Artists**: 17,662 unique
    - **Genres**: 3,153 categories
    """)

# ===========================
# 🎵 Footer
# ===========================

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; padding: 20px;">'
    '🎵 Amazon Music Clustering Dashboard — Powered by Streamlit & Machine Learning 🎵'
    '</div>',
    unsafe_allow_html=True
)
