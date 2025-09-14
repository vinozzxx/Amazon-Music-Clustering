# 🎧 Amazon Music Clustering Dashboard

**Unsupervised Learning for Music Discovery & Playlist Intelligence**

` Transform raw audio data into intelligent music clusters — and build AI-powered playlists that feel human.`

The Amazon Music Clustering Dashboard is an end-to-end unsupervised machine learning project designed to discover hidden patterns in music listening behavior using audio features like danceability, energy, valence, and tempo. Built for music analysts, data scientists, and streaming platform teams, this tool reveals how songs naturally group together — not by genre labels, but by how they sound.

Using K-Means, DBSCAN, and Hierarchical Clustering, we uncover 4–6 distinct musical “personas” across 95,000+ tracks — from chill acoustic ballads to high-energy dance anthems. The results are visualized through an interactive Streamlit dashboard, enabling users to explore clusters, generate smart playlists, search songs, and export insights — all without writing a single line of code.

This project demonstrates the full lifecycle of an unsupervised ML application:
✅ Exploratory Data Analysis (EDA)
✅ Feature engineering & normalization
✅ Multi-algorithm clustering
✅ Dimensionality reduction (PCA)
✅ Cluster interpretation & labeling
✅ Interactive deployment via Streamlit

Perfect for building personalized recommendation engines, optimizing radio stations, or understanding listener segmentation in music streaming services.

---
## 🎯 Goal
The primary goal is to automatically group similar songs into meaningful musical clusters based on their audio characteristics — and turn those clusters into actionable, user-friendly tools for music discovery and playlist generation.

This enables:

Music platforms to auto-generate mood-based playlists (e.g., “Chill Vibes,” “Workout Energy”)
Artists & labels to understand how their music fits within broader listener trends
Data teams to validate and refine genre classifications using objective audio metrics
Listeners to discover new music aligned with their sonic preferences — not just popularity

---

## 📊 Dataset Insight
The dataset contains 95,837 songs from Spotify/Amazon-style metadata, enriched with:

| **Feature Type**      | **Fields**                                                                                                              |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 🎵 **Audio Features** | `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo` |
| 📑 **Metadata**       | `name_song`, `name_artists`, `genres`, `popularity_songs`, `duration_ms`, `explicit`                                    |
| 🔖 **Cluster Labels** | `cluster (K-Means)`, `cluster_dbscan`, `cluster_hc (Hierarchical)`                                                      |
| 👨‍🎤 **Artist Info**    | `artist_popularity`, `follower_count`                                                                                   |

`💡 Key Insight: Songs are grouped not by human-assigned genres (which can be inconsistent), but by measurable sonic traits — revealing true musical DNA. `


**Example Cluster Interpretations:**

| **Cluster** | **Label**                  | **Description**                                              |
| ----------- | -------------------------- | ------------------------------------------------------------ |
| 0           | 🎵 Acoustic & Instrumental | High acousticness, low energy/danceability, moderate valence |
| 1           | 🔥 High Energy & Dance     | High danceability, energy, tempo; loud, low speechiness      |
| 2           | 🎤 Rap & Spoken Word       | High speechiness, low instrumentalness/acousticness          |
| 3           | 😊 Happy Pop & Upbeat      | High valence, danceability, energy; medium tempo             |

---

## 🛠 Tech Stack

| **Category**            | **Tools**           | **Purpose**                                                      |
| ----------------------- | ------------------- | ---------------------------------------------------------------- |
| **Core Language**       | Python              | Primary scripting and logic engine                               |
| **Data Manipulation**   | Pandas              | Loading, cleaning, filtering 95K+ rows                           |
| **Numerical Computing** | NumPy               | Mathematical operations on audio features                        |
| **Machine Learning**    | Scikit-learn        | K-Means, DBSCAN, Hierarchical Clustering, PCA, scaling           |
| **Model Persistence**   | Pickle              | Saving trained clustering pipelines                              |
| **Visualization**       | Plotly              | Interactive charts: radar plots, PCA scatter, heatmaps, boxplots |
| **Dashboard**           | Streamlit           | Zero-code web app for real-time exploration                      |
| **Optional Visuals**    | Matplotlib, Seaborn | Static EDA plots during development                              |


`All tools are open-source, Python-native, and optimized for data science workflows. `


## 🚀 Key Features

- **Multi-Algorithm Clustering** — Compare results from K-Means, DBSCAN, and Hierarchical Clustering side-by-side  
- **Interactive Cluster Exploration** — Filter by genre, artist, popularity, duration, explicit content, and audio feature ranges  
- **Radar Charts** — Visually compare cluster profiles across 9 audio dimensions  
- **PCA Visualization** — See high-dimensional clusters projected into 2D space  
- **Correlation Heatmap** — Understand relationships between audio features (e.g., energy ↔ danceability)  
- **Smart Playlist Generator** — Build custom playlists per cluster with adjustable size and similarity threshold  
- **AI-Powered Similarity Engine** — Find songs most similar to any track using cosine distance on normalized audio features  
- **Song & Artist Search** — Instantly locate songs or artists (case-insensitive, partial match support)  
- **Artist Analysis** — View which clusters an artist dominates, and their average audio profile  
- **Export Capabilities** — Download filtered data as CSV/JSON, save generated playlists, or export summary reports  
- **Real-Time Filtering** — Sliders and selectors update all visuals instantly — no page reload needed  
- **Mobile-Friendly UI** — Fully responsive design works on desktop, tablet, and phone  

---


## 🎛️ Dashboard Sections  

### 1. 📈 Dataset Overview  
- Total songs: **95,837**  
- Unique artists: **17,662**  
- Genres: **3,153 categories**  
- Active clusters: *Dynamic based on selected method*  

### 2. 🎯 Cluster Distribution  
- Pie & bar charts showing **% distribution per cluster**  
- Toggle between **K-Means, DBSCAN, and Hierarchical results**  
- Filter clusters by **song count or dominance**  

### 3. 🎵 Audio Feature Analysis  
- **Box Plots** → See feature spread within each cluster *(e.g., “Is tempo really higher in Dance clusters?”)*  
- **Radar Charts** → Compare cluster “sonic fingerprints” at a glance  
- **Correlation Matrix** → Heatmap of feature interdependencies  
- **PCA Scatter Plot** → 2D projection of clusters using principal components  

### 4. 🎶 Smart Playlist Generator  
- Select a cluster → Set **playlist size (1–50 songs)**  
- Adjust **similarity threshold (0.7–1.0)** for tighter/fuzzier matches  
- View **ranked songs with similarity scores**  
- One-click **download as CSV**  

### 5. 🔍 Search & Recommendations  
- **Search Bar** → Find songs/artists instantly *(e.g., “Taylor Swift”, “Blinding Lights”)*  
- **Similar Songs** → Click any song to see its **top 10 most similar neighbors**  
- **Artist Profile** → See which clusters an artist appears in + their average audio score  

### 6. 📥 Export & Reports  
- Export **filtered dataset** as CSV or JSON  
- Save **generated playlists** as `.csv`  
- Generate **one-click Summary Report** with stats: *mean, std, cluster sizes, dominant genres*  

---

## 🚀 Quick Start
Prerequisites
Python 3.8+
pip package manager
##  Getting Started  

Follow these steps to set up and run the dashboard locally:  

```bash
# Clone the repository
git clone https://github.com/yourusername/amazon-music-clustering-dashboard.git
cd amazon-music-clustering-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run enhanced_app.py
```
Launch
The app will open automatically at: (http://localhost:8501)[http://localhost:8502/]

`💡 Tip: The dataset (amazon_music_clusters_all_methods.csv) must be in the same folder as enhanced_app.py.` 

---

### 📁 Project Structure
```bash
📁 amazon-music-clustering-dashboard/
│
├── 📄 enhanced_app.py           # Main Streamlit dashboard (interactive UI)
├── 📄 app.py                    # Original basic version (for comparison)
├── 📄 AMC.ipynb                 # Jupyter notebook with full EDA + clustering pipeline
├── 📄 amazon_music_clusters_all_methods.csv  # Primary dataset (95K+ songs)
├── 📄 requirements.txt          # All Python dependencies
├── 📄 README.md                 # This file
│
├── 📂 models/                   # Saved preprocessing & clustering pipelines
│   ├── scaler.pkl               # StandardScaler for audio features
│   └── kmeans_model.pkl         # Trained K-Means model
│
├── 📂 cluster_plots/            # Optional static visualizations (for documentation)
│   ├── cluster_heatmap.png
│   ├── clusters_pca.png
│   └── clusters_tsne.png
│
└── 📂 assets/                   # Icons, logos, etc.
    └── logo.png
```
---

## 📊 Sample Insights (From Real Data)

- **High Speechiness ↔ Low Instrumentalness**  
  Confirms **Rap / Hip-Hop clusters** are driven by vocals rather than instruments.  

- **Valence (Happiness) ↔ Danceability**  
  Strong correlation shows **upbeat pop songs** tend to also be highly danceable.  

- **Loudness Variation Across Clusters**  
  Even within the same genre cluster, mastering levels differ — useful for **EQ optimization in playlists**.  

- **DBSCAN Identifies Outliers**  
  Unique tracks (e.g., experimental jazz, ambient noise) are separated, revealing **niche or non-mainstream clusters**.  

---

## 🧪 Model Evaluation Metrics****

| **Algorithm**    | **Silhouette Score** | **# Clusters** | **Key Takeaway**                                |
| ---------------- | -------------------- | -------------- | ----------------------------------------------- |
| **K-Means**      | 0.28                 | 4              | Optimal balance of cohesion & separation        |
| **DBSCAN**       | N/A                  | \~5 + noise    | Excellent at finding **rare genres** & outliers |
| **Hierarchical** | 0.26                 | 4              | Good interpretability via **dendrogram**        |

---

## 📄 License  

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this project, provided proper credit is given to the original author.  

---

##🙏 Acknowledgments
Built with ❤️ using industry-leading open-source tools:

- **Streamlit** – *“Bridges experimentation and production.” – Emmanuel Ameisen*  
  Lets data scientists focus on **insights**, not HTML.  

- **Plotly** – Interactive, publication-quality visualizations for rich storytelling.  

- **Scikit-learn** – Robust and scalable clustering algorithms (K-Means, DBSCAN, Hierarchical).  

- **Pandas** – Data wrangling powerhouse for handling 95K+ rows with ease.  

- **NumPy** – The numerical backbone of machine learning and preprocessing.  

- **Pickle** – Simple and effective **model persistence** for saving trained pipelines.  

`“Streamlit lets data scientists focus on insights — not HTML.”` 

---

## ❓ FAQ  

**Q1: Why clustering for music data?**  
Clustering helps group songs into meaningful genres when explicit labels are missing. It uncovers **hidden patterns** in music features.  

**Q2: Why K-Means, DBSCAN, and Hierarchical?**  
- **K-Means** → Balanced cohesion and separation.  
- **DBSCAN** → Great for finding **rare/less common genres** and handling noise.  
- **Hierarchical** → Offers **interpretability** with dendrograms.  

**Q3: How do I run the project?**  
1. Clone the repository.  
2. Install dependencies (`pip install -r requirements.txt`).  
3. Run `streamlit run app.py`.  

**Q4: Do I need GPU or high specs?**  
No 🚫. The dataset (~95K rows) can be processed on a normal laptop with 8GB RAM.  

**Q5: Can I use my own dataset?**  
Yes ✅. Replace the input CSV with your dataset — make sure it has numeric/audio feature columns.  

**Q6: Is the model perfect at genre detection?**  
Not 100%. Clustering is **unsupervised**, so it groups based on similarity, not fixed labels. It’s great for **exploration & insights**.  

**Q7: License?**  
MIT License — free to use and modify.  






















































