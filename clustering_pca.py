import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

data = pd.read_csv('Spotify-2000.csv')

data_names = data[["Title", "Artist"]]

# dropping non-numeric columns and convert remaining columns to numeric
data = data.drop(columns=["Index", "Title", "Artist", "Top Genre"])

for col in data.columns:
    data[col] = data[col].replace(',', '', regex=True).astype(float)

data = data.dropna()

# relevant features for clustering
data2 = data[["Beats Per Minute (BPM)", "Loudness (dB)", 
              "Liveness", "Valence", "Acousticness", 
              "Speechiness"]]

# standardizing
scaler = StandardScaler()
data2_scaled = scaler.fit_transform(data2)

# PCA APPLICATION
pca = PCA()
data_pca = pca.fit_transform(data2_scaled)

# explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# plotting cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCs')
plt.grid()
plt.show()

# computer optimal number of components
threshold = 0.9  # 90% explained variance threshold
optimal_components = np.argmax(cumulative_explained_variance >= threshold)+1
print(f"Optimal number of components: {optimal_components}")

# applying PCA again with the optimal number of components
pca_optimal = PCA(n_components=optimal_components)
data_pca_optimal = pca_optimal.fit_transform(data2_scaled)

# Elbow Method
inertia = []
silhouette_scores = []
K = range(2, 15)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_pca_optimal)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_pca_optimal, kmeans.labels_))

# plot the Elbow Test
plt.figure(figsize=(10, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid()
plt.show()

# plot the Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k')
plt.grid()
plt.show()

# KMeans clustering with the selected optimal number of clusters
optimal_clusters = 4  # change based on assessment of plots
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
clusters = kmeans.fit_predict(data_pca_optimal)

# visualization

data["Music Segments"] = clusters + 1 

data["Music Segments"] = data["Music Segments"].map({
    1: "Cluster 1: High BPM, Low Acousticness",
    2: "Cluster 2: Moderate Energy, High Valence",
    3: "Cluster 3: Low Danceability, High Loudness",
    4: "Cluster 4: High Speechiness, Moderate Energy",
})

data["Title"] = data_names["Title"]
data["Artist"] = data_names["Artist"]

# plotting 3D scatter plot using PCA components
PLOT = go.Figure()
for i in list(data["Music Segments"].unique()):
    subset = data[data["Music Segments"] == i]
    PLOT.add_trace(go.Scatter3d(
        x=data_pca_optimal[data["Music Segments"] == i][:, 0],
        y=data_pca_optimal[data["Music Segments"] == i][:, 1],
        z=data_pca_optimal[data["Music Segments"] == i][:, 2],
        mode='markers',
        marker=dict(size=6, line=dict(width=1)),
        text=subset.apply(lambda row: f"Title: {row['Title']}<br>Artist: {row['Artist']}", axis=1),
        name=str(i)
    ))
PLOT.update_traces(hovertemplate='Title: %{text}<br>PCA1: %{x} <br>PCA2: %{y} <br>PCA3: %{z}')

PLOT.update_layout(
    width=1100, height=1100, autosize=True, showlegend=True,
    scene=dict(
        xaxis=dict(title='PCA1', titlefont_color='black'),
        yaxis=dict(title='PCA2', titlefont_color='black'),
        zaxis=dict(title='PCA3', titlefont_color='black')
    ),
    font=dict(family="Gilroy", color='black', size=12)
)

PLOT.show()
