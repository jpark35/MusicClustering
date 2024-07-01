import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

# loading the dataset
data = pd.read_csv('Spotify-2000.csv')

# preview data
print(data.head())

data_names = data[["Title", "Artist"]]

# dropping non-numeric columns
data = data.drop(columns=["Index", "Title", "Artist", "Top Genre"])

for col in data.columns:
    data[col] = data[col].replace(',', '', regex=True).astype(float)

# relevant features for clustering
data2 = data[["Beats Per Minute (BPM)", "Loudness (dB)", 
              "Liveness", "Valence", "Acousticness", 
              "Speechiness"]]

# normalize the data
scaler = MinMaxScaler()
data2_scaled = scaler.fit_transform(data2)

# DECISION MAKING: EVALUATE THE NUMBER OF CLUSTERS

inertia = []
silhouette_scores = []
K = range(2, 15)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data2_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data2_scaled, kmeans.labels_))

# Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k')
plt.show()

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(data2_scaled)

data["Music Segments"] = clusters + 1

# mapping cluster numbers to cluster names
data["Music Segments"] = data["Music Segments"].map({
    1: "Cluster 1: High BPM, Low Acousticness",
    2: "Cluster 2: Moderate Energy, High Valence",
    3: "Cluster 3: Low Danceability, High Loudness",
    4: "Cluster 4: High Speechiness, Moderate Energy"
})

data["Title"] = data_names["Title"]
data["Artist"] = data_names["Artist"]

# 3D scatter plot
PLOT = go.Figure()
for i in list(data["Music Segments"].unique()):
    subset = data[data["Music Segments"] == i]
    PLOT.add_trace(go.Scatter3d(
        x=subset['Beats Per Minute (BPM)'],
        y=subset['Energy'],
        z=subset['Danceability'],
        mode='markers',
        marker=dict(size=6, line=dict(width=1)),
        text=subset.apply(lambda row: f" {row['Title']}<br>Artist: {row['Artist']}", axis=1),
        name=str(i)
    ))
PLOT.update_traces(hovertemplate='Title: %{text}<br>Beats Per Minute (BPM): %{x} <br>Energy: %{y} <br>Danceability: %{z}')

PLOT.update_layout(
    width=1100, height=1100, autosize=True, showlegend=True,
    scene=dict(
        xaxis=dict(title='Beats Per Minute (BPM)', titlefont_color='black'),
        yaxis=dict(title='Energy', titlefont_color='black'),
        zaxis=dict(title='Danceability', titlefont_color='black')
    ),
    font=dict(family="Gilroy", color='black', size=12)
)

PLOT.show()