# MusicClustering

This is a compilation of Python code that clusters music data with and without Principal Component Analysis. 

## Introduction

The goal of this project was to analyze various music data and put them into clusters based on similar features. This can serve as a recommendation to those who wish to get similar music in the designated clusters.

## Features

The data contains statistics of the top 2000 tracks on the Spotify platform. This dataset was found on Kaggle at the link: https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset?resource=download

These are all the features provided in the dataset. 

* Artist
* Top Genre
* Year
* Beats Per Minute (BPM)
* Energy
* Danceability
* Loudness (dB)
* Liveness
* Valence
* Length (Duration)
* Acousticness
* Speechiness
* Popularity

## Installation

This project was completed on Python and the necessary packages are included at the top of the files.

clustering.py:

`import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go`

clustering_pca.py:

`import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go`

## Credit

This project was inspired by Aman Kharwal at https://thecleverprogrammer.com/2022/04/05/clustering-music-genres-with-machine-learning/

