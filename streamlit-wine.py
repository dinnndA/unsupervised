import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

# Load the K-Means model
st.title("Wine Clustering with K-Means")

try:
    # Load model .pkl
    with open("k-meanswine.pkl", "rb") as file:
        kmeans_model = pickle.load(file)

    # Simulate a dataset
    wine_data = pd.DataFrame({
        "alcohol": np.random.uniform(10, 15, 200),
        "total_phenols": np.random.uniform(0.1, 5, 200),
    })
    wine_features = wine_data.values

    # Input for maximum K
    max_k = st.slider("Select Maximum K", min_value=2, max_value=20, value=17)

    # Button to display Elbow Method graph
    if st.button("Show Elbow Method Graph"):
        # Calculate SSE for each K value
        sse = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=10)
            kmeans.fit(wine_features)
            sse.append(kmeans.inertia_)

        # Plot Elbow Method graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), sse, marker='o')
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Sum of Squared Errors (SSE)")
        plt.title("Elbow Method for Optimal K")
        plt.grid(True)

        # Display the graph in Streamlit
        st.pyplot(plt)

except FileNotFoundError:
    st.error("Model kmeans_wine.pkl not found! Please ensure the file is in the correct directory.")
