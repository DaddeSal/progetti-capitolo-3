# 1. Data Exploration & Preprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


iris = load_iris()
X = iris.data
feature_names = iris.feature_names


df = pd.DataFrame(X, columns=feature_names)
print(df.describe())

scaler = StandardScaler()
X_std = scaler.fit_transform(X)



# 2. Stima di K con DBSCAN


min_samples = 5
eps_values = [0.35, 0.45, 0.55]

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_std)

    n_noise = np.sum(labels == -1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    print(
        f"eps = {eps:.2f} | min_samples = {min_samples} | "
        f"Cluster trovati (K): {n_clusters} | Punti di rumore (-1): {n_noise}"
    )



# 3. Stima di K con Elbow Method


wcss = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k, init="k-means++", n_init="auto", random_state=42
    )
    kmeans.fit(X_std)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_values, wcss, "o-", color="blue", markersize=8)

k_opt = 3
plt.axvline(
    x=k_opt,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"K ottimale scelto = {k_opt}",
)

plt.xlabel("Numero di Cluster (K)")
plt.ylabel("WCSS (Inerzia)")
plt.xticks(k_values)
plt.title("Elbow Method per Dataset Iris")
plt.legend()
plt.grid(True)
plt.show()



# 4. Applicazione Finale & Visualizzazione


k_final = 3
kmeans_final = KMeans(
    n_clusters=k_final, init="k-means++", n_init="auto", random_state=42
)
clusters_final = kmeans_final.fit_predict(X_std)
centroids_final = kmeans_final.cluster_centers_

sil_score = silhouette_score(X_std, clusters_final)
print(f"Silhouette Score per K={k_final}: {sil_score:.4f}")

plt.figure(figsize=(8, 6))

plt.scatter(
    X_std[:, 0],
    X_std[:, 1],
    c=clusters_final,
    cmap="viridis",
    s=50,
    alpha=0.7,
    label="Dati",
)

plt.scatter(
    centroids_final[:, 0],
    centroids_final[:, 1],
    c="red",
    s=200,
    marker="X",
    edgecolor="black",
    label="Centroidi",
)

plt.title(
    f"K-Means su Iris (K={k_final}) - Prime 2 Feature\nSilhouette Score: {sil_score:.2f}"
)
plt.xlabel(f"{feature_names[0]} (Standardizzata)")
plt.ylabel(f"{feature_names[1]} (Standardizzata)")
plt.legend()
plt.grid(True)
plt.show()