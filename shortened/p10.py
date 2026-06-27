import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

d = load_breast_cancer()
X_s = StandardScaler().fit_transform(d.data)
X_p = PCA(2).fit_transform(X_s)
km = KMeans(n_clusters=2, random_state=42)
y_km = km.fit_predict(X_s)

print(f"Confusion Matrix:\n{confusion_matrix(d.target, y_km)}")
print(f"\nClassification Report:\n{classification_report(d.target, y_km)}")

plots = [
    (y_km, "Set1", "K-Means Clustering of Breast Cancer Dataset"),
    (d.target, "coolwarm", "True Labels of Breast Cancer Dataset"),
    (y_km, "Set1", "Clusters with Centroids")
]

for i, (hue, pal, ttl) in enumerate(plots):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X_p[:, 0], y=X_p[:, 1], hue=hue, palette=pal, alpha=0.9)
    if i == 2:
        ctr = PCA(2).fit(X_s).transform(km.cluster_centers_)
        plt.scatter(ctr[:, 0], ctr[:, 1], s=200, c="black", marker="X", label="Centroids")
    plt.legend()
    plt.title(ttl)
    plt.show()
