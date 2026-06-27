from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris=load_iris()
reduced_data=PCA(n_components=2).fit_transform(iris.data)

for i,color in enumerate(['r','g','b']):
    plt.scatter(
        reduced_data[iris.target==i,0],
        reduced_data[iris.target==i,1],
        label=iris.target_names[i],
        color=color
    )
plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()