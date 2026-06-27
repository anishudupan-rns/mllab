P1 = r"""import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset and get columns
df = fetch_california_housing(as_frame=True).frame
features = df.columns

# Histograms
plt.figure(figsize=(15, 10))
for i, col in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True, bins=30)
plt.tight_layout()
plt.show()

# Box Plots
plt.figure(figsize=(15, 10))
for i, col in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col])
plt.tight_layout()
plt.show()

# Outliers (Readable but short)
print("Outliers Detection:")
for col in features:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"{col}: {outliers} outliers")

# Summary
print("\nDataset Summary:\n", df.describe())
"""

P2 = r"""import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data=fetch_california_housing(as_frame=True).frame
corr_matrix=data.corr()

sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")
plt.show()

sns.pairplot(data,diag_kind='kde')
plt.show()
"""

P3 = r"""from sklearn.datasets import load_iris
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
"""

P4 = r"""import pandas as pd

data=pd.read_csv("../data.csv")
attributes=data.columns[:-1]
class_label=data.columns[-1]
hypothesis=['0' for _ in attributes]


for index,row in data.iterrows():
    if row[class_label]=='yes':
        for i,value in enumerate(row[attributes]):
            if hypothesis[i]=='0' or hypothesis[i]==value:
                hypothesis[i]=value
            else:
                hypothesis[i]='?'

print(hypothesis)
"""

P5 = r"""import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
data = np.random.rand(100)
labels = ["Class1" if x <= 0.5 else "Class2" for x in data[:50]]

def euclidean_distance(x1, x2):
    return abs(x1 - x2)

def knn_classifier(train_data, train_labels, test_point, k):
    distances = [(euclidean_distance(test_point, train_data[i]), train_labels[i]) for i in range(len(train_data))]
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]
    k_nearest_labels = [label for _, label in k_nearest_neighbors]
    return Counter(k_nearest_labels).most_common(1)[0][0]

train_data = data[:50]
train_labels = labels
test_data = data[50:]
k_values = [1, 2, 3, 4, 5, 20, 30]


print("--- k-Nearest Neighbors Classification ---")
print("Training dataset: First 50 points labeled based on the rule (x <= 0.5 -> Class1, x > 0.5 -> Class2)")
print("Testing dataset: Remaining 50 points to be classified\n")
results = {}

for k in k_values:
    print(f"Results for k = {k}:")
    classified_labels = [knn_classifier(train_data, train_labels, test_point, k) for test_point in test_data]
    results[k] = classified_labels
    for i, label in enumerate(classified_labels, start=51):
        print(f"Point x{i} (value: {test_data[i - 51]:.4f}) is classified as {label}")
        print("\n")
print("Classification complete.\n")

for k in k_values:
    classified_labels = results[k]
    class1_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i] == "Class1"]
    class2_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i] == "Class2"]
    plt.figure(figsize=(10, 6))
    plt.scatter(train_data, [0] * len(train_data), c=["blue" if label == "Class1" else "red" for label in train_labels],label="Training Data", marker="o")
    plt.scatter(class1_points, [1] * len(class1_points), c="blue", label="Class1 (Test)", marker="x")
    plt.scatter(class2_points, [1] * len(class2_points), c="red", label="Class2 (Test)", marker="x")
    plt.title(f"k-NN Classification Results for k = {k}")
    plt.xlabel("Data Points")
    plt.ylabel("Classification Level")
    plt.legend()
    plt.grid(True)
    plt.show()
"""

P6 = r"""import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
x_test = np.linspace(0, 2 * np.pi, 200)
x_bias = np.c_[np.ones(100), X]
y = np.sin(X) + 0.1 * np.random.randn(100)
print("x_test",x_test)


def predict(xi, tau=0.5):
    W = np.diag(np.exp(-np.sum((x_bias - xi) ** 2, axis=1) / (2 * tau**2)))
    return xi @ (np.linalg.inv(x_bias.T @ W @ x_bias) @ x_bias.T @ W @ y)


# Execution & Visualization
y_pred = [predict([1, x]) for x in x_test]

plt.scatter(X, y, alpha=0.7, label="Training Data")
plt.plot(x_test, y_pred, lw=2, label="LWR Fit")
plt.legend()
plt.title("Locally Weighted Regression")
plt.show()
"""

P7 = r"""import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_and_plot(X, y, model, title, xlabel, ylabel, plot_method):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.fit(X_train, y_train).predict(X_test)

    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plot_method(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    print(f"{title}\nMean Squared Error: {mean_squared_error(y_test, y_pred)}\nR^2 Score: {r2_score(y_test, y_pred)}")

print("Demonstrating Linear Regression and Polynomial Regression\n")

# --- California Housing ---
cali = fetch_california_housing()
evaluate_and_plot(
    X=cali.data[:, 2:3], y=cali.target, model=LinearRegression(),
    title="Linear Regression - California Housing Dataset",
    xlabel="Average number of rooms (AveRooms)", ylabel="Median value of homes ($100,000)",
    plot_method=plt.plot
)

# --- Auto MPG ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
df = pd.read_csv(url, sep=r'\s+', usecols=[0, 2], names=["mpg", "displacement"], na_values="?").dropna()
evaluate_and_plot(
    X=df[["displacement"]], y=df["mpg"],
    model=make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression()),
    title="Polynomial Regression - Auto MPG Dataset",
    xlabel="Displacement", ylabel="Miles per gallon (mpg)",
    plot_method=plt.scatter
)
"""

P8 = r"""import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = load_breast_cancer()
xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42).fit(xtr, ytr)
yp = clf.predict(xte)

print(f"Model Accuracy: {accuracy_score(yte, yp) * 100:.2f}%")
res = clf.predict(xte[0].reshape(1, -1))
print(f"Predicted Class for the new sample: {'Benign' if res==1 else 'Malignant'}")

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Decision Tree - Breast Cancer Dataset")
plt.show()
"""

P9 = r"""import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB

data = fetch_olivetti_faces(shuffle=True, random_state=42)
xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

gnb = GaussianNB().fit(xtr, ytr)
yp = gnb.predict(xte)

print(f"Accuracy: {accuracy_score(yte, yp):.2f}")
print(f"Cross-validation accuracy: {cross_val_score(gnb, data.data, data.target, cv=5).mean():.2f}")
print("\nClassification Report:\n", classification_report(yte, yp, zero_division=1))
print("\nConfusion Matrix:\n", confusion_matrix(yte, yp))

fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for ax, img, lbl, pred in zip(axes.ravel(), xte, yte, yp):
    ax.imshow(img.reshape(64, 64), cmap="gray")
    ax.set_title(f"True:{lbl}, Pred:{pred}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""

P10 = r"""import matplotlib.pyplot as plt
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
"""

def program1():
    print(P1)

def program2():
    print(P2)

def program3():
    print(P3)

def program4():
    print(P4)

def program5():
    print(P5)

def program6():
    print(P6)

def program7():
    print(P7)

def program8():
    print(P8)

def program9():
    print(P9)

def program10():
    print(P10)
