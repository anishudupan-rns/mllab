import matplotlib.pyplot as plt
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
