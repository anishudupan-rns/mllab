import sklearn
import importlib

target=["accuracy_score","train_test_split","DecisionTreeClassifier","plot_tree","classification_report", "confusion_matrix","cross_val_score","GaussianNB","KMeans","PCA","StandardScaler"]

for i in dir(sklearn):
    try:
        dirs=importlib.import_module(f"sklearn.{i}")
        for j in target:
            if hasattr(dirs,j):
                print(f"from sklearn.{i} import {j}")
    except Exception:
        pass

