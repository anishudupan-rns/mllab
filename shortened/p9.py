import matplotlib.pyplot as plt
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
