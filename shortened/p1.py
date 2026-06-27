import seaborn as sns
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
