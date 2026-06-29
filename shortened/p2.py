import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data=fetch_california_housing(as_frame=True).frame
corr_matrix=data.corr()

sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")
plt.show()

sns.pairplot(data,diag_kind='kde')
plt.show()