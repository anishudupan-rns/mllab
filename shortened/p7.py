import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# 1. Linear Regression
cal = fetch_california_housing()
xtr, xte, ytr, yte = train_test_split(cal.data[:,[2]], cal.target, test_size=0.2, random_state=42)
yp=make_pipeline(PolynomialFeatures(1), LinearRegression()).fit(xtr, ytr).predict(xte)
print(f"\nR^2 score,Mean Squared Error: {r2_score(yte,yp)},{mean_squared_error(yte, yp)}")
plt.scatter(xte, yte, c="b", label="Actual")
plt.plot(xte, yp, c="r", label="Predicted")
plt.legend()
plt.show()


# 2. Polynomial Regression
csv=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", sep="\\s+", names=["mpg", "c", "disp", "h", "w", "a", "y", "o"], na_values="?").dropna()
xtr, xte, ytr, yte = train_test_split(csv[["disp"]], csv["mpg"], test_size=0.2, random_state=42)
yp=make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(xtr, ytr).predict(xte)
print(f"\nR^2 score,Mean Squared Error: {r2_score(yte,yp)},{mean_squared_error(yte, yp)}")
plt.scatter(xte, yte, c="b", label="Actual")
plt.scatter(xte, yp, c="r", label="Predicted")
plt.legend()
plt.show()

# csv=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",sep="\\s+",header=None,na_values="?").dropna()
# run(csv[[3]],csv[1],2)