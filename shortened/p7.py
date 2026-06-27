import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def run(X, y, deg):
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    yp = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(xtr, ytr).predict(xte)
    print(f"\nR^2 score,Mean Squared Error: {r2_score(yte,yp)},{mean_squared_error(yte, yp)}")
    plt.scatter(xte, yte, c="b", label="Actual")
    plot_func = plt.plot if deg == 1 else plt.scatter
    plot_func(xte, yp, c="r", label="Predicted")
    plt.legend()
    plt.show()

print("Demonstrating Linear Regression and Polynomial Regression\n")
# 1. Linear Regression
cal = fetch_california_housing()
run(cal.data[:, [2]], cal.target, 1)

# 2. Polynomial Regression
csv=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", sep="\\s+", names=["mpg", "c", "disp", "h", "w", "a", "y", "o"], na_values="?").dropna()
run(csv[["disp"]],csv["mpg"], 2)

# csv=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",sep="\\s+",header=None,na_values="?").dropna()
# run(csv[[3]],csv[1],2)