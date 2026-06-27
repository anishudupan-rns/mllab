import pandas as pd
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