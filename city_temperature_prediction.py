import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from polynomial_fitting import PolynomialFitting

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename,parse_dates=["Date"])
    df = df[df["Temp"] > -100]
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    israel_df = df[df["Country"] == "Israel"].copy()

    plt.figure()
    for year in israel_df["Year"].unique():
        year_data = israel_df[israel_df["Year"] == year]
        plt.scatter(year_data["DayOfYear"], year_data["Temp"], label=str(year), s=10)

    plt.xlabel("Day of Year")
    plt.ylabel("Temperature")
    plt.title("Daily Temperature in Israel by Day of Year")
    plt.legend()
    plt.savefig("israel_temp_scatter.png")
    plt.close()

    # Monthly std plot
    monthly_std = israel_df.groupby("Month")["Temp"].std()

    plt.figure()
    monthly_std.plot(kind="bar")
    plt.xlabel("Month")
    plt.ylabel("Temperature STD")
    plt.title("Temperature Standard Deviation by Month in Israel")
    plt.savefig("israel_monthly_std.png")
    plt.close()

    # Question 4 - Fitting model for different values of `k`
    X = israel_df["DayOfYear"].to_numpy()
    y = israel_df["Temp"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    degrees = range(1, 11)
    test_errors = []
    for k in degrees:
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)

        error = model.loss(X_test, y_test)
        error = round(error, 2)

        test_errors.append(error)
        print(f"k={k}, test error={error}")

    best_k = list(degrees)[np.argmin(test_errors)]
    print(f"Best k: {best_k}")

    plt.figure()
    plt.bar(degrees, test_errors)
    plt.xlabel("Polynomial degree k")
    plt.ylabel("Test MSE")
    plt.title("Test Error for Polynomial Degrees")
    plt.xticks(list(degrees))
    plt.savefig("polynomial_test_errors.png")
    plt.close()


