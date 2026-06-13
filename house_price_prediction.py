import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression


def preprocess_train(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    X = X.copy()
    y = y.copy()
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    X = X.select_dtypes(include=[np.number])
    return X, y



def preprocess_test(X: pd.DataFrame) -> pd.DataFrame:
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    X = X.copy()
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.mean())
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> None:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    os.makedirs(output_path, exist_ok=True)
    for feature in X.columns:
        corr = X[feature].corr(y)
        plt.figure()
        plt.scatter(X[feature], y)
        plt.xlabel(feature)
        plt.ylabel("price")
        plt.title(f"{feature}, Pearson corr = {corr:.3f}")
        plt.savefig(os.path.join(output_path, f"{feature}.png"))
        plt.close()



if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Question 3 - preprocessing of housing prices train dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Question 4 - preprocess the test data
    X_train, y_train = preprocess_train(X_train, y_train)

    X_test = X_test[X_train.columns]
    feature_evaluation(X_train, y_train, output_path="feature_plots")
    # Question 5 - Fit model over increasing percentages of the overall training data
    percentages = range(10, 101)
    mean_losses = []
    std_losses = []

    for p in percentages:
        losses = []

        for i in range(10):
            X_sample = X_train.sample(frac=p / 100, random_state=i)
            y_sample = y_train.loc[X_sample.index]

            model = LinearRegression(include_intercept=True)
            model.fit(X_sample.to_numpy(), y_sample.to_numpy())

            loss = model.loss(X_test.to_numpy(), y_test.to_numpy())
            losses.append(loss)

        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)

    plt.figure()
    plt.plot(percentages, mean_losses, label="Mean test loss")
    plt.fill_between(
        percentages,
        mean_losses - 2 * std_losses,
        mean_losses + 2 * std_losses,
        alpha=0.2,
        label="mean ± 2 std"
    )
    plt.xlabel("Training set percentage")
    plt.ylabel("MSE loss")
    plt.title("Test loss as function of training size")
    plt.legend()
    plt.savefig("training_size_loss.png")
    # plt.show()

    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

