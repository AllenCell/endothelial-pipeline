from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def examine(path, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(path)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.iloc[:, :-1])

    # Apply PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Colorize by the timepoint
    colors = df["T"]

    # Plot the data
    plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.5, c=colors, cmap="viridis")
    plt.colorbar(label="T")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA with Colorized by T")
    plt.savefig(save_dir / "pca_plot.png")
    plt.close()

    X = df.iloc[:, :256]  # Features (all columns except 'T')
    y = df["T"]  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    plt.scatter(y_test, y_pred, s=1)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted Values")
    plt.savefig(save_dir / "true_vs_predicted.png")
    plt.close()

    # Extract the coefficients
    coefficients = model.coef_

    # Pair the coefficients with the feature names
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": coefficients}
    )

    # Sort the features by importance
    feature_importance = feature_importance.sort_values(
        by="Importance", ascending=False
    )

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance["Feature"], feature_importance["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.savefig(save_dir / "feature_importance.png")
    plt.close()


if __name__ == "__main__":
    fire.Fire(examine)
