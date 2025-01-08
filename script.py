import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Load CSV file
data = "resources/Frontier_Data.csv"
df = pd.read_csv(data)

# Clean data
def clean_data(df: pd.DataFrame):
    # Abbreviate header names to snake_case convention
    df.columns = [col.lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_") for col in df.columns]

    df.columns = [col.replace("subloop1", "s1").replace("subloop2", "s2").replace("subloop3", "s3")
                  .replace("coolant_","") for col in df.columns]
    
    df = df.rename(columns={
        "overall_supply_temp": "supply_temp",
        "frontier_total_power": "total_power"
    })

    # Replace "#DIV/0!" that is present in some rows with null values
    df = df.replace("#DIV/0!", np.nan)

    # Drop second row (units) & last two empty columns
    df = df.iloc[1:, :-2]

    # Remove any rows with null values
    df = df.dropna()

    # Remove redundant variables
    df = df.drop(columns=["overall_average_return_temp",
                          "overall_flow",
                          "overall_wasteheat",
                          "frontier_compute_power",
                          "frontier_facility_accessory_power",
                          "power_usage_effectiveness"])

    return df

# Set type of data to float and time
def setTypes(df: pd.DataFrame):
    df = df.astype({
        "s1_return_temp": "float",
        "s1_flow": "float",
        "s2_return_temp": "float",
        "s2_flow": "float",
        "s3_return_temp": "float",
        "s3_flow": "float",
        "supply_temp": "float",
        "s1_wasteheat": "float",
        "s2_wasteheat": "float",
        "s3_wasteheat": "float",
        "total_power": "float"
    })

    df['date_time'] = pd.to_datetime(df['date_time'], format='%d/%m/%Y %H:%M')

    return df

# Calculate & print summary statistics
def print_summary_statistics(df: pd.DataFrame):
    df = df.iloc[:, 1:]
    stats = {
        "Minimum": df.min(),
        "Median": df.median(),
        "Maximum": df.max(),
        "Mean": df.mean(),
        "Std Dev": df.std(),
        "Skewness": df.apply(skew),
        "Kurtosis": df.apply(kurtosis)
    }

    summary_df = pd.DataFrame(stats)
    summary_df = summary_df.round(2) # Round values to 2 decimal places
    print(summary_df)

# Plot histogram
def plot_histogram(df: pd.DataFrame, variable: str):
    plt.hist(df[variable], bins=20, color='blue', edgecolor='black')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {variable}')
    plt.show()

# Remove outliers
def remove_outliers(df: pd.DataFrame):
    data = df.iloc[:, 1:]

    # Calculate Z-scores for each column
    z_scores = (data - data.mean()) / data.std()

    # Find rows with any Z-scores > 3 or < -3
    outliers = (np.abs(z_scores) > 3).any(axis=1)

    # Remove outliers
    df = df[~outliers]

    return df

# Plot correlation matrix
def plot_correlation_matrix(df: pd.DataFrame):
    data = df.iloc[:, 1:]
    matrix = data.corr()

    plt.figure(figsize=(10, 8)) # Set figure size
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix Heatmap")
    plt.show()

# Plot Scatterplot
def show_scatterplot(df: pd.DataFrame, x: str, y: str):
    data = df.iloc[:, 1:]
    sns.scatterplot(data=data, x=x, y=y)
    plt.title(f'Scatterplot of {x} vs {y}')
    plt.show()

# Plot time graph
def plot_time_graph(df: pd.DataFrame, variable: str):
    plt.figure(figsize=(10, 6))
    plt.plot(df["date_time"], df[variable], linestyle="-", label="Power demands over time")
    plt.title("Time vs total_power")
    plt.xlabel("Time")
    plt.ylabel("Total Power")
    plt.grid(True)
    plt.legend()
    plt.show()

# Feature selection
def select_features(df: pd.DataFrame, *features):
    df = df[list[features]]
    return df

# Save dataframe to file
def save_to_csv(df: pd.DataFrame, name: str):
    data = f"resources/{name}.csv"
    df.to_csv(data, index=False)

# Split train/test data
def split_data(df: pd.DataFrame, target_column: str, test_size: float):
    data = df.iloc[:, 1:]
    features = data.drop(columns=[target_column])
    target = data[target_column]

    # Split data to training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

# Evaluate model with metrics - MSE, RMSE, MAE, MAPE & R2
def evaluate_model(model: str, y, y_pred):
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"---{model}---")
    print(f"MSE: {mse:.4f}   RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}   MAPE: {mape:.4f}")
    print(f"R2: {r2:.4f}\n")

# Linear regression
def linear_regression(X_train, X_test, y_train, y_test):
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the training & test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate model
    evaluate_model("Linear Regression - Train", y_train, y_train_pred)
    evaluate_model("Linear Regression - Test", y_test, y_test_pred)

# Polynomial regression
def polynomial_regression(X_train, X_test, y_train, y_test):
    # Set up the pipeline
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(include_bias=False)),
        ('linear_reg', LinearRegression())
    ])

    # Define the parameter grid for degree of polynomial
    param_grid = {
        'poly_features__degree': [2, 3, 4]
    }

    # Use GridSearchCV for hyperparameter tuning (5-fold cross-validation)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='r2', verbose=2)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_degree = grid_search.best_params_['poly_features__degree']

    # Predict on the test set
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Evaluate model
    evaluate_model("Polynomial Regression - Train", y_train, y_train_pred)
    evaluate_model("Polynomial Regression - Test", y_test, y_test_pred)
    print(f"Best degree: {best_degree}")

# Random Forest Regression
def random_forest_regression(X_train, X_test, y_train, y_test):
    # Initialise model
    model = RandomForestRegressor(random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],        # Number of trees
        'max_depth': [None, 10, 20],           # Depth of trees
        'min_samples_split': [2, 5, 10],       # Minimum samples to split a node
        'min_samples_leaf': [1, 2, 4]          # Minimum samples per leaf
    }

    # Perform Randomized Search with 10 iterations (3-fold)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10,
                                       cv=3, scoring='r2', random_state=42, verbose=2, n_jobs=-1)
    
    # Fit the random search
    random_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Predict on the test set
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Evaluate model
    evaluate_model("Random Forest Regression - Train", y_train, y_train_pred)
    evaluate_model("Random Forest Regression - Test", y_test, y_test_pred)
    print(f"Best parameters: {best_params}")

# Support Vector Regression (SVR)
def support_vector_regression(X_train, X_test, y_train, y_test):
    # Define the pipeline with scaling and SVR
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    # Define the parameter grid
    param_grid = {
        'svr__C': [0.1, 1, 10],                # Regularization strength
        'svr__gamma': ['scale', 0.1, 0.01]     # Kernel coefficient
    }

    # Perform Randomized Search
    random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=5,
                                       cv=3, scoring='r2', random_state=42, verbose=2, n_jobs=-1)
    
    # Fit the random search
    random_search.fit(X_train, y_train)
    
    # Get the best model and its parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Predict on the test set
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Evaluate model
    evaluate_model("Support Vector Regression - Train", y_train, y_train_pred)
    evaluate_model("Support Vector Regression - Test", y_test, y_test_pred)
    print(f"Best parameters: {best_params}")

# XGBoost
def xgboost(X_train, X_test, y_train, y_test):
    # Initialise model
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
    }

    # Perform Randomized Search
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10,
                                       cv=3, scoring='r2', random_state=42, verbose=2, n_jobs=-1)

    # Fit the random search
    random_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Predict on the test set
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Evaluate model
    evaluate_model("XGBoost - Train", y_train, y_train_pred)
    evaluate_model("XGBoost - Test", y_test, y_test_pred)
    print(f"Best parameters: {best_params}")

    visualize_performance("XGBoost", y_train, y_train_pred, y_test, y_test_pred)

# Visualize model performance
def visualize_performance(model, y_train, y_train_pred, y_test, y_test_pred):
    
    # Create subplots for train and test set comparisons
    plt.figure(figsize=(12, 6))

    # Training set predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, edgecolor='k', label="Predicted vs Actual")
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='r', linestyle='--', label="Ideal Line")
    plt.title(f"{model} - Training Set")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)

    # Testing set predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolor='k', label="Predicted vs Actual")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', label="Ideal Line")
    plt.title(f"{model} - Testing Set")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)

    # Show plots
    plt.tight_layout()
    plt.show()

# Main
df = clean_data(df)
df = setTypes(df)
save_to_csv(df, "Frontier_Data_Cleaned")

df = remove_outliers(df)
save_to_csv(df, "Frontier_Data_Preprocessed")

df = select_features(df, "s1_wasteheat, s2_wasteheat, s3_wasteheat, s1_return_temp, s2_return_temp, s3_return_temp")

X_train, X_test, y_train, y_test = split_data(df, "total_power", 0.3)

linear_regression(X_train, X_test, y_train, y_test)
polynomial_regression(X_train, X_test, y_train, y_test)
random_forest_regression(X_train, X_test, y_train, y_test)
support_vector_regression(X_train, X_test, y_train, y_test)
xgboost(X_train, X_test, y_train, y_test)
