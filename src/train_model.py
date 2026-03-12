from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Make sure invoice_day is datetime
    df["invoice_day"] = pd.to_datetime(df["invoice_day"])

    # Calendar features
    df["day_of_week_num"] = df["invoice_day"].dt.dayofweek   # Monday=0, Sunday=6
    df["day_of_month"] = df["invoice_day"].dt.day
    df["month_num"] = df["invoice_day"].dt.month
    df["week_num"] = df["invoice_day"].dt.isocalendar().week.astype(int)
    df["year"] = df["invoice_day"].dt.year
    df["is_weekend"] = df["day_of_week_num"].isin([5, 6]).astype(int)

    # Lag features: previous values of revenue
    df["lag_1"] = df["daily_revenue"].shift(1)
    df["lag_7"] = df["daily_revenue"].shift(7)
    df["lag_14"] = df["daily_revenue"].shift(14)

    # Rolling averages (using past days only)
    df["rolling_mean_7"] = df["daily_revenue"].shift(1).rolling(window=7).mean()
    df["rolling_mean_14"] = df["daily_revenue"].shift(1).rolling(window=14).mean()

    return df


def calculate_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]

    input_path = base_dir / "data" / "processed" / "daily_sales.csv"
    models_dir = base_dir / "models"
    processed_dir = base_dir / "data" / "processed"

    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "random_forest_daily_revenue.joblib"
    predictions_path = processed_dir / "daily_sales_predictions.csv"
    metrics_path = processed_dir / "model_metrics.csv"

    print("Reading daily sales data...")
    df = pd.read_csv(input_path)

    print(f"Original daily sales shape: {df.shape}")

    # Create features
    print("Creating time-based features...")
    df = add_time_features(df)

    # Drop rows where lag/rolling features are not available
    before_drop = len(df)
    df = df.dropna().copy()
    after_drop = len(df)
    print(f"Removed rows with missing lag features: {before_drop - after_drop}")

    # Sort by date just to be safe
    df = df.sort_values("invoice_day").reset_index(drop=True)

    # Features for the model
    feature_cols = [
        "day_of_week_num",
        "day_of_month",
        "month_num",
        "week_num",
        "year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "total_orders",
        "total_items_sold",
        "unique_customers",
    ]

    target_col = "daily_revenue"

    X = df[feature_cols]
    y = df[target_col]

    # Time-based split: first 80% for training, last 20% for testing
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    test_dates = df["invoice_day"].iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Baseline model: predict yesterday's revenue
    baseline_pred = X_test["lag_1"].values

    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = calculate_rmse(y_test, baseline_pred)

    print("\nBaseline model results:")
    print(f"MAE: {baseline_mae:.2f}")
    print(f"RMSE: {baseline_rmse:.2f}")

    # Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    rf_pred = model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = calculate_rmse(y_test, rf_pred)

    print("\nRandom Forest results:")
    print(f"MAE: {rf_mae:.2f}")
    print(f"RMSE: {rf_rmse:.2f}")

    # Save trained model
    print("\nSaving trained model...")
    joblib.dump(model, model_path)

    # Save predictions for comparison
    predictions_df = pd.DataFrame(
        {
            "invoice_day": test_dates.values,
            "actual_daily_revenue": y_test.values,
            "baseline_prediction": baseline_pred,
            "random_forest_prediction": rf_pred,
        }
    )

    predictions_df.to_csv(predictions_path, index=False)

    # Save metrics
    metrics_df = pd.DataFrame(
        [
            {"model": "baseline_lag_1", "MAE": baseline_mae, "RMSE": baseline_rmse},
            {"model": "random_forest", "MAE": rf_mae, "RMSE": rf_rmse},
        ]
    )

    metrics_df.to_csv(metrics_path, index=False)

    # Feature importance
    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importance_path = processed_dir / "feature_importance.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False)

    print("\nDone.")
    print(f"Model saved to: {model_path}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Feature importance saved to: {feature_importance_path}")

    print("\nTop 10 most important features:")
    print(feature_importance_df.head(10))


if __name__ == "__main__":
    main()