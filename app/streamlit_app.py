from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Retail Analytics & Forecasting",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "data" / "processed"

    daily_sales_path = processed_dir / "daily_sales.csv"
    predictions_path = processed_dir / "daily_sales_predictions.csv"
    metrics_path = processed_dir / "model_metrics.csv"
    feature_importance_path = processed_dir / "feature_importance.csv"
    sales_only_path = processed_dir / "retail_cleaned_sales_only.parquet"

    missing_files = []
    for file_path in [
        daily_sales_path,
        predictions_path,
        metrics_path,
        feature_importance_path,
        sales_only_path,
    ]:
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing_files)
        )

    daily_sales = pd.read_csv(daily_sales_path)
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    feature_importance = pd.read_csv(feature_importance_path)
    sales_only = pd.read_parquet(sales_only_path)

    daily_sales["invoice_day"] = pd.to_datetime(daily_sales["invoice_day"])
    predictions["invoice_day"] = pd.to_datetime(predictions["invoice_day"])
    sales_only["InvoiceDate"] = pd.to_datetime(sales_only["InvoiceDate"])
    sales_only["invoice_day"] = pd.to_datetime(sales_only["invoice_day"])

    return daily_sales, predictions, metrics, feature_importance, sales_only


def format_currency(value: float) -> str:
    return f"€{value:,.0f}"


def main():
    st.title("Retail Analytics & Forecasting Dashboard")
    st.write(
        "End-to-end retail analytics project with data cleaning, KPI tracking, and daily revenue forecasting."
    )

    try:
        daily_sales, predictions, metrics, feature_importance, sales_only = load_data()
    except Exception as e:
        st.error("Could not load app data.")
        st.code(str(e))
        st.info("Make sure you already ran:")
        st.code(
            r""".\.venv\Scripts\python.exe .\src\clean_data.py
.\.venv\Scripts\python.exe .\src\train_model.py"""
        )
        st.stop()

    # Sidebar filters
    st.sidebar.header("Filters")

    min_date = daily_sales["invoice_day"].min().date()
    max_date = daily_sales["invoice_day"].max().date()

    selected_dates = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif isinstance(selected_dates, list) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    filtered_daily = daily_sales[
        (daily_sales["invoice_day"].dt.date >= start_date)
        & (daily_sales["invoice_day"].dt.date <= end_date)
    ].copy()

    filtered_sales = sales_only[
        (sales_only["invoice_day"].dt.date >= start_date)
        & (sales_only["invoice_day"].dt.date <= end_date)
    ].copy()

    # KPI section
    total_revenue = filtered_sales["sales_amount"].sum()
    total_orders = filtered_sales["InvoiceNo"].nunique()
    total_customers = filtered_sales["CustomerID"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", format_currency(total_revenue))
    c2.metric("Total Orders", f"{total_orders:,}")
    c3.metric("Unique Customers", f"{total_customers:,}")
    c4.metric("Average Order Value", format_currency(avg_order_value))

    st.divider()

    # Revenue trend
    st.subheader("Daily Revenue Trend")
    revenue_fig = px.line(
        filtered_daily,
        x="invoice_day",
        y="daily_revenue",
        title="Daily Revenue Over Time",
    )
    st.plotly_chart(revenue_fig, width="stretch")

    # Orders and customers
    col1, col2 = st.columns(2)

    with col1:
        orders_fig = px.line(
            filtered_daily,
            x="invoice_day",
            y="total_orders",
            title="Daily Orders",
        )
        st.plotly_chart(orders_fig, width="stretch")

    with col2:
        customers_fig = px.line(
            filtered_daily,
            x="invoice_day",
            y="unique_customers",
            title="Daily Unique Customers",
        )
        st.plotly_chart(customers_fig, width="stretch")

    st.divider()

    # Top countries
    st.subheader("Top Countries by Revenue")
    country_revenue = (
        filtered_sales.groupby("Country", as_index=False)["sales_amount"]
        .sum()
        .sort_values("sales_amount", ascending=False)
        .head(10)
    )

    country_fig = px.bar(
        country_revenue,
        x="Country",
        y="sales_amount",
        title="Top 10 Countries by Revenue",
    )
    st.plotly_chart(country_fig, width="stretch")

    # Top products
    st.subheader("Top Products by Revenue")
    product_revenue = (
        filtered_sales.groupby("Description", as_index=False)["sales_amount"]
        .sum()
        .sort_values("sales_amount", ascending=False)
        .head(10)
    )

    product_fig = px.bar(
        product_revenue,
        x="Description",
        y="sales_amount",
        title="Top 10 Products by Revenue",
    )
    product_fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(product_fig, width="stretch")

    st.divider()

    # Model performance
    st.subheader("Forecast Model Performance")

    metric_col1, metric_col2 = st.columns(2)

    baseline_row = metrics[metrics["model"] == "baseline_lag_1"].iloc[0]
    rf_row = metrics[metrics["model"] == "random_forest"].iloc[0]

    with metric_col1:
        st.write("Baseline Model")
        st.metric("MAE", f"{baseline_row['MAE']:.2f}")
        st.metric("RMSE", f"{baseline_row['RMSE']:.2f}")

    with metric_col2:
        st.write("Random Forest")
        st.metric("MAE", f"{rf_row['MAE']:.2f}")
        st.metric("RMSE", f"{rf_row['RMSE']:.2f}")

    st.write(
        "Lower MAE and RMSE mean the model's predictions are closer to the actual revenue."
    )

    # Prediction comparison chart
    prediction_fig = px.line(
        predictions,
        x="invoice_day",
        y=[
            "actual_daily_revenue",
            "baseline_prediction",
            "random_forest_prediction",
        ],
        title="Actual vs Predicted Daily Revenue",
    )
    st.plotly_chart(prediction_fig, width="stretch")

    # Feature importance
    st.subheader("Feature Importance")
    top_features = feature_importance.head(10)

    importance_fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Most Important Features",
    )
    importance_fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(importance_fig, width="stretch")

    # Raw data preview
    st.subheader("Preview Tables")
    preview_option = st.selectbox(
        "Choose table to preview",
        ["Daily Sales", "Predictions", "Feature Importance", "Sales Transactions"],
    )

    if preview_option == "Daily Sales":
        st.dataframe(filtered_daily.tail(30), width="stretch")
    elif preview_option == "Predictions":
        st.dataframe(predictions.tail(30), width="stretch")
    elif preview_option == "Feature Importance":
        st.dataframe(feature_importance, width="stretch")
    else:
        st.dataframe(filtered_sales.head(50), width="stretch")


if __name__ == "__main__":
    main()