from pathlib import Path
import pandas as pd


def main() -> None:
    # Project folders
    base_dir = Path(__file__).resolve().parents[1]
    raw_csv_path = base_dir / "data" / "raw" / "online_retail_raw.csv"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    cleaned_full_path = processed_dir / "retail_cleaned_full.parquet"
    cleaned_sales_only_path = processed_dir / "retail_cleaned_sales_only.parquet"
    daily_sales_parquet_path = processed_dir / "daily_sales.parquet"
    daily_sales_csv_path = processed_dir / "daily_sales.csv"

    print("Reading raw CSV...")
    df = pd.read_csv(raw_csv_path, low_memory=False)

    print(f"Initial shape: {df.shape}")

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Convert data types
    print("Fixing data types...")

    text_columns = ["InvoiceNo", "StockCode", "Description", "CustomerID", "Country"]
    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()

    # CustomerID often comes in as values like 17850.0 after Excel/CSV roundtrip
    df["CustomerID"] = df["CustomerID"].str.replace(".0", "", regex=False)

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Remove duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)
    print(f"Removed duplicate rows: {before_dedup - after_dedup}")

    # Remove rows missing critical fields
    critical_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "Country",
    ]

    before_dropna = len(df)
    df = df.dropna(subset=critical_columns)
    after_dropna = len(df)
    print(f"Removed rows with missing critical values: {before_dropna - after_dropna}")

    # Remove blank descriptions
    before_blank_desc = len(df)
    df = df[df["Description"].str.len() > 0]
    after_blank_desc = len(df)
    print(f"Removed rows with blank descriptions: {before_blank_desc - after_blank_desc}")

    # Create useful flags and features
    print("Creating features...")
    df["is_cancellation"] = df["InvoiceNo"].str.startswith("C", na=False)
    df["has_customer_id"] = df["CustomerID"].notna()
    df["sales_amount"] = df["Quantity"] * df["UnitPrice"]

    df["invoice_day"] = df["InvoiceDate"].dt.floor("D")
    df["invoice_month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["invoice_hour"] = df["InvoiceDate"].dt.hour
    df["invoice_weekday"] = df["InvoiceDate"].dt.day_name()

    # Save cleaned full dataset
    print("Saving cleaned full dataset...")
    df.to_parquet(cleaned_full_path, index=False)

    # Create sales-only dataset
    # We keep only normal completed sales for forecasting:
    # - not cancellation
    # - positive quantity
    # - positive unit price
    sales_df = df[
        (~df["is_cancellation"]) &
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0) &
        (df["sales_amount"] > 0)
    ].copy()

    print(f"Sales-only shape: {sales_df.shape}")

    print("Saving sales-only dataset...")
    sales_df.to_parquet(cleaned_sales_only_path, index=False)

    # Create daily sales table
    print("Creating daily sales table...")
    daily_sales = (
        sales_df.groupby("invoice_day", as_index=False)
        .agg(
            daily_revenue=("sales_amount", "sum"),
            total_orders=("InvoiceNo", "nunique"),
            total_items_sold=("Quantity", "sum"),
            unique_customers=("CustomerID", "nunique"),
        )
        .sort_values("invoice_day")
    )

    # Create a complete calendar so every day exists in the time series
    all_days = pd.DataFrame(
        {
            "invoice_day": pd.date_range(
                start=daily_sales["invoice_day"].min(),
                end=daily_sales["invoice_day"].max(),
                freq="D",
            )
        }
    )

    daily_sales = all_days.merge(daily_sales, on="invoice_day", how="left")

    # Fill missing days with zeros
    daily_sales["daily_revenue"] = daily_sales["daily_revenue"].fillna(0)
    daily_sales["total_orders"] = daily_sales["total_orders"].fillna(0).astype(int)
    daily_sales["total_items_sold"] = daily_sales["total_items_sold"].fillna(0).astype(int)
    daily_sales["unique_customers"] = daily_sales["unique_customers"].fillna(0).astype(int)

    # Add simple calendar features for later modeling
    daily_sales["day_of_week"] = daily_sales["invoice_day"].dt.day_name()
    daily_sales["month"] = daily_sales["invoice_day"].dt.month
    daily_sales["week"] = daily_sales["invoice_day"].dt.isocalendar().week.astype(int)
    daily_sales["year"] = daily_sales["invoice_day"].dt.year

    print("Saving daily sales table...")
    daily_sales.to_parquet(daily_sales_parquet_path, index=False)
    daily_sales.to_csv(daily_sales_csv_path, index=False)

    # Final summary
    print("\nDone.")
    print(f"Cleaned full dataset saved to: {cleaned_full_path}")
    print(f"Sales-only dataset saved to: {cleaned_sales_only_path}")
    print(f"Daily sales parquet saved to: {daily_sales_parquet_path}")
    print(f"Daily sales CSV saved to: {daily_sales_csv_path}")

    print("\nSummary:")
    print(f"Final cleaned full shape: {df.shape}")
    print(f"Final sales-only shape: {sales_df.shape}")
    print(f"Daily sales shape: {daily_sales.shape}")

    print("\nDaily sales preview:")
    print(daily_sales.head())


if __name__ == "__main__":
    main()