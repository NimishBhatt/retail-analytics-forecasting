from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import pandas as pd

# Official UCI download link for Online Retail dataset
DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ZIP_FILE_NAME = "online_retail.zip"
EXCEL_FILE_NAME = "Online Retail.xlsx"


def main() -> None:
    # Main project folder
    base_dir = Path(__file__).resolve().parents[1]

    # Raw data folder
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / ZIP_FILE_NAME
    excel_path = raw_dir / EXCEL_FILE_NAME
    raw_csv_path = raw_dir / "online_retail_raw.csv"
    preview_csv_path = raw_dir / "preview_first_100_rows.csv"

    # Step 1: Download zip file if it does not already exist
    if not zip_path.exists():
        print("Downloading dataset zip file from UCI...")
        urlretrieve(DATASET_ZIP_URL, zip_path)
        print(f"Downloaded: {zip_path}")
    else:
        print(f"Zip file already exists: {zip_path}")

    # Step 2: Extract the Excel file if needed
    if not excel_path.exists():
        print("Extracting Excel file from zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        print(f"Extracted: {excel_path}")
    else:
        print(f"Excel file already exists: {excel_path}")

    # Step 3: Read the Excel file
    print("Reading Excel file...")
    df = pd.read_excel(excel_path, engine="openpyxl")

    # Step 4: Clean column names
    df.columns = [col.strip() for col in df.columns]

    expected_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Step 5: Save raw data as CSV
    print("Saving raw dataset as CSV...")
    df.to_csv(raw_csv_path, index=False)

    # Step 6: Save first 100 rows as preview CSV
    df.head(100).to_csv(preview_csv_path, index=False)

    # Step 7: Print summary
    print("\nDone.")
    print(f"Raw CSV saved to: {raw_csv_path}")
    print(f"Preview CSV saved to: {preview_csv_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()