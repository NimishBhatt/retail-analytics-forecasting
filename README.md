# Retail Analytics & Forecasting Dashboard

An end-to-end data analytics and forecasting project built with Python, pandas, scikit-learn, Plotly, and Streamlit.

This project transforms raw retail transaction data into analytics-ready datasets, builds a daily revenue forecasting model, and presents the results through an interactive dashboard.

## Live App
[(https://retail-analytics-forecasting-nimishbhattde.streamlit.app/)]

## GitHub Repository
[(https://github.com/NimishBhatt/retail-analytics-forecasting)]

## Project Overview

The goal of this project is to demonstrate a complete data workflow:

- data ingestion
- data cleaning and validation
- feature engineering
- business analytics
- machine learning forecasting
- interactive dashboard deployment

The final dashboard allows users to monitor revenue trends, sales activity, customer behavior, and model performance.

## Dataset

This project uses the **Online Retail** dataset.

### Dataset summary
- transactional retail dataset
- includes invoice, product, customer, price, quantity, and country information
- suitable for sales analytics and time-based forecasting

### Main variables
- `InvoiceNo` – transaction ID
- `StockCode` – product code
- `Description` – product name
- `Quantity` – number of units sold
- `InvoiceDate` – date and time of purchase
- `UnitPrice` – price per unit
- `CustomerID` – customer identifier
- `Country` – customer country

## Project Structure

```text
retail-analytics-forecasting/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── model_report.md
├── models/
│   └── random_forest_daily_revenue.joblib
├── src/
│   ├── download_data.py
│   ├── clean_data.py
│   └── train_model.py
├── requirements.txt
├── .gitignore
└── README.md