# Retail Analytics & Forecasting Dashboard

An end-to-end data analytics and forecasting project built with Python, pandas, scikit-learn, Plotly, and Streamlit.

## Project Goal
Build a retail analytics dashboard that:
- cleans transaction data
- creates KPIs
- forecasts daily revenue
- deploys an interactive dashboard

## Tech Stack
- Python
- pandas
- scikit-learn
- Plotly
- Streamlit
- joblib

## Project Structure
```text
retail-analytics-forecasting/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── download_data.py
│   ├── clean_data.py
│   └── train_model.py
├── requirements.txt
├── .gitignore
└── README.md