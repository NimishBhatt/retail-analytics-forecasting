# Model Report: Retail Analytics & Daily Revenue Forecasting

## 1. Objective

The objective of this project was to build a daily revenue forecasting model using cleaned retail transaction data.

The goal was not only to create a machine learning model, but to build a small end-to-end analytics product that combines:

- transaction data cleaning
- business KPI generation
- time-based aggregation
- forecasting
- dashboard visualization

## 2. Data Description

The project uses a retail transaction dataset containing invoice-level and product-level purchase records.

### Core variables
- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

### Business meaning
This data allows analysis at multiple levels:
- transaction level
- product level
- customer level
- country level
- day level

That makes it appropriate for both descriptive analytics and time-series forecasting.

## 3. Data Preparation

### Cleaning steps
The following preprocessing steps were applied:

- standardized column names
- converted dates and numeric fields into usable formats
- removed duplicate rows
- removed rows with missing critical values
- removed unusable transaction lines
- identified cancelled invoices using `InvoiceNo`
- created `sales_amount = Quantity * UnitPrice`

### Why this matters
Without cleaning:
- cancellations would distort revenue
- invalid rows would weaken model quality
- mixed data types would break transformations
- time-series analysis would learn from noisy signals

## 4. Aggregation Strategy

To forecast revenue, the transaction-level data was transformed into a **daily sales table**.

### Daily target variable
- `daily_revenue`

### Daily supporting variables
- `total_orders`
- `total_items_sold`
- `unique_customers`

This aggregation converts raw transactions into a time-series dataset with one row per day.

## 5. Features Used in Forecasting

The forecasting model used the following feature groups.

### A. Calendar features
- `day_of_week_num`
- `day_of_month`
- `month_num`
- `week_num`
- `year`
- `is_weekend`

These features help the model learn recurring demand patterns such as weekly or monthly seasonality.

### B. Lag features
- `lag_1`
- `lag_7`
- `lag_14`

These capture short-term temporal dependence by providing recent revenue history.

### C. Rolling features
- `rolling_mean_7`
- `rolling_mean_14`

These smooth out daily fluctuations and help the model understand recent momentum.

### D. Business activity features
- `total_orders`
- `total_items_sold`
- `unique_customers`

These directly connect revenue to transaction and customer behavior.

## 6. Modeling Approach

Two approaches were compared.

### Baseline model
A naive forecasting benchmark that predicts revenue using the previous day's revenue.

This gives a minimum performance threshold and helps evaluate whether the machine learning model adds real value.

### Random Forest Regressor
A supervised machine learning model trained using:
- calendar features
- lag features
- rolling averages
- business activity features

Random Forest was chosen because it is:
- easy to interpret
- strong on structured/tabular data
- robust for a portfolio-scale project

## 7. Train/Test Strategy

Because this is a time-series problem, the data was not randomly shuffled.

Instead:
- the first 80% of observations were used for training
- the last 20% were used for testing

This preserves time order and makes evaluation more realistic.

## 8. Evaluation Metrics

The models were evaluated using:

### MAE
**Mean Absolute Error**  
Measures the average absolute difference between predictions and actual values.

### RMSE
**Root Mean Squared Error**  
Measures error while penalizing larger mistakes more strongly.

Lower values are better for both metrics.

## 9. Model Results

### Baseline model
- **MAE:** 25,438.90
- **RMSE:** 33,581.24

### Random Forest model
- **MAE:** 6,656.15
- **RMSE:** 15,947.79

## 10. Interpretation of Results

The Random Forest model significantly outperformed the baseline on both error metrics.

This suggests that the engineered variables capture meaningful structure in the sales process.

The improvement indicates that daily revenue is influenced not only by yesterday's revenue, but also by:
- transaction volume
- customer activity
- short-term historical trends
- calendar-based demand patterns

## 11. Feature Importance

The most important variables in the model were:

1. `total_items_sold`
2. `total_orders`
3. `unique_customers`
4. `lag_1`
5. `lag_7`
6. `day_of_month`
7. `rolling_mean_7`
8. `lag_14`
9. `week_num`
10. `rolling_mean_14`

## 12. Significance of Key Variables

### total_items_sold
This was the strongest feature because revenue is fundamentally driven by how many items are sold. Higher sales volume usually leads directly to higher revenue.

### total_orders
This captures how many distinct purchase events happened in a day. More orders often indicate stronger overall demand.

### unique_customers
This reflects customer activity breadth. A larger active customer base on a given day is often associated with stronger revenue.

### lag_1, lag_7, lag_14
These variables show that recent revenue history matters. Retail activity often has short-term memory and repeated patterns.

### rolling_mean_7 and rolling_mean_14
These help the model recognize short-term trends while reducing daily volatility.

### day_of_month and week_num
These indicate that revenue also contains periodic calendar patterns.

## 13. Business Interpretation

The model is driven mainly by real business activity:
- how many items were sold
- how many orders were placed
- how many customers were active

This is useful because the model is not a black box in business terms.  
Its key drivers are understandable and explainable.

## 14. Limitations

This project is intentionally lightweight and designed for portfolio deployment, so it has limitations:

- no holiday or promotion variables were included
- no external business context was added
- Random Forest is not the most advanced forecasting method
- no retraining pipeline is automated
- no uncertainty intervals are shown

## 15. Future Improvements

Possible improvements include:
- adding XGBoost or LightGBM
- testing Prophet or other forecasting models
- including holiday and promotional effects
- forecasting by product category or country
- adding automated retraining and monitoring
- building a full CI/CD workflow

## 16. Conclusion

This project demonstrates a complete analytics workflow from raw retail transactions to deployed business dashboard.

The final result is not just a model, but a usable analytics application that combines:
- data preparation
- business intelligence
- forecasting
- deployment

The Random Forest model substantially improved forecasting performance over a naive baseline, and the most important variables aligned well with business intuition.