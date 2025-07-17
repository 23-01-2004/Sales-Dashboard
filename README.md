# Sales Analysis Dashboard

A **Sales Analysis Dashboard** built using **Dash (Plotly)** that allows users to visualize and analyze sales data, make forecasts using different models, and predict profits using machine learning. The dashboard provides various interactive charts and graphs to help users gain insights into business performance across multiple dimensions.

## Features

- **Interactive Filters**: 
  - Date Range
  - Countries
  - Products
- **Visualizations**:
  - Daily and Monthly Revenue Trends
  - Sales by Product and Country
  - Profit Margin by Product
  - Bubble Chart for Units Sold, Revenue, and Profit
- **Forecasting**:
  - ARIMA, Exponential Smoothing, and Prophet for predicting future sales data
- **Machine Learning**:
  - Random Forest for Profit Prediction
  - Actual vs Predicted profit comparison with feature importance visualization
- **Risk Analysis**: Identify products with low margins or declining performance

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/23-01-2004/Sales-Dashboard.git
cd sales_analysis
```
### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
### 4. Run the App

```bash
python src/app.py
```

**Using the Dashboard**
1. Filters: Select the date range, country, and product(s) to filter the data.

2. Forecasting: Choose a forecast model (ARIMA, Exponential Smoothing, Prophet) to predict future sales data for the selected country and product.

3. Sales Insights: View charts displaying sales trends, revenue, and profit margins.

4. Machine Learning: Get profit predictions using a trained Random Forest model and see the importance of different features.


