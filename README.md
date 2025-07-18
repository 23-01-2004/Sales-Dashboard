# Sales Analysis Dashboard

A **Sales Analysis Dashboard** built using **Dash (Plotly)** that allows users to visualize and analyze sales data, make forecasts using different models, and predict profits using machine learning. The dashboard provides various interactive charts and graphs to help users gain insights into business performance across multiple dimensions.

## Features

- **Interactive Filters**: 
  - Date Range
  - Countries
  - Products
  - Interested Plots
  - Forecast Model
  - 
- **Visualizations**:
  - Daily Revenue Trend: Tracks daily sales fluctuations of revenue of the company. 
  -  Monthly Revenue Trend: Tracks the monthly revenue fluctuations. 
  - Sales by Product and Country: Individual Bar Charts which shows Sales of Product of each country. 
  - Profit Margin by Product: A plot to understand which products are most/least profitable. 
  - Bubble Chart: A multi-metric view including Units Sold, Revenue and Profit Margin. 
- **Forecasting**:
  - Auto Regressive Integrated Moving Average (ARIMA): This plot is the best for linear trends.
  - Exponential Smoothing: This plot is ideal for seasonal or trend based forecasting.
  - Prophet: Prophet introduced by meta is designed for daily trends with seasonal effects. 
- **Machine Learning**:
  - Random Forest for Profit Prediction
  - Actual vs Predicted profit comparison with feature importance visualization.
- **Risk Analysis**: 
- Identify products with low margins or declining performance.
- Revenue Volatility: Coefficient of Variation is used to plot overall stability of products in every country.
- Concentration Risk: This metric along with individual pie charts are plotted to examine over-dependency on any particular product or any country.
- Profitability Risk: This metric highlights low margin products which could cause potential risks which highlights revneue fluctuations, unexpected exprenses and Economic Downturns. 
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
pip install -r sales_analysis/requirements.txt
```
### 4. Run the App

```bash
python src/app.py
```

**Using the Dashboard**
1. Filters: Select the time period, interested products, countries and analysis you want to display on the dashboard. Attached plots will display automatically. 

2. Forecasting:
 - Further Filters are given to forecast future values based on 3 target variable (Revenue, Units Sold, Profit)the user can choose from.
 - Forecasting is done via 3 different models
 - 
a. Auto-Regressive Integrated Moving Average Model (ARIMA).

b. Exponential Smoothing.

c. Prophet (Meta). 


3. Sales Insights: Bar Charts, Pie Plots, Grouped Bar Charts, Histograms and KDE Distributions are displayed for multiple combination of features to find overall revenue on each or multiple countries for each or multiple products. Revenue and Profit Margin of each product were also displayed so further improvements could be done on minor classes. 

4. Machine Learning: Get profit predictions using a trained Random Forest model and see the importance of different features.


