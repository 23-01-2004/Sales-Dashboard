# Sales Performance & Business Intelligence (BI) Dashboard

An interactive **Sales Performance Dashboard** provides end-to-end analytics for a speciality product company operating across multi-country markets. Built using **Dash-Plotly** that delivers 
- Sales Insights, profitability analysis, and Risk Monitoring
- AI-Powered forecasts and Machine Learning (ML) modeling
- Visualizations of different types on sales data

The dashboard provides various interactive charts and graphs to help users gain insights into business performance across multiple dimensions.

---

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

  # Below is a overall brief about the Dashboard features :

 | **Component** | **Description** |
|---------------|----------------|
| Data Loading | Load and prepare data using the `load_and_clean_data()` function. |
| Dash App Initialization | Initialize a Dash app with a Bootstrap theme. |
| Sidebar | Create a sidebar with filters for date range, countries, products, plots, forecast models, and target variables. |
| | Date Range Picker | Allow users to select a date range for filtering data. |
| | Country Dropdown | Dropdown to select countries for filtering data. |
| | Product Dropdown | Dropdown to select products for filtering data. |
| | Plot Selector | Dropdown to select which plots to display. |
| | Forecast Model Selector | Dropdown to select the forecasting model (ARIMA, Exponential Smoothing, Prophet). |
| | Target Variable Selector | Dropdown to select the target variable for forecasting (Units Sold, Revenue, Profit). |
| Main Content Area | Define the main content area for displaying plots and analysis results. |
| Layout | Structure the layout of the app with a sidebar and main content area. |
| Summary Stats | Generate summary statistics for total revenue, cost, profit, and average profit margin. |
| | Summary Cards | Create cards to display summary statistics. |
| Plot Generation | Generate various plots based on user selection. |
| | Daily Revenue Trend | Line plot showing daily revenue fluctuations over time. |
| | Monthly Revenue Trend | Line plot showing aggregated monthly revenue trend. |
| | Revenue by Month | Bar plot showing average revenue per month across all years. |
| | Country-wise Monthly Revenue | Line plot showing monthly revenue by country. |
| | Sales by Country | Bar plot showing total revenue generated per country. |
| | Sales by Product | Bar plot showing total revenue per product type. |
| | Profit Margin by Product | Bar plot showing average profit margin per product. |
| | Grouped Sales by Product & Country | Grouped bar plot visualizing sales per product in each country. |
| | Bubble Chart | Bubble chart showing profit margin vs units sold vs revenue. |
| Forecasting Analysis | Integrate forecasting analysis using selected models and target variables. |
| | Forecast Plot | Plot showing actual data in blue and forecasted data in red. |
| | Forecast Table | Table displaying forecasted data. |
| Risk Analysis | Generate a risk analysis section. |
| Callback Function | Define a callback function to update the dashboard based on user inputs and selections. |
| | Filter Data | Filter data based on selected date range, countries, and products. |
| | Generate Plots | Generate selected plots based on user input. |
| | Forecasting | Perform forecasting using the selected model and target variable, and display the results. |
| Run App | Run the Dash app with debug mode enabled. |


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
 - Forecasting is done via 3 different models. 
   
a. Auto-Regressive Integrated Moving Average Model (ARIMA).

b. Exponential Smoothing.

c. Prophet (Meta). 


3. Sales Insights: Bar Charts, Pie Plots, Grouped Bar Charts, Histograms and KDE Distributions are displayed for multiple combination of features to find overall revenue on each or multiple countries for each or multiple products. Revenue and Profit Margin of each product were also displayed so further improvements could be done on minor classes. 

4. Machine Learning: Get profit predictions using a trained Random Forest model and see the importance of different features.


