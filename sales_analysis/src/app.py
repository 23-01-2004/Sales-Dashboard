import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import base64
import io
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ML and forecasting imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# ==================== DATA LOADING AND CLEANING ====================
def load_and_clean_data(df):
    """
    Load and clean the uploaded CSV data
    """
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Check required columns
        required_columns = ['Date', 'Country', 'Product', 'Units Sold', 'Revenue', 'Profit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to map common column name variations
            column_mapping = {
                'date': 'Date',
                'country': 'Country',
                'product': 'Product',
                'units_sold': 'Units Sold',
                'units sold': 'Units Sold',
                'revenue': 'Revenue',
                'profit': 'Profit',
                'sales': 'Revenue',
                'quantity': 'Units Sold',
                'qty': 'Units Sold'
            }
            
            # Apply mapping
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col in required_columns:
                    df = df.rename(columns={old_col: new_col})
        
        # Check again for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # # Clean and convert data types - handle Excel serial dates
        # try:
        #     # First try converting directly to datetime
        #     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        #     # For any remaining NaT values, try converting from Excel serial numbers
        #     mask = df['Date'].isna()
        #     if mask.any():
        #         # Excel's date system starts from 1900-01-01 (with known bugs)
        #         # Excel serial date 1 = 1900-01-01, but Excel incorrectly treats 1900 as a leap year
        #         excel_epoch = pd.Timestamp('1899-12-30')  # Correct Excel epoch
                
        #         # Convert the numeric values to timedelta and add to epoch
        #         numeric_dates = pd.to_numeric(df.loc[mask, 'Date'], errors='coerce')
        #         df.loc[mask, 'Date'] = excel_epoch + pd.to_timedelta(numeric_dates, unit='D')
                
        # except Exception as e:
        #     # Final fallback - try treating all as Excel serial numbers
        #     try:
        #         excel_epoch = pd.Timestamp('1899-12-30')
        #         numeric_dates = pd.to_numeric(df['Date'], errors='coerce')
        #         df['Date'] = excel_epoch + pd.to_timedelta(numeric_dates, unit='D')
        #     except:
        #         # If all fails, try a more flexible approach
        #         df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
        def convert_excel_dates(excel_date):
            return pd.to_datetime('1899-12-30') + pd.to_timedelta(excel_date, unit = 'D')
        
        # Convert other numeric columns
        df['Date']       = df['Date'].apply(convert_excel_dates)
        df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')
        df['Revenue']    = pd.to_numeric(df['Revenue'], errors='coerce')
        df['Profit']      = pd.to_numeric(df['Profit'], errors='coerce')
        
        # Remove rows with invalid dates or negative values
        #df = df.dropna(subset=['Date'])
        df = df[df['Units Sold'] >= 0]
        df = df[df['Revenue'] >= 0]
        
        # Remove any dates that are clearly wrong (before 1900 or after 2100)
        #df = df[(df['Date'] >= '1900-01-01') & (df['Date'] <= '2100-12-31')]
        
        # Calculate profit margin
        df['Profit Margin (%)'] = (df['Profit'] / df['Revenue'] * 100).round(2)
        df['Profit Margin (%)'] = df['Profit Margin (%)'].fillna(0)
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        raise Exception(f"Error in data cleaning: {str(e)}")

# ==================== DATA PROCESSING FUNCTIONS ====================

def filter_data(df, start_date, end_date, countries, products):
    """Filter data based on date range, countries, and products"""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Filter by date range
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[
            (filtered_df['Date'] >= start_date) & 
            (filtered_df['Date'] <= end_date)
        ]
    
    # Filter by countries
    if countries and 'Country' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    
    # Filter by products
    if products and 'Product' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Product'].isin(products)]
    
    return filtered_df

def get_summary_stats(df):
    """Generate summary statistics"""
    if df.empty:
        return {
            "current": {"Total Revenue": 0, "Total Profit": 0, "Total Units Sold": 0, "Average Profit Margin (%)": 0},
            "delta_percent": {},
            "delta_color": {}
        }
    
    # Current period stats
    current_stats = {
        "Total Revenue": df['Revenue'].sum(),
        "Total Profit": df['Profit'].sum(),
        "Total Units Sold": df['Units Sold'].sum(),
        "Average Profit Margin (%)": df['Profit Margin (%)'].mean()
    }
    
    # Calculate deltas (for demo, using random values)
    delta_percent = {
        "Total Revenue Δ (%)": np.random.uniform(-5, 15),
        "Total Profit Δ (%)": np.random.uniform(-10, 20),
        "Total Units Sold Δ (%)": np.random.uniform(-3, 12),
        "Avg Profit Margin Δ (%)": np.random.uniform(-2, 8)
    }
    
    delta_color = {
        k: "green" if v > 0 else "red" for k, v in delta_percent.items()
    }
    
    return {
        "current": current_stats,
        "delta_percent": delta_percent,
        "delta_color": delta_color
    }

def get_sales_by_country(df):
    """Get sales data by country"""
    if df.empty:
        return pd.DataFrame(columns=['Country', 'Revenue'])
    return df.groupby('Country')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False)

def get_sales_by_product(df):
    """Get sales data by product"""
    if df.empty:
        return pd.DataFrame(columns=['Product', 'Revenue'])
    return df.groupby('Product')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False)

def get_profit_margin_by_product(df):
    """Get profit margin by product"""
    if df.empty:
        return pd.DataFrame(columns=['Product', 'Profit Margin (%)'])
    return df.groupby('Product')['Profit Margin (%)'].mean().reset_index().sort_values('Profit Margin (%)', ascending=False)

def get_grouped_bar_data(df):
    """Get grouped bar data for products by country"""
    if df.empty:
        return pd.DataFrame()
    return df.groupby(['Country', 'Product'])['Revenue'].sum().unstack(fill_value=0)

def get_bubble_data(df):
    """Get bubble chart data"""
    if df.empty:
        return pd.DataFrame(columns=['Product', 'Units_Sold', 'Revenue', 'Profit_Margin'])
    
    bubble_data = df.groupby('Product').agg({
        'Units Sold': 'sum',
        'Revenue': 'sum',
        'Profit Margin (%)': 'mean'
    }).reset_index()
    
    bubble_data = bubble_data.rename(columns={
        'Units Sold': 'Units_Sold',
        'Profit Margin (%)': 'Profit_Margin'
    })
    
    return bubble_data

def get_monthly_revenue_trend(df):
    """Get monthly revenue trend"""
    if df.empty:
        return pd.DataFrame(columns=['Date', 'Revenue'])
    
    try:
        # Ensure proper datetime format and set as index
        monthly_data = df.copy()
        monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])
        monthly_data = monthly_data.set_index('Date')
        
        # Resample to monthly frequency and sum revenue
        monthly_data = monthly_data['Revenue'].resample('M').sum().reset_index()
        
        return monthly_data
        
    except Exception as e:
        print(f"Error in monthly trend: {str(e)}")
        return pd.DataFrame(columns=['Date', 'Revenue'])

def get_revenue_by_month(df):
    """Get average revenue by month across all years"""
    if df.empty:
        return pd.DataFrame(columns=['Month', 'Revenue'])
    
    df['Month'] = df['Date'].dt.month_name()
    monthly_avg = df.groupby('Month')['Revenue'].mean().reset_index()
    
    # Order months properly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=month_order, ordered=True)
    monthly_avg = monthly_avg.sort_values('Month')
    
    return monthly_avg

def get_country_time_trend(df):
    """Get country-wise monthly revenue trend"""
    if df.empty:
        return pd.DataFrame(columns=['Date', 'Country', 'Revenue'])
    
    monthly_country = df.set_index('Date').groupby('Country').resample('M')['Revenue'].sum().reset_index()
    return monthly_country

# ==================== FORECASTING FUNCTIONS ====================

def prepare_data(df, target_column):
    """Prepare data for forecasting"""
    if df.empty:
        return pd.DataFrame()
    
    # Group by date and sum the target column
    daily_data = df.groupby('Date')[target_column].sum().reset_index()
    daily_data = daily_data.sort_values('Date')
    
    return daily_data

def arima_forecast(df, target_column, periods=30):
    """ARIMA forecasting"""
    try:
        data = prepare_data(df, target_column)
        if len(data) < 10:
            return None, pd.Series([0] * periods)
        
        # Simple ARIMA(1,1,1) model
        model = ARIMA(data[target_column], order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=periods)
        
        return fitted_model, forecast
        
    except Exception:
        # Fallback to simple moving average
        data = prepare_data(df, target_column)
        if len(data) < 3:
            return None, pd.Series([data[target_column].mean()] * periods)
        
        # Simple moving average forecast
        ma_value = data[target_column].rolling(window=min(7, len(data))).mean().iloc[-1]
        forecast = pd.Series([ma_value] * periods)
        
        return None, forecast

def exponential_smoothing_forecast(df, target_column, periods=30):
    """Exponential smoothing forecasting"""
    try:
        data = prepare_data(df, target_column)
        if len(data) < 10:
            return None, pd.Series([0] * periods)
        
        # Simple exponential smoothing
        model = ExponentialSmoothing(data[target_column], trend=None, seasonal=None)
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=periods)
        
        return fitted_model, forecast
        
    except Exception:
        # Fallback
        data = prepare_data(df, target_column)
        if len(data) < 3:
            return None, pd.Series([data[target_column].mean()] * periods)
        
        # Simple exponential smoothing fallback
        alpha = 0.3
        last_value = data[target_column].iloc[-1]
        forecast = pd.Series([last_value] * periods)
        
        return None, forecast

def prophet_forecast(df, target_column, periods=30):
    """Prophet forecasting (simplified version)"""
    try:
        data = prepare_data(df, target_column)
        if len(data) < 10:
            return None, pd.DataFrame({'yhat': [0] * periods})
        
        # Simple linear trend forecast (Prophet alternative)
        data['Days'] = range(len(data))
        
        # Linear regression for trend
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(data[['Days']], data[target_column])
        
        # Forecast
        future_days = range(len(data), len(data) + periods)
        forecast = model.predict([[day] for day in future_days])
        
        forecast_df = pd.DataFrame({'yhat': forecast})
        
        return model, forecast_df
        
    except Exception:
        # Fallback
        data = prepare_data(df, target_column)
        if len(data) < 3:
            mean_value = data[target_column].mean() if not data.empty else 0
            return None, pd.DataFrame({'yhat': [mean_value] * periods})
        
        # Simple trend forecast
        trend = (data[target_column].iloc[-1] - data[target_column].iloc[0]) / len(data)
        last_value = data[target_column].iloc[-1]
        forecast = [last_value + trend * i for i in range(1, periods + 1)]
        
        return None, pd.DataFrame({'yhat': forecast})

# ==================== MACHINE LEARNING FUNCTIONS ====================

def train_profit_predictor(df):
    """Train ML model to predict profit"""
    try:
        if df.empty or len(df) < 10:
            return None, pd.DataFrame(), None
        
        # Prepare features
        ml_df = df.copy()
        
        # Create features
        ml_df['Month'] = ml_df['Date'].dt.month
        ml_df['Year'] = ml_df['Date'].dt.year
        ml_df['DayOfWeek'] = ml_df['Date'].dt.dayofweek
        ml_df['Quarter'] = ml_df['Date'].dt.quarter
        
        # Encode categorical variables
        le_country = LabelEncoder()
        le_product = LabelEncoder()
        
        ml_df['Country_encoded'] = le_country.fit_transform(ml_df['Country'])
        ml_df['Product_encoded'] = le_product.fit_transform(ml_df['Product'])
        
        # Features and target
        features = ['Units Sold', 'Revenue', 'Month', 'Year', 'DayOfWeek', 'Quarter', 'Country_encoded', 'Product_encoded']
        X = ml_df[features]
        y = ml_df['Profit']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Create prediction DataFrame
        prediction_df = pd.DataFrame({
            'Actual_Profit': y_test.values,
            'Predicted_Profit': y_pred,
            'Difference': y_test.values - y_pred
        })
        prediction_df['Accuracy'] = (100 - abs(prediction_df['Difference'] / prediction_df['Actual_Profit'] * 100)).round(2)
        
        return model, prediction_df.head(10), None
        
    except Exception as e:
        return None, pd.DataFrame(), None

# ==================== RISK ANALYSIS ====================

def generate_risk_section(df):
    """Generate risk analysis section"""
    if df.empty:
        return html.Div("No data available for risk analysis.")
    
    try:
        # Calculate various risk metrics
        revenue_volatility = df['Revenue'].std() / df['Revenue'].mean() * 100
        profit_volatility = df['Profit'].std() / df['Profit'].mean() * 100
        
        # Country concentration risk
        country_revenue = df.groupby('Country')['Revenue'].sum()
        country_concentration = (country_revenue.max() / country_revenue.sum() * 100)
        
        # Product concentration risk
        product_revenue = df.groupby('Product')['Revenue'].sum()
        product_concentration = (product_revenue.max() / product_revenue.sum() * 100)
        
        # Risk indicators
        risks = []
        
        if revenue_volatility > 50:
            risks.append("High revenue volatility detected")
        if profit_volatility > 60:
            risks.append("High profit volatility detected")
        if country_concentration > 60:
            risks.append("High country concentration risk")
        if product_concentration > 60:
            risks.append("High product concentration risk")
        
        if not risks:
            risks.append("No significant risks detected")
        
        risk_items = [html.Li(risk) for risk in risks]
        
        return html.Div([
            html.H4("Risk Analysis", style={'marginTop': '30px', 'color': '#d32f2f'}),
            html.Ul(risk_items),
            html.P(f"Revenue Volatility: {revenue_volatility:.1f}%"),
            html.P(f"Profit Volatility: {profit_volatility:.1f}%"),
            html.P(f"Country Concentration: {country_concentration:.1f}%"),
            html.P(f"Product Concentration: {product_concentration:.1f}%")
        ])
        
    except Exception:
        return html.Div("Error generating risk analysis.")

# ==================== PLOT OPTIONS ====================

PLOT_OPTIONS = {
    "revenue-trend": {
        "label": "Daily Revenue Trend",
        "description": "Displays daily revenue fluctuations over time. This chart shows how your business's revenue has changed day-to-day, helping you identify patterns or anomalies in your sales."
    },
    "monthly-revenue-trend": {
        "label": "Monthly Revenue Trend",
        "description": "Shows aggregated monthly revenue trend. This chart gives you a higher-level view of your business's revenue performance over time."
    },
    "revenue-by-month": {
        "label": "Revenue by Month",
        "description": "Average revenue per month across all years. This chart provides a summary view of your business's monthly revenue performance."
    },
    "country-time-trend": {
        "label": "Country-wise Monthly Revenue",
        "description": "Shows how your business's revenue has changed in each country over time, helping you identify regional trends."
    },
    "sales-by-country": {
        "label": "Sales by Country",
        "description": "Bar chart showing total revenue generated per country. This chart gives you a snapshot of revenue performance by region."
    },
    "sales-by-product": {
        "label": "Sales by Product",
        "description": "Bar chart showing total revenue per product type. This chart shows which products are driving sales."
    },
    "profit-margin-by-product": {
        "label": "Profit Margin by Product",
        "description": "Bar chart showing average profit margin (%) per product. This chart shows which products are most profitable."
    },
    "grouped-bar-sales": {
        "label": "Grouped Sales by Product & Country",
        "description": "Visualizes sales per product in each country. This chart shows product performance across different markets."
    },
    "bubble-chart": {
        "label": "Bubble Chart: Profit vs Units Sold vs Revenue",
        "description": "Multi-metric view of products showing relationships between units sold, revenue, and profit margin."
    }
}

FORECASTING_OPTIONS = {
    "arima": {"label": "ARIMA"},
    "exponential_smoothing": {"label": "Exponential Smoothing"},
    "prophet": {"label": "Prophet (Linear Trend)"}
}

# ==================== LAYOUT ====================

# 2. Fix the sidebar layout to be more responsive
sidebar = html.Div([
    html.H4("Filters", className="display-6", style={'color': '#68A3D0'}),
    html.Hr(),
    
    # Upload CSV
    html.Div([
        html.H5("Upload CSV File"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'marginBottom': '10px'
            }
        ),
    ]),
    html.Hr(),
    
    # Conditional rendering for filters
    html.Div(id='filter-section', children=[
        html.Label("Select Date Range", style={'color':'#788E9C'}),
        dcc.DatePickerRange(
            id='date-range',
            start_date=None,
            end_date=None,
            disabled=True,  # Initially disabled
            style={'background-color':'#DCC5B2'}
        ),
        html.Br(), html.Br(),
        
        html.Label("Select Countries", style={'color':'#788E9C'}),
        dcc.Dropdown(
            id='country-dropdown',
            options=[],
            value=[],
            multi=True,
            placeholder="Upload data first...",
            style={'background-color':"#C7DDF2", 'maxHeight': '150px', 'overflowY': 'scroll'}
        ),
        html.Br(),
        
        html.Label("Select Products", style={'color':'#788E9C'}),
        dcc.Dropdown(
            id='product-dropdown',
            options=[],
            value=[],
            multi=True,
            placeholder="Upload data first...",
            style={'background-color':'#C7DDF2', 'maxHeight': '150px', 'overflowY': 'scroll'}
        ),
        html.Br(),
        
        html.Label("Select Plots to Display", style={'color':"#788E9C"}),
        dcc.Dropdown(
            id='plot-selector',
            options=[{"label": info["label"], "value": key} for key, info in PLOT_OPTIONS.items()],
            value=["revenue-trend", "sales-by-country", "profit-margin-by-product"],
            multi=True,
            style={'background-color':"#C7DDF2", 'maxHeight':'150px', 'overflowY':'scroll'},
        ),
        html.Br(),
        
        html.Label("Select Forecast Model:", style={'color': "#9FA6DD"}),
        dcc.Dropdown(
            id='forecast-selector',
            options=[{"label": info["label"], "value": key} for key, info in FORECASTING_OPTIONS.items()],
            value="arima",
            multi=False,
            style={'background-color': "#B7BFEC"}
        ),
        html.Br(),
        
        html.Label("Select Target Column:", style={'color': "#9FA6DD"}),
        dcc.Dropdown(
            id='target-selector',
            options=[
                {"label": "Units Sold", "value": "Units Sold"},
                {"label": "Revenue", "value": "Revenue"},
                {"label": "Profit", "value": "Profit"}
            ],
            value="Profit",
            multi=False,
            style={'background-color': "#B7BFEC"}
        ),
        
        html.Label("Select Forecast Country:", style={'color': "#9FA6DD", 'marginTop': '10px'}),
        dcc.Dropdown(
            id='forecast-country-dropdown',
            options=[],
            value=None,
            multi=False,
            placeholder="Upload data first...",
            style={'background-color': "#B7BFEC"}
        ),
        
        html.Label("Select Forecast Product:", style={'color': "#9FA6DD", 'marginTop': '10px'}),
        dcc.Dropdown(
            id='forecast-product-dropdown',
            options=[],
            value=None,
            multi=False,
            placeholder="Upload data first...",
            style={'background-color': "#B7BFEC"}
        ),
        
        html.Div(dbc.Button("Apply Filters", id='submit-button', color='primary', className='mt-3', 
                           style={'background-color':"#18576A", 'margin-left':'30%'}))
    ])
], style={
    'padding': '20px', 
    'background-color': "#D9E6F5", 
    'border-radius': '20px', 
    'width': '100%',  # Make it responsive
    'height': 'auto',  # Remove fixed height
    'min-height': '100vh'  # Ensure it takes full viewport height
})

# Main layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, width=3, style={'padding-right': '0px'}),
        dbc.Col([
            html.H1("Sales Performance Dashboard", 
                   style={"textAlign": "center", 'color':"#0C436D", 
                          'margin-bottom':'30px','margin-top':'20px'}),
            html.Div(id='summary-stats'),
            html.Div(id='selected-plots'),
            html.Div(id='risk-analysis-section'),
            html.Div(id='forecast-section'),
            html.Div(id='ml-section'),
            html.Div(id='upload-status', 
                    style={'margin-top': '20px', 'font-size': '16px', 'color': 'green'})
        ], width=9, style={'padding-left': '20px'})
    ], style={'margin': '0px'})
], fluid=True, style={'background-color':"#F1FAFE", 'color':"#5D5595", 'padding': '0px'})

# ==================== UTILITY FUNCTIONS ====================
# Add a helper function to show/hide filters based on data availability
def create_initial_message():
    """Create initial message when no data is uploaded"""
    return html.Div([
        html.Div([
            html.H3("📊 Welcome to Sales Performance Dashboard", 
                   style={'color': '#0C436D', 'textAlign': 'center', 'marginBottom': '30px'}),
            html.Div([
                html.H5("To get started:", style={'color': '#0C436D', 'marginBottom': '20px'}),
                html.Ul([
                    html.Li("Upload a CSV file with your sales data"),
                    html.Li("Required columns: Date, Country, Product, Units Sold, Revenue, Profit"),
                    html.Li("Select your desired filters and visualizations"),
                    html.Li("Click 'Apply Filters' to generate insights")
                ], style={'color': '#5D5595', 'fontSize': '16px'})
            ], style={'textAlign': 'left', 'maxWidth': '600px', 'margin': '0 auto'})
        ], style={
            'background-color': 'white',
            'padding': '40px',
            'border-radius': '10px',
            'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
            'margin': '50px auto',
            'maxWidth': '800px'
        })
    ])

def apply_bar_colors(fig, df, value_col):
    """Apply colors to bar chart based on max/min values"""
    if df.empty:
        return fig
        
    max_val = df[value_col].max()
    min_val = df[value_col].min()
    colors = ['lightblue'] * len(df)
    
    max_idx = df[df[value_col] == max_val].index[0]
    min_idx = df[df[value_col] == min_val].index[0]
    
    colors[max_idx] = 'lightgreen'
    colors[min_idx] = 'lightcoral'
    
    fig.update_traces(marker_color=colors)
    return fig

def generate_summary_cards(stats):
    """Generate summary statistic cards"""
    cards = []
    for k, v in stats["current"].items():
        delta_key = f"{k} Δ (%)" if k != "Average Profit Margin (%)" else "Avg Profit Margin Δ (%)"
        delta_val = stats["delta_percent"].get(delta_key, 0)
        color = stats["delta_color"].get(delta_key, "black")
        
        # Format values
        if "Revenue" in k or "Profit" in k:
            formatted_val = f"${v:,.2f}"
        elif "Margin" in k:
            formatted_val = f"{v:.1f}%"
        else:
            formatted_val = f"{v:,.0f}"
            
        cards.append(html.Div([
            html.H5(k),
            html.P(formatted_val),
            html.Span(f"{delta_val:.1f}% ", style={"color": color})
        ], style={'border': '1px solid #ccc', 'padding': '10px', 'width': '18%', 'textAlign': 'center'}))
    
    return html.Div(cards, style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'})

# ==================== CALLBACKS ====================

@app.callback(
    [
        Output('country-dropdown', 'options'),
        Output('country-dropdown', 'value'),
        Output('product-dropdown', 'options'),
        Output('product-dropdown', 'value'),
        Output('forecast-country-dropdown', 'options'),
        Output('forecast-country-dropdown', 'value'),
        Output('forecast-product-dropdown', 'options'),
        Output('forecast-product-dropdown', 'value'),
        Output('date-range', 'start_date'),
        Output('date-range', 'end_date'),
        Output('date-range', 'disabled'),  # Add this to disable when no data
    ],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_options(uploaded_contents, filename):
    """Update dropdown options based on uploaded data"""
    if uploaded_contents is None:
        # Return empty options and disable date range when no file uploaded
        return ([], [], [], [], [], None, [], None, 
                None, None, True)  # disabled=True
    
    try:
        # Parse uploaded file
        content_type, content_string = uploaded_contents.split(',')
        decoded = base64.b64decode(content_string)
        df_uploaded = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Clean data
        processed_df = load_and_clean_data(df_uploaded)
        
        # Update options
        country_options = [{'label': c, 'value': c} for c in sorted(processed_df['Country'].unique())]
        product_options = [{'label': p, 'value': p} for p in sorted(processed_df['Product'].unique())]
        
        # Update date range with actual data range
        min_date = processed_df['Date'].min()
        max_date = processed_df['Date'].max()

        # DEBUG: Print date range values
        print(f"=== DEBUG: DATE RANGE ===")
        print(f"Min date: {min_date} (type: {type(min_date)})")
        print(f"Max date: {max_date} (type: {type(max_date)})")
        
        # Convert to date strings if they're timestamps
        if hasattr(min_date, 'date'):
            min_date_str = min_date.date()
            max_date_str = max_date.date()
        else:
            min_date_str = min_date
            max_date_str = max_date
            
        print(f"Min date string: {min_date_str}")
        print(f"Max date string: {max_date_str}")
        
        return (country_options, list(processed_df['Country'].unique()), 
                product_options, list(processed_df['Product'].unique()),
                country_options, country_options[0]['value'] if country_options else None,
                product_options, product_options[0]['value'] if product_options else None,
                min_date, max_date, False)  # disabled=False
        
    except Exception as e:
        return ([], [], [], [], [], None, [], None, None, None, True)  # disabled=True on error
    
@app.callback(
    [
        Output('summary-stats', 'children'),
        Output('selected-plots', 'children'),
        Output('forecast-section', 'children'),
        Output("risk-analysis-section", "children"),
        Output('ml-section', 'children'),
        Output('upload-status', 'children')
    ],
    [
        Input('submit-button', 'n_clicks'),
        Input('upload-data', 'contents')
    ],
    [
        State('upload-data', 'filename'),
        State('date-range', 'start_date'),
        State('date-range', 'end_date'),
        State('country-dropdown', 'value'),
        State('product-dropdown', 'value'),
        State('plot-selector', 'value'),
        State('forecast-selector', 'value'),
        State('target-selector', 'value'),
        State('forecast-country-dropdown', 'value'),
        State('forecast-product-dropdown', 'value')
    ]
)

def update_dashboard(n_clicks, uploaded_contents, filename, start_date, end_date, countries, products, selected_plots,
                     fc_model, target, fc_country, fc_product):
    """Main callback to update dashboard"""
    
    # Check for uploaded file
    if uploaded_contents is None:
        initial_message = create_initial_message()
        return [initial_message, None, None, None, None, ""]
    
    try:
        # Process uploaded file
        content_type, content_string = uploaded_contents.split(',')
        decoded = base64.b64decode(content_string)
        df_uploaded = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Clean data
        processed_df = load_and_clean_data(df_uploaded)
        
        # Filter data based on inputs
        filtered_df = filter_data(processed_df, start_date, end_date, countries, products)
        
        # Generate summary stats
        stats = get_summary_stats(filtered_df)
        summary_cards = generate_summary_cards(stats)
        
        # Generate selected plots
        plot_components = []
        
        for plot_type in selected_plots:
            if plot_type == "revenue-trend":
                trend_data = filtered_df.groupby('Date')['Revenue'].sum().reset_index()
                fig = px.line(trend_data, x='Date', y='Revenue', 
                              title="Daily Revenue Trend", 
                              template="plotly_white")
                fig.update_layout(height=400)
                plot_components.append(dcc.Graph(figure=fig))
                
            elif plot_type == "monthly-revenue-trend":
                monthly_data = get_monthly_revenue_trend(filtered_df)
                if not monthly_data.empty:
                    # Format the date for better x-axis labels
                    monthly_data['Month_Year'] = monthly_data['Date'].dt.strftime('%b %Y')
                    
                    fig = px.line(monthly_data, 
                                x='Month_Year', 
                                y='Revenue', 
                                title="Monthly Revenue Trend",
                                template="plotly_white",
                                markers=True)
                    
                    # Improve layout
                    fig.update_layout(
                        height=400,
                        xaxis_title="Month",
                        yaxis_title="Revenue",
                        xaxis={'type': 'category', 'tickangle': 45}
                    )
                    
                    # Add hover information
                    fig.update_traces(
                        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>"
                    )
                    
                    plot_components.append(dcc.Graph(figure=fig))
                else:
                    plot_components.append(html.Div("No monthly revenue data available for the selected filters."))
                
            elif plot_type == "revenue-by-month":
                monthly_avg = get_revenue_by_month(filtered_df)
                fig = px.bar(monthly_avg, x='Month', y='Revenue', 
                             title="Average Revenue by Month",
                             template="plotly_white")
                fig.update_layout(height=400)
                plot_components.append(dcc.Graph(figure=fig))
                
            elif plot_type == "country-time-trend":
                country_trend = get_country_time_trend(filtered_df)
                if not country_trend.empty:
                    fig = px.line(country_trend, x='Date', y='Revenue', color='Country',
                                 title="Country-wise Monthly Revenue",
                                 template="plotly_white")
                    fig.update_layout(height=400)
                    plot_components.append(dcc.Graph(figure=fig))
                    
            elif plot_type == "sales-by-country":
                country_sales = get_sales_by_country(filtered_df)
                fig = px.bar(country_sales, x='Country', y='Revenue',
                             title="Sales by Country",
                             template="plotly_white")
                fig = apply_bar_colors(fig, country_sales, 'Revenue')
                fig.update_layout(height=400)
                plot_components.append(dcc.Graph(figure=fig))
                
            elif plot_type == "sales-by-product":
                product_sales = get_sales_by_product(filtered_df)
                fig = px.bar(product_sales, x='Product', y='Revenue',
                             title="Sales by Product",
                             template="plotly_white")
                fig = apply_bar_colors(fig, product_sales, 'Revenue')
                fig.update_layout(height=400)
                plot_components.append(dcc.Graph(figure=fig))
                
            elif plot_type == "profit-margin-by-product":
                profit_margin = get_profit_margin_by_product(filtered_df)
                fig = px.bar(profit_margin, x='Product', y='Profit Margin (%)',
                             title="Profit Margin by Product",
                             template="plotly_white")
                fig = apply_bar_colors(fig, profit_margin, 'Profit Margin (%)')
                fig.update_layout(height=400)
                plot_components.append(dcc.Graph(figure=fig))
                
            elif plot_type == "grouped-bar-sales":
                grouped_data = get_grouped_bar_data(filtered_df)
                if not grouped_data.empty:
                    fig = px.bar(grouped_data, barmode='group',
                                title="Grouped Sales by Product & Country",
                                template="plotly_white")
                    fig.update_layout(height=400)
                    plot_components.append(dcc.Graph(figure=fig))
                    
            elif plot_type == "bubble-chart":
                bubble_data = get_bubble_data(filtered_df)
                if not bubble_data.empty:
                    fig = px.scatter(bubble_data, x="Units_Sold", y="Revenue",
                                      size="Profit_Margin", color="Product",
                                      hover_name="Product", 
                                      title="Bubble Chart: Profit vs Units Sold vs Revenue",
                                      template="plotly_white")
                    fig.update_layout(height=500)
                    plot_components.append(dcc.Graph(figure=fig))
        
        # Generate forecast section
        forecast_components = []
        
        if fc_model and target:
            # Filter data for forecast if country/product selected
            forecast_df = filtered_df.copy()
            if fc_country:
                forecast_df = forecast_df[forecast_df['Country'] == fc_country]
            if fc_product:
                forecast_df = forecast_df[forecast_df['Product'] == fc_product]
                
            if not forecast_df.empty:
                forecast_components.append(html.H3("Forecasting", style={'marginTop': '30px', 'color': '#0C436D'}))
                
                # Get forecast based on selected model
                if fc_model == "arima":
                    model, forecast = arima_forecast(forecast_df, target)
                    model_name = "ARIMA"
                elif fc_model == "exponential_smoothing":
                    model, forecast = exponential_smoothing_forecast(forecast_df, target)
                    model_name = "Exponential Smoothing"
                else:  # prophet
                    model, forecast = prophet_forecast(forecast_df, target)
                    model_name = "Linear Trend"
                
                # Create forecast plot
                historical = prepare_data(forecast_df, target)
                last_date = historical['Date'].max()
                
                # Create future dates
                future_dates = [last_date + timedelta(days=i) for i in range(1, len(forecast)+1)]
                
                # Create figure
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=historical['Date'],
                    y=historical[target],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='green', dash='dash')
                ))
                
                # Add confidence interval (simplified)
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast * 1.1,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast * 0.9,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)',
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f"{model_name} Forecast for {target}",
                    xaxis_title="Date",
                    yaxis_title=target,
                    template="plotly_white",
                    height=500
                )
                
                forecast_components.append(dcc.Graph(figure=fig))
                
                # Add forecast values table
                forecast_table = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': forecast
                })
                
                forecast_components.append(html.H5("Forecast Values"))
                forecast_components.append(dash_table.DataTable(
                    data=forecast_table.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in forecast_table.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    page_size=10
                ))
        
        # Generate ML section
        ml_components = []
        if not filtered_df.empty:
            ml_components.append(html.H3("Machine Learning Insights", style={'marginTop': '30px', 'color': '#0C436D'}))
            
            # Train profit predictor
            model, predictions, error = train_profit_predictor(filtered_df)
            
            if model is not None and not predictions.empty:
                # Feature importance plot
                importances = model.feature_importances_
                features = ['Units Sold', 'Revenue', 'Month', 'Year', 'DayOfWeek', 'Quarter', 'Country', 'Product']
                
                fig = px.bar(x=features, y=importances, 
                             title="Feature Importance for Profit Prediction",
                             labels={'x': 'Features', 'y': 'Importance'},
                             template="plotly_white")
                fig.update_layout(height=400)
                ml_components.append(dcc.Graph(figure=fig))
                
                # Prediction results table
                ml_components.append(html.H5("Sample Predictions vs Actual Values"))
                ml_components.append(dash_table.DataTable(
                    data=predictions.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in predictions.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    page_size=10
                ))
        
        # Generate risk analysis
        risk_section = generate_risk_section(filtered_df)
        
        return [
            summary_cards,
            html.Div(plot_components),
            html.Div(forecast_components),
            risk_section,
            html.Div(ml_components),
            f"Data loaded successfully: {filename}"
        ]
        
    except Exception as e:
        return [
            None,
            None,
            None,
            None,
            None,
            f"Error processing file: {str(e)}"
        ]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
