import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from ml_models import train_profit_predictor
from forecasting import prepare_data, arima, exponential_smoothing_forecast, prophet
import warnings
warnings.filterwarnings("ignore")

# Import utility functions
from utils.data_loader import load_and_clean_data
from utils.data_preprocessor import (
    filter_data,
    get_summary_stats,
    get_sales_by_country,
    get_sales_by_product,
    get_profit_margin_by_product,
    get_grouped_bar_data,
    get_bubble_data,
    get_monthly_revenue_trend,
    get_revenue_by_month,
    get_country_time_trend
)

# Import risk analysis module
from risk_analysis import generate_risk_section

# Load and prepare data
df = load_and_clean_data()

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def apply_bar_colors(fig, df, value_col):
    """Highlight highest and lowest bars with light colors"""
    max_val = df[value_col].max()
    min_val = df[value_col].min()
    max_idx = df[df[value_col] == max_val].index[0]
    min_idx = df[df[value_col] == min_val].index[0]
    colors = ['lightblue'] * len(df)
    colors[max_idx] = 'lightgreen'
    colors[min_idx] = 'lightcoral'
    fig.update_traces(marker_color=colors)
    return fig

# Define available plots with labels and descriptions

FORECASTING_OPTIONS ={
    "arima" : {"label":"ARIMA"},
    "exponential_smoothing":{"label":"Exponential Smoothing"},
    "prophet":{"label":"Prophet"}
}

TARGET_OPTIONS = {
    "Units Sold" : {"label": "Units Sold"},
    "Revenue"  : {"label" : "Revenue"},
    "Profit"  : {"label" : "Profit"}
}
PLOT_OPTIONS = {
    "revenue-trend": {
        "label": "Daily Revenue Trend",
        "description": "Displays daily revenue fluctuations over time. This chart shows how your business's revenue has changed day-to-day, helping you identify patterns or anomalies in your sales. For example, you might notice that your revenue tends to spike on Mondays and slow down on Fridays."
    },
    "monthly-revenue-trend": {
        "label": "Monthly Revenue Trend",
        "description": "Shows aggregated monthly revenue trend. This chart gives you a higher-level view of your business's revenue performance over time. You can see how your overall revenue has changed from month to month, helping you identify trends or seasonality in your sales."
    },
    "revenue-by-month": {
        "label": "Revenue by Month",
        "description": "Average revenue per month across all years. This chart provides a summary view of your business's monthly revenue performance over the course of multiple years. You can quickly compare how your revenue has changed from year to year and spot any significant shifts."
    },
    "country-time-trend": {
        "label": "Country-wise Monthly Revenue",
        "description": "your business's revenue has changed in each country over time, helping you identify trends or differences in performance across regions. For example, you might notice that a particular region is experiencing faster growth than others."
    },
    "sales-by-country": {
        "label": "Sales by Country",
        "description": "Bar chart showing total revenue generated per country. This chart gives you a snapshot view of how your business's revenue is performing in each country. You can see which countries are generating the most revenue and identify any patterns or differences in performance."
    },
    "sales-by-product": {
        "label": "Sales by Product",
        "description": "Bar chart showing total revenue per product type.This chart gives you an overall idea of how different products are contributing in driving the sales of your company.You can see which products are contributing the most to your company."
    },
    "profit-margin-by-product": {
        "label": "Profit Margin by Product",
        "description": "Bar chart showing average profit margin (%) per product. This chart provides us a detailed view of how each product is helping your company in terms of profitability. You can visualize which products are generating lower or higher profits and work on improvements."
    },
    "grouped-bar-sales": {
        "label": "Grouped Sales by Product & Country",
        "description": "Visualizes sales per product in each country. This chart combines the insights from product and country and shows you how each product is doing in each country. It will give you an overall idea of which product is performing better or less in each market and can act accordingly."
    },
    "bubble-chart": {
        "label": "Bubble Chart: Profit vs Units Sold vs Revenue",
        "description": "Multi-metric view of products based on units sold, revenue, and profit margin. This plot gives us a clear idea of how much profit to revenue each product is doing. Each product is plotted in a spherical structure but with different sizes, as the bubble size displays the profit margin i.e. Higher the bubble higher the margin and vice versa."
    }
}

# Sidebar
sidebar = html.Div([
            html.H4(
            "Filters",
            className="display-6",
            style={
                'color': '#68A3D0',
                'transition': 'width 2s, height 4s',
                'width': '100px',  # Initial width
                'height': '50px',  # Initial height
            }
        ) ,   
        html.Hr(),
    html.Label("Select Date Range",style = {'color':'#788E9C'}),
    dcc.DatePickerRange(
        id='date-range',
        min_date_allowed=df['Date'].min(),
        max_date_allowed=df['Date'].max(),
        start_date=df['Date'].min(),
        end_date=df['Date'].max(),
        style = {'background-color':'#DCC5B2'},
    ),
    html.Br(), html.Br(),
    html.Label("Select Countries", style = {'color':'#788E9C'}),
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': c, 'value': c} for c in df['Country'].unique()],
        value=df['Country'].unique().tolist(),
        multi=True,
        style={'background-color':"#C7DDF2",'maxHeight': '150px', 'overflowY': 'scroll'}
    ),
    html.Br(),
    html.Label("Select Products",style = {'color':'#788E9C'}),
    dcc.Dropdown(
        id='product-dropdown',
        options=[{'label': p, 'value': p} for p in df['Product'].unique()],
        value=df['Product'].unique().tolist(),
        multi=True,
        style={'background-color':'#C7DDF2','maxHeight': '150px', 'overflowY': 'scroll'}
    ),
    html.Br(),
    html.Label("Select Plots to Display",style = {'color':"#788E9C"}),
    dcc.Dropdown(
        id='plot-selector',
        options=[{"label": info["label"], "value": key} for key, info in PLOT_OPTIONS.items()],
        value=["revenue-trend", "sales-by-country", "profit-margin-by-product"],
        multi=True,
        style={'background-color':"#C7DDF2",'color':'black','maxHeight':'150px','overflowY':'scroll'},
                
    ),html.Br(),
    html.Label("Select Forecast Model :", style={'color': "#9FA6DD"}),
    dcc.Dropdown(
        id = 'forecast-selector',
        options=[{"label": info["label"], "value": key} for key, info in FORECASTING_OPTIONS.items()],
        value="arima",
        multi = False,
        style = {'background-color': "#B7BFEC"}
    ),html.Br(),
    html.Label("Select Target Column:",style = {'color': "#9FA6DD"}),
    dcc.Dropdown(
        id = 'target-selector',
        options=[{"label": info["label"], "value": key} for key, info in TARGET_OPTIONS.items()],
        value = "Profit",
        multi = False,
        style = {'background-color': "#B7BFEC"}
    ),
    html.Label("Select Forecast Country:", style={'color': "#9FA6DD"}),
dcc.Dropdown(
    id='forecast-country-dropdown',
    options=[{'label': c, 'value': c} for c in df['Country'].unique()],
    value=df['Country'].unique()[0],
    multi=False,
    style={'background-color': "#B7BFEC"}
),

html.Label("Select Forecast Product:", style={'color': "#9FA6DD", 'marginTop': '10px'}),
dcc.Dropdown(
    id='forecast-product-dropdown',
    options=[{'label': p, 'value': p} for p in df['Product'].unique()],
    value=df['Product'].unique()[0],
    multi=False,
    style={'background-color': "#B7BFEC"}
),

    html.Div(dbc.Button("Apply Filters", color="primary", id="submit-button", className="mt-3",style = {'background-color':"#18576A", 'margin-left':'30%'} )),

 ], style={'padding': '20px', 'background-color': "#D9E6F5",'border-radius': '20px','width':'350px', 'height':'1050px'})

# Main content area
content = html.Div(id="page-content")

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col([html.H1("Sales Performance Dashboard", style={"textAlign": "center", 'color':"#0C436D", 'margin-bottom':'50px','margin-top':'30px'}),

            html.Div(id='summary-stats'),
            html.Div(id='selected-plots'),
            html.Div(id='risk-analysis-section'),
            html.Div(id = 'forecast-section'),
            html.Div(id ='ml-section')
        ], width=9)
    ])
], fluid=True,style = {'background-color':"#F1FAFE", 'color':"#5D5595"})

def generate_summary_cards(stats):
    stat_cards = []
    for k, v in stats["current"].items():
        delta_key = f"{k} Δ (%)" if k != "Average Profit Margin (%)" else "Avg Profit Margin Δ (%)"
        delta_val = stats["delta_percent"].get(delta_key, 0)
        color = stats["delta_color"].get(delta_key, "black")
        stat_cards.append(html.Div([
            html.H5(k),
            html.P(f"${v:,.2f}" if "Revenue" in k or "Profit" in k else f"{v:,.0f}"),
            html.Span(f"{delta_val:.1f}% ", style={"color": color})
        ], style={'border': '1px solid #ccc', 'padding': '10px', 'width': '18%', 'textAlign': 'center'}))
    return html.Div(stat_cards, style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'})

@app.callback(
    [
        Output('summary-stats', 'children'),
        Output('selected-plots', 'children'),
        Output('forecast-section', 'children'),
        Output("risk-analysis-section", "children"),
        Output('ml-section', 'children')  # New section

    ],
    Input('submit-button', 'n_clicks'),
    [
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
def update_dashboard(
    n_clicks, start_date, end_date, countries, products, selected_plots,
    fc_model, target, fc_country, fc_product):

    filtered_df = filter_data(df, start_date, end_date, countries, products)

    # Summary Stats
    stats = get_summary_stats(filtered_df)
    summary_cards = generate_summary_cards(stats)

    # Generate selected plots
    plot_elements = []

    if not selected_plots:
        plot_elements.append(html.Div("Please select at least one plot from the filters.", style={"textAlign": "center"}))

    # Daily Revenue Trend
    if "revenue-trend" in selected_plots:
        daily_rev = filtered_df.resample('D', on='Date')['Revenue'].sum().reset_index()
        fig = px.line(daily_rev, x='Date', y='Revenue', title="Daily Revenue Trend")
        plot_elements.append(html.Div([
            html.Div(PLOT_OPTIONS['revenue-trend']['description'], className="mb-2 mt-4"),
            dcc.Graph(figure=fig)
        ]))

    # Monthly Revenue Trend
    if "monthly-revenue-trend" in selected_plots:
        monthly_rev = get_monthly_revenue_trend(filtered_df)
        fig = px.line(monthly_rev, x='Date', y='Revenue', title="Monthly Revenue Trend")
        plot_elements.append(html.Div([
            html.Div(PLOT_OPTIONS['monthly-revenue-trend']['description'], className="mb-2 mt-4"),
            dcc.Graph(figure=fig)
        ]))

    # Revenue by Month
    if "revenue-by-month" in selected_plots:
        revenue_by_month = get_revenue_by_month(filtered_df)
        fig = px.bar(revenue_by_month, x='Month', y='Revenue', title="Average Revenue by Month")
        fig = apply_bar_colors(fig, revenue_by_month, 'Revenue')
        plot_elements.append(html.Div([
            html.Div(PLOT_OPTIONS['revenue-by-month']['description'], className="mb-2 mt-4"),
            dcc.Graph(figure=fig)
        ]))

    # Country-wise Time Series
    if "country-time-trend" in selected_plots:
        country_time = get_country_time_trend(filtered_df)
        fig = px.line(country_time, x='Date', y='Revenue', color='Country', title="Monthly Revenue by Country")
        plot_elements.append(html.Div([
            html.Div(PLOT_OPTIONS['country-time-trend']['description'], className="mb-2 mt-4"),
            dcc.Graph(figure=fig)
        ]))

    # Sales by Country
    if "sales-by-country" in selected_plots:
        country_data = get_sales_by_country(filtered_df)
        fig = px.bar(country_data, x='Country', y='Revenue', title="Sales by Country")
        fig = apply_bar_colors(fig, country_data, 'Revenue')
        plot_elements.append(html.Div([
            html.Div(PLOT_OPTIONS['sales-by-country']['description'], className="mb-2 mt-4"),
            dcc.Graph(figure=fig)
        ]))

    # Sales by Product
    if "sales-by-product" in selected_plots:
        product_data = get_sales_by_product(filtered_df)
        fig = px.bar(product_data, x='Product', y='Revenue', title="Sales by Product")
        fig = apply_bar_colors(fig, product_data, 'Revenue')
        plot_elements.append(html.Div([
            html.Div(PLOT_OPTIONS['sales-by-product']['description'], className="mb-2 mt-4"),
            dcc.Graph(figure=fig)
        ]))

    # Profit Margin by Product
    if "profit-margin-by-product" in selected_plots:
        profit_margin_data = get_profit_margin_by_product(filtered_df)
        fig = px.bar(profit_margin_data, x='Product', y='Profit Margin (%)', title="Profit Margin by Product")
        fig = apply_bar_colors(fig, profit_margin_data, 'Profit Margin (%)')
        plot_elements.append(html.Div([
            html.Div(PLOT_OPTIONS['profit-margin-by-product']['description'], className="mb-2 mt-4"),
            dcc.Graph(figure=fig)
        ]))

    # Grouped Bar Plot
    if "grouped-bar-sales" in selected_plots:
        grouped_data = get_grouped_bar_data(filtered_df)
        fig = go.Figure()
        for prod in grouped_data.columns:
            fig.add_trace(go.Bar(name=prod, x=grouped_data.index, y=grouped_data[prod]))
        fig.update_layout(barmode='group', title="Sales by Product per Country")
        plot_elements.append(html.Div([
            html.Div(PLOT_OPTIONS['grouped-bar-sales']['description'], className="mb-2 mt-4"),
            dcc.Graph(figure=fig)
        ]))

    # Bubble Chart
    if "bubble-chart" in selected_plots:
        bubble_data = get_bubble_data(filtered_df)
        fig = px.scatter(bubble_data, x='Units_Sold', y='Revenue',
                         size='Profit_Margin', color='Product',
                         hover_name='Product', size_max=60,
                         title="Bubble Chart: Profit Margin vs Units Sold vs Revenue")
        plot_elements.append(html.Div([
            html.Div(PLOT_OPTIONS['bubble-chart']['description'], className="mb-2 mt-4"),
            dcc.Graph(figure=fig)
        ]))

    fc_df = df[(df["Country"]==fc_country) & (df["Product"]==fc_product)].copy()
    if fc_df.empty:
        forecast_section = html.Div("No data for selected forecasting filters.")
    else:
        fc_ready = prepare_data(fc_df, target)
        if fc_model=="arima":
            _, fc = arima(fc_df, target)
        elif fc_model=="exponential_smoothing":
            _, fc = exponential_smoothing_forecast(fc_df, target)
        else:
            _, fc_df_pred = prophet(fc_ready, target)
            fc = fc_df_pred['yhat']
        start_date = fc_df['Date'].max()+pd.Timedelta(days=1)
        fc = pd.Series(fc.values, index=pd.date_range(start_date, periods=len(fc), freq='D'))

        ffig = go.Figure()
        ffig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df[target], mode='lines', name='Actual', line=dict(color='blue')))
        ffig.add_trace(go.Scatter(x=fc.index, y=fc, mode='lines', name='Forecast', line=dict(color='red')))
        ffig.update_layout(title=f"{fc_country} - {fc_product} Forecast ({FORECASTING_OPTIONS[fc_model]['label']})",
                            xaxis_title="Date", yaxis_title=target)

        ftable = html.Div([
            html.H3("Forecasted Data", style={'textAlign':'left'}),
            dash.dash_table.DataTable(
                columns=[{"name":"Date","id":"Date"},{"name":target,"id":target}],
                data=[{"Date": d.strftime('%Y-%m-%d'), target: round(v,2)} for d,v in zip(fc.index, fc)]
            )
        ], style={'margin-bottom':'50px','margin-right': '30px'})
        forecast_section = html.Div([dcc.Graph(figure=ffig), ftable])
        model, prediction_df, importance_image= train_profit_predictor(filtered_df)

    if model and not prediction_df.empty:
        ml_section = html.Div([
            html.H3("ML-Based Profit Prediction (Random Forest)", style={'marginTop': '30px'}),
            html.Img(src=importance_image, style={'maxWidth': '100%', 'height': 'auto', 'marginBottom': '20px'}),
            dash.dash_table.DataTable(
                columns=[{"name": col, "id": col} for col in prediction_df.columns],
                data=prediction_df.to_dict('records'),
                style_table={'overflowX': 'auto'}
            )
        ])
    else:
        ml_section = html.Div("Not enough data for ML model.")



    risk = generate_risk_section(df)
    return summary_cards, html.Div(plot_elements), forecast_section, risk, ml_section

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')