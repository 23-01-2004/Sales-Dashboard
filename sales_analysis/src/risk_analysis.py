import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

def calculate_revenue_volatility(df):
    """
    Calculate revenue volatility by country using Coefficient of Variation (CV)
    CV = Standard Deviation / Mean → Higher value = more volatile
    """
    # Resample daily revenue by country
    df_daily_country = df.resample('D', on='Date').sum(numeric_only=True).reset_index()
    df_daily_country = df_daily_country[['Date', 'Revenue']].copy()

    # Add country-level daily revenue
    daily_revenue_by_country = df.groupby(['Date', 'Country'])['Revenue'].sum().reset_index()

    # Pivot for volatility analysis
    pivoted = daily_revenue_by_country.pivot(index='Date', columns='Country', values='Revenue').fillna(0)

    # Compute CV per country
    volatility = pivoted.std() / pivoted.mean()
    volatility = volatility.reset_index(name="Volatility")

    # Plot
    fig = px.bar(volatility, x='Country', y='Volatility',
                 title="Revenue Volatility by Country (Coefficient of Variation)",
                 labels={"Volatility": "Revenue Volatility (CV)"},
                 color='Volatility', color_continuous_scale='Blues')

    fig.update_layout(template='plotly_white')
    return html.Div([
        html.Div("1. Revenue Volatility: High CV indicates unstable revenue patterns.", className="mb-2 mt-4"),
        dcc.Graph(figure=fig)
    ])

def analyze_concentration_risk(df):
    """Analyze market and product concentration using Pareto Principle (80/20 rule)"""

    # Market Concentration
    country_revenue = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False)
    country_revenue_pct = (country_revenue / country_revenue.sum()) * 100
    top_countries = country_revenue_pct.head(2).sum()

    # Product Concentration
    product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
    product_revenue_pct = (product_revenue / product_revenue.sum()) * 100
    top_products = product_revenue_pct.head(2).sum()

    # Visualize Country Concentration
    fig_country = px.pie(country_revenue_pct.reset_index(), names='Country', values='Revenue',
                         title=f"Revenue Distribution by Country (Top 2: {top_countries:.1f}%)",
                         color_discrete_sequence=px.colors.sequential.RdBu)

    # Visualize Product Concentration
    fig_product = px.pie(product_revenue_pct.reset_index(), names='Product', values='Revenue',
                         title=f"Revenue Contribution by Product (Top 2: {top_products:.1f}%)",
                         color_discrete_sequence=px.colors.sequential.Viridis)

    return html.Div([
        html.Div("2. Concentration Risk: High dependency on few markets or products can be risky.", className="mb-2 mt-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_country), width=6),
            dbc.Col(dcc.Graph(figure=fig_product), width=6)
        ])
    ])

def analyze_profitability_risk(df):
    """Identify low-margin products and countries"""

    # By Product
    product_margin = df.groupby('Product')['Profit Margin (%)'].mean().sort_values()
    fig_product = px.bar(product_margin.reset_index(), x='Product', y='Profit Margin (%)',
                         title="Profit Margin by Product",
                         color='Profit Margin (%)', color_continuous_scale='RdYlGn_r')

    # By Country
    country_margin = df.groupby('Country')['Profit Margin (%)'].mean().sort_values()
    fig_country = px.bar(country_margin.reset_index(), x='Country', y='Profit Margin (%)',
                         title="Profit Margin by Country",
                         color='Profit Margin (%)', color_continuous_scale='RdYlGn_r')

    # Highlight low margin
    fig_product.update_traces(marker_color=['red' if val < 50 else "#68A3D0" for val in product_margin])
    fig_country.update_traces(marker_color=['red' if val < 50 else '#68A3D0' for val in country_margin])

    return html.Div([
        html.Div("3. Profitability Risk: Low profit margins indicate financial risk.", className="mb-2 mt-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_product), width=6),
            dbc.Col(dcc.Graph(figure=fig_country), width=6)
        ])
    ])
def summary_risk(df):
    return html.Div([
        html.H3("Summary of Risk Factors : ", className="mb-2 mt-4"), html.Br(),
        html.Div(
            "1. Revenue Volatility: This shows overall how much volatile the revenue is across countries i.e. how much the overall coefficient of revenue fluctuates across different places.",
            style={'textAlign': 'left'}
        ),
        html.Hr(),html.Br(),
        html.Div(
            "2. Concentration Risk: This metric shows two different pie charts for product and revenue across all countries and all products to find out which are mostly driving the sales. Biasing towards few assets or places could increase the risk of concentration thus any disruption of certain products or markets can heavily impact the overall sales of the company. Evenly distributed risk is much more stable."
        ),
        html.Hr(),html.Br(),
        html.Div(
            "3. Profitability Risk: This section identifies low-margin products and countries which may pose a threat to the financial stability of the company. Identifying these areas and taking corrective measures can help mitigate this risk."
        )
    ],style={'margin-bottom':'100px'})

def generate_risk_section(df):
    """Generate all three risk components"""
    volatility_card = calculate_revenue_volatility(df)
    concentration_card = analyze_concentration_risk(df)
    profitability_card = analyze_profitability_risk(df)
    summary_card = summary_risk(df)

    return html.Div([
        html.H3(" Risk Analysis", style={'textAlign': 'center'}),
        html.Hr(),
        volatility_card,
        concentration_card,
        profitability_card,
        summary_card
    ])