# utils/data_preprocessor.py
import pandas as pd

def filter_data(df, start_date, end_date, countries, products):
    """Filter DataFrame based on user input"""
    filtered_df = df[
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))
    ]

    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]

    if products:
        filtered_df = filtered_df[filtered_df['Product'].isin(products)]

    return filtered_df

def get_summary_stats(df, lookback_days=30):
    """Get KPI summary with percentage change from previous period"""

    latest_date = df['Date'].max()
    current_start = latest_date - pd.Timedelta(days=lookback_days)
    prev_start = current_start - pd.Timedelta(days=lookback_days)

    current_df = df[df['Date'] > current_start]
    prev_df = df[(df['Date'] > prev_start) & (df['Date'] <= current_start)]

    def calc_delta(current, previous):
        if previous == 0 or pd.isna(previous):
            return float('inf')
        return ((current - previous) / previous) * 100

    stats_current = {
        "Total Revenue": current_df["Revenue"].sum(),
        "Total Units Sold": current_df["Units Sold"].sum(),
        "Total Profit": current_df["Profit"].sum(),
        "Average Profit Margin (%)": current_df["Profit Margin (%)"].mean()
    }

    stats_prev = {
        "Total Revenue": prev_df["Revenue"].sum(),
        "Total Units Sold": prev_df["Units Sold"].sum(),
        "Total Profit": prev_df["Profit"].sum(),
        "Average Profit Margin (%)": prev_df["Profit Margin (%)"].mean()
    }

    deltas = {
        "Total Revenue Δ (%)": calc_delta(stats_current["Total Revenue"], stats_prev["Total Revenue"]),
        "Total Units Sold Δ (%)": calc_delta(stats_current["Total Units Sold"], stats_prev["Total Units Sold"]),
        "Total Profit Δ (%)": calc_delta(stats_current["Total Profit"], stats_prev["Total Profit"]),
        "Avg Profit Margin Δ (%)": calc_delta(stats_current["Average Profit Margin (%)"], stats_prev["Average Profit Margin (%)"])
    }

    colors = {
        k: "darkgreen" if deltas[k] > 0 else "darkred" for k in deltas
    }

    return {
        "current": stats_current,
        "delta_percent": deltas,
        "delta_color": colors
    }

def get_sales_by_country(df):
    return df.groupby("Country")["Revenue"].sum().reset_index()

def get_sales_by_product(df):
    return df.groupby("Product")["Revenue"].sum().reset_index()

def get_profit_margin_by_product(df):
    return df.groupby("Product")["Profit Margin (%)"].mean().reset_index()

def get_grouped_bar_data(df):
    return df.groupby(['Country', 'Product'])['Revenue'].sum().unstack(fill_value=0)

def get_bubble_data(df):
    return df.groupby("Product").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units Sold", "sum"),
        Profit_Margin=("Profit Margin (%)", "mean")
    ).reset_index()

def get_monthly_revenue_trend(df):
    return df.resample('MS', on='Date')['Revenue'].sum().reset_index()

def get_revenue_by_month(df):
    df['Month'] = df['Date'].dt.month_name()
    return df.groupby('Month')['Revenue'].mean().reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]).reset_index()

def get_country_time_trend(df):
    return df.groupby([pd.Grouper(key='Date', freq='MS'), 'Country'])['Revenue'].sum().reset_index()