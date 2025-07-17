# utils/data_loader.py
import pandas as pd
from datetime import datetime, timedelta

def excel_date(excel_date):
    """Convert Excel serial date to Python datetime"""
    return datetime(1900, 1, 1) + timedelta(days=int(excel_date) - 2)

def load_and_clean_data(filepath="../data/sales_data .csv"):
    """
    Load and clean raw sales data with Excel dates and missing column names.
    """

    # Load without header
    df = pd.read_csv(filepath, header=None)

    # Manually assign column names
    df.columns = ['Date', 'Country', 'Product', 'Units Sold', 'Revenue', 'Cost', 'Profit']

    # Remove rows where Date is still the word 'Date'
    df = df[df['Date'] != 'Date']

    # Convert Excel date
    df['Date'] = df['Date'].apply(lambda x: excel_date(x))
    df['Date'] = pd.to_datetime(df['Date'])

    # Fix numeric columns
    numeric_cols = ['Units Sold', 'Revenue', 'Cost', 'Profit']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Add derived metrics
    df['Profit Margin (%)'] = (df['Profit'] / df['Revenue']) * 100
    df['Unit Price'] = df['Revenue'] / df['Units Sold']

    df = df.sort_values('Date').reset_index(drop = True)

    return df
