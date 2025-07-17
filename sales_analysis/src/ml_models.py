# src/ml_models.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

def train_profit_predictor(df):
    df = df.dropna()

    features = ['Units Sold', 'Revenue', 'Cost']
    target = 'Profit'

    # Guard clause for insufficient or missing data
    if df[features + [target]].isnull().any().any() or df.shape[0] < 20:
        return None, pd.DataFrame(), None

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on full data for display
    df['Predicted Profit'] = model.predict(X)

    # Feature importance plot
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(
        importance_df,
        x='Feature',
        y='Importance',
        title='Feature Importance for Profit Prediction (Random Forest)',
        color='Importance',
        color_continuous_scale='Blues'
    )

    # Return model, predictions DataFrame, and importance plot
    return model, df[['Date', 'Country', 'Product', 'Profit', 'Predicted Profit']], fig
