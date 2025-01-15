import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio

# Set the default renderer to browser
pio.renderers.default = "browser"

def demark_sequential_and_combo(df, column='close'):
    """
    Calculate the DeMark Sequential and Combo indicators for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the price data.
    column (str): The column name for the price data (default is 'close').

    Returns:
    pd.DataFrame: DataFrame with added columns for Setup, Countdown, and Combo.
    """
    df = df.copy()
    df['setup'] = 0
    df['countdown'] = 0
    df['combo'] = 0

    # Setup phase
    for i in range(4, len(df)):
        if df[column].iloc[i] > df[column].iloc[i - 4]:
            df.at[df.index[i], 'setup'] = df['setup'].iloc[i - 1] + 1 if df['setup'].iloc[i - 1] >= 0 else 1
        elif df[column].iloc[i] < df[column].iloc[i - 4]:
            df.at[df.index[i], 'setup'] = df['setup'].iloc[i - 1] - 1 if df['setup'].iloc[i - 1] <= 0 else -1
        else:
            df.at[df.index[i], 'setup'] = 0

    # Countdown phase
    for i in range(9, len(df)):
        if df['setup'].iloc[i] == 9:
            countdown = 0
            combo = 0
            for j in range(i + 1, len(df)):
                if df[column].iloc[j] < df[column].iloc[j - 2]:
                    countdown += 1
                    df.at[df.index[j], 'countdown'] = countdown
                    if countdown == 13:
                        break
                else:
                    df.at[df.index[j], 'countdown'] = 0

                # Combo phase
                if df[column].iloc[j] < df[column].iloc[i]:
                    combo += 1
                    df.at[df.index[j], 'combo'] = combo
                    if combo == 13:
                        break
                else:
                    df.at[df.index[j], 'combo'] = 0

    return df

def calculate_bollinger_bands(df, column='close', window=20, num_std_dev=2):
    """
    Calculate Bollinger Bands for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the price data.
    column (str): The column name for the price data (default is 'close').
    window (int): The number of periods for the moving average (default is 20).
    num_std_dev (int): The number of standard deviations for the bands (default is 2).

    Returns:
    pd.DataFrame: DataFrame with added columns for the middle, upper, and lower bands.
    """
    df = df.copy()
    df['middle_band'] = df[column].rolling(window=window).mean()
    df['std_dev'] = df[column].rolling(window=window).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * num_std_dev)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * num_std_dev)
    return df

# Generate fake data for Bollinger Bands example
np.random.seed(0)  # For reproducibility
dates = [datetime(2023, 10, 1) + timedelta(days=i) for i in range(60)]
close_prices = np.random.rand(60) * 100  # Random prices between 0 and 100

# Create DataFrame
bollinger_data = pd.DataFrame({'close': close_prices}, index=dates)

# Calculate Bollinger Bands
bollinger_result = calculate_bollinger_bands(bollinger_data)

# Plot Bollinger Bands with Plotly
fig = go.Figure()

# Add close price line
fig.add_trace(go.Scatter(x=bollinger_result.index, y=bollinger_result['close'], mode='lines', name='Close Price'))

# Add Bollinger Bands
fig.add_trace(go.Scatter(x=bollinger_result.index, y=bollinger_result['upper_band'], mode='lines', name='Upper Band', line=dict(color='rgba(255, 0, 0, 0.5)')))
fig.add_trace(go.Scatter(x=bollinger_result.index, y=bollinger_result['middle_band'], mode='lines', name='Middle Band', line=dict(color='rgba(0, 0, 255, 0.5)')))
fig.add_trace(go.Scatter(x=bollinger_result.index, y=bollinger_result['lower_band'], mode='lines', name='Lower Band', line=dict(color='rgba(0, 255, 0, 0.5)')))

# Update layout
fig.update_layout(title='Bollinger Bands',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  template='plotly_white')

# Show plot
fig.show()
