import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio

# Set the default renderer to browser
pio.renderers.default = "browser"

def demark_sequential(df, column='close'):
    """
    Calculate the DeMark Sequential indicator for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the price data.
    column (str): The column name for the price data (default is 'close').

    Returns:
    pd.DataFrame: DataFrame with added columns for Setup and Countdown.
    """
    df = df.copy()
    df['setup'] = 0
    df['countdown'] = 0

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
            for j in range(i + 1, len(df)):
                if df[column].iloc[j] < df[column].iloc[j - 2]:
                    countdown += 1
                    df.at[df.index[j], 'countdown'] = countdown
                    if countdown == 13:
                        break
                else:
                    df.at[df.index[j], 'countdown'] = 0

    return df

# Generate fake data
np.random.seed(0)  # For reproducibility
dates = [datetime(2023, 10, 1) + timedelta(days=i) for i in range(30)]
close_prices = np.random.rand(30) * 100  # Random prices between 0 and 100

# Create DataFrame
fake_data = pd.DataFrame({'close': close_prices}, index=dates)

# Apply DeMark Sequential
result = demark_sequential(fake_data)

# Plot with Plotly
fig = go.Figure()

# Add close price line
fig.add_trace(go.Scatter(x=result.index, y=result['close'], mode='lines', name='Close Price'))

# Add setup numbers in red
for i in range(len(result)):
    if result['setup'].iloc[i] != 0:
        fig.add_annotation(x=result.index[i], y=result['close'].iloc[i],
                           text=str(result['setup'].iloc[i]),
                           showarrow=True, arrowhead=1, ax=0, ay=-20,
                           font=dict(color='red'))

# Add countdown numbers in purple
for i in range(len(result)):
    if result['countdown'].iloc[i] != 0:
        fig.add_annotation(x=result.index[i], y=result['close'].iloc[i],
                           text=str(result['countdown'].iloc[i]),
                           showarrow=True, arrowhead=1, ax=0, ay=20,
                           font=dict(color='purple'))

# Update layout
fig.update_layout(title='DeMark Sequential Indicator',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  template='plotly_white')

# Show plot
fig.show()
