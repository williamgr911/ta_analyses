import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash.dependencies import Input, Output

# Generate trending fake data
np.random.seed(42)  # for reproducibility
dates = [datetime(2023, 10, 1) + timedelta(days=i) for i in range(60)]

# Create a trending series with some volatility
trend = np.linspace(50, 100, 60)  # Upward trend from 50 to 100
volatility = np.random.normal(0, 2, 60)  # Random noise
seasonality = 5 * np.sin(np.linspace(0, 4*np.pi, 60))  # Add some waves
close_prices = trend + volatility + seasonality

# Create DataFrame
data = pd.DataFrame({'close': close_prices}, index=dates)

# Calculate DeMark Sequential and Combo
def demark_sequential_and_combo(df, column='close'):
    df = df.copy()
    df['setup'] = 0
    df['countdown'] = 0
    df['combo'] = 0

    for i in range(4, len(df)):
        if df[column].iloc[i] > df[column].iloc[i - 4]:
            df.at[df.index[i], 'setup'] = df['setup'].iloc[i - 1] + 1 if df['setup'].iloc[i - 1] >= 0 else 1
        elif df[column].iloc[i] < df[column].iloc[i - 4]:
            df.at[df.index[i], 'setup'] = df['setup'].iloc[i - 1] - 1 if df['setup'].iloc[i - 1] <= 0 else -1
        else:
            df.at[df.index[i], 'setup'] = 0

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

                if df[column].iloc[j] < df[column].iloc[i]:
                    combo += 1
                    df.at[df.index[j], 'combo'] = combo
                    if combo == 13:
                        break
                else:
                    df.at[df.index[j], 'combo'] = 0

    return df

# Calculate Bollinger Bands
def calculate_bollinger_bands(df, column='close', window=20, num_std_dev=2):
    df = df.copy()
    df['middle_band'] = df[column].rolling(window=window).mean()
    df['std_dev'] = df[column].rolling(window=window).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * num_std_dev)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * num_std_dev)
    return df

# Apply indicators
demark_result = demark_sequential_and_combo(data)
bollinger_result = calculate_bollinger_bands(data)

# Determine traffic light color based on DeMark setup values
def get_traffic_light_color(setup_values):
    latest_setup = setup_values.iloc[-1]
    if abs(latest_setup) < 5:
        return "green"
    elif abs(latest_setup) < 8:
        return "orange"
    else:
        return "red"

# Create plots
def create_demark_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'))
    for i in range(len(df)):
        if df['setup'].iloc[i] != 0:
            # Determine color based on setup direction
            # Positive setup (buy) = red, Negative setup (sell) = green
            setup_color = 'red' if df['setup'].iloc[i] > 0 else 'green'
            fig.add_annotation(x=df.index[i], y=df['close'].iloc[i],
                             text=str(abs(df['setup'].iloc[i])),  # Use absolute value
                             showarrow=False,  # Remove arrow
                             yshift=10,  # Slight shift above the price
                             font=dict(color=setup_color))
        if df['countdown'].iloc[i] != 0:
            fig.add_annotation(x=df.index[i], y=df['close'].iloc[i],
                             text=str(df['countdown'].iloc[i]),
                             showarrow=False,  # Remove arrow
                             yshift=-10,  # Slight shift below the price
                             font=dict(color='purple'))
        if df['combo'].iloc[i] != 0:
            fig.add_annotation(x=df.index[i], y=df['close'].iloc[i],
                             text=str(df['combo'].iloc[i]),
                             showarrow=False,  # Remove arrow
                             yshift=-20,  # Further shift below the price
                             font=dict(color='blue'))
    fig.update_layout(title='DeMark Sequential and Combo Indicators',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     template='plotly_white')
    return fig

def create_bollinger_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], mode='lines', name='Upper Band', line=dict(color='rgba(255, 0, 0, 0.5)')))
    fig.add_trace(go.Scatter(x=df.index, y=df['middle_band'], mode='lines', name='Middle Band', line=dict(color='rgba(0, 0, 255, 0.5)')))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], mode='lines', name='Lower Band', line=dict(color='rgba(0, 255, 0, 0.5)')))
    fig.update_layout(title='Bollinger Bands',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      template='plotly_white')
    return fig

# Add this function to count signals by color
def count_signals_by_color(data_dict):
    color_counts = {'green': 0, 'orange': 0, 'red': 0}
    for symbol, df in data_dict.items():
        color = get_traffic_light_color(df['setup'])
        color_counts[color] += 1
    return color_counts

# Create multiple data series (example with 3 different trends)
data_dict = {
    'uptrend': pd.DataFrame({
        'close': trend + volatility + seasonality
    }, index=dates),
    'downtrend': pd.DataFrame({
        'close': np.linspace(100, 50, 60) + volatility - seasonality
    }, index=dates),
    'sideways': pd.DataFrame({
        'close': np.ones(60) * 75 + volatility + seasonality * 0.5
    }, index=dates)
}

# Calculate indicators for all series
for symbol in data_dict:
    data_dict[symbol] = demark_sequential_and_combo(data_dict[symbol])

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Modify the layout to include an interval component and an ID for the traffic light
app.layout = dbc.Container([
    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds (1 second)
        n_intervals=0
    ),
    # Add summary row at the top
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Signal Summary", className="text-center"),
                html.Div([
                    html.Span(id='green-count', className="mx-2"),
                    html.Span(id='orange-count', className="mx-2"),
                    html.Span(id='red-count', className="mx-2")
                ], className="text-center")
            ])
        ]), width=12, className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.Div([
                html.Div([
                    "DeMark Sequential and Combo - ",
                    html.Span(id='setup-text'),
                    " - ",
                    html.Span(id='combo-text'),
                    " ",
                    html.Span(id='traffic-light', style={'fontSize': '24px'})
                ])
            ])),
            dbc.CardBody(dcc.Graph(figure=create_demark_plot(data_dict['uptrend'])))
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Bollinger Bands"),
            dbc.CardBody(dcc.Graph(figure=create_bollinger_plot(bollinger_result)))
        ]), width=6)
    ])
], fluid=True)

# Add callback to update the traffic light
@app.callback(
    Output('traffic-light', 'style'),
    Output('traffic-light', 'children'),
    Output('setup-text', 'children'),
    Output('combo-text', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_header_info(n):
    # Recalculate DeMark indicators
    demark_result = demark_sequential_and_combo(data)
    color = get_traffic_light_color(demark_result['setup'])
    
    # Get latest values
    setup_value = abs(demark_result['setup'].iloc[-1])
    combo_value = demark_result['combo'].iloc[-1]
    
    setup_text = f"Setup is at {setup_value}"
    combo_text = f"Combo signal is at {combo_value}"
    
    return {'color': color, 'fontSize': '24px'}, ' ●', setup_text, combo_text

# Add callback for the summary counts
@app.callback(
    [Output('green-count', 'children'),
     Output('orange-count', 'children'),
     Output('red-count', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_signal_counts(n):
    # Recalculate indicators for all series
    for symbol in data_dict:
        data_dict[symbol] = demark_sequential_and_combo(data_dict[symbol])
    
    counts = count_signals_by_color(data_dict)
    
    return [
        html.Span([
            f"Green: {counts['green']}",
            html.I(className="fas fa-circle", style={'color': 'green', 'marginLeft': '5px'})
        ]),
        html.Span([
            f"Orange: {counts['orange']}",
            html.I(className="fas fa-circle", style={'color': 'orange', 'marginLeft': '5px'})
        ]),
        html.Span([
            f"Red: {counts['red']}",
            html.I(className="fas fa-circle", style={'color': 'red', 'marginLeft': '5px'})
        ])
    ]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
