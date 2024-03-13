import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.graph_objects as go
import geopandas as gpd
import json
import dash_bootstrap_components as dbc
import joblib
from dash.dependencies import State
import numpy as np
from joblib import load



import datetime
from datetime import datetime
import matplotlib.dates as dates
from pmdarima.arima import auto_arima
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset


df = pd.read_csv(r'/Users/Francesca/PycharmProjects/project3dashboard/.venv/lib/python3.12/APP/pricepaiddata.csv')

external_stylesheets = ['https://fonts.googleapis.com/css?family=Roboto&display=swap']

app = dash.Dash(__name__)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('London House Price Dashboard', style={'textAlign': 'center', 'color': 'black'}),
    html.Div('A tool to explore London House Prices over time', style={'textAlign': 'center', 'color': 'black'}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='User Guide', children=[
            html.P("Welcome to the User Guide!", style={'color': 'black', 'text-align': 'center'}),
            html.Div([
                html.H3('1) The purpose of the dashboard', style={'color': 'black', 'text-align': 'center'}),
                html.P(
                    "This dashboard allows you to explore how london house prices have changed from 1995 to 2023 using HMRC Price Paid Data",
                    style={'color': 'black', 'text-align': 'center'}),
                html.P("Begin exploring by selecting any one of the tabs above.", style={'color': 'black', 'text-align': 'center'}),
            ]),
            html.Div([
                html.H3('2) Navigating Tabs', style={'color': 'black', 'text-align': 'center'}),
                html.P("Use the tabs at the top to switch between different sections and discover its functionality.", style={'color': 'black', 'text-align': 'center'}),
                html.P("Refer to the 'How to use' section in each tab first to ensure ease of use of the exploratory tools available.", style={'color': 'black', 'text-align': 'center', 'text-decoration': 'underline'}),
                html.P(
                    "Please allow some time for the graphs to update after selecting options from the dropdown due to the large nature of the dataset.",
                    style={'color': 'black', 'text-align': 'center'}),
            ]),
            html.Div([
                html.H3('3) The Predictive Model', style={'color': 'black', 'text-align': 'center'}),
                html.P("Select the final tab to start predicting London House Prices", style={'color': 'black', 'text-align': 'center'}),
                html.P("Choose values from the available dropdowns to recieve a prediction", style={'color': 'black', 'text-align': 'center'}),
            ])
        ]),

        dcc.Tab(label='Compare types sold by district', children=[
            html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
                # Sidebar for General Comparison
                html.Div([
                    html.H2('How to Use', style={'color': 'black', 'marginBottom': '10px'}),  # Header with instructions
                    html.H4('Steps', style={'color': 'black', 'marginBottom': '20px', 'text-decoration': 'underline'}),
                    html.P('1) Please select a district - City of London is the default',
                           style={'color': 'black', 'marginBottom': '20px'}),
                    html.P(
                        '2) Choose any selection of postcodes or years (you can choose multiple of each variable) to filter the data further',
                        style={'color': 'black', 'marginBottom': '20px'}),
                    html.P('3) Hover over the bars on the chart to view the number of houses sold in specific postcodes. Zoom into the graph by clicking and dragging your mouse to select a section or use any of the tools available in the top right corner - use the "Reset axes" button to rescale the graph',
                           style={'color': 'black', 'marginBottom': '20px'}),
                    html.P('4) Clear selection of postcodes and years and choose a new district to explore',
                           style={'color': 'black', 'marginBottom': '20px'}),
                    dcc.Dropdown(
                        id='geo-dropdown',
                        options=[{'label': i, 'value': i} for i in sorted(df['District'].unique())],
                        placeholder='Select district...',
                        className='custom-dropdown'
                    ),
                    dcc.Dropdown(
                        id='postcode-dropdown',
                        options=[{'label': str(k), 'value': str(k)}
                                 for k in sorted(df['Area Code'].unique())],
                        placeholder='Select postcodes...',
                        className='custom-dropdown',
                        multi=True,
                    ),
                    dcc.Dropdown(
                        id='year-dropdown',
                        options=[{'label': int(j), 'value': int(j)}
                                 for j in df['Year'].unique() if not pd.isna(j)],
                        placeholder='Select year of sale...',
                        className='custom-dropdown',
                        multi=True,
                    ),
                ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

                # Main Content for General Comparison
                html.Div([
                    # Price Graph - Full width
                    dcc.Graph(id='bar-plot', style={'width': '100%'})
                ], style={'width': '80%', 'display': 'flex', 'flexDirection': 'column', 'marginTop': '60px'}),
            ])
        ]),

        dcc.Tab(label='Average prices by year', children=[
            html.Div([
                html.Div([
                    html.H2('How to Use', style={'color': 'black', 'marginBottom': '10px'}),
                    html.H4('Steps', style={'color': 'black', 'marginBottom': '20px', 'text-decoration': 'underline'}),
                    html.P('1) Select a year to view the average house price in each district that year',
                           style={'color': 'black', 'marginBottom': '20px'}),
                    html.P('2) Hover over an area on the map to view its average house price. Use any of the tools available in the top right corner to adjust the map view - use the "Reset view" button to rescale the graph',
                           style={'color': 'black', 'marginBottom': '20px'}),
                    dcc.Dropdown(
                        id='year-dropdown2',
                        options=[{'label': int(j), 'value': int(j)} for j in
                                 sorted(df['Year'].unique(), reverse=True)],
                        placeholder='Select year...', className='custom-dropdown'
                    ),
                ], style={'width': '20%', 'display': 'inline-block',
                          'verticalAlign': 'top', 'padding': '10px'}),

                # Main Content for Choropleth Map
                html.Div([
                    html.H5(id='map-title', style={'textAlign': 'center', 'color': 'black', 'marginBottom': '20px'}),
                    dcc.Graph(id='borough-map'),
                ], style={'width': '80%', 'display': 'inline-block', 'textAlign': 'center', 'verticalAlign': 'top',
                          'marginTop': '20px'}),
            ], style={'display': 'flex'})
        ]),
        dcc.Tab(label='Percentage sold per district', children=[
            html.Div([
                html.Div([
                    html.H2('How to Use', style={'color': 'black', 'marginBottom': '10px'}),
                    html.H4('Steps', style={'color': 'black', 'marginBottom': '20px', 'text-decoration': 'underline'}),
                    html.P('1) Select a year to view the percentage of houses sold in each district that year',
                           style={'color': 'black', 'marginBottom': '20px'}),
                    html.P('2) Hover over a section of the piechart the number and percentage of houses sold in a specific district',
                           style={'color': 'black', 'marginBottom': '20px'}),
                    html.P('3) Click on specific districts in the list on the right-hand side to remove or add them into your analysis',
                           style={'color': 'black', 'marginBottom': '20px'}),


                    dcc.Dropdown(
                        id='year-dropdown3',
                        options=[{'label': int(j), 'value': int(j)} for j in sorted(df['Year'].unique(), reverse=True)],
                        placeholder='Select year...', className='custom-dropdown'
                    ),
                ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

                html.Div([
                    dcc.Graph(id='pie-chart'),
                ], style={'width': '75%', 'display': 'inline-block', 'marginTop': '20px'}),
            ])
        ]),

        # Postcode Comparison Tab
        dcc.Tab(label='Average house price comparison by postcode', children=[
            html.Div([
                html.Div([
                    html.H2('How to Use', style={'color': 'black', 'marginBottom': '10px'}),
                    html.H4('Steps', style={'color': 'black', 'marginBottom': '20px', 'text-decoration': 'underline'}),
                    html.P('1) Please select a district - City of London is the default',
                           style={'color': 'black', 'marginBottom': '20px'}),
                    html.P(
                        '2) Choose two postcodes to compare the median house prices in both areas',
                        style={'color': 'black', 'marginBottom': '20px'}),
                    html.P('3) Hover over the line on the graph to view the median house price in each year',
                           style={'color': 'black', 'marginBottom': '20px'}),
                    html.P('4) Clear selection of postcodes and choose a new district to explore',
                           style={'color': 'black', 'marginBottom': '20px'}),
                    dcc.Dropdown(
                        id='geo-dropdown2',
                        options=[{'label': b, 'value': b} for b in sorted(df['District'].unique())],
                        placeholder='Select district...',
                        className='custom-dropdown'
                    ),
                    dcc.Dropdown(
                        id='postcode-dropdown-2',
                        options=[{'label': str(m), 'value': str(m)} for m in sorted(df['Area Code'].unique())],
                        placeholder='Select postcode 1...', className='custom-dropdown'
                    ),
                    dcc.Dropdown(
                        id='postcode-dropdown-3',
                        options=[{'label': str(c), 'value': str(c)} for c in sorted(df['Area Code'].unique())],
                        placeholder='Select postcode 2...', className='custom-dropdown'
                    ),
                ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

                # Main Content for Postcode Comparison
                html.Div([
                    dcc.Graph(id='average-plot', style={'flex': '1'}),
                    dcc.Graph(id='postcode-comp', style={'width': '100%'})
                ], style={'width': '75%', 'display': 'inline-block', 'padding': '10px'}),
            ]),
        ]),
        dcc.Tab(label='Forecasted Mean House Prices', children=[
            html.Div([
                html.Div(children=[
                html.H2('How to Use', style={'color': 'black', 'marginBottom': '10px'}),
                html.P('Select a number of months to forecast the mean house price in London - 12 months is the default', style={'color': 'black'}),
                dcc.Dropdown(
                    id='forecast-months-dropdown',
                    options=[{'label': f"{i} months", 'value': i} for i in range(6, 121, 6)],  # Up to 60 months
                    value=12,  # Default to 12 months
                    className='custom-dropdown',
                ),], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                html.Div(children=[
                dcc.Graph(id='forecast-graph', style={'width': '100%'}),
            ], style={'width': '75%', 'display': 'inline-block', 'padding': '10px'}),
            ])
        ]),
        dcc.Tab(label='Prediction Tool')
  ])
])


# update first postcode dropdown

@app.callback(
    Output('postcode-dropdown', 'options'),
    [Input('year-dropdown', 'value'),
     Input('geo-dropdown', 'value')]
)
def update_postcode(selected_years, selected_district):
    if not selected_district:
        return []

    filtered_df = df[df['District'] == selected_district]

    if selected_years:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]

    postcode_options = [{'label': postcode, 'value': postcode} for postcode in
                        sorted(filtered_df['Area Code'].unique())]
    return postcode_options


# update second postcode dropdown

@app.callback(
    Output('postcode-dropdown-2', 'options'),
    [Input('geo-dropdown2', 'value')]
)
def update_postcode(selected_district):
    if not selected_district:
        return []

    filtered_df = df[df['District'] == selected_district]

    postcode_options_2 = [{'label': postcode, 'value': postcode} for postcode in
                          sorted(filtered_df['Area Code'].unique())]
    return postcode_options_2


# update third postcode dropdown

@app.callback(
    Output('postcode-dropdown-3', 'options'),
    [Input('geo-dropdown2', 'value')]
)
def update_postcode(selected_district):
    if not selected_district:
        return []

    filtered_df = df[df['District'] == selected_district]

    postcode_options_3 = [{'label': postcode, 'value': postcode} for postcode in
                          sorted(filtered_df['Area Code'].unique())]
    return postcode_options_3


# Update first year dropdown

@app.callback(
    Output('year-dropdown', 'options'),
    [Input('postcode-dropdown', 'value'),
     Input('geo-dropdown', 'value')]
)
def update_year(selected_postcodes, selected_district):
    if not selected_district:
        return []

    filtered_df = df[df['District'] == selected_district]

    if selected_postcodes:
        filtered_df = filtered_df[filtered_df['Area Code'].isin(selected_postcodes)]

    year_options = [{'label': year, 'value': year} for year in sorted(filtered_df['Year'].unique(), reverse=True)]
    return year_options


# update map title

@app.callback(
    Output('map-title', 'children'),
    [Input('year-dropdown2', 'value')]
)
def update_map_title(selected_year):
    if selected_year:
        return f'Average House Prices for {selected_year}'
    else:
        return f'Average House Prices for Latest Year'


# Bar Chart

@app.callback(
    Output('bar-plot', 'figure'),
    [Input('geo-dropdown', 'value'),
     Input('postcode-dropdown', 'value'),
     Input('year-dropdown', 'value')
     ]
)
def update_bar_plot(selected_district, selected_postcodes, selected_year):
    if not selected_district:
        selected_district = "CITY OF LONDON"

    filtered_df = df[df['District'] == selected_district]

    if selected_postcodes:
        filtered_df = filtered_df[filtered_df['Area Code'].isin(selected_postcodes)]

    if selected_year:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_year)]

    total_df = filtered_df.groupby(['Type', 'Area Code']).size().reset_index(name='Count')

    title = [f'Houses Sold in {selected_district}'] if selected_district else ['House Sold']
    if selected_postcodes:
        title.append(f'Postcodes: {", ".join(map(str, selected_postcodes))}')
    if selected_year:
        title.append(f'Year: {", ".join(map(str, selected_year))}')

    blue_colors = ['#E1EFF6', '#CDEBFA', '#B8E1FA', '#A0D2EB', '#89C2D9', '#71B5D7', '#5CAFE2', '#47A9DE', '#3293D8', '#2D8CDE', '#2775D8', '#2165C1', '#1B56AA', '#174794', '#13397E', '#0F2B68', '#0B1F5B', '#081747', '#061030', '#050922', '#3399CC', '#0099CC', '#0077B3', '#005C99', '#004C8C', '#003D73', '#002D5C', '#001E43', '#001234', '#000F1F', '#D0E7FF', '#00416A', '#001D3D']

    bar_fig = px.bar(total_df, x="Type", y="Count", color='Area Code', barmode='group',
                     color_discrete_sequence=blue_colors)

    bar_fig.update_layout(xaxis_title='House Type', yaxis_title='Number of Houses Sold',
                          title=' - '.join(title),
                          paper_bgcolor= '#f4f4f4',
                          plot_bgcolor='#f4f4f4',
                          font_color='black')

    return bar_fig


# Borough Map

@app.callback(
    Output('borough-map', 'figure'),
    [Input('year-dropdown2', 'value')]
)
def update_map(selected_year):
    if selected_year:
        if not isinstance(selected_year, list):
            selected_year = [selected_year]
        filtered_df = df[df['Year'].isin(selected_year)]
    else:
        latest_year = df['Year'].max()
        filtered_df = df[df['Year'] == latest_year]

    avg_prices = filtered_df.groupby('District')['House Price'].mean().reset_index()

    with open(
            '/Users/Francesca/PycharmProjects/project3dashboard/.venv/lib/python3.12/london-boroughs_1179.geojson') as f:
        london_boroughs = json.load(f)

    # Match GeoJSON features with the average prices by district
    for feature in london_boroughs['features']:
        feature['properties']['AveragePrice'] = avg_prices.loc[
            avg_prices['District'] == feature['properties']['name'], 'House Price'].squeeze()

    colorscale = ["#fff2cc",
      "#428bcc",
       "#16537e"]

    map = px.choropleth_mapbox(avg_prices,
                               geojson=london_boroughs,
                               locations='District',
                               color='House Price',
                               color_continuous_scale=colorscale,
                               featureidkey="properties.name",
                               mapbox_style="carto-positron",
                               zoom=9,
                               center={"lat": 51.5074, "lon": -0.1278},
                               opacity=0.5,
                               labels={'House Price': 'Average House Price'}
                               )

    map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      plot_bgcolor='#f4f4f4',  # Sets the plot background color
                      paper_bgcolor='#f4f4f4',
                      font_color='black')

    return map


# Pie chart

@app.callback(
    Output('pie-chart', 'figure'),
    [Input('year-dropdown3', 'value')]
)
def update_pie_chart(selected_year):
    if selected_year and not isinstance(selected_year, list):
        selected_year = [selected_year]

    if selected_year:
        filtered_df = df[df['Year'].isin(selected_year)]
    else:
        latest_year = df['Year'].max()
        filtered_df = df[df['Year'] == latest_year]

    pie_data = filtered_df.groupby('District').size().reset_index(name='Number of Houses Sold')

    blue = ['#E1EFF6', '#CDEBFA', '#B8E1FA', '#A0D2EB', '#89C2D9', '#71B5D7', '#5CAFE2', '#47A9DE', '#3293D8', '#2D8CDE', '#2775D8', '#2165C1', '#1B56AA', '#174794', '#13397E', '#0F2B68', '#0B1F5B', '#081747', '#061030', '#050922', '#3399CC', '#0099CC', '#0077B3', '#005C99', '#004C8C', '#003D73', '#002D5C', '#001E43', '#001234', '#000F1F', '#D0E7FF', '#00416A', '#001D3D']

    fig = px.pie(pie_data, names='District', values='Number of Houses Sold',
                 title=f'Percentage of House Sales by Borough in {selected_year}',
                 color_discrete_sequence=blue)

    fig.update_layout({
        'plot_bgcolor': '#f4f4f4',
        'paper_bgcolor': '#f4f4f4',
        'font_color': 'black',
        'title': {
            'text': f'Percentage of House Sales by Borough in {", ".join(map(str, selected_year)) if selected_year else "Latest Year"}',
            'font': {'size': 20}}
    })

    return fig


# Average house price plot

@app.callback(
    Output('average-plot', 'figure'),
    [Input(component_id='geo-dropdown2', component_property='value')]
)
def update_average_plot(selected_district):
    if not selected_district:
        selected_district = "CITY OF LONDON"

    filtered_df = df[df['District'] == selected_district]

    average_df = filtered_df.groupby('Year')['House Price'].median().reset_index()

    avg_title = [f'Median House Price in {selected_district}']

    average_fig = px.line(
        average_df,
        x='Year',
        y='House Price')

    average_fig.update_traces(line_color='navy')

    average_fig.update_layout(xaxis_title='Year',
                              yaxis_title='Median Price',
                              title=' - '.join(avg_title),
                              xaxis=dict(
                                  tickmode='linear',
                                  tick0=average_df['Year'].min(),
                                  dtick=1,
                                  tickformat='.0f'),
                              plot_bgcolor='#f4f4f4',
                              paper_bgcolor='#f4f4f4',
                              font_color='black')

    return average_fig


# postcode comparison graph

@app.callback(
    Output('postcode-comp', 'figure'),
    [Input('postcode-dropdown-2', 'value'),
     Input('postcode-dropdown-3', 'value')]
)
def update_postcode_comparison_graph(postcode1, postcode2):
    filtered_df1 = df[df['Area Code'] == postcode1].groupby('Year')['House Price'].median().reset_index()
    filtered_df2 = df[df['Area Code'] == postcode2].groupby('Year')['House Price'].median().reset_index()

    comp_title = [f'Median House Price in {postcode1} and {postcode2}']

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=filtered_df1['Year'], y=filtered_df1['House Price'],
                              mode='lines+markers', name=f'Area Code {postcode1}',
                              line=dict(color='lightblue')))
    fig2.add_trace(go.Scatter(x=filtered_df2['Year'], y=filtered_df2['House Price'],
                              mode='lines+markers', name=f'Area Code {postcode2}',
                              line=dict(color='navy')))

    fig2.update_layout(title=' - '.join(comp_title),
                       xaxis_title='Year',
                       yaxis_title='Median Price',
                       xaxis=dict(
                           tickmode='linear',
                           tick0=filtered_df1['Year'].min(),
                           dtick=1,
                           tickformat='.0f'),
                       plot_bgcolor='#f4f4f4',
                       paper_bgcolor='#f4f4f4',
                       font_color='black')

    return fig2

@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('forecast-months-dropdown', 'value')]
)
def update_forecast_graph(forecast_months):
    if forecast_months is None:
        forecast_months = 12
    # Load and preprocess your data (simplified for this example)
    df1 = pd.read_csv('pricepaidnew.csv')

    # changing dates to datetime approapriate and arranging ascending
    df1['Date Sold'] = pd.to_datetime(df1['Date Sold'], format='mixed', dayfirst=True)
    df1 = df1.sort_values(by='Date Sold')

    # using IQR to remove outliers outside the range of 1.5 times the IQR
    grouped = df.groupby('Year')
    for year, group in grouped:
        median_price = group['House Price'].median()

    Q1 = df1['House Price'].quantile(0.25)
    Q3 = df1['House Price'].quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5
    outliers = df1[(df1['House Price'] < Q1 - threshold * IQR) | (df1['House Price'] > Q3 + threshold * IQR)]
    df1.loc[outliers.index, 'House Price'] = median_price

    df1.set_index('Date Sold', inplace=True)
    ts = df1.resample('M').mean()
    ts = ts.fillna(ts.groupby(ts.index.month).transform('mean'))
    ts = ts['House Price']

    # split the timeseries into 25 years training and 3 year testing data
    msk = (ts.index < ts.index[-1] - pd.Timedelta(days=1095))
    ts_train = ts[msk].copy()
    ts_test = ts[~msk].copy()

    future_index = pd.date_range(start=ts.index[-1], end=ts.index[-1] + DateOffset(months=forecast_months), freq='M')
    future_data = pd.Series(index=future_index)

    # Extend the original time series data by concatenating with the future data
    ts_extended = pd.concat([ts, future_data])

    # Fit SARIMAX model to the extended time series data
    fit = sm.tsa.statespace.SARIMAX(ts_extended, order=(1, 1, 1), seasonal_order=(1, 1, 2, 12)).fit()

    # Make future predictions
    predictions = fit.predict(start=ts.index[0], end=future_index[-1])

    # Plot the original data and future predictions
    fig3 = px.line(
        predictions)

    fig3.update_traces(line_color='navy')

    fig3.update_layout(xaxis_title='Year', yaxis_title='Predicted Mean House Price',
                          title='Future predictions with SARIMA model',
                          paper_bgcolor='#f4f4f4',
                          plot_bgcolor='#f4f4f4',
                          font_color='black')

    return fig3

if __name__ == '__main__':
    app.run_server(debug=True, port=8055)

