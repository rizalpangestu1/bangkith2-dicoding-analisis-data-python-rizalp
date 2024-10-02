import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# Load the data
day_df = pd.read_csv("https://raw.githubusercontent.com/rizalpangestu1/belajar-analisis-data-python-rizal-dicoding/refs/heads/main/day_df_cleaned.csv")
hour_df = pd.read_csv("https://raw.githubusercontent.com/rizalpangestu1/belajar-analisis-data-python-rizal-dicoding/refs/heads/main/hour_df_cleaned.csv")

# Convert 'dteday' to datetime for filtering
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

# Set page layout
st.set_page_config(layout="wide")

# Sidebar selection for different visualizations
visualization = st.sidebar.selectbox(
    "Choose Visualization",
    ["Overview", "Time Series - Working Day vs. Holiday", "Hourly Distribution - Hour Data", "Pie Chart - User Types", "Regression Analysis", "All Visualizations Combined"]
)

# Define a date range filter for each visualization except "Regression Analysis"
if visualization != "Regression Analysis":
    start_date = st.sidebar.date_input("Start Date", day_df['dteday'].min())
    end_date = st.sidebar.date_input("End Date", day_df['dteday'].max())
    
    # Validate the date input
    if start_date > end_date:
        st.error("Error: End Date must fall after Start Date.")
        st.stop()

    # Apply the filter to day_df and hour_df based on selected date range
    filtered_day_df = day_df[(day_df['dteday'] >= pd.to_datetime(start_date)) & (day_df['dteday'] <= pd.to_datetime(end_date))]
    filtered_hour_df = hour_df[(hour_df['dteday'] >= pd.to_datetime(start_date)) & (hour_df['dteday'] <= pd.to_datetime(end_date))]
else:
    filtered_day_df = day_df
    filtered_hour_df = hour_df

# Pre-compute hourly distribution for all visualizations
hourly_usage_cnt = filtered_hour_df.groupby('hr')['cnt'].sum().reset_index()
hourly_usage_casual = filtered_hour_df.groupby('hr')['casual'].sum().reset_index()
hourly_usage_registered = filtered_hour_df.groupby('hr')['registered'].sum().reset_index()

# Overview of the datasets
if visualization == "Overview":
    st.title('Bike Sharing Data Dashboard 🚴')
    st.caption("Made by: Rizal Pangestu/rizal.pangestu0601@mail.ugm.ac.id/rizal-pangestu0601")
    st.image("https://github.com/rizalpangestu1/belajar-analisis-data-python-rizal-dicoding/blob/main/image1_hH9B4gs.width-1300.jpg?raw=true")
    st.caption("Source image: Google/Andrew Hyatt")
    st.write('''
        Bike sharing systems automate the process of bike rentals, allowing users to easily rent and return bikes at different locations.
        With over 500 programs globally, these systems are important for addressing traffic, environmental, and health issues.
        The data generated from bike sharing, such as travel duration and location details, makes it valuable for research,
        acting as a virtual sensor network to monitor city mobility and detect significant urban events.
    ''')
    st.write('''
        The bike-sharing rental patterns are closely linked to environmental and seasonal factors such as weather, precipitation,
        day of the week, season, and time of day. The dataset contains two years of historical records from the Capital Bikeshare
        system in Washington D.C. (2011-2012), available at the Capital Bikeshare website. The data has been aggregated into two-hour
        and daily intervals, with corresponding weather and seasonal details added, sourced from freemeteo.com.
    ''')

    st.header("Overview of Bike Sharing Datasets")
    st.subheader("📝 Notes: ")
    st.write("1. day.csv: Bike sharing counts aggregated on daily basis.")
    st.write("2. hour.csv: Bike sharing counts aggregated on hourly basis.")
    st.write(" ")
    st.write("📊 These are the first 5 rows of each dataset within the selected date range.")
    st.subheader("Day Data")
    st.write(filtered_day_df.head())

    st.subheader("Hour Data")
    st.write(filtered_hour_df.head())

# Line plot of bike usage
elif visualization == "Time Series - Working Day vs. Holiday":
    st.header("Trends of Bike Usage Over Time: Working Days vs Holidays")

    # Create line plot for bike usage over time using Plotly
    fig = px.line(filtered_day_df, x='dteday', y='cnt', color='workingday',
                  labels={"dteday": "Date", "cnt": "Total Bike Usage (cnt)", "workingday": "Working Day"},
                  title="Trends of Bike Usage Over Time: Working Days vs Holidays")
    st.plotly_chart(fig)

# Bar plot for hourly distribution
elif visualization == "Hourly Distribution - Hour Data":
    st.header("Distribution of Bike Usage by Hour")
    
    # Plot cnt vs hr using Plotly
    fig1 = px.bar(hourly_usage_cnt, x='hr', y='cnt', 
                  labels={"hr": "Hour of the Day", "cnt": "Total Bike Rentals (cnt)"},
                  title="Distribution of Bike Rentals by Hour of the Day")
    st.plotly_chart(fig1)

    # Plot casual vs hr
    fig2 = px.bar(hourly_usage_casual, x='hr', y='casual', 
                  labels={"hr": "Hour of the Day", "casual": "Unregistered Bike Rentals"},
                  title="Distribution of Bike for Unregistered User Usage by Hour of the Day")
    st.plotly_chart(fig2)

    # Plot registered vs hr
    fig3 = px.bar(hourly_usage_registered, x='hr', y='registered', 
                  labels={"hr": "Hour of the Day", "registered": "Registered Bike Rentals"},
                  title="Distribution of Bike for Registered User Usage by Hour of the Day")
    st.plotly_chart(fig3)

# Pie chart for user type distribution
elif visualization == "Pie Chart - User Types":
    st.header("Distribution of Bike Usage Between Unregistered and Registered Bike Users")

    # Calculate the total usage by user type based on the filtered data
    usage_counts = [filtered_day_df['casual'].sum(), filtered_day_df['registered'].sum()]
    labels = ['Unregistered', 'Registered']

    # Create pie chart using Plotly
    fig4 = go.Figure(data=[go.Pie(labels=labels, values=usage_counts, 
                                  hole=0.3, marker=dict(colors=['orange', 'green']))])
    fig4.update_layout(title_text='Distribution of Bike Usage Between Unregistered and Registered Bike Users')
    st.plotly_chart(fig4)

# Regression analysis section without date filter
elif visualization == "Regression Analysis":
    st.header("Regression Analysis on Bike Usage and Weather Variables")

    # Create a copy of day_df to transform 'cnt' without affecting the original
    reg_day_df = day_df.copy()
    reg_day_df['cnt'] = np.log(reg_day_df['cnt'])

    # Define independent and dependent variables for the model
    independent_var = reg_day_df[['weathersit', 'temp', 'hum', 'windspeed']]
    dependent_var = reg_day_df['cnt']

    # Add a constant to the features for intercept
    independent_var_with_constant = sm.add_constant(independent_var)

    # Build the OLS regression model
    model = sm.OLS(dependent_var, independent_var_with_constant).fit()

    # Display the model summary
    st.subheader("OLS Regression Model Summary")
    st.text(model.summary())

    # Create scatter plots for regression analysis using Plotly
    fig = px.scatter_matrix(reg_day_df, dimensions=['weathersit', 'temp', 'hum', 'windspeed'], color='cnt',
                            title="Scatter Plots of Weather Variables vs. Bike Rentals (Log-Transformed)",
                            labels={col: col for col in ['weathersit', 'temp', 'hum', 'windspeed', 'cnt']})
    st.plotly_chart(fig)

# Caption for references
st.caption('''References:
    Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", 
    Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.
''')
