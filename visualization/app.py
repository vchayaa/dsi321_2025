import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io

# Add error handling for lakefs_client import
try:
    from lakefs_client.client import LakeFSClient
    LAKEFS_AVAILABLE = True
except ImportError:
    LAKEFS_AVAILABLE = False
    st.warning("lakefs_client not installed. Some features may not work.")

# Page config
st.set_page_config(page_title="Weather Data Analysis", layout="wide")
st.title("🌦️ Weather Data Analysis Dashboard")

# LakeFS connection
if LAKEFS_AVAILABLE:
    client = LakeFSClient(
        endpoint=os.environ["LAKEFS_ENDPOINT"],
        access_key=os.environ["LAKEFS_ACCESS_KEY"],
        secret_key=os.environ["LAKEFS_SECRET_KEY"]
    )

    repo = os.environ["LAKEFS_REPO"]
    branch = os.environ["LAKEFS_BRANCH"]
    path = "weather_data/temperature.csv"

    try:
        # Load data
        obj = client.objects.get_object(repo, branch, path)
        df = pd.read_csv(io.BytesIO(obj.read()))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sidebar
        st.sidebar.header("Data Analysis Options")
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type",
            ["Weather Overview", "Temperature Trends", "Weather Distribution"]
        )
        
        # Location filter
        if 'location' in df.columns:
            locations = ['All'] + sorted(df['location'].unique().tolist())
            selected_location = st.sidebar.selectbox("Select Location", locations)
            
            # Filter data based on location
            if selected_location != 'All':
                filtered_df = df[df['location'] == selected_location]
            else:
                filtered_df = df.copy()
        else:
            filtered_df = df.copy()
            
        # Date range filter
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [filtered_df['timestamp'].min().date(), filtered_df['timestamp'].max().date()],
            min_value=filtered_df['timestamp'].min().date(),
            max_value=filtered_df['timestamp'].max().date()
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (filtered_df['timestamp'].dt.date >= start_date) & (filtered_df['timestamp'].dt.date <= end_date)
            filtered_df = filtered_df[mask]
        
        if analysis_type == "Weather Overview":
            st.header("Current Weather Metrics")
            
            # Get the most recent data point
            latest_data = filtered_df.sort_values('timestamp').iloc[-1]
            
            # Display key metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Temperature", 
                    f"{latest_data['temperature']:.1f}°C",
                    delta=f"{latest_data['temperature'] - filtered_df.iloc[-2]['temperature']:.1f}°C" if len(filtered_df) > 1 else None
                )
            
            with col2:
                if 'humidity' in filtered_df.columns:
                    st.metric(
                        "Humidity", 
                        f"{latest_data['humidity']:.1f}%",
                        delta=f"{latest_data['humidity'] - filtered_df.iloc[-2]['humidity']:.1f}%" if len(filtered_df) > 1 else None
                    )
                else:
                    st.metric("Humidity", "N/A")
            
            with col3:
                if 'wind_speed' in filtered_df.columns:
                    st.metric(
                        "Wind Speed", 
                        f"{latest_data['wind_speed']:.1f} m/s",
                        delta=f"{latest_data['wind_speed'] - filtered_df.iloc[-2]['wind_speed']:.1f}" if len(filtered_df) > 1 else None
                    )
                else:
                    st.metric("Wind Speed", "N/A")
            
            # Interactive multi-metric chart
            st.subheader("Weather Metrics Over Time")
            
            metrics = ["temperature"]
            if 'humidity' in filtered_df.columns:
                metrics.append("humidity")
            if 'wind_speed' in filtered_df.columns:
                metrics.append("wind_speed")
                
            selected_metrics = st.multiselect(
                "Select Metrics to Display",
                options=metrics,
                default=["temperature"]
            )
            
            if selected_metrics:
                fig = go.Figure()
                
                for metric in selected_metrics:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['timestamp'],
                        y=filtered_df[metric],
                        mode='lines',
                        name=metric.capitalize()
                    ))
                
                fig.update_layout(
                    title="Weather Metrics Trend",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    legend_title="Metrics",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Weather condition
            if 'weather_main' in filtered_df.columns:
                st.subheader("Current Weather Condition")
                st.info(f"🌤️ {latest_data['weather_main']} - {latest_data.get('weather_description', '')}")
        
        elif analysis_type == "Temperature Trends":
            st.header("Temperature Trends Over Time")
            
            # Visualization 1: Temperature Line Chart
            fig1 = px.line(filtered_df, 
                          x='timestamp', 
                          y='temperature',
                          title='Temperature Variation Over Time',
                          labels={'temperature': 'Temperature (°C)', 
                                 'timestamp': 'Time'})
            st.plotly_chart(fig1, use_container_width=True)
            
            # Temperature heatmap by hour and day
            if len(filtered_df) > 24:
                st.subheader("Temperature Patterns by Hour and Day")
                
                filtered_df['day'] = filtered_df['timestamp'].dt.day
                filtered_df['hour'] = filtered_df['timestamp'].dt.hour
                
                # Create pivot table for heatmap
                temp_pivot = filtered_df.pivot_table(
                    values='temperature', 
                    index='day',
                    columns='hour',
                    aggfunc='mean'
                )
                
                fig_heatmap = px.imshow(
                    temp_pivot,
                    labels=dict(x="Hour of Day", y="Day of Month", color="Temperature (°C)"),
                    x=temp_pivot.columns,
                    y=temp_pivot.index,
                    color_continuous_scale="RdBu_r"
                )
                
                fig_heatmap.update_layout(title="Temperature Heatmap by Hour and Day")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Insight
            st.info("📊 Insight: The temperature data shows clear daily patterns with peaks during midday hours and troughs during early morning.")
            
        elif analysis_type == "Weather Distribution":
            st.header("Weather Condition Distribution")
            
            # Visualization 2: Weather Distribution
            if 'weather_main' in filtered_df.columns:
                weather_counts = filtered_df['weather_main'].value_counts()
                fig2 = px.bar(weather_counts, 
                            title='Distribution of Weather Conditions',
                            labels={'value': 'Count', 
                                    'index': 'Weather Condition'})
                st.plotly_chart(fig2, use_container_width=True)
                
                # Insight
                st.info("📊 Insight: The distribution shows the predominant weather patterns in the region.")
                
                # Weather conditions by time of day
                st.subheader("Weather Conditions by Time of Day")
                filtered_df['hour'] = filtered_df['timestamp'].dt.hour
                weather_hour = filtered_df.groupby(['weather_main', 'hour']).size().reset_index(name='count')
                
                fig_weather_hour = px.bar(
                    weather_hour,
                    x='hour',
                    y='count',
                    color='weather_main',
                    title='Weather Conditions by Hour of Day',
                    labels={'hour': 'Hour of Day', 'count': 'Frequency', 'weather_main': 'Weather Condition'}
                )
                st.plotly_chart(fig_weather_hour, use_container_width=True)
            else:
                st.warning("Weather condition data not available in the dataset.")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("No data found. Please run the pipeline first.")
else:
    st.error("LakeFS client not available. Please install the lakefs_client package.")
    
    # Display sample data for demonstration
    st.subheader("Sample Data (Demo Mode)")
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    sample_data = {
        'timestamp': dates,
        'temperature': np.sin(np.arange(100)/10) * 5 + 25 + np.random.normal(0, 1, 100),
        'humidity': np.cos(np.arange(100)/10) * 10 + 60 + np.random.normal(0, 3, 100),
        'wind_speed': np.abs(np.random.normal(5, 2, 100)),
        'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain', 'Drizzle'], 100)
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Display sample visualization
    st.line_chart(sample_df.set_index('timestamp')[['temperature', 'humidity', 'wind_speed']])
    st.warning("This is sample data. Connect to LakeFS to see real data.")
