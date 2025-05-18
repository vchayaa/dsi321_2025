import streamlit as st
# Page config must be the first Streamlit command
st.set_page_config(page_title="Weather Data Analysis", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io
from datetime import datetime
import pytz

# Define Thailand timezone
thai_tz = pytz.timezone('Asia/Bangkok')

# Add a function to convert timestamps to Thailand time
def convert_to_thai_time(df):
    """Convert timestamp column to Thailand timezone"""
    if 'timestamp' in df.columns and not df.empty:
        # Check if timestamps already have timezone info
        has_tzinfo = False
        if pd.api.types.is_datetime64_dtype(df['timestamp']):
            sample = df['timestamp'].iloc[0]
            has_tzinfo = sample.tzinfo is not None
        
        if not has_tzinfo:
            # If timestamps don't have timezone, assume they're UTC and convert to Thai time
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(thai_tz)
        else:
            # If they have timezone but it's not Thai time, convert
            if df['timestamp'].iloc[0].tzinfo != thai_tz:
                df['timestamp'] = df['timestamp'].dt.tz_convert(thai_tz)
    
    return df

# Function to optimize data types
def optimize_dtypes(df):
    """Optimize data types to reduce memory usage and improve performance."""
    df_optimized = df.copy()
    
    # Convert object columns to string or category
    for col in df.select_dtypes(include=['object']).columns:
        # If the number of unique values is less than 50% of the data, convert to category
        if df[col].nunique() / len(df) < 0.5:
            df_optimized[col] = df[col].astype('category')
        else:
            # Use pandas string type instead of object
            df_optimized[col] = df[col].astype('string')
    
    # Convert float64 columns to float32 if possible
    for col in df.select_dtypes(include=['float64']).columns:
        # Check if the column can be safely converted to float32
        if df[col].min() > np.finfo(np.float32).min and df[col].max() < np.finfo(np.float32).max:
            df_optimized[col] = df[col].astype(np.float32)
    
    return df_optimized

# Initialize session state if not already done
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

# Add error handling for lakefs_client import
try:
    from lakefs_client import Configuration, ApiClient
    from lakefs_client.apis import ObjectsApi
    LAKEFS_AVAILABLE = True
except ImportError as e:
    LAKEFS_AVAILABLE = False
    st.sidebar.error(f"Import error: {str(e)}")

# Title and sidebar
st.title("🌦️ Weather Data Analysis Dashboard")

# Remove the LakeFS connection header and info from sidebar
# st.sidebar.header("LakeFS Connection")
# st.sidebar.info("Using default LakeFS connection")

# Hardcoded LakeFS connection settings (not visible to users)
lakefs_endpoint = "http://lakefs-dev:8000/"
lakefs_access_key = "access_key"
lakefs_secret_key = "secret_key"
lakefs_repo = "weather"
lakefs_branch = "main"

# Function to load data from LakeFS
def load_data_from_lakefs():
    try:
        # Create configuration
        config = Configuration()
        config.host = lakefs_endpoint
        config.username = lakefs_access_key
        config.password = lakefs_secret_key
        
        # Create API client
        api_client = ApiClient(config)
        objects_api = ObjectsApi(api_client)
        
        path = "weather.parquet"
        
        # Configure storage options for LakeFS (S3-compatible)
        storage_options = {
            "key": lakefs_access_key,
            "secret": lakefs_secret_key,
            "client_kwargs": {
                "endpoint_url": lakefs_endpoint
            }
        }
        
        # Construct the full LakeFS S3-compatible path
        lakefs_s3_path = f"s3a://{lakefs_repo}/{lakefs_branch}/{path}"
        
        # Load data using pandas read_parquet
        df = pd.read_parquet(
            lakefs_s3_path,
            storage_options=storage_options
        )
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Convert to Thailand timezone
            df = convert_to_thai_time(df)
        
        # Optimize data types
        df = optimize_dtypes(df)
        
        return df, True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, False

# Function to generate sample data for demo mode
def generate_sample_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H', tz=thai_tz)
    sample_data = {
        'timestamp': dates,
        'main.temp': np.sin(np.arange(100)/10) * 5 + 25 + np.random.normal(0, 1, 100),
        'main.humidity': np.cos(np.arange(100)/10) * 10 + 60 + np.random.normal(0, 3, 100),
        'wind.speed': np.abs(np.random.normal(5, 2, 100)),
        'precipitation': np.abs(np.random.exponential(0.5, 100)) * (np.random.random(100) > 0.7),
        'weather.main': np.random.choice(['Clear', 'Clouds', 'Rain', 'Drizzle'], 100),
        'province': np.random.choice(['Bangkok', 'Chiang Mai', 'Phuket', 'Pathum Thani'], 100)
    }
    sample_df = pd.DataFrame(sample_data)
    return optimize_dtypes(sample_df)

# Function to filter data based on user selections
def filter_data(df, location_col, selected_locations, date_range):
    filtered_df = df.copy()
    
    # Filter by location
    if location_col and selected_locations:
        filtered_df = filtered_df[filtered_df[location_col].isin(selected_locations)]
    
    # Filter by date range
    if 'timestamp' in filtered_df.columns and date_range and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (filtered_df['timestamp'].dt.date >= start_date) & (filtered_df['timestamp'].dt.date <= end_date)
        filtered_df = filtered_df[mask]
    
    return filtered_df

# Function to create weather overview visualizations
def show_weather_overview(filtered_df, location_col, selected_locations):
    st.header("Current Weather Metrics")
    
    # Get the most recent data point
    if 'timestamp' in filtered_df.columns:
        latest_data = filtered_df.sort_values('timestamp').iloc[-1]
    else:
        latest_data = filtered_df.iloc[-1]
    
    # Create metrics display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Temperature
    if 'main.temp' in filtered_df.columns:
        col1.metric("🌡️ Temperature", f"{latest_data['main.temp']:.1f} °C")
    elif 'temperature' in filtered_df.columns:
        col1.metric("🌡️ Temperature", f"{latest_data['temperature']:.1f} °C")
    
    # Humidity
    if 'main.humidity' in filtered_df.columns:
        col2.metric("💧 Humidity", f"{latest_data['main.humidity']:.1f} %")
    elif 'humidity' in filtered_df.columns:
        col2.metric("💧 Humidity", f"{latest_data['humidity']:.1f} %")
    
    # Wind Speed
    if 'wind.speed' in filtered_df.columns:
        col3.metric("🌬️ Wind Speed", f"{latest_data['wind.speed']:.1f} m/s")
    elif 'wind_speed' in filtered_df.columns:
        col3.metric("🌬️ Wind Speed", f"{latest_data['wind_speed']:.1f} m/s")
    
    # Rainfall (1h) - with fallback to 0.0 if not present
    rain_1h = 0.0
    if 'rain.1h' in filtered_df.columns:
        rain_1h = latest_data['rain.1h']
    col4.metric("🌧️ Rainfall (1h)", f"{rain_1h:.1f} mm")
    
    # Rainfall (3h) - with fallback to 0.0 if not present
    rain_3h = 0.0
    if 'rain.3h' in filtered_df.columns:
        rain_3h = latest_data['rain.3h']
    col5.metric("🌧️ Rainfall (3h)", f"{rain_3h:.1f} mm")
    
    # Weather condition - make it bigger
    if 'weather.main' in filtered_df.columns:
        weather_col = 'weather.main'
        desc_col = 'weather.description' if 'weather.description' in filtered_df.columns else None
    elif 'weather_main' in filtered_df.columns:
        weather_col = 'weather_main'
        desc_col = 'weather_description' if 'weather_description' in filtered_df.columns else None
    else:
        weather_col = None
        
    if weather_col:
        st.header("Current Weather Condition")  # Changed from subheader to header
        weather_desc = latest_data.get(desc_col, '') if desc_col else ''
        st.markdown(f"### 🌤️ {latest_data[weather_col]} - {weather_desc}")  # Made bigger with markdown
    
    # Weather metrics over time
    st.subheader("Weather Metrics Over Time")
    
    # Define the specific metrics we want to display
    specific_metrics = []
    # Temperature
    if 'main.temp' in filtered_df.columns:
        specific_metrics.append(('main.temp', 'Temperature'))
    elif 'temperature' in filtered_df.columns:
        specific_metrics.append(('temperature', 'Temperature'))

    # Humidity
    if 'main.humidity' in filtered_df.columns:
        specific_metrics.append(('main.humidity', 'Humidity'))
    elif 'humidity' in filtered_df.columns:
        specific_metrics.append(('humidity', 'Humidity'))

    # Wind Speed
    if 'wind.speed' in filtered_df.columns:
        specific_metrics.append(('wind.speed', 'Wind Speed'))
    elif 'wind_speed' in filtered_df.columns:
        specific_metrics.append(('wind_speed', 'Wind Speed'))

    # Always add rainfall metrics even if columns don't exist yet
    specific_metrics.append(('rain.1h', 'Rainfall (1h)'))
    specific_metrics.append(('rain.3h', 'Rainfall (3h)'))

    # Let user select from only these specific metrics
    metric_options = [name for _, name in specific_metrics]
    selected_metric_names = st.multiselect(
        "Select Metrics to Display", 
        options=metric_options,
        default=metric_options  # Default to all metrics selected
    )

    # Map selected names back to column names
    selected_metrics = [col for col, name in specific_metrics if name in selected_metric_names]
    
    if 'timestamp' in filtered_df.columns and selected_metrics:
        # Create a working copy of the dataframe
        plot_df = filtered_df.copy()
        
        # Ensure timestamp column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(plot_df['timestamp']):
            try:
                plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
            except Exception as e:
                st.error(f"Could not convert timestamp column to datetime format: {str(e)}")
                return
        
        # Convert all selected metrics to numeric type before plotting
        for metric in selected_metrics:
            if metric in plot_df.columns:
                plot_df[metric] = pd.to_numeric(plot_df[metric], errors='coerce')
        
        # Sort dataframe by timestamp to ensure chronological order
        plot_df = plot_df.sort_values('timestamp')
        
        # Create the figure
        fig = go.Figure()

        # Note: Day separators were removed due to compatibility issues with Plotly and pandas timestamps
        
        # Determine if we should color by location
        color_by_location = False
        if location_col and len(selected_locations) > 1:
            color_by_location = True
        
        # Add traces for each metric
        for metric in selected_metrics:
            display_name = metric.split('.')[-1].capitalize() if '.' in metric else metric.capitalize()
            
            if color_by_location:
                # Create separate traces for each location
                for location in selected_locations:
                    location_df = plot_df[plot_df[location_col] == location].copy()
                    if not location_df.empty:
                        try:
                            fig.add_trace(go.Scatter(
                                x=location_df['timestamp'],
                                y=location_df[metric],
                                mode='lines+markers',
                                name=f"{display_name} - {location}",
                                hovertemplate=f'{location}: %{{y:.1f}} - %{{x}}<extra></extra>'
                            ))
                        except Exception as e:
                            st.warning(f"Could not plot {metric} for {location}: {str(e)}")
            else:
                try:
                    fig.add_trace(go.Scatter(
                        x=plot_df['timestamp'],
                        y=plot_df[metric],
                        mode='lines+markers',
                        name=display_name,
                        hovertemplate='%{y:.1f} - %{x}<extra></extra>'
                    ))
                except Exception as e:
                    st.warning(f"Could not plot {metric}: {str(e)}")
        
        fig.update_layout(
            title="Weather Metrics Trend",
            xaxis_title="Time",
            yaxis_title="Value",
            legend_title="Metrics",
            hovermode="x unified",
            xaxis=dict(
                tickformat="%H:%M\n%b %d",
                tickangle=-45
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Function to show temperature trends
def show_temperature_trends(filtered_df):
    # Remove the "Temperature Trends Over Time" header
    # st.header("Temperature Trends Over Time")
    
    # Determine temperature column
    if 'main.temp' in filtered_df.columns:
        temp_col = 'main.temp'
    elif 'temperature' in filtered_df.columns:
        temp_col = 'temperature'
    else:
        temp_col = None
        
    if temp_col and 'timestamp' in filtered_df.columns:
        # Ensure temperature column is numeric
        filtered_df[temp_col] = pd.to_numeric(filtered_df[temp_col], errors='coerce')
        
        # Create a copy of the dataframe for plotting
        plot_df = filtered_df.copy()
        plot_df = plot_df.sort_values('timestamp')
        
        # Determine location column
        location_col = None
        for col_name in ['requested_province', 'province', 'location', 'city']:
            if col_name in plot_df.columns:
                location_col = col_name
                break
        
        # 1. FIRST: Temperature Insights
        st.subheader("Temperature Insights")
        
        # Calculate temperature change rate
        if len(filtered_df) > 1:
            temp_change = filtered_df[temp_col].diff().mean() * 24  # Average hourly change * 24 for daily rate
            
            if abs(temp_change) < 0.1:
                trend_message = "อุณหภูมิค่อนข้างคงที่"
            elif temp_change > 0:
                trend_message = f"อุณหภูมิเริ่มสูงขึ้นประมาณ{abs(temp_change):.1f}°C ต่อวัน"
            else:
                trend_message = f"อุณหภูมิเริ่มลดลงประมาณ{abs(temp_change):.1f}°C ต่อวัน"
            
            st.info(f"📊 Insight: {trend_message}")
        
        # Time of day insight
        # Calculate hourly stats for insights
        hour_stats = filtered_df.copy()
        hour_stats['hour'] = hour_stats['timestamp'].dt.hour
        hourly_temps = hour_stats.groupby('hour')[temp_col].agg(['mean', 'min', 'max']).reset_index()
        
        hottest_hour = hourly_temps.loc[hourly_temps['mean'].idxmax(), 'hour']
        coldest_hour = hourly_temps.loc[hourly_temps['mean'].idxmin(), 'hour']
        
        st.info(f"🌡️ เวลาที่ร้อนที่สุดของวันโดยทั่วไปจะอยู่ที่ประมาณ {hottest_hour}:00,ในขณะที่ช่วงที่หนาวที่สุดอยู่ราวๆ {coldest_hour}:00.")
        
        # Daily pattern insight
        daily_temp_range = hourly_temps['mean'].max() - hourly_temps['mean'].min()
        if daily_temp_range > 10:
            variation_msg = "มีการเปลี่ยนแปลงของอุณหภูมิมากตลอดทั้งวัน"
        elif daily_temp_range > 5:
            variation_msg = "มีการเปลี่ยนแปลงของอุณหภูมิในระดับปานกลางตลอดทั้งวัน"
        else:
            variation_msg = "อุณหภูมิจะค่อนข้างคงที่ตลอดทั้งวัน"
        
        st.info(f"🌤️ {variation_msg} ช่วงอุณหภูมิที่เปลี่ยนแปลงในแต่ละวันโดยประมาณ {daily_temp_range:.1f}°C.")
        
        # 2. SECOND: Temperature Statistics
        st.subheader("Temperature Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        avg_temp = filtered_df[temp_col].mean()
        with col1:
            st.metric("🌡️ Average Temperature", f"{avg_temp:.1f} °C")
        
        with col2:
            max_temp = filtered_df[temp_col].max()
            max_temp_time = filtered_df.loc[filtered_df[temp_col].idxmax(), 'timestamp']
            st.metric("🔥 Maximum Temperature", f"{max_temp:.1f} °C")
            st.caption(f"Recorded at: {max_temp_time.strftime('%Y-%m-%d %H:%M')}")
        
        with col3:
            min_temp = filtered_df[temp_col].min()
            min_temp_time = filtered_df.loc[filtered_df[temp_col].idxmin(), 'timestamp']
            st.metric("❄️ Minimum Temperature", f"{min_temp:.1f} °C")
            st.caption(f"Recorded at: {min_temp_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Temperature variation by time of day
        st.subheader("Temperature Variation by Time of Day")
        
        # Group by hour and calculate statistics
        hour_stats = filtered_df.copy()
        hour_stats['hour'] = hour_stats['timestamp'].dt.hour
        hourly_temps = hour_stats.groupby('hour')[temp_col].agg(['mean', 'min', 'max']).reset_index()
        
        # Create hourly temperature chart
        fig_hourly = go.Figure()
        
        # Add range (min to max)
        fig_hourly.add_trace(go.Scatter(
            x=hourly_temps['hour'],
            y=hourly_temps['max'],
            fill=None,
            mode='lines',
            line_color='rgba(231,107,243,0.2)',
            showlegend=False,
            name='Max'
        ))
        
        fig_hourly.add_trace(go.Scatter(
            x=hourly_temps['hour'],
            y=hourly_temps['min'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(231,107,243,0.2)',
            name='Temperature Range'
        ))
        
        # Add mean line
        fig_hourly.add_trace(go.Scatter(
            x=hourly_temps['hour'],
            y=hourly_temps['mean'],
            mode='lines+markers',
            line=dict(color='rgb(231,107,243)', width=3),
            name='Average Temperature'
        ))
        
        fig_hourly.update_layout(
            title="Average Temperature by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Temperature (°C)",
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 24, 2)),
                ticktext=[f"{h}:00" for h in range(0, 24, 2)]
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # 3. THIRD: Temperature Patterns by Hour and Day
        # Temperature heatmap by hour and day
        if len(filtered_df) > 24:
            st.subheader("Temperature Patterns by Hour and Day")
            
            # Extract day and hour for heatmap
            heatmap_df = filtered_df.copy()
            heatmap_df['day'] = heatmap_df['timestamp'].dt.day
            heatmap_df['hour'] = heatmap_df['timestamp'].dt.hour
            
            # Create pivot table for heatmap
            temp_pivot = heatmap_df.pivot_table(
                values=temp_col, 
                index='day',
                columns='hour',
                aggfunc='mean'
            )
            
            # Create an enhanced heatmap
            fig_heatmap = px.imshow(
                temp_pivot,
                labels=dict(x="Hour of Day", y="Day of Month", color="Temperature (°C)"),
                x=temp_pivot.columns,
                y=temp_pivot.index,
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            
            fig_heatmap.update_layout(
                title="Temperature Heatmap by Hour and Day",
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(0, 24, 3)),
                    ticktext=[f"{h}:00" for h in range(0, 24, 3)]
                ),
                coloraxis_colorbar=dict(
                    title="°C",
                ),
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("Temperature or timestamp data not available in the dataset.")

# Main app logic
if LAKEFS_AVAILABLE:
    # Load data automatically instead of using a button
    if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
        with st.spinner("Loading data from LakeFS..."):
            df, success = load_data_from_lakefs()
            if success:
                st.session_state['df'] = df
                st.session_state['data_loaded'] = True
                st.success("Data loaded successfully!")
            else:
                st.session_state['data_loaded'] = False
                st.error("Failed to load data. Please check your LakeFS connection.")
    
    # Check if data is loaded
    if st.session_state.get('data_loaded', False):
        df = st.session_state['df']
        
        # Sidebar options
        st.sidebar.header("Data Analysis Options")
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type",
            ["Weather Overview", "Temperature Trends"]
        )
        
        # Determine location column
        location_col = None
        for col_name in ['location', 'location_name', 'requested_province', 'province', 'city']:
            if col_name in df.columns:
                location_col = col_name
                break
        
        # Location filter using dropdown
        selected_locations = []
        if location_col:
            # แปลงข้อมูลให้เป็น string แล้ว
            all_locations = df[location_col].astype(str).unique().tolist()
            all_locations.sort()  # เรียงลำ
            
            # Add an "All Locations" option
            dropdown_options = ["All Locations"] + all_locations
            
            # Create the dropdown
            selected_option = st.sidebar.selectbox(
                "Select Location",
                options=dropdown_options,
                index=0  # Default to "All Locations"
            )
            
            # Process the selection
            if selected_option == "All Locations":
                selected_locations = all_locations
            else:
                selected_locations = [selected_option]
        
        # Date range filter
        date_range = None
        if 'timestamp' in df.columns:
            date_range = st.sidebar.date_input(
                "Select Date Range",
                [df['timestamp'].min().date(), df['timestamp'].max().date()],
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date()
            )
        
        # Filter data
        filtered_df = filter_data(df, location_col, selected_locations, date_range)
        
        # Show selected analysis
        if analysis_type == "Weather Overview":
            show_weather_overview(filtered_df, location_col, selected_locations)
        elif analysis_type == "Temperature Trends":
            show_temperature_trends(filtered_df)
    else:
        st.info("👈 Please configure your LakeFS connection settings in the sidebar and click 'Load Data'")
else:
    st.error("LakeFS client not available. Please install the lakefs_client package with: pip install lakefs-client")
    
    # Display sample data for demonstration
    st.subheader("Sample Data (Demo Mode)")
    sample_df = generate_sample_data()
    st.line_chart(sample_df.set_index('timestamp')[['main.temp', 'main.humidity', 'wind.speed', 'precipitation']])
    st.warning("This is sample data. Install lakefs-client and connect to LakeFS to see real data.")

# Debug section
if st.checkbox("Show Debug Info", False):
    st.subheader("Debug Information")
    
    if 'filtered_df' in locals():
        # Show dataframe info
        st.write("DataFrame Info:")
        buffer = io.StringIO()
        filtered_df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        # Show column types
        st.write("Column Types:")
        st.write(filtered_df.dtypes)
        
        # Show sample data
        st.write("Sample Data:")
        st.write(filtered_df.head())
    else:
        st.warning("No filtered data available yet.")
        
        # Show session state for debugging
        st.write("Session State Keys:")
        st.write(list(st.session_state.keys()))
        
        if 'df' in st.session_state:
            st.write("Raw dataframe is available in session state")
            st.write("Sample from raw dataframe:")
            st.write(st.session_state['df'].head())
