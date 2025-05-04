import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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
            ["Temperature Trends", "Weather Distribution", "ML Predictions"]
        )
        
        if analysis_type == "Temperature Trends":
            st.header("Temperature Trends Over Time")
            
            # Visualization 1: Temperature Line Chart
            fig1 = px.line(df, 
                          x='timestamp', 
                          y='temperature',
                          title='Temperature Variation Over Time',
                          labels={'temperature': 'Temperature (°C)', 
                                 'timestamp': 'Time'})
            st.plotly_chart(fig1, use_container_width=True)
            
            # Insight
            st.info("📊 Insight: The temperature data shows clear daily patterns with peaks during midday hours and troughs during early morning.")
            
        elif analysis_type == "Weather Distribution":
            st.header("Weather Condition Distribution")
            
            # Visualization 2: Weather Distribution
            weather_counts = df['weather_main'].value_counts()
            fig2 = px.bar(weather_counts, 
                          title='Distribution of Weather Conditions',
                          labels={'value': 'Count', 
                                 'index': 'Weather Condition'})
            st.plotly_chart(fig2, use_container_width=True)
            
            # Insight
            st.info("📊 Insight: The distribution shows the predominant weather patterns in the region.")
            
        else:  # ML Predictions
            st.header("Temperature Prediction Model")
            
            # Prepare data for ML
            df['hour'] = df['timestamp'].dt.hour
            X = df[['hour']]
            y = df['temperature']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Model metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{r2:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}°C")
            
            # Visualization of predictions
            results_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred,
                'Hour': X_test['hour']
            })
            
            fig3 = px.scatter(results_df, 
                             x='Hour', 
                             y=['Actual', 'Predicted'],
                             title='Actual vs Predicted Temperature by Hour',
                             labels={'value': 'Temperature (°C)', 
                                    'Hour': 'Hour of Day'})
            st.plotly_chart(fig3, use_container_width=True)
            
            # Model insights
            st.info("🤖 Model Insight: The linear regression model shows how temperature varies predictably with the time of day.")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("No data found. Please run the pipeline first.")
