# Real-Time Weather Data Pipeline

A comprehensive data engineering solution that collects, processes, and visualizes real-time weather data from OpenWeatherMap API for Thai provinces and international cities.

## Key Features

- **Automated Data Collection**: Fetches weather data for Satitram and other 14 locations nearby every 15 minutes
- **Workflow Orchestration**: Leverages Prefect 3 for reliable, scheduled pipeline execution
- **Data Versioning**: Implements LakeFS for data versioning and lineage tracking
- **Bilingual Insights**: Interactive Streamlit dashboard with weather analytics 
- **Containerized Architecture**: Docker-based deployment for consistent environments
- **Optimized Storage**: Partitioned Parquet files with appropriate data types for efficient querying

## Technology Stack

- **Prefect 3**: Workflow orchestration and task management
- **Docker & Docker Compose**: Service containerization and orchestration
- **LakeFS**: Data lake management with Git-like versioning
- **Streamlit**: Interactive data visualization dashboard
- **Pandas/PyArrow**: Efficient data processing and storage

## Quick Start

1. Clone the repository
2. install requirement.txt by `pip install -r requirements.txt`
3. Run `docker-compose up -d` in the docker directory
4. Access the dashboard at http://localhost:8503
5. Explore data versions in LakeFS at http://localhost:8001

The system collects temperature, humidity, wind speed, rainfall data (1h/3h measurements) and precipitation , providing valuable insights through time-series analysis and daily patterns with Thai language support.

## Project Structure

```
projectfolder/
├── docker/                       # Docker configuration
│   ├── Dockerfile                # Streamlit app container
│   
├── make/                         # Build configurations
│   ├── Dockerfile.jupyter        # Jupyter notebook container
│   ├── Dockerfile.prefect-worker # Prefect worker container
│   ├── requirements.txt          # Python dependencies
│   └── wait-for-server.sh        # Service startup script
├── visualization/                # Dashboard components
│   └── app.py                    # Streamlit visualization app
├── work/                         # Workflow definitions
│   ├── myflow/                   # Prefect flow modules
│   │   └── 03_main_flow/         # Main data pipeline
│   │       ├── deploy.py         # Deployment configuration
│   │       ├── deploy-local.py   # Local deployment script
│   │       └── flow.py           # Main pipeline implementation
│  
├── docker-compose.yml        # Multi-container orchestration
├── README.md                     # Project documentation
├── SCHEMA.md                     # Data schema documentation
└── requirements.txt              # Python dependencies
```

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- OpenWeatherMap API key
- LakeFS repository (already created: "weather")

### Installation Steps

1. Clone the repository:
   ```
   git clone <repository-url>
   cd projectfolder
   ```

2. Ensure your LakeFS repository "weather" is set up and accessible

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

4. Start the Docker containers:
   ```
   cd docker
   docker-compose up -d
   ```

## Running the Pipeline

### Scheduled Execution

To create a scheduled deployment: open jupyter and open terminal then run

```
python deploy-local.py
```

This will deploy the pipeline to run at the interval specified in your configuration.

if it's not working may be your workpool don't run properly 

```
prefect worker start --pool 'default-agent-pool'
```

## Data Visualization

To visualize the weather data with Streamlit:

```
cd visualization
streamlit run app.py
```

This will start a Streamlit server and open a web browser with the visualization dashboard.

## LakeFS Integration

The pipeline is configured to work with your existing LakeFS repository "project324":

- Data is stored in the "main" branch by default
- Both raw and processed data are versioned

You can access the LakeFS UI at http://localhost:8001 (or your configured port) to browse, compare, and manage your data versions.

## Configuration

All configuration is centralized in `myflow/03_main_flow/flow.py`:

- Weather API settings
- Province and city definitions
- Data storage paths
- LakeFS connection details
- Prefect settings


## Customization

You can customize the pipeline by:

1. Adding more provinces and locations in the `PROVINCES` and `location` variables in `myflow/03_main_flow/flow.py`
2. Adding more data processing steps in `myflow/03_main_flow/flow.py`
3. Extending the visualization in `visualization/app.py`

## Troubleshooting

If you encounter issues:

1. Ensure LakeFS is running and accessible
2. Check that your OpenWeatherMap API key is valid
3. Verify that the data directories exist and are writable
4. Check the Prefect UI for any task failures

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenWeatherMap for providing the weather data API
- The Prefect, Docker, and LakeFS communities for their excellent documentation

## Visualization

![image alt](https://github.com/vchayaa/dsi321_2025/blob/ea33bc92f721253a041f9420398332064568929c/satit1.png)

- in "Current Weahter Metrics" will appear 5 Metrics as Temperature, Humidity, Wind speed, Rainfall(1h), Rainfall(3h)
- in "Current Weather Condition" will display how weather condition is from the last record and a description of main weather condition
- in "Weather Metrics Over Time" will show a graph that can be select which metric, location and Time range to display


![image alt](https://github.com/vchayaa/dsi321_2025/blob/ea33bc92f721253a041f9420398332064568929c/satit2.png) 

- in "Temperature Insights" will show summarize insights or data of the location, Time range that user selected in Thai
- in "Temperature Statsitics" will show Average Temp, Max temp and Min temp
- in "Temperature Variation by time by day" will show the temperature variation at different times of the day.



![image alt](https://github.com/vchayaa/dsi321_2025/blob/ea33bc92f721253a041f9420398332064568929c/satit3.png)

- in "Temperature Patterns by hour by day" shows a heatmap of temperature pattern of the day



![image alt](https://github.com/vchayaa/dsi321_2025/blob/ea33bc92f721253a041f9420398332064568929c/satit4.png)

- This page shows a K-means clustering machine learning that classifies weather conditions into 3 groups using temperature, humidity, wind speed and precipitaion, and provides recommendations for action in different weather conditions.
