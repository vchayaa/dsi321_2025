# Real-Time Weather Data Pipeline

A real-time data pipeline for collecting, processing, and analyzing weather data using Prefect 3, Docker, Jupyter, and LakeFS.

## Project Overview

This project implements a data pipeline that:

1. Collects real-time weather data from the OpenWeatherMap API
2. Processes and transforms the data
3. Stores both raw and processed data in LakeFS (using the existing "project324" repository)
4. Provides tools for data analysis using Jupyter notebooks
5. Visualizes weather data with Streamlit

## Technologies Used

- **Prefect 3**: Workflow orchestration
- **Docker**: Containerization
- **Jupyter**: Interactive data analysis
- **LakeFS**: Data versioning and management
- **Python**: Programming language
- **Pandas/PyArrow**: Data manipulation
- **Streamlit/Plotly**: Data visualization

## Project Structure

```
project_repo/
├── docker/
│   ├── Dockerfile                # Docker image configuration
│   └── docker-compose.yml        # Docker services configuration
├── notebooks/
│   └── weather_analysis.ipynb    # Jupyter notebook for data analysis
├── src/
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration settings (single source of truth)
│   ├── deploy.py                 # Deployment script for Prefect
│   ├── main.py                   # Main pipeline implementation
│   └── utils.py                  # Utility functions
├── visualization/
│   └── app.py                    # Streamlit visualization app
├── .env                          # Environment variables
├── README.md                     # Project documentation
├── STRUCTURE.md                  # Project structure documentation
└── requirements.txt              # Python dependencies
```

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- OpenWeatherMap API key
- LakeFS repository (already created: "project324")

### Installation Steps

1. Clone the repository:
   ```
   git clone <repository-url>
   cd project_repo
   ```

2. Ensure your LakeFS repository "project324" is set up and accessible

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

### Manual Execution

To run the pipeline manually:

```
python -m src.pipeline
```

This will:
- Collect weather data for Thai provinces (Pathum Thani, Bangkok, Chiang Mai, Phuket)
- Collect weather data for international cities
- Process and save the data
- Upload the data to your LakeFS repository "project324"

### Scheduled Execution

To create a scheduled deployment:

```
python -m src.deploy
```

This will deploy the pipeline to run at the interval specified in your configuration.

## Data Analysis

1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open the `notebooks/weather_analysis.ipynb` notebook

3. Follow the instructions in the notebook to analyze the collected weather data

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
- Province data is stored in the "province" directory
- City data is stored in the "city" directory
- Both raw and processed data are versioned

You can access the LakeFS UI at http://localhost:8001 (or your configured port) to browse, compare, and manage your data versions.

## Configuration

All configuration is centralized in `src/config.py`:

- Weather API settings
- Province and city definitions
- Data storage paths
- LakeFS connection details
- Prefect settings

You can override these settings by updating the `.env` file.

## Customization

You can customize the pipeline by:

1. Adding more provinces or cities in the `PROVINCES` and `CITIES` variables in `src/config.py`
2. Modifying the data collection interval in `.env`
3. Adding more data processing steps in `src/main.py`
4. Extending the visualization in `visualization/app.py`

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