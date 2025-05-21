# Weather Data Schema

## Data Structure

The weather data pipeline collects and processes weather data with the following schema:

| Column Name         | Data Type                     | Description                                           |
|---------------------|-------------------------------|-------------------------------------------------------|
| timestamp           | datetime64[ns, Asia/Bangkok]  | Date and time when the weather data was collected     |
| minute              | int64                         | Minute component of the timestamp                     |
| created_at          | datetime64[ns, Asia/Bangkok]  | Date and time when the record was created             |
| province            | category                      | Thai province where data was collected                |
| location_name       | category                      | Name of the location as provided in configuration     |
| api_location        | category                      | Location name returned by the OpenWeatherMap API      |
| weather_main        | category                      | Main weather condition (Clear, Clouds, Rain, etc.)    |
| weather_description | category                      | Detailed weather description                          |
| main.temp           | float32                       | Temperature in Celsius                                |
| main.humidity       | int64                         | Humidity percentage                                   |
| wind.speed          | float32                       | Wind speed in meters per second                       |
| rain.1h             | float32                       | Rainfall volume for the last 1 hour in mm             |
| rain.3h             | float32                       | Rainfall volume for the last 3 hours in mm            |
| precipitation       | float32                       | Total precipitation (sum of rainfall and snowfall)    |
| year                | category                      | Year component of the timestamp (for partitioning)    |
| month               | category                      | Month component of the timestamp (for partitioning)   |
| day                 | category                      | Day component of the timestamp (for partitioning)     |
| hour                | category                      | Hour component of the timestamp (for partitioning)    |

## Key Columns

The following columns are used as key metrics for analysis:
- main.temp
- main.humidity
- wind.speed
- rain.1h
- rain.3h
- precipitation

## Data Format

The data is stored in Parquet format in the LakeFS repository, with partitioning by year, month, day, and hour for efficient querying.

## Data Storage

The data is stored in the LakeFS repository "weather" with the following structure:

```
weather/
├── main/                  # Main branch
│   └── weather.parquet/   # Partitioned parquet dataset
│       ├── year=2025/
│       │   ├── month=4/
│       │   │   ├── day=10/
│       │   │   │   ├── hour=7/
│       │   │   │   │   └── [parquet files]
│       │   │   │   ├── hour=8/
│       │   │   │   │   └── [parquet files]
│       │   │   │   └── ...
│       │   │   └── ...
│       │   └── ...
│       └── ...
```

## Optimization

The schema is optimized for storage and query performance:
- Categorical data types are used for columns with repeated values
- Float32 is used instead of Float64 for numeric measurements
- Partitioning is applied to support time-based queries
- Timestamps include timezone information (Asia/Bangkok)
