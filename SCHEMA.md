# Weather Data Schema

## Data Structure

The weather data pipeline collects and processes weather data with the following schema:

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| timestamp   | TEXT      | Date and time when the weather data was collected |
| city        | TEXT      | Name of the city |
| temperature | REAL      | Temperature in Celsius |
| humidity    | INTEGER   | Humidity percentage |
| wind_speed  | REAL      | Wind speed in meters per second |

## Key Columns

The following columns are used as key metrics for analysis:
- temperature
- humidity
- wind_speed

## Data Format

The data is stored in both CSV and Parquet formats in the LakeFS repository.

## Data Storage

The data is stored in the LakeFS repository "project324" with the following structure:

```
project324/
├── main/                  # Main branch
│   ├── province/          # Province weather data
│   │   └── province_data_{timestamp}.parquet
│   └── city/              # City weather data
│       └── city_data_{timestamp}.parquet
```