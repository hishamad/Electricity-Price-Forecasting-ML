import os
import datetime
import time
import requests
import pandas as pd
import json
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import openmeteo_requests
import requests_cache
from retry_requests import retry
import hopsworks
import hsfs
from pathlib import Path
from datetime import datetime, timedelta

def get_historical_weather(start_date,  end_date, latitude, longitude):

    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max", "wind_direction_10m_dominant", "sunshine_duration"]
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(3).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(4).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s"),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
    daily_data["sunshine_duration"] = daily_sunshine_duration

    daily_dataframe = pd.DataFrame(data = daily_data)
    daily_dataframe = daily_dataframe.dropna()
    return daily_dataframe




import pandas as pd
import os

def process_energy_data():
    target_rows = [
        "Hydro Water Reservoir - Actual Aggregated [MW]",
        "Nuclear - Actual Aggregated [MW]",
        "Other - Actual Aggregated [MW]",
        "Wind Onshore - Actual Aggregated [MW]"
    ]
    data_folder = "../data"
    aggregated_data = pd.DataFrame()

    files_to_read = [
        "Actual Generation per Production Type_202201010000-202301010000.csv",
        "Actual Generation per Production Type_202301010000-202401010000.csv",
        "Actual Generation per Production Type_202401010000-202501010000.csv"
    ]


    for filename in os.listdir(data_folder):
        if filename.endswith(".csv") and filename in files_to_read:
            file_path = os.path.join(data_folder, filename)

            with open(file_path, "r") as file:
                lines = file.readlines()

            descriptor = [col.strip('"') for col in lines[0].strip().split(",")]
            data = [line.strip().split(",") for line in lines[1:]]

            df = pd.DataFrame(data, columns=descriptor)

            df['Hour'] = df["MTU"].str.split(" ").str[1]
            df = df[df["Hour"].str.endswith(":00")]

            df['Date'] = df["MTU"].str.split(" ").str[0]

            for column in target_rows:
                df[column] = df[column].str.replace('"', '', regex=False)
                df[column] = pd.to_numeric(df[column], errors='coerce')

            daily_avg = df.groupby('Date')[target_rows].mean().reset_index()
            aggregated_data = pd.concat([aggregated_data, daily_avg], ignore_index=True)

    return aggregated_data



def fetch_data_for_today(ELECTRICITY_API_TOKEN):

    base_url = "https://web-api.tp.entsoe.eu/api"
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    time_interval = f"{yesterday.strftime('%Y-%m-%dT%H:%MZ')}/{today.strftime('%Y-%m-%dT%H:%MZ')}"

    full_url = (
        f"{base_url}?"
        f"securityToken={ELECTRICITY_API_TOKEN}&"
        f"documentType=A73&"
        f"processType=A16&"
        f"in_Domain=10YSE-1--------K&"
        f"timeInterval={time_interval}"
    )

    try:
        response = requests.get(full_url)
        response.raise_for_status()
        data = response.content
        print("Data fetched successfully:")
        print(data)

        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None