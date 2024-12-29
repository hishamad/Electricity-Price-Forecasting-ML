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
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO


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


def get_daily_weather(past_days, forecast_days, latitude, longitude):
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max", "wind_direction_10m_dominant", "sunshine_duration"],
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(3).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(4).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["sunshine_duration"] = daily_sunshine_duration
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant

    daily_dataframe = pd.DataFrame(data = daily_data)
    print(daily_dataframe)
    return(daily_dataframe)



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



def parse(data, target_psr_types=["B12", "B14", "B19", "B20"]):
    root = ET.parse(BytesIO(data)).getroot()
    ns = {'ns': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0'}
    psr_type_mapping = {
        "B12": "hydro_mw",
        "B14": "nuclear_mw",
        "B19": "wind_mw",
        "B20": "other_mw"
    }
    aggregated_data = {psr: {"total_quantity": 0, "count": 0} for psr in target_psr_types}

    for times in root.findall('ns:TimeSeries', ns):
        psrType = times.find('ns:MktPSRType/ns:psrType', ns).text
        if psrType in target_psr_types:
            for period in times.findall('ns:Period', ns):
                for point in period.findall('ns:Point', ns):
                    quantity = float(point.find('ns:quantity', ns).text)
                    aggregated_data[psrType]["total_quantity"] += quantity
                    aggregated_data[psrType]["count"] += 1

    data_row = {
        psr_type_mapping[psrType]: (
            aggregated_data[psrType]["total_quantity"] / aggregated_data[psrType]["count"]
            if aggregated_data[psrType]["count"] > 0 else 0
        )
        for psrType in target_psr_types
    }

    ordered_columns = ["hydro_mw", "nuclear_mw", "other_mw", "wind_mw"]
    ordered_row = {col: data_row[col] for col in ordered_columns}

    df = pd.DataFrame([ordered_row])
    return df



def fetch_data_for_yesterday(ELECTRICITY_API_TOKEN):

    base_url = "https://web-api.tp.entsoe.eu/api"
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    time_interval = f"{yesterday.strftime('%Y-%m-%dT%H:%MZ')}/{today.strftime('%Y-%m-%dT%H:%MZ')}"

    full_url = (
        f"{base_url}?"
        f"securityToken={ELECTRICITY_API_TOKEN}&"
        f"documentType=A75&"
        f"processType=A16&"
        f"in_Domain=10YSE-1--------K&"
        f"timeInterval={time_interval}"
    )

    try:
        response = requests.get(full_url)
        response.raise_for_status()
        data = response.content
        print("Data fetched successfully:")
        parsed_data = parse(data)
        print(parsed_data)

        return parsed_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    return None


def get_el_price(date):
    el_price_df = pd.DataFrame()
    region = "SE3"
    url = f'https://www.elprisetjustnu.se/api/v1/prices/{date.year}/{date.month:02}-{date.day:02}_{region}.json'
    response = requests.get(url)
    if response.status_code == 200:
        data_json = response.json()
        price = 0
        for item in data_json:  
            price += item['SEK_per_kWh']
        price /= len(data_json)

        el_price_df['price'] = [price]
        el_price_df['date'] = [date.strftime('%Y-%m-%d')]
    else:
        print(f"ELpris API Error")

    return el_price_df