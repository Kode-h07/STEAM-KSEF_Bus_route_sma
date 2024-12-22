import pandas as pd
import numpy as np
import os
import chardet
import math
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the terminal/console handles EUC-KR encoding
import sys

sys.stdout.reconfigure(encoding="euc-kr")


def apply_hyperparameters(df, pheromone_scale, pheromone_offset):
    df["pheromone"] = (
        df["normalized_passenger_count"] * pheromone_scale
    ) + pheromone_offset
    return df


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# File path for bus stations CSV
file_path = "bus_stations.csv"

# Read the main data CSV using the EUC-KR encoding
df = pd.read_csv(file_path, encoding="euc-kr")

# Filter rows for specific bus stop locations
filtered_df = df[df["위치"].str.contains("경기도 가평군 설악면", na=False)].copy()

# Add latitude, longitude, and some additional columns for calculations
filtered_df.loc[:, "lat"] = filtered_df["WGS84lat"]
filtered_df.loc[:, "lon"] = filtered_df["WGS84lon"]

# Normalize latitude and longitude to a custom grid
min_lon = filtered_df["lon"].min()
min_lat = filtered_df["lat"].min()

filtered_df.loc[:, "x"] = filtered_df["lon"].apply(
    lambda x: int((x - min_lon) * (10**3))
)
filtered_df.loc[:, "y"] = filtered_df["lat"].apply(
    lambda y: int((y - min_lat) * (10**3))
)

# Adjust x, y to ensure all coordinates are positive and shifted appropriately
min_x = filtered_df["x"].min()
min_y = filtered_df["y"].min()
filtered_df.loc[:, "x"] = filtered_df["x"].apply(lambda x: int(x - min_x) + 10)
filtered_df.loc[:, "y"] = filtered_df["y"].apply(lambda y: int(y - min_y) + 10)

# Create a 'nodes' column combining x, y as tuples
filtered_df.loc[:, "nodes"] = filtered_df[["x", "y"]].apply(tuple, axis=1)

# Drop duplicate nodes and reset index
filtered_df = filtered_df.drop_duplicates(subset=["nodes"]).reset_index(drop=True)

# Remove unnecessary columns
filtered_df = filtered_df.drop(columns=["시군명", "중앙차로여부", "관할관청", "위치"])

# Drop duplicates in bus stop names
filtered_df = filtered_df.drop_duplicates(subset=["정류소명"])


# filtered_df.to_csv("seorak.csv", encoding="euc-kr", index=True)

file_path = "seorak.csv"

# Read the main data CSV using the EUC-KR encoding
df = pd.read_csv(file_path, encoding="euc-kr")

filtered_df = df


# Function to detect file encoding using chardet
def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        raw_data = file.read()
    return chardet.detect(raw_data)["encoding"]


# Function to safely read CSV with handling for encoding issues
def safe_read_csv(file_path, encoding):
    with open(file_path, mode="r", encoding=encoding, errors="replace") as file:
        return pd.read_csv(file)


# Initialize an array to store passenger counts
passenger_counts = []

# Path to the folder containing passenger CSV files
passengers_folder = "passengers"

# Iterate through each bus stop name in the filtered dataframe
for stop_name in filtered_df["정류소명"]:
    csv_file = f"{stop_name}.csv"
    file_path = os.path.join(passengers_folder, csv_file)

    if os.path.exists(file_path):
        try:
            # Detect the file encoding of the passenger CSV file
            file_encoding = detect_encoding(file_path)

            # Read the CSV file with the detected encoding (using safe_read_csv)
            stop_df = safe_read_csv(file_path, file_encoding)

            # Slice the data starting from the 2nd row and 7th column to the right and down
            passenger_data = stop_df.iloc[
                1:, 6:
            ]  # Start from 2nd row and 7th column (index 1, 6)

            # Convert the values to numeric, ignoring any errors (coerce invalid values to NaN)
            passenger_data = passenger_data.apply(pd.to_numeric, errors="coerce")

            # Sum all the numeric values (ignore NaN values)
            passenger_count = passenger_data.sum().sum()

            # Append the result to the passenger counts list
            passenger_counts.append(passenger_count)
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            passenger_counts.append(
                999
            )  # Use 999 as an error indicator if there's a problem
    else:
        passenger_counts.append(999)  # Use 999 if the file doesn't exist

# Add the passenger counts to the filtered dataframe
filtered_df["passenger_count"] = passenger_counts

valid_stations = filtered_df[filtered_df["passenger_count"] != 999].copy()

# Step 2: Create a mask for stations with missing passenger counts (999)
missing_stations = filtered_df[filtered_df["passenger_count"] == 999].copy()

# Step 3: For each missing station, find the closest valid station
for index, missing_station in missing_stations.iterrows():
    min_distance = float("inf")
    nearest_station_passenger_count = (
        999  # Default to 999 if no station is found (safety check)
    )

    # Get the x, y coordinates of the missing station
    missing_x = missing_station["x"]
    missing_y = missing_station["y"]

    # Loop through each valid station to calculate the distance
    for _, valid_station in valid_stations.iterrows():
        valid_x = valid_station["x"]
        valid_y = valid_station["y"]
        valid_passenger_count = valid_station["passenger_count"]

        # Calculate the Euclidean distance between the current valid station and the missing station
        distance = calculate_distance(missing_x, missing_y, valid_x, valid_y)

        # If this valid station is closer, update the minimum distance and passenger count
        if distance < min_distance:
            min_distance = distance
            nearest_station_passenger_count = valid_passenger_count
    filtered_df.loc[index, "passenger_count"] = nearest_station_passenger_count

    # Update the missing station with the passenger count of the nea

min_passenger_count = filtered_df["passenger_count"].min()
max_passenger_count = filtered_df["passenger_count"].max()

# Avoid division by zero in case all values are the same
if min_passenger_count != max_passenger_count:
    filtered_df["normalized_passenger_count"] = (
        filtered_df["passenger_count"] - min_passenger_count
    ) / (max_passenger_count - min_passenger_count)
else:
    filtered_df["normalized_passenger_count"] = (
        0  # All values are the same, so set to 0 (or 1 depending on the choice)
    )

# Print the normalized passenger count for verification

# Multiply the normalized passenger count by 15 and add 5
# hyperparameter
filtered_df["pheromone"] = (
    filtered_df["normalized_passenger_count"] * 19.861105709313733
) + 8.738

print(type(filtered_df))
# Optionally, save the results to a new CSV file
filtered_df.to_csv("seorak.csv", encoding="euc-kr", index=False)
