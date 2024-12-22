import csv
import numpy as np
import pandas as pd

# File paths
REFERENCE_CSV = "lines.csv"  # CSV containing reference numbers
ORIGINAL_CSV = "stations_final.csv"  # CSV to be filtered

# Header to insert as the first row
header = [
    "Unnamed: 0",
    "�����Ҹ�",
    "station_name",
    "station_id",
    "station_num",
    "WGS84lat",
    "WGS84lon",
    "lat",
    "lon",
    "value",
    "x",
    "y",
    "nodes",
    "passenger_count",
    "normalized_passenger_count",
    "adjusted_passenger_count",
    "pheromone",
]


# Step 1: Extract unique numbers from the reference CSV
def extract_reference_numbers(file_path):
    """Extracts unique numeric values from the given CSV file."""
    numbers = set()
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            for value in row:
                if value.strip().isdigit():
                    numbers.add(int(value))
    return numbers


reference_numbers = extract_reference_numbers(REFERENCE_CSV)
print(f"Extracted reference numbers: {reference_numbers}")


# Step 2: Filter rows based on the first column and store the result in a NumPy variable
def filter_csv_by_first_column(input_file, valid_numbers):
    """Filters rows where the first column value exists in valid_numbers and returns as a NumPy array."""
    valid_rows = []

    with open(input_file, mode="r", encoding="utf-8") as infile:
        reader = csv.reader(infile)

        for row in reader:
            if (
                row
                and row[0].strip().isdigit()
                and int(row[0].strip()) in valid_numbers
            ):
                valid_rows.append(row)

    # Convert the list of valid rows into a NumPy array
    return np.array(valid_rows)


# Get the filtered data as a NumPy array
valid_stations = filter_csv_by_first_column(ORIGINAL_CSV, reference_numbers)

# Insert the header as the first row
valid_stations_with_header = np.vstack([header, valid_stations])

filtered_df = pd.DataFrame(
    valid_stations_with_header[1:], columns=valid_stations_with_header[0]
)

# Ensure the 'normalized_passenger_count' column is numeric
filtered_df["normalized_passenger_count"] = pd.to_numeric(
    filtered_df["normalized_passenger_count"], errors="coerce"
)

# Check if the column contains NaN values and handle them (optional)
filtered_df["normalized_passenger_count"].fillna(0, inplace=True)
filtered_df["x"] = pd.to_numeric(filtered_df["x"], errors="coerce")
filtered_df["y"] = pd.to_numeric(filtered_df["y"], errors="coerce")
filtered_df["value"] = pd.to_numeric(filtered_df["value"], errors="coerce")


# Calculate the 'pheromone' column
filtered_df["pheromone"] = (
    filtered_df["normalized_passenger_count"] * 19.861105709313733
) + 8.738

output_txt_file = "filtered_data.txt"  # Path to your output text file

# Write the string representation of the DataFrame to the text file
with open(output_txt_file, "w", encoding="utf-8") as file:
    file.write(
        filtered_df.to_string(index=False)
    )  # index=False excludes the index from the output
