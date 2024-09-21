import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
file_path = "버스정류소현황.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path, encoding="euc-kr")

# '위치' 컬럼에서 '경기도 가평군 설악면'이 포함된 행 필터링
filtered_df = df[df["위치"].str.contains("경기도 가평군 설악면", na=False)].copy()

# Extract latitude and longitude using .loc[]
filtered_df.loc[:, "lat"] = filtered_df["WGS84위도"]
filtered_df.loc[:, "lon"] = filtered_df["WGS84경도"]
filtered_df.loc[:, "value"] = 4

# Get minimum latitude and longitude
min_lon = filtered_df["lon"].min()
min_lat = filtered_df["lat"].min()

# Scale lon and lat values
filtered_df.loc[:, "x"] = filtered_df["lon"].apply(
    lambda x: int((x - min_lon) * (10**3))
)
filtered_df.loc[:, "y"] = filtered_df["lat"].apply(
    lambda y: int((y - min_lat) * (10**3))
)

# Adjust x and y values to ensure they start from a base of 10
min_x = filtered_df["x"].min()
min_y = filtered_df["y"].min()
filtered_df.loc[:, "x"] = filtered_df["x"].apply(lambda x: int(x - min_x) + 10)
filtered_df.loc[:, "y"] = filtered_df["y"].apply(lambda y: int(y - min_y) + 10)

# Create nodes as tuples
filtered_df.loc[:, "nodes"] = filtered_df[["x", "y"]].apply(tuple, axis=1)

# Remove duplicates based on the nodes
filtered_df = filtered_df.drop_duplicates(subset=["nodes"])

filtered_df = filtered_df.drop(columns=["시군명", "중앙차로여부", "관할관청", "위치"])

# Save the result to a new CSV file
filtered_df.to_csv("seorak.csv", encoding="euc-kr", index=False)


plt.figure(figsize=(10, 8))
plt.scatter(filtered_df["x"], filtered_df["y"], c="blue", marker="o")

# Adding labels for each node

plt.title("Bus Stop Nodes")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid()
plt.xlim(0, filtered_df["x"].max() + 10)  # Adjust x limits for better visibility
plt.ylim(0, filtered_df["y"].max() + 10)  # Adjust y limits for better visibility
plt.show()
