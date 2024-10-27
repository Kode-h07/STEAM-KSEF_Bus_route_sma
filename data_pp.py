import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "버스정류소현황.csv"

df = pd.read_csv(file_path, encoding="euc-kr")

filtered_df = df[df["위치"].str.contains("경기도 가평군 설악면", na=False)].copy()

filtered_df.loc[:, "lat"] = filtered_df["WGS84위도"]
filtered_df.loc[:, "lon"] = filtered_df["WGS84경도"]
filtered_df.loc[:, "value"] = 4

min_lon = filtered_df["lon"].min()
min_lat = filtered_df["lat"].min()

filtered_df.loc[:, "x"] = filtered_df["lon"].apply(
    lambda x: int((x - min_lon) * (10**3))
)
filtered_df.loc[:, "y"] = filtered_df["lat"].apply(
    lambda y: int((y - min_lat) * (10**3))
)

min_x = filtered_df["x"].min()
min_y = filtered_df["y"].min()
filtered_df.loc[:, "x"] = filtered_df["x"].apply(lambda x: int(x - min_x) + 10)
filtered_df.loc[:, "y"] = filtered_df["y"].apply(lambda y: int(y - min_y) + 10)

filtered_df.loc[:, "nodes"] = filtered_df[["x", "y"]].apply(tuple, axis=1)

filtered_df = filtered_df.drop_duplicates(subset=["nodes"])

filtered_df = filtered_df.drop(columns=["시군명", "중앙차로여부", "관할관청", "위치"])


filtered_df = filtered_df.reset_index(drop=True)


filtered_df = filtered_df.drop_duplicates(subset=["정류소명"])

"pheromone = attrativeness of a food to the slime => ex)num of passengers in that station can be set"
filtered_df["pheromone"] = np.random.randint(1, 11, size=len(filtered_df))

filtered_df.to_csv("seorak.csv", encoding="euc-kr", index=True)
