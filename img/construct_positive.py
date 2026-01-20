import pandas as pd
import numpy as np
from geopy.distance import great_circle


city="Shenzhen"
csv_path = f""
df = pd.read_csv(csv_path)

df["region_id"] = np.arange(len(df), dtype=int)


def parse_coord(coord_str):
    coord_str = coord_str.strip("()")
    lat, lon = map(float, coord_str.split(","))
    return lat, lon

df[["lower_lat", "lower_lon"]] = df["WGS84_lower_left"].apply(lambda x: pd.Series(parse_coord(x)))
df[["upper_lat", "upper_lon"]] = df["WGS84_upper_right"].apply(lambda x: pd.Series(parse_coord(x)))

df["center_lat"] = (df["lower_lat"] + df["upper_lat"]) / 2
df["center_lon"] = (df["lower_lon"] + df["upper_lon"]) / 2

centers = df[["center_lat", "center_lon"]].to_numpy()
nearest_ids = []

for i, (lat1, lon1) in enumerate(centers):
    dists = np.array([
        great_circle((lat1, lon1), (lat2, lon2)).km
        for j, (lat2, lon2) in enumerate(centers)
    ])
    dists[i] = np.inf
    nearest_idx = dists.argmin()
    nearest_ids.append(nearest_idx)

output_path = f""
with open(output_path, "w") as f:
    for nid in nearest_ids:
        f.write(f"{nid}\n")
