import pandas as pd

data = pd.read_csv("baseline_config.csv")
load = data["total_building_load"]
load.values
