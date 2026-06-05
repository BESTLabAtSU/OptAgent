import pandas as pd

wea = pd.read_csv('SFH_1.csv')
selected = wea[['Time', 'temp_outdoor', 'solar']]

selected = selected.rename(columns={
    'Time': 'Time',
    'temp_outdoor': 'outdoor_temperature',
    'solar': 'solar_radiation'
})

selected.to_csv('weather.csv')