import pandas as pd

def init_dataset(uni_dim=False):

    data = pd.read_csv("examples/data/AirQuality/AirQuality.csv", sep=";")

    if uni_dim:
        data = data[["CO(GT)"]]

    else:
        data = data.drop(columns=['Date', 'Time'])

    data = data.applymap(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    data = data.fillna(0)

    return data.values.T

