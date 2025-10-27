import pandas as pd
from future_stars.predict import predict


df_fw = pd.DataFrame([{
    "Player": "Rahaf",
    "Pos": "FW",
    "Nationality": "Saudi Arabia",
    "Age": 22,
    "Min": 900,
    "G+A": 3,
    "xG": 2.5,
    "xAG": 1.0,
    "Tkl": 5,
    "Blocks_stats_defense": 2,
    "Clr": 1,
    "PrgP": 10,
    "Save_Pct": None,
    "CSPercent": None
}])

results_fw = predict(data=df_fw, model_path="model/model.pkl")
print(results_fw)



df_gk = pd.DataFrame([{
    "Player": "Sara",
    "Pos": "GK",
    "Nationality": "Saudi Arabia",
    "Age": 23,
    "Min": 1200,
    "G+A": 0,
    "xG": 0.0,
    "xAG": 0.0,
    "Tkl": 0,
    "Blocks_stats_defense": 0,
    "Clr": 0,
    "PrgP": 0,
    "Save_Pct": 0.82,          #
    "CSPercent": 0.55
}])

results_gk = predict(data=df_gk, model_path="model/model.pkl")
print(results_gk)
