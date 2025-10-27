import pandas as pd
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from future_stars.predict import predict
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title=os.getenv("APP_NAME", "Future Stars API"))

@app.get("/")
def root():
    return {"message": os.getenv("APP_MESSAGE", "Hello from Future Stars API!")}


# Schema for one player (to handle user input)
class PlayerData(BaseModel):
    player_name: str
    position: str
    nationality: str
    age: int
    minutes_played: int
    goals_assists: int | float = 0
    expected_goals: float = 0.0
    expected_assists: float = 0.0
    tackles: int | float = 0
    blocks: int | float = 0
    clearances: int | float = 0
    progressive_passes: int | float = 0
    save_percent: float = 0.0


def json_to_scout_row(p: PlayerData) -> pd.DataFrame:
    """Convert JSON schema into a row for prediction."""
    row = {
        "Player": p.player_name,
        "Nation": p.nationality,
        "Age": p.age,
        "Pos": p.position,
        "Min": p.minutes_played,
        "G+A": p.goals_assists,
        "xG": p.expected_goals,
        "xAG": p.expected_assists,
        "Tkl": p.tackles,
        "Blocks_stats_defense": p.blocks,
        "Clr": p.clearances,
        "PrgP": p.progressive_passes,
        "Save%": p.save_percent
    }
    return pd.DataFrame([row])


# Predict one player
@app.post("/predict_one")
def predict_one(player: PlayerData):
    raw_df = json_to_scout_row(player)
    pred_df = predict(data=raw_df)
    return pred_df.iloc[0].to_dict()


# Predict File CSV
@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    raw_df = pd.read_csv(file.file)
    pred_df = predict(data=raw_df)  
    return pred_df.to_dict(orient="records")
