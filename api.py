import os
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title=os.getenv("APP_NAME", "MyAPI"))

@app.get("/")
def root():
    return {"message": os.getenv("APP_MESSAGE", "Hello from API!")}

