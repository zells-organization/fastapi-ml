from fastapi import FastAPI, HTTPException
from model import convert, predict
from models import StockIn, StockOut

app = FastAPI()


@app.get('/ping')
def pong():
    return {"ping": "pong"}
