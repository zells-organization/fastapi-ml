from pydantic import BaseModel


class StockIn(BaseModel):
    ticker: str
