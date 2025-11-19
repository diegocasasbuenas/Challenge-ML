import fastapi
import pandas as pd
from pydantic import BaseModel, validator
from typing import List
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from challenge.model import DelayModel

app = fastapi.FastAPI()


model = DelayModel()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("MES")
    def validate_mes(cls, v):
        if v < 1 or v > 12:
            raise ValueError("MES must be between 1 and 12")
        return v

    @validator("TIPOVUELO")
    def validate_tipovuelo(cls, v):
        if v not in ["N", "I"]:
            raise ValueError("TIPOVUELO must be N or I")
        return v


class PredictRequest(BaseModel):
    flights: List[Flight]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    flights_data = [flight.dict() for flight in request.flights]
    df = pd.DataFrame(flights_data)

    features = model.preprocess(df)
    predictions = model.predict(features)

    return {"predict": predictions}
