import json
from typing import List
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field, ConfigDict, model_validator
from predict import predict_data, model, MODEL_PATH

METRICS_PATH = MODEL_PATH.parent / "metrics.json"

app = FastAPI(
    title="California Housing Price API",
    description="Predicts the median house value of a California district, "
                "using a model trained on the 1990 California census data.",
)

class HouseData(BaseModel):
    """
    Pydantic BaseModel representing one California district.

    Attributes:
        median_income (float): Median income in $10,000s (8.3 means $83,000).
        house_age (float): Median house age in years.
        avg_rooms (float): Average number of rooms per household.
        avg_bedrooms (float): Average number of bedrooms per household.
        population (float): Number of people living in the district.
        avg_occupancy (float): Average number of people per household.
        latitude (float): Latitude, limited to California (32 to 42).
        longitude (float): Longitude, limited to California (-125 to -114).
    """
    median_income: float = Field(gt=0, description="Median income in $10,000s")
    house_age: float = Field(ge=0, description="Median house age in years")
    avg_rooms: float = Field(gt=0, description="Average rooms per household")
    avg_bedrooms: float = Field(gt=0, description="Average bedrooms per household")
    population: float = Field(gt=0, description="People living in the district")
    avg_occupancy: float = Field(gt=0, description="Average people per household")
    latitude: float = Field(ge=32, le=42, description="Must be inside California")
    longitude: float = Field(ge=-125, le=-114, description="Must be inside California")

    # Example shown in /docs when you click "Try it out" (a real district near Berkeley)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "median_income": 8.3252,
            "house_age": 41,
            "avg_rooms": 6.98,
            "avg_bedrooms": 1.02,
            "population": 322,
            "avg_occupancy": 2.56,
            "latitude": 37.88,
            "longitude": -122.23,
        }
    })

    @model_validator(mode="after")
    def check_bedrooms(self):
        """A household can't have more bedrooms than rooms."""
        if self.avg_bedrooms > self.avg_rooms:
            raise ValueError("avg_bedrooms cannot be greater than avg_rooms")
        return self

    def to_features(self):
        """Return the features in the same order the model was trained on."""
        return [self.median_income, self.house_age, self.avg_rooms, self.avg_bedrooms,
                self.population, self.avg_occupancy, self.latitude, self.longitude]

class PriceResponse(BaseModel):
    predicted_price: float

class BatchRequest(BaseModel):
    houses: List[HouseData] = Field(min_length=1, max_length=100)

class BatchResponse(BaseModel):
    predictions: List[float]


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Check that the API is running and which model is loaded."""
    return {"status": "healthy", "model": type(model).__name__}

@app.post("/predict", response_model=PriceResponse)
async def predict_price(house: HouseData):
    """
    Predict the median house value for one district.
    Example:
        POST /predict
        {
            "median_income": 8.3252, "house_age": 41, "avg_rooms": 6.98, "avg_bedrooms": 1.02,
            "population": 322, "avg_occupancy": 2.56, "latitude": 37.88, "longitude": -122.23
        }
        Response:
        {
            "predicted_price": <price in dollars>
        }
        The real median value for this district is $452,600.
    """
    try:
        prediction = predict_data([house.to_features()])
        return PriceResponse(predicted_price=round(float(prediction[0]), 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Predict house values for up to 100 districts in one request."""
    try:
        predictions = predict_data([house.to_features() for house in request.houses])
        return BatchResponse(predictions=[round(float(p), 2) for p in predictions])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """Return the loaded model type and the test scores saved by train.py."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found. Run train.py first.")
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return {
        "model_type": type(model).__name__,
        "features": list(HouseData.model_fields.keys()),
        **metrics,
    }
