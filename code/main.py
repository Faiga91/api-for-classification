"""
API for a classification task, written using FastAPI.
"""
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from joblib import load
from pydantic import BaseModel  # pylint: disable=no-name-in-module

SCALER_FILE = './models/scaler.pkl'
MODEL_FILE =  './models/model.pkl'

# load the scaler
with open(SCALER_FILE, 'rb') as scaler_file:
    standard_scaler = load(scaler_file)

# load the model
with open(MODEL_FILE, 'rb') as classifier_file:
    decision_tree_classifier = load(classifier_file)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    GET function on the root folder.
    """
    data = {
        "page": "Home Page"
    }
    return templates.TemplateResponse("page.html", {"request": request, "data": data})

class Point(BaseModel): # pylint: disable=too-few-public-methods
    """
    The input to the model, it is a point with x and y coordinates.
    """
    x : float
    y : float


@app.post("/predict")
async def predict(point: Point):
    """
    Predict function that defines the end point for the forecastor.
    Input: X and Y.
    Output: the class as a ressult of the classification model.
    """
    input_array = np.array([[point.x], [point.y]]).reshape(1,2)
    scaled_input = standard_scaler.transform(input_array)
    model_output = decision_tree_classifier.predict(scaled_input.reshape(1,2))
    output_  = model_output[0].tolist()
    return {
        "response" : output_
    }
