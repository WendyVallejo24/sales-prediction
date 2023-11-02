from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title = 'Big Mart Sales Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

model = load(pathlib.Path('model/train-v1.joblib'))

class InputData(BaseModel):
    Item_Weight:float = 12.9,
    Item_Visibility:float = 0.023721223,
    Item_MRP:float = 32.2432,
    Outlet_Establishment_Year:int = 1997


class OutputData(BaseModel):
    score:float=0.5029458945548542

@app.post('/score', response_model = OutputData)
def score(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict(model_input)

    return {'score':result}
