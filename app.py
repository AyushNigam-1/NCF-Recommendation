import numpy as np
import os
import certifi
import sys
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from src.exception.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline
from dotenv import load_dotenv

load_dotenv()
ca = certifi.where()

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/recommend")
async def recommend_route(user_id: int, n: int = 5):
    try:
        predict_pipeline = PredictPipeline()
        recommendations = predict_pipeline.recommend_items(user_id=user_id, n=n)
        
        return JSONResponse(content={"recommended_movies": recommendations})

    except CustomException as ce:
        raise HTTPException(status_code=400, detail=str(ce))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during the prediction process.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)
