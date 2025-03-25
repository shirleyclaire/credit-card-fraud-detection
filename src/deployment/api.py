from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from .inference_pipeline import RealTimeInference
import uvicorn
from typing import Dict, Any
from ..investigation.investigation_system import InvestigationSystem

class Transaction(BaseModel):
    card_id: str
    amount: float
    merchant: str
    timestamp: int
    additional_features: Dict[str, Any] = {}

app = FastAPI()
inference_pipeline = RealTimeInference(
    model_path='models/fraud_detector.joblib',
    preprocessor_path='models/preprocessor.joblib'
)

investigation_system = InvestigationSystem(
    model=inference_pipeline.model,
    preprocessor=inference_pipeline.preprocessor
)

@app.post("/predict")
async def predict_transaction(transaction: Transaction):
    try:
        result = inference_pipeline.predict_transaction(transaction.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    stats = inference_pipeline.get_performance_stats()
    return {
        "status": "healthy",
        "performance": stats
    }

@app.get("/investigation/case/{case_id}")
async def get_case_report(case_id: str):
    try:
        transaction = await get_transaction(case_id)
        card_history = await get_card_history(transaction['card_id'])
        report = investigation_system.generate_case_report(transaction, card_history)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/investigation/feedback/{case_id}")
async def submit_feedback(case_id: str, feedback: dict):
    try:
        investigation_system.add_investigator_feedback(case_id, feedback)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/investigation/queue")
async def get_investigation_queue(limit: int = 100):
    try:
        queue = investigation_system.get_investigation_queue(max_cases=limit)
        return queue
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)
