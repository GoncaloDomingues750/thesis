from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.pdb_utils import load_data_from_db
from pymongo import MongoClient
from gridfs import GridFS

from fastapi.responses import StreamingResponse
from bson import ObjectId

from fastapi.responses import JSONResponse

from app.auth.users import get_current_user_id
from datetime import datetime
from fastapi import Depends




import os

from app.pdb_utils import fetch_and_store_protein
from app.ml.ml_pipeline import (
    load_and_preprocess,
    train_and_evaluate,
    train_stacking_model,
    export_all_results_to_pdf_pdfpages
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

router = APIRouter()

class PDBRequest(BaseModel):
    pdb_ids: List[str]
    n_before: int = 3
    n_inside: int = 4
    models: List[str]  # e.g. ["Decision Tree", "Random Forest", "SVM"]

@router.post("/submit_ids")
async def submit_protein_ids(request: PDBRequest, user_id: str = Depends(get_current_user_id)):
    try:
        # Step 1: Fetch and store protein data
        for pdb_id in request.pdb_ids:
            await fetch_and_store_protein(pdb_id, request.n_before, request.n_inside)

        # Step 2: Load and preprocess data from MongoDB
        X, y = await load_data_from_db()

        # Step 3: Train selected models
        results = []
        for model_name in request.models:
            if model_name == "Decision Tree":
                results.append(train_and_evaluate(DecisionTreeClassifier(class_weight="balanced"), X, y, model_name))
            elif model_name == "Random Forest":
                results.append(train_and_evaluate(RandomForestClassifier(class_weight="balanced"), X, y, model_name, param_grid={"clf__n_estimators": [50, 100]}))
            elif model_name == "Logistic Regression":
                results.append(train_and_evaluate(LogisticRegression(max_iter=1000, class_weight="balanced"), X, y, model_name, param_grid={"clf__C": [0.1, 1.0, 10.0]}))
            elif model_name == "SVM":
                results.append(train_and_evaluate(SVC(probability=True, class_weight="balanced"), X, y, model_name, param_grid={"clf__C": [0.1, 1, 10]}, use_feature_selection=True))
            elif model_name == "KNN":
                results.append(train_and_evaluate(KNeighborsClassifier(), X, y, model_name, param_grid={"clf__n_neighbors": [3, 5, 7]}))
            elif model_name == "Gradient Boosting":
                results.append(train_and_evaluate(GradientBoostingClassifier(), X, y, model_name, param_grid={"clf__n_estimators": [100, 200]}))
            elif model_name == "Extra Trees":
                results.append(train_and_evaluate(ExtraTreesClassifier(class_weight="balanced"), X, y, model_name, param_grid={"clf__n_estimators": [100, 200]}))
            elif model_name == "Stacking":
                results.append(train_stacking_model(X, y))

        pdf_bytes = export_all_results_to_pdf_pdfpages(results)
        timestamp = datetime.utcnow()

        mongo_client = MongoClient("mongodb://mongo:27017")
        fs = GridFS(mongo_client["protein_db"])
        pdf_id = fs.put(pdf_bytes.read(),
        filename=f"evaluation_summary_{timestamp}.pdf",
        metadata={
            "user_id": user_id,
            "models": request.models,
            "pdb_ids": request.pdb_ids,
            "n_before": request.n_before,
            "n_inside": request.n_inside,
            "timestamp": datetime.utcnow()
        })

        return {
            "status": "success",
            "metrics": results,
            "pdf_id": str(pdf_id)
        }

    except Exception as e:
        import traceback
        print("❌ Exception occurred:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")



@router.get("/download_report/{file_id}")
async def download_report(file_id: str):
    fs = GridFS(MongoClient("mongodb://mongo:27017")["protein_db"])
    try:
        file = fs.get(ObjectId(file_id))
        return StreamingResponse(file, media_type="application/pdf", headers={
            "Content-Disposition": f"attachment; filename={file.filename}"
        })
    except Exception as e:
        raise HTTPException(status_code=404, detail="PDF not found")
    

@router.get("/list_reports")
async def list_reports(user_id: str = Depends(get_current_user_id)):
    fs = GridFS(MongoClient("mongodb://mongo:27017")["protein_db"])
    try:
        files = fs.find({"metadata.user_id": user_id})
        reports = [
            {"filename": file.filename, "file_id": str(file._id), "upload_date": file.upload_date.isoformat()}
            for file in files if file.filename.endswith(".pdf")
        ]
        return JSONResponse(content=reports)
    except Exception as e:
        import traceback
        print("❌ Exception occurred:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")