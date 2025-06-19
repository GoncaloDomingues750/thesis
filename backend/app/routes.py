from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
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
async def submit_protein_ids(request: PDBRequest):
    try:
        # Step 1: Fetch and store protein data
        for pdb_id in request.pdb_ids:
            await fetch_and_store_protein(pdb_id, request.n_before, request.n_inside)

        # Step 2: Load and preprocess data from MongoDB
        X, y = load_and_preprocess()

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

        # Step 4: Save PDF report
        export_all_results_to_pdf_pdfpages(results)

        return {"status": "success", "metrics": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
