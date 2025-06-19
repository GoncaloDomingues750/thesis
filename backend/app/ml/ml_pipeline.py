# app/ml_pipeline.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def load_and_preprocess(csv_dir: str):
    all_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]
    dfs = [pd.read_csv(f) for f in all_files if os.path.getsize(f) > 0]
    dfs = [df for df in dfs if not df.empty and "label" in df.columns]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    aa_cols = [col for col in df.columns if col.startswith("aa_")]
    num_cols = [col for col in df.columns if col not in aa_cols + ["label"]]

    ct = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), aa_cols)
    ])
    X = ct.fit_transform(df)
    y = df["label"].values
    return X, y

def train_and_evaluate(model, X, y, name, param_grid=None, use_feature_selection=False):
    print(f"\n🔍 Running: {name}")

    steps = [('var', VarianceThreshold(threshold=0.0))]
    if use_feature_selection:
        steps.append(('select', SelectKBest(score_func=f_classif, k=20)))
    steps.append(('clf', model))
    pipeline = Pipeline(steps)

    clf = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1) if param_grid else pipeline

    scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    avg_f1 = np.mean(scores)
    print("F1 scores:", scores)
    print("Avg F1 score:", avg_f1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"plot_{name.replace(' ', '_').lower()}_confusion.png")
    plt.close()

    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]

        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"ROC Curve - {name}")
        plt.savefig(f"plot_{name.replace(' ', '_').lower()}_roc.png")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title(f"Precision-Recall Curve - {name}")
        plt.savefig(f"plot_{name.replace(' ', '_').lower()}_pr.png")
        plt.close()

    return {
        "model": name,
        "f1_scores": scores.tolist(),
        "avg_f1": float(avg_f1),
        "classification_report": report
    }

def train_stacking_model(X, y):
    print("\n🤖 Running: Stacking Classifier")

    base_estimators = [
        ('dt', DecisionTreeClassifier(class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced')),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(probability=True, class_weight='balanced'))
    ]
    meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')

    clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    avg_f1 = np.mean(scores)
    print("F1 scores:", scores)
    print("Avg F1 score:", avg_f1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.title("Confusion Matrix - Stacking")
    plt.tight_layout()
    plt.savefig("plot_stacking_confusion.png")
    plt.close()

    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]

        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("ROC Curve - Stacking")
        plt.savefig("plot_stacking_roc.png")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title("Precision-Recall Curve - Stacking")
        plt.savefig("plot_stacking_pr.png")
        plt.close()

    return {
        "model": "Stacking",
        "f1_scores": scores.tolist(),
        "avg_f1": float(avg_f1),
        "classification_report": report
    }

def export_all_results_to_pdf_pdfpages(metrics_list, pdf_path='output_csvs/evaluation_summary.pdf'):
    with PdfPages(pdf_path) as pdf:
        for metrics in metrics_list:
            name = metrics["model"]
            f1_scores = metrics["f1_scores"]
            avg_f1 = metrics["avg_f1"]
            classification_rep = metrics["classification_report"]

            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            report_text = f"Model: {name}\n\n"
            report_text += f"F1 Scores (Cross-Validation): {f1_scores}\n"
            report_text += f"Average F1 Score: {avg_f1:.4f}\n\n"
            report_text += "Classification Report:\n" + classification_rep
            ax.text(0, 1, report_text, ha='left', va='top', wrap=True, fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

            for plot_type in ['confusion', 'roc', 'pr', 'learning']:
                plot_file = f"plot_{name.replace(' ', '_').lower()}_{plot_type}.png"
                if os.path.exists(plot_file):
                    img = plt.imread(plot_file)
                    fig, ax = plt.subplots(figsize=(8.27, 11.69))
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig)
                    plt.close(fig)

    print(f"📄 PDF report saved to {pdf_path}")
