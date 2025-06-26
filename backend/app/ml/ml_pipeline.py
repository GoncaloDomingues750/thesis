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
from io import BytesIO
import matplotlib.table as tbl
from PIL import Image
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold






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

    # Extract feature names
    num_features = ct.named_transformers_["num"].get_feature_names_out(num_cols)
    cat_features = ct.named_transformers_["cat"].get_feature_names_out(aa_cols)
    feature_names = np.concatenate([num_features, cat_features])

    return X, y, feature_names.tolist()


def train_and_evaluate(model, X, y, name, param_grid=None, use_feature_selection=False, feature_names=None):
    print(f"\n🔍 Running: {name}")

    steps = [('var', VarianceThreshold(threshold=0.0))]
    if use_feature_selection:
        steps.append(('select', SelectKBest(score_func=f_classif, k=20)))
    steps.append(('clf', model))
    pipeline = Pipeline(steps)

    clf = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1) if param_grid else pipeline


    y_proba = cross_val_predict(clf, X, y, cv=5, method='predict_proba')
    y_pred = np.argmax(y_proba, axis=1)
    report = classification_report(y, y_pred, output_dict=True)
    avg_f1 = report["macro avg"]["f1-score"]
    print("Avg F1 score:", avg_f1)
    avg_f1 = report["macro avg"]["f1-score"]
    print("Avg F1 score:", avg_f1)


    print(report)

    ConfusionMatrixDisplay(confusion_matrix(y, y_pred)).plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"plot_{name.replace(' ', '_').lower()}_confusion.png")
    plt.close()

    try:
        RocCurveDisplay.from_predictions(y, y_proba[:, 1])
        plt.title(f"ROC Curve - {name}")
        plt.savefig(f"plot_{name.replace(' ', '_').lower()}_roc.png")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y, y_proba[:, 1])
        plt.title(f"Precision-Recall Curve - {name}")
        plt.savefig(f"plot_{name.replace(' ', '_').lower()}_pr.png")
        plt.close()
    except AttributeError:
        print(f"⚠️ Model {name} does not support predict_proba, skipping ROC/PR plots.")

    # Try to access feature importances if available
    feature_importance_path = None

    clf.fit(X, y)
    if isinstance(clf, GridSearchCV):
        final_model = clf.best_estimator_
    else:
        final_model = clf


    # Unwrap inner classifier if final_model is a Pipeline
    if isinstance(final_model, Pipeline):
        estimator = final_model.named_steps['clf']
    else:
        estimator = final_model

    if hasattr(estimator, "feature_importances_") and feature_names is not None:
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_k = min(20, len(importances))
        top_indices = indices[:top_k]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]

        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_k} Feature Importances - {name}")
        plt.barh(range(top_k), top_importances[::-1])
        plt.yticks(range(top_k), top_features[::-1])
        plt.xlabel("Importance")
        plt.tight_layout()

        feature_importance_path = f"plot_{name.replace(' ', '_').lower()}_importance.png"
        plt.savefig(feature_importance_path)
        plt.close()

    if feature_names is not None:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=clf,
            X=X, y=y,
            cv=5,
            scoring='f1',
            train_sizes=np.linspace(0.1, 1.0, 5),
            n_jobs=-1
        )

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training F1 Score')
        plt.plot(train_sizes, test_mean, label='Validation F1 Score')
        plt.title(f"Learning Curve - {name}")
        plt.xlabel("Training Set Size")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        learning_curve_path = f"plot_{name.replace(' ', '_').lower()}_learning.png"
        plt.savefig(learning_curve_path)
        plt.close()
    else:
        learning_curve_path = None

    scores = cross_val_score(clf, X, y, cv=5, scoring='f1')

    return {
        "model": name,
        "f1_scores": scores.tolist(),
        "avg_f1": float(avg_f1),
        "classification_report": report,
        "feature_importance_path": feature_importance_path,
        "learning_curve_path": learning_curve_path
    }


def train_stacking_model(X, y, feature_names=None):
    print("\n🤖 Running: Stacking Classifier")

    base_estimators = [
        ('dt', DecisionTreeClassifier(max_depth=5, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(probability=True, kernel='rbf', C=1, class_weight='balanced'))
    ]
    meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')

    clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )


    y_proba = cross_val_predict(clf, X, y, cv=5, method='predict_proba')
    y_pred = np.argmax(y_proba, axis=1)
    report = classification_report(y, y_pred, output_dict=True)
    avg_f1 = report["macro avg"]["f1-score"]
    print("Avg F1 score:", avg_f1)
    avg_f1 = report["macro avg"]["f1-score"]
    print("Avg F1 score:", avg_f1)

    ConfusionMatrixDisplay(confusion_matrix(y, y_pred)).plot()
    plt.title("Confusion Matrix - Stacking")
    plt.tight_layout()
    plt.savefig("plot_stacking_confusion.png")
    plt.close()

    if hasattr(clf, "predict_proba"):
        RocCurveDisplay.from_predictions(y, y_proba[:, 1])
        plt.title("ROC Curve - Stacking")
        plt.savefig("plot_stacking_roc.png")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y, y_proba[:, 1])
        plt.title("Precision-Recall Curve - Stacking")
        plt.savefig("plot_stacking_pr.png")
        plt.close()

    learning_curve_path = None
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=clf, X=X, y=y, cv=5, scoring='f1',
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training F1')
        plt.plot(train_sizes, test_mean, label='Validation F1')
        plt.title("Learning Curve - Stacking")
        plt.xlabel("Training Set Size")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        learning_curve_path = "plot_stacking_learning.png"
        plt.savefig(learning_curve_path)
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not generate learning curve for stacking: {e}")

    scores = cross_val_score(clf, X, y, cv=5, scoring='f1')

    return {
        "model": "Stacking",
        "f1_scores": scores.tolist(),
        "avg_f1": float(avg_f1),
        "classification_report": report,
        "learning_curve_path": learning_curve_path
    }



def export_all_results_to_pdf_pdfpages(metrics_list, logo_path="app/Logo.png"):
    from matplotlib.backends.backend_pdf import PdfPages
    from io import BytesIO
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import os

    pdf_bytes = BytesIO()
    page_num = 1


    def add_header_bar(fig, logo_path="app/Logo.png"):
        """Draws a light blue header bar with resized logo on the left and title on the right."""
        dpi = fig.dpi
        fig_width, fig_height = fig.get_size_inches()

        header_height_inches = 0.8  # ~0.8 inches header
        header_height_rel = header_height_inches / fig_height

        # Create header axes spanning the top of the page
        header_ax = fig.add_axes([0, 1 - header_height_rel, 1, header_height_rel])
        header_ax.set_facecolor('#ADD8E6')
        header_ax.set_xticks([])
        header_ax.set_yticks([])
        header_ax.set_xlim(0, 1)
        header_ax.set_ylim(0, 1)

        # Add title on the right
        header_ax.text(0.98, 0.5, "Classification Report", ha="right", va="center", fontsize=16, weight="bold")

        # Load and plot logo on the left
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path)
            logo_height_px = int(header_height_inches * dpi * 0.8)
            aspect_ratio = logo_img.width / logo_img.height
            logo_width_px = int(logo_height_px * aspect_ratio)
            logo_img = logo_img.resize((logo_width_px, logo_height_px), Image.LANCZOS)
            logo_array = np.array(logo_img)

            # Create inset Axes for logo inside header
            logo_ax = fig.add_axes([0.02, 0.90, 0.12, 0.12])  # [left, bottom, width, height] in 0-1 figure coords
            logo_ax.imshow(mpimg.imread(logo_path))
            logo_ax.axis('off')  # Hide borders

    with PdfPages(pdf_bytes) as pdf:
        for metrics in metrics_list:
            name = metrics["model"]
            report_dict = metrics["classification_report"]
            labels = [k for k in report_dict if isinstance(report_dict[k], dict)]
            columns = ['precision', 'recall', 'f1-score', 'support']
            table_data = [[label] + [f"{report_dict[label].get(metric, 0):.2f}" for metric in columns] for label in labels]
            table_data.insert(0, ['Class'] + [c.capitalize() for c in columns])

            # --- PAGE 1: Table + Confusion ---
            fig, axs = plt.subplots(2, 1, figsize=(8.27, 11.69), gridspec_kw={"height_ratios": [2, 3]})
            fig.subplots_adjust(top=0.85, hspace=0.4)

            add_header_bar(fig)

            axs[0].set_title(f"{name}", fontsize=14, weight='bold', pad=20)


            axs[0].axis('off')
            table = axs[0].table(cellText=table_data, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)

            confusion_path = f"plot_{name.replace(' ', '_').lower()}_confusion.png"
            if os.path.exists(confusion_path):
                img = mpimg.imread(confusion_path)
                axs[1].imshow(img)
                axs[1].axis('off')

            fig.text(0.95, 0.02, f"Page {page_num}", ha='right', fontsize=9, color='gray')
            pdf.savefig(fig)
            plt.close(fig)
            page_num += 1

            # --- PAGE 2: ROC + PR ---
            fig, axs = plt.subplots(2, 1, figsize=(8.27, 11.69))
            fig.subplots_adjust(top=0.85, hspace=0.4)

            add_header_bar(fig)

            for i, plot_type in enumerate(['roc', 'pr']):
                axs[i].axis('off')
                img_path = f"plot_{name.replace(' ', '_').lower()}_{plot_type}.png"
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    axs[i].imshow(img)

            fig.text(0.95, 0.02, f"Page {page_num}", ha='right', fontsize=9, color='gray')
            pdf.savefig(fig)
            plt.close(fig)
            page_num += 1

            # --- PAGE 3: Feature Importance ---
            if metrics.get("feature_importance_path"):
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                add_header_bar(fig)
                ax.axis('off')

                img = mpimg.imread(metrics["feature_importance_path"])
                ax.imshow(img)
                fig.text(0.95, 0.02, f"Page {page_num}", ha='right', fontsize=9, color='gray')
                pdf.savefig(fig)
                plt.close(fig)
                page_num += 1


            # --- PAGE 4: Learning Curve ---
            if metrics.get("learning_curve_path"):
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                add_header_bar(fig)
                ax.axis('off')

                img = mpimg.imread(metrics["learning_curve_path"])
                ax.imshow(img)
                fig.text(0.95, 0.02, f"Page {page_num}", ha='right', fontsize=9, color='gray')
                pdf.savefig(fig)
                plt.close(fig)
                page_num += 1

    pdf_bytes.seek(0)

    # Clean temp plots
    for metrics in metrics_list:
        name = metrics["model"]
        for plot_type in ['confusion', 'roc', 'pr', 'learning', 'importance']:
            plot_file = f"plot_{name.replace(' ', '_').lower()}_{plot_type}.png"
            if os.path.exists(plot_file):
                os.remove(plot_file)

    return pdf_bytes
