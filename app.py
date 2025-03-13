from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
import numpy as np

app = Flask(__name__)
CORS(app) 

def detect_problem_type(y):
    """Detect whether the target column is for classification or regression."""
    if y.dtypes == 'object' or len(y.unique()) < 10:
        return 'classification'
    else:
        return 'regression'

def get_data_insights(data):
    insights = {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "missing_values": data.isnull().sum().to_dict(),
        "categorical_features": data.select_dtypes(include=['object']).columns.tolist(),
        "numerical_features": data.select_dtypes(include=['number']).columns.tolist(),
        "data_sample": data.head(5).to_dict(orient='records'),
        "target_distribution": data.iloc[:, -1].value_counts().to_dict() if detect_problem_type(data.iloc[:, -1]) == 'classification' else {
            "min": float(data.iloc[:, -1].min()),
            "max": float(data.iloc[:, -1].max()),
            "mean": float(data.iloc[:, -1].mean()),
            "median": float(data.iloc[:, -1].median())
        }
    }
    return insights

def train_best_model(data):
    X = data.iloc[:, :-1]  # Features (all columns except last)
    y = data.iloc[:, -1]   # Target (last column)

    problem_type = detect_problem_type(y)
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    if problem_type == 'classification':
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42)
        }
        scoring = 'accuracy'
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "SVR": SVR(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
            "AdaBoost": AdaBoostRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42)
        }
        # Using negative RMSE as scorer for regression since cross_val_score maximizes
        rmse_scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
        scoring = rmse_scorer

    # Implement K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    model_scores = {}
    
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
            if problem_type == 'classification':
                avg_score = np.mean(cv_scores)
            else:
                # Convert back from negative RMSE
                avg_score = -np.mean(cv_scores)
            
            results[name] = avg_score
            model_scores[name] = cv_scores.tolist() if problem_type == 'classification' else [-score for score in cv_scores.tolist()]
        except Exception as e:
            results[name] = float('-inf') if problem_type == 'classification' else float('inf')
            model_scores[name] = f"Error: {str(e)}"
    
    # Find best model
    best_model = max(results, key=results.get) if problem_type == 'classification' else min(results, key=results.get)
    
    # Train the best model on full data for future predictions
    models[best_model].fit(X, y)
    
    return {
        "problem_type": problem_type,
        "best_model": best_model,
        "score": results[best_model],
        "all_scores": model_scores,
        "feature_importance": get_feature_importance(models[best_model], X.columns) if hasattr(models[best_model], 'feature_importances_') else None
    }

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    return None

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Please upload a CSV or Excel file."}), 400
        
        if data.shape[0] < 10:
            return jsonify({"error": "Dataset too small. Please provide at least 10 rows of data."}), 400
            
        insights = get_data_insights(data)
        result = train_best_model(data)
        return jsonify({"insights": insights, "model_recommendation": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)