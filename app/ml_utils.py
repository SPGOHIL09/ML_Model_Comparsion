import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                             AdaBoostClassifier, AdaBoostRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
import json
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64



def detect_problem_type(y):
    """Detect whether the target column is for classification or regression."""
    if y.dtypes == 'object' or len(np.unique(y)) < 10:
        return 'classification'
    else:
        return 'regression'

def get_data_insights(data):
    """Extract insights from the data"""
    target_col = data.columns[-1]
    
    # Generate data visualizations
    visualizations = generate_visualizations(data)
    
    insights = {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "column_names": data.columns.tolist(),
        "missing_values": data.isnull().sum().to_dict(),
        "categorical_features": data.select_dtypes(include=['object']).columns.tolist(),
        "numerical_features": data.select_dtypes(include=['number']).columns.tolist(),
        "data_sample": data.head(5).to_dict(orient='records'),
        "visualizations": visualizations,
        "target_column": target_col,
        "target_distribution": data[target_col].value_counts().to_dict() if detect_problem_type(data[target_col]) == 'classification' else {
            "min": float(data[target_col].min()),
            "max": float(data[target_col].max()),
            "mean": float(data[target_col].mean()),
            "median": float(data[target_col].median()),
            "std": float(data[target_col].std())
        }
    }
    return insights

def generate_visualizations(data):
    """Generate visualizations for the data"""
    visualizations = {}
    
    # Target distribution visualization
    plt.figure(figsize=(10, 6))
    target_col = data.columns[-1]
    
    if detect_problem_type(data[target_col]) == 'classification':
        sns.countplot(y=target_col, data=data)
        plt.title(f'Distribution of {target_col}')
    else:
        sns.histplot(data[target_col], kde=True)
        plt.title(f'Distribution of {target_col}')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    visualizations['target_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Correlation heatmap for numerical features
    numerical_cols = data.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation = data[numerical_cols].corr()
        mask = np.triu(correlation)
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['correlation'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    return visualizations

def train_best_model(data):
    """Train and evaluate multiple models to find the best one"""
    X = data.iloc[:, :-1]  # Features (all columns except last)
    y = data.iloc[:, -1]   # Target (last column)
    target_col = data.columns[-1]
    
    problem_type = detect_problem_type(y)
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Set up the models
    if problem_type == 'classification':
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "AdaBoost": AdaBoostClassifier(random_state=42)
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
        }
        
            
        # Using negative RMSE as scorer for regression
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
            
            results[name] = float(avg_score)
            model_scores[name] = [float(score) if problem_type == 'classification' else float(-score) for score in cv_scores]
        except Exception as e:
            results[name] = float('-inf') if problem_type == 'classification' else float('inf')
            model_scores[name] = str(e)
    
    # Find best model
    best_model_name = max(results, key=results.get) if problem_type == 'classification' else min(results, key=results.get)
    
    # Train the best model on full data for future predictions
    best_model = models[best_model_name]
    best_model.fit(X, y)
    
    # Visualize model comparison
    model_comparison_chart = visualize_model_comparison(results, problem_type)
    
    # Get feature importance if available
    feature_importance = None
    feature_importance_chart = None
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        feature_importance = dict(zip(X.columns, importance))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Visualize feature importance
        feature_importance_chart = visualize_feature_importance(feature_importance)
    
    return {
        "problem_type": problem_type,
        "target_column": target_col,
        "best_model": best_model_name,
        "score": results[best_model_name],
        "all_scores": model_scores,
        "model_comparison_chart": model_comparison_chart,
        "feature_importance": feature_importance,
        "feature_importance_chart": feature_importance_chart
    }

def visualize_model_comparison(results, problem_type):
    """Visualize model comparison"""
    plt.figure(figsize=(12, 8))
    
    # Sort models by performance
    if problem_type == 'classification':
        # Higher is better for classification
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    else:
        # Lower is better for regression
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))
    
    bars = plt.barh(list(sorted_results.keys()), list(sorted_results.values()))
    
    # Add performance values on bars
    for i, bar in enumerate(bars):
        value = list(sorted_results.values())[i]
        plt.text(
            bar.get_width() + (0.01 if problem_type == 'classification' else 0.05), 
            bar.get_y() + bar.get_height()/2, 
            f'{value:.4f}', 
            va='center'
        )
    
    metric = 'Accuracy' if problem_type == 'classification' else 'RMSE (lower is better)'
    plt.xlabel(metric)
    plt.title(f'Model Performance Comparison ({metric})')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return chart

def visualize_feature_importance(feature_importance):
    """Visualize feature importance"""
    plt.figure(figsize=(12, 8))
    
    # Take top 20
    if len(feature_importance) > 20:
        feature_importance = dict(list(feature_importance.items())[:20])
    
    # Sort for visualization
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1]))
    
    plt.barh(list(sorted_importance.keys()), list(sorted_importance.values()))
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return chart