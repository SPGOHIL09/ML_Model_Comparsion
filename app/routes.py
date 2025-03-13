from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, flash
import pandas as pd
from app.ml_utils import get_data_insights, train_best_model

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the file
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Please upload a CSV or Excel file."}), 400
        
        # Check if dataset is valid
        if data.shape[0] < 10 or data.shape[1] < 2:
            return jsonify({"error": "Dataset too small. Please provide at least 10 rows and 2 columns."}), 400
            
        # Get insights
        insights = get_data_insights(data)
        
        # Train models
        model_results = train_best_model(data)
        
        # Prepare and return results
        results = {
            "insights": insights,
            "model_results": model_results,
            "filename": file.filename
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500