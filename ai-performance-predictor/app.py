from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import pandas as pd
import pickle
import os
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_folder='.')

# Enable CORS for all routes (helps with local development)
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Handle OPTIONS requests for CORS preflight
@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    return '', 204

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Check if models exist, if not train them
if not (os.path.exists('models/random_forest_model.pkl') and 
        os.path.exists('models/logistic_regression_model.pkl')):
    from train_models import train_and_save_models
    train_and_save_models()

# Load the trained models
try:
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
        
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
        
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    # Create empty models as fallback
    rf_model = RandomForestClassifier()
    lr_model = LogisticRegression()
    scaler = StandardScaler()

@app.route('/')
def index():
    return send_from_directory('.', 'page.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Print request for debugging
        print("Received prediction request")
        data = request.json
        print(f"Request data: {data}")
        
        # Extract features
        courses = data.get('courses', [])
        attendance = data.get('attendance', '')
        cgpa = data.get('cgpa', '')
        
        # Validate input
        if not courses or not attendance or not cgpa:
            return jsonify({"error": "Missing required data"}), 400
        
        # Process course data
        total_grade_points = 0
        total_credits = 0
        
        for course in courses:
            grade_value = 0
            grade_range = course.get('grade', '')
            credit = course.get('credits', 0)
            
            # Convert grade ranges to numeric values
            if grade_range == "91-100": grade_value = 10    # O
            elif grade_range == "81-90": grade_value = 9    # A+
            elif grade_range == "71-80": grade_value = 8    # A
            elif grade_range == "61-70": grade_value = 7    # B+
            elif grade_range == "56-60": grade_value = 6    # B
            elif grade_range == "51-55": grade_value = 5    # C
            elif grade_range == "0-50": grade_value = 2     # F
            
            total_grade_points += grade_value * credit
            total_credits += credit
        
        # Avoid division by zero
        avg_grade = total_grade_points / max(total_credits, 1)
        
        # Process attendance
        attendance_value = 0
        if attendance == ">95": attendance_value = 95
        elif attendance == ">85": attendance_value = 85
        elif attendance == ">75": attendance_value = 75
        elif attendance == ">65": attendance_value = 65
        elif attendance == ">50": attendance_value = 50
        
        # Process CGPA
        cgpa_value = 0
        if cgpa == "9~10": cgpa_value = 9.5
        elif cgpa == "8~9": cgpa_value = 8.5
        elif cgpa == "7~8": cgpa_value = 7.5
        elif cgpa == "6~7": cgpa_value = 6.5
        elif cgpa == "5~6": cgpa_value = 5.5
        
        # Create feature vector
        features = np.array([[attendance_value, cgpa_value, avg_grade]])
        print(f"Processed features: {features}")
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        grade_prediction = rf_model.predict(features_scaled)[0]
        risk_prediction = lr_model.predict(features_scaled)[0]
        
        # Convert grade prediction to grade range
        grade_ranges = ["F (Below 50%)", "C to B (50-60%)", "B to B+ (60-70%)", 
                       "B+ to A (70-80%)", "A to A+ (80-90%)", "A+ to O (90-100%)"]
        predicted_grade = grade_ranges[grade_prediction]
        
        # Convert risk prediction to risk level and performance category
        risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
        performance_categories = ["Performing Well", "Needs Improvement", "Needs Immediate Intervention"]
        
        risk_level = risk_levels[risk_prediction]
        performance_category = performance_categories[risk_prediction]
        
        # Generate recommendations
        recommendations = []
        
        if attendance_value < 75:
            recommendations.append("Improve class attendance to at least 75%")
        
        if avg_grade < 6:
            recommendations.append("Focus on improving grades in current courses")
        
        if cgpa_value < 7:
            recommendations.append("Develop better study habits to improve overall CGPA")
        
        if risk_prediction >= 1:  # Moderate or high risk
            recommendations.append("Consider seeking academic counseling or tutoring")
        
        if len(recommendations) == 0:
            recommendations.append("Continue with current academic performance")
        
        # Return prediction results
        result = {
            "predictedGrade": predicted_grade,
            "performanceCategory": performance_category,
            "riskLevel": risk_level,
            "recommendations": recommendations,
            "predictedGradeValue": float(avg_grade) * 10  # Convert to percentage scale
        }
        
        print(f"Returning result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run with 0.0.0.0 to make accessible from other devices on the network
    app.run(host='0.0.0.0', debug=True, port=5000)