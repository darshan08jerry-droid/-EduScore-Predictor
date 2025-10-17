from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level to project root
frontend_dir = os.path.join(project_root, 'frontend')

app = Flask(__name__, 
            template_folder=frontend_dir,
            static_folder=frontend_dir)

# Enable CORS response to clinet sissde code
from flask_cors import CORS
CORS(app)

# --- Global Artifacts Loading ---
try:
    # Load Preprocessor (Scaler)
    SCALER = joblib.load('models/scaler.pkl')
    # Load Models
    MODELS = {
        'Linear Regression': joblib.load('models/regression.pkl'),
        'KNN': joblib.load('models/knn.pkl'),
        'Random Forest': joblib.load('models/randomforest.pkl')
    }
    # Load Metrics
    with open('metrics.json', 'r') as f:
        MODEL_SCORES = json.load(f)
    # Load the list of features the model was trained on
    FEATURE_NAMES = joblib.load('models/feature_names.pkl')
        
    print("✅ All models and metrics loaded successfully.")
    MODELS_LOADED = True
except Exception as e:
    print(f"❌ ERROR: Failed to load models/metrics. Error: {e}")
    MODELS_LOADED = False

# --- Helper Functions --- #convert the cotrgorical 
def categorize_score(score):
    """Converts continuous score to a category (A, B, C) for matrix visualization."""
    if score >= 80:
        return 'A'
    elif score >= 60:
        return 'B'
    else:
        return 'C'

#confuseion matrix
def generate_confusion_matrix_plot(y_true, y_pred, model_name):
    """Generate and return a base64 encoded confusion matrix plot."""
    categories = ['A', 'B', 'C']
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories,
                cbar_kws={'label': 'Number of Students'})
    plt.title(f'Confusion Matrix - {model_name}\n(A: 80-100, B: 60-79, C: 0-59)')
    plt.xlabel('Predicted Grade')
    plt.ylabel('Actual Grade')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

# --- API Endpoints --- connect frontend serving
@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')

@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files."""
    return send_from_directory(os.path.join(app.static_folder), filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files."""
    return send_from_directory(os.path.join(app.static_folder), filename)

#metrics display
@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Returns the performance metrics for all trained models."""
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded. Please run train_model.py first."}), 500
    return jsonify(MODEL_SCORES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles real-time prediction request."""
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded. Please run train_model.py first."}), 500

    try:
        data = request.form.to_dict()
        print(f"📥 Received prediction request: {data}")

        input_data = {
            'gender': [int(data.get('gender', 0))],
            'age': [int(data.get('age', 18))],
            'parental_education': [int(data.get('parental_education', 3))],
            'study_hours_per_day': [float(data.get('study_hours_per_day', 3.0))],
            'attendance_percentage': [float(data.get('attendance_percentage', 90))],
            'previous_grade': [float(data.get('previous_grade', 75))],
            'test_preparation_course': [int(data.get('test_preparation_course', 0))],
            'internet_access': [int(data.get('internet_access', 1))],
            'health_status': [int(data.get('health_status', 4))],
            'absences': [int(data.get('absences', 2))],
            'co_curricular_participation': [int(data.get('co_curricular_participation', 1))],
            'hours_on_phone_daily': [float(data.get('hours_on_phone_daily', 4.0))],
            'sleep_hours': [float(data.get('sleep_hours', 7.5))],
        }
        
        input_df = pd.DataFrame(input_data)
        input_filtered = input_df[FEATURE_NAMES]
        input_scaled = SCALER.transform(input_filtered)

        predictions = {}
        for name, model in MODELS.items():
            prediction = model.predict(input_scaled)[0]
            prediction_clamped = max(0, min(100, prediction)) 
            predictions[name] = round(prediction_clamped, 2)

        print(f"📤 Sending predictions: {predictions}")
        return jsonify(predictions)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

@app.route('/matrix_data', methods=['GET'])
def get_matrix_data():
    """Calculates and returns categorized performance matrix data using the test set."""
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded."}), 500

    try:
        DATASET_PATH = r"C:\Users\sanmu\Downloads\student_performance_2500.csv"
        df = pd.read_csv(DATASET_PATH)
        
        X = df.drop('final_exam_score', axis=1).select_dtypes(include=np.number)
        y = df['final_exam_score']
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_true_cat = y_test.apply(categorize_score)
        categories = ['A', 'B', 'C']
        
        matrix_data = {}
        classification_metrics = {}
        confusion_matrices = {}
        
        X_test_scaled = SCALER.transform(X_test)

        for name, model in MODELS.items():
            y_pred = model.predict(X_test_scaled)
            y_pred_cat = pd.Series(y_pred).apply(categorize_score)

            cm = confusion_matrix(y_true_cat, y_pred_cat, labels=categories)
            
            accuracy = accuracy_score(y_true_cat, y_pred_cat)
            precision = precision_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
            recall = recall_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
            f1 = f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
            
            cm_plot = generate_confusion_matrix_plot(y_true_cat, y_pred_cat, name)
            
            matrix_data[name] = cm.tolist()
            confusion_matrices[name] = cm_plot
            classification_metrics[name] = {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4)
            }

        return jsonify({
            'confusion_matrices': confusion_matrices,
            'classification_metrics': classification_metrics,
            'matrix_data': matrix_data
        })

    except Exception as e:
        print(f"❌ Matrix data error: {e}")
        return jsonify({"error": f"Failed to generate matrix data: {e}"}), 500

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """Returns feature importance for Random Forest model."""
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded."}), 500

    try:
        rf_model = MODELS['Random Forest']
        feature_importance = dict(zip(FEATURE_NAMES, rf_model.feature_importances_))
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        return jsonify(sorted_importance)
        
    except Exception as e:
        return jsonify({"error": f"Failed to get feature importance: {e}"}), 500

@app.route('/dataset_stats', methods=['GET'])
def get_dataset_stats():
    """Returns basic statistics about the dataset."""
    try:
        DATASET_PATH = r"C:\Users\sanmu\Downloads\student_performance_2500.csv"
        df = pd.read_csv(DATASET_PATH)
        
        stats = {
            'total_students': len(df),
            'features_count': len(df.columns) - 1,
            'score_distribution': {
                'A (80-100)': len(df[df['final_exam_score'] >= 80]),
                'B (60-79)': len(df[(df['final_exam_score'] >= 60) & (df['final_exam_score'] < 80)]),
                'C (0-59)': len(df[df['final_exam_score'] < 60])
            },
            'average_score': round(df['final_exam_score'].mean(), 2),
            'std_score': round(df['final_exam_score'].std(), 2),
            'min_score': int(df['final_exam_score'].min()),
            'max_score': int(df['final_exam_score'].max())
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": f"Failed to get dataset stats: {e}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": MODELS_LOADED,
        "loaded_models": list(MODELS.keys()) if MODELS_LOADED else []
    })

if __name__ == '__main__':
    print("🚀 Starting EduScore Predictor Flask Server...")
    print("📊 Available endpoints:")
    print("   GET  /              - Frontend interface")
    print("   GET  /metrics       - Model performance metrics")
    print("   POST /predict       - Make predictions")
    print("   GET  /matrix_data   - Confusion matrices")
    print("   GET  /feature_importance - Feature importance")
    print("   GET  /dataset_stats - Dataset statistics")
    print("   GET  /health        - Health check")
    print(f"📁 Template folder: {app.template_folder}")
    print(f"📁 Static folder: {app.static_folder}")
    
    app.run(debug=True, port=5000, host='0.0.0.0')