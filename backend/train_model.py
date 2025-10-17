import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os

# --- Configuration ---
DATASET_PATH = r"C:\Users\sanmu\Downloads\student_performance_2500.csv"
TARGET_COLUMN = 'final_exam_score'
MODELS_DIR = 'models'

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Define models to train
REGRESSORS = {
    'regression': LinearRegression(),
    'knn': KNeighborsRegressor(n_neighbors=5),
    'randomforest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
}

def categorize_score(score):
    """Converts continuous score to a category (A, B, C) for classification metrics."""
    if score >= 80:
        return 'A'
    elif score >= 60:
        return 'B'
    else:
        return 'C'

#r loading dataset, preprocessing, training models, evaluating metrics, and saving artifacts.
def train_and_save_models():
    """Loads data, cleans non-numeric features, trains models, and saves all artifacts."""
    try:
        print(" Loading dataset...")
        df = pd.read_csv(DATASET_PATH)
        print(f"Dataset loaded successfully with {len(df)} records")
    except FileNotFoundError:
        print(f" ERROR: Dataset not found at {DATASET_PATH}. Please check the path.")
        return False
    except Exception as e:
        print(f" ERROR: Failed to load dataset: {e}")
        return False

    # Separate features (X) and target (y)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Automatically select only numeric columns for model training
    X_numeric = X.select_dtypes(include=np.number)
    
    non_numeric_cols = X.columns.difference(X_numeric.columns)
    if not non_numeric_cols.empty:
        print(f"ℹ️ INFO: Excluding non-numeric columns from features: {non_numeric_cols.tolist()}")
    
    X = X_numeric

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Store feature names for consistency when scaling input in app.py
    FEATURE_NAMES = X.columns.tolist() 
    print(f"🔧 Using {len(FEATURE_NAMES)} features: {FEATURE_NAMES}")

    # --- Preprocessing Setup (StandardScaler) ---
    scaler = StandardScaler()
    
    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the fitted scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(FEATURE_NAMES, os.path.join(MODELS_DIR, 'feature_names.pkl')) 
    print("✅ Scaler and Feature Names saved.")

    # --- Model Training and Metric Calculation ---
    model_metrics = {}

    for name, model in REGRESSORS.items():
        print(f" Training {name}...")
        
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Calculate regression metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate classification metrics (after categorization)
            y_true_cat = y_test.apply(categorize_score)
            y_pred_cat = pd.Series(y_pred).apply(categorize_score)
            
            accuracy = accuracy_score(y_true_cat, y_pred_cat)
            precision = precision_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
            recall = recall_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
            f1 = f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
            
            # Store metrics (Using proper display names)
            display_name = name.replace('regression', 'Linear Regression').replace('knn', 'KNN').replace('randomforest', 'Random Forest')

            model_metrics[display_name] = {
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'R2': round(r2, 4),
                'Accuracy': round(accuracy, 4),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1_Score': round(f1, 4)
            }
            
            # Save the trained model
            joblib.dump(model, os.path.join(MODELS_DIR, f'{name}.pkl'))
            print(f"✅ Model {display_name} trained and saved. R2: {model_metrics[display_name]['R2']}, Accuracy: {model_metrics[display_name]['Accuracy']}")
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue

    # --- Save Metrics to JSON ---
    with open('metrics.json', 'w') as f:
        json.dump(model_metrics, f, indent=4)
    print("✅ Metrics saved to metrics.json")
    
    # Print dataset summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   Features used: {len(FEATURE_NAMES)}")
    print(f"   Target distribution:")
    print(f"     A (80-100): {len(y[y >= 80])} students")
    print(f"     B (60-79): {len(y[(y >= 60) & (y < 80)])} students")
    print(f"     C (0-59): {len(y[y < 60])} students")
    print(f"   Average score: {y.mean():.2f}")
    print(f"   Score range: {y.min():.1f} - {y.max():.1f}")
    
    return True

if __name__ == '__main__':
    print("🎯 Starting EduScore Predictor Model Training...")
    success = train_and_save_models()
    if success:
        print("\n🎉 Training completed successfully! You can now run app.py to start the web server.")
    else:
        print("\n💥 Training failed. Please check the errors above.")