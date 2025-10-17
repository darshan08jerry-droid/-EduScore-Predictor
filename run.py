# run.py - Main launcher for EduScore Predictor
import os
import sys
import webbrowser
import threading
import time

def main():
    print("🎓 EduScore Predictor - Starting Application...")
    
    # Add backend to Python path
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    sys.path.insert(0, backend_dir)
    
    # Check if models exist
    models_dir = os.path.join(backend_dir, 'models')
    required_files = ['scaler.pkl', 'feature_names.pkl', 'regression.pkl', 'knn.pkl', 'randomforest.pkl']
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(models_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing model files. Please run train_model.py first.")
        print("Missing files:", missing_files)
        response = input("Do you want to train models now? (y/n): ")
        if response.lower() == 'y':
            from backend import train_model
            print("🤖 Training models...")
            train_model.train_and_save_models()
        else:
            print("Please run 'python backend/train_model.py' first.")
            return
    
    # Start Flask server
    from backend.app import app
    
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    print("🌐 Starting web server...")
    threading.Thread(target=open_browser).start()
    
    # Run Flask app
    app.run(debug=True, port=5000, use_reloader=False)

if __name__ == '__main__':
    main()