# 🎓 Student Performance Prediction using Regression, KNN, and Random Forest

## 📘 Project Description
This project predicts students’ final exam performance using **Linear Regression**, **K-Nearest Neighbors (KNN)**, and **Random Forest Regressor**.  
It integrates a **Flask backend** with an **HTML/CSS/JS frontend** for real-time predictions.  
The system visualizes **graphs, evaluation metrics, and confusion matrices** for model comparison and insights.

---

## 🚀 Features
- Predicts final exam scores or performance categories.  
- Implements **Regression, KNN, and Random Forest** models for accuracy comparison.  
- Attractive, interactive frontend for dynamic input and output.  
- Visualizes performance metrics and confusion matrices.  
- Dataset contains **2,500 student records** with academic and behavioral attributes.

---

## 🧠 Tech Stack
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Python Flask  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Dataset:** `student_performance_2500.csv`

---

## ⚙️ How It Works
1. User enters student academic and behavioral data.  
2. Flask backend loads trained ML models and predicts final exam scores.  
3. Frontend displays predictions, graphs, metrics, and confusion matrix.

---

## 📊 Machine Learning Models
| Model | Type | Description |
|-------|------|-------------|
| Linear Regression | Regression | Baseline model for linear trends. |
| KNN Regressor (K=5) | Non-parametric | Predicts based on nearest neighbor similarity. |
| Random Forest Regressor | Ensemble | Captures complex non-linear relationships, highest accuracy. |

---

## 🏁 Results & Inference
- Random Forest achieved the **highest accuracy (R² ≈ 0.91)**.  
- Key positive factors: study hours, attendance, previous grades.  
- Negative factors: absences, excessive phone usage.  
- Provides actionable insights for educators to **identify at-risk students early**.

---

## 💼 Business Impact
- Enables early intervention for struggling students.  
- Helps allocate mentoring resources efficiently.  
- Supports data-driven academic management and policy decisions.  
- Scalable to different institutions and subjects.

---

## 📂 Dataset
- Path: `C:\Users\sanmu\Downloads\student_performance_2500.csv`  
- Alternative: [Students Performance Dataset (Kaggle)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

---

## 📸 Screenshots
*(Add images of your frontend UI, prediction results, and graphs here)*

---

## 🧾 License
This project is open-source under the **MIT License**.
