// --- Enhanced Frontend JavaScript (script.js) ---

const form = document.getElementById('prediction-form');
const outputDiv = document.getElementById('prediction-output');
const predictionScores = document.getElementById('prediction-scores');
const errorDiv = document.getElementById('error-message');
const metricsTableBody = document.querySelector('#metrics-table tbody');
const classificationTableBody = document.querySelector('#classification-table tbody');
const featureImportanceDiv = document.getElementById('feature-importance');
const confusionMatricesDiv = document.getElementById('confusion-matrices');
const datasetStatsDiv = document.getElementById('dataset-stats');

let r2ChartInstance, rmseChartInstance, accuracyChartInstance;

// API Base URL
const API_BASE = 'http://localhost:5000';

// 1. Prediction Handler
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorDiv.textContent = '';
    errorDiv.style.display = 'none';
    
    // Show loading state
    predictionScores.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>Calculating predictions...</p>
        </div>
    `;
    outputDiv.style.display = 'block';

    try {
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        console.log('📤 Sending prediction data:', data);

        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server returned status ${response.status}`);
        }

        const predictions = await response.json();
        console.log('📥 Received predictions:', predictions);
        
        // Display predictions
        let scoresHTML = '';
        for (const [model, score] of Object.entries(predictions)) {
            // Determine score category for styling
            let scoreClass = 'score-average';
            let category = 'Average (B)';
            
            if (score >= 85) {
                scoreClass = 'score-excellent';
                category = 'Excellent (A)';
            } else if (score >= 70) {
                scoreClass = 'score-good';
                category = 'Good (B)';
            } else if (score < 60) {
                scoreClass = 'score-poor';
                category = 'Needs Support (C)';
            }
            
            scoresHTML += `
                <div class="score-card">
                    <h4>${model}</h4>
                    <div class="score-value ${scoreClass}">${score.toFixed(1)}</div>
                    <small style="color: var(--gray); font-size: 0.8rem;">${category}</small>
                </div>
            `;
        }
        predictionScores.innerHTML = scoresHTML;

    } catch (error) {
        console.error('❌ Prediction error:', error);
        errorDiv.textContent = `Prediction Error: ${error.message}`;
        errorDiv.style.display = 'block';
        outputDiv.style.display = 'none';
    }
});

// 2. Load Dataset Statistics
async function loadDatasetStats() {
    try {
        datasetStatsDiv.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Loading dataset statistics...</p>
            </div>
        `;

        const response = await fetch(`${API_BASE}/dataset_stats`);
        if (!response.ok) throw new Error('Failed to fetch dataset statistics');
        
        const stats = await response.json();
        
        let statsHTML = `
            <div class="stat-card">
                <div class="stat-value">${stats.total_students.toLocaleString()}</div>
                <div class="stat-label">Total Students</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.features_count}</div>
                <div class="stat-label">Features Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.average_score}</div>
                <div class="stat-label">Average Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.score_distribution['A (80-100)'].toLocaleString()}</div>
                <div class="stat-label">Grade A (80-100)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.score_distribution['B (60-79)'].toLocaleString()}</div>
                <div class="stat-label">Grade B (60-79)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.score_distribution['C (0-59)'].toLocaleString()}</div>
                <div class="stat-label">Grade C (0-59)</div>
            </div>
        `;
        
        datasetStatsDiv.innerHTML = statsHTML;
        
    } catch (error) {
        console.error('❌ Dataset stats error:', error);
        datasetStatsDiv.innerHTML = `<p class="error-message">Failed to load dataset statistics: ${error.message}</p>`;
    }
}

// 3. Metrics and Graph Initialization
async function loadMetricsAndDrawGraphs() {
    try {
        // Show loading states
        metricsTableBody.innerHTML = '<tr><td colspan="5" style="text-align: center;"><div class="loading"><div class="spinner"></div></div></td></tr>';
        classificationTableBody.innerHTML = '<tr><td colspan="5" style="text-align: center;"><div class="loading"><div class="spinner"></div></div></td></tr>';
        featureImportanceDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading feature importance...</p></div>';
        confusionMatricesDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading confusion matrices...</p></div>';

        const response = await fetch(`${API_BASE}/metrics`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Failed to fetch model metrics (Status: ${response.status})`);
        }
        const metrics = await response.json();

        const labels = Object.keys(metrics);
        const r2Scores = labels.map(label => metrics[label]['R2']);
        const rmseScores = labels.map(label => metrics[label]['RMSE']);
        const maeScores = labels.map(label => metrics[label]['MAE']);
        const accuracyScores = labels.map(label => metrics[label]['Accuracy'] || 0);

        console.log('📊 Loaded metrics:', metrics);

        // Update Metrics Table
        metricsTableBody.innerHTML = '';
        labels.forEach(model => {
            const row = metricsTableBody.insertRow();
            row.insertCell().textContent = model;
            row.insertCell().textContent = metrics[model]['RMSE'].toFixed(4);
            row.insertCell().textContent = metrics[model]['MAE'].toFixed(4);
            row.insertCell().textContent = metrics[model]['R2'].toFixed(4);
            row.insertCell().textContent = (metrics[model]['Accuracy'] || 0).toFixed(4);
        });

        // Load classification metrics and confusion matrices
        try {
            const matrixResponse = await fetch(`${API_BASE}/matrix_data`);
            if (matrixResponse.ok) {
                const matrixData = await matrixResponse.json();
                
                // Update classification table
                classificationTableBody.innerHTML = '';
                labels.forEach(model => {
                    if (matrixData.classification_metrics[model]) {
                        const row = classificationTableBody.insertRow();
                        row.insertCell().textContent = model;
                        row.insertCell().textContent = matrixData.classification_metrics[model]['accuracy'].toFixed(4);
                        row.insertCell().textContent = matrixData.classification_metrics[model]['precision'].toFixed(4);
                        row.insertCell().textContent = matrixData.classification_metrics[model]['recall'].toFixed(4);
                        row.insertCell().textContent = matrixData.classification_metrics[model]['f1_score'].toFixed(4);
                    }
                });
                
                // Display confusion matrices
                let matricesHTML = '';
                for (const [model, matrixImage] of Object.entries(matrixData.confusion_matrices)) {
                    matricesHTML += `
                        <div class="matrix-container">
                            <div class="matrix-title">${model}</div>
                            <img src="data:image/png;base64,${matrixImage}" alt="Confusion Matrix for ${model}" class="matrix-image">
                            <p style="font-size: 0.8rem; color: var(--gray); margin-top: 8px;">
                                A: 80-100, B: 60-79, C: 0-59
                            </p>
                        </div>
                    `;
                }
                confusionMatricesDiv.innerHTML = matricesHTML;
            }
        } catch (error) {
            console.error("❌ Error loading confusion matrices:", error);
            confusionMatricesDiv.innerHTML = '<p class="error-message">Confusion matrices not available. Please check server logs.</p>';
        }

        // Load feature importance data
        try {
            const featureResponse = await fetch(`${API_BASE}/feature_importance`);
            if (featureResponse.ok) {
                const featureData = await featureResponse.json();
                displayFeatureImportance(featureData);
            }
        } catch (error) {
            console.error("❌ Error loading feature importance:", error);
            featureImportanceDiv.innerHTML = '<p class="error-message">Feature importance data not available</p>';
        }

        const chartColors = ['#4361ee', '#7209b7', '#4cc9f0', '#fca311', '#2ecc71'];

        // Draw R2 Score Chart
        const r2Ctx = document.getElementById('r2Chart').getContext('2d');
        if (r2ChartInstance) r2ChartInstance.destroy();
        r2ChartInstance = new Chart(r2Ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'R² Score (Closer to 1 is better)',
                    data: r2Scores,
                    backgroundColor: chartColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { 
                        beginAtZero: true, 
                        max: 1.0, 
                        title: { 
                            display: true, 
                            text: 'R² Score' 
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.7)',
                        titleFont: {
                            size: 14
                        },
                        bodyFont: {
                            size: 13
                        }
                    }
                }
            }
        });

        // Draw RMSE Chart
        const rmseCtx = document.getElementById('rmseChart').getContext('2d');
        if (rmseChartInstance) rmseChartInstance.destroy();
        rmseChartInstance = new Chart(rmseCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'RMSE (Closer to 0 is better)',
                    data: rmseScores,
                    backgroundColor: chartColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { 
                        beginAtZero: true, 
                        title: { 
                            display: true, 
                            text: 'RMSE' 
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.7)',
                        titleFont: {
                            size: 14
                        },
                        bodyFont: {
                            size: 13
                        }
                    }
                }
            }
        });

        // Draw Accuracy Chart
        const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
        if (accuracyChartInstance) accuracyChartInstance.destroy();
        accuracyChartInstance = new Chart(accuracyCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Accuracy (Closer to 1 is better)',
                    data: accuracyScores,
                    backgroundColor: chartColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { 
                        beginAtZero: true, 
                        max: 1.0, 
                        title: { 
                            display: true, 
                            text: 'Accuracy' 
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.7)',
                        titleFont: {
                            size: 14
                        },
                        bodyFont: {
                            size: 13
                        }
                    }
                }
            }
        });

        // Add export functionality
        document.getElementById('export-r2').addEventListener('click', () => {
            exportChartAsImage(r2ChartInstance, 'r2-score-comparison');
        });
        
        document.getElementById('export-rmse').addEventListener('click', () => {
            exportChartAsImage(rmseChartInstance, 'rmse-comparison');
        });
        
        document.getElementById('export-accuracy').addEventListener('click', () => {
            exportChartAsImage(accuracyChartInstance, 'accuracy-comparison');
        });
        
        document.getElementById('export-prediction').addEventListener('click', () => {
            exportPredictionData();
        });

    } catch (error) {
        console.error("❌ Error loading metrics:", error);
        metricsTableBody.innerHTML = `<tr><td colspan="5" class="error-message" style="text-align: center;">Error: ${error.message}. Please ensure models are trained and server is running.</td></tr>`;
    }
}

// 4. Display Feature Importance
function displayFeatureImportance(featureData) {
    let featureHTML = '';
    
    // Sort features by importance
    const sortedFeatures = Object.entries(featureData)
        .sort((a, b) => b[1] - a[1]);
    
    // Find max value for scaling
    const maxValue = Math.max(...Object.values(featureData));
    
    sortedFeatures.forEach(([feature, value]) => {
        const percentage = (value / maxValue) * 100;
        featureHTML += `
            <div class="feature-bar">
                <div class="feature-name">${formatFeatureName(feature)}</div>
                <div class="feature-value">
                    <div class="feature-fill" style="width: ${percentage}%"></div>
                </div>
                <div class="feature-percent">${(value * 100).toFixed(1)}%</div>
            </div>
        `;
    });
    
    featureImportanceDiv.innerHTML = featureHTML;
}

// 5. Format feature names for display
function formatFeatureName(feature) {
    // Convert snake_case to Title Case with spaces
    return feature
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// 6. Export chart as image
function exportChartAsImage(chart, filename) {
    const link = document.createElement('a');
    link.download = `${filename}.png`;
    link.href = chart.toBase64Image();
    link.click();
}

// 7. Export prediction data
function exportPredictionData() {
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    const predictions = {};
    const scoreCards = document.querySelectorAll('.score-card');
    scoreCards.forEach(card => {
        const model = card.querySelector('h4').textContent;
        const score = card.querySelector('.score-value').textContent;
        const category = card.querySelector('small').textContent;
        predictions[model] = { score, category };
    });
    
    const exportData = {
        input: data,
        predictions: predictions,
        timestamp: new Date().toISOString(),
        project: "EduScore Predictor"
    };
    
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportData, null, 2));
    const link = document.createElement('a');
    link.download = "eduscore-prediction.json";
    link.href = dataStr;
    link.click();
}

// Health check function
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            const health = await response.json();
            console.log('✅ Server health:', health);
            return true;
        }
    } catch (error) {
        console.error('❌ Server health check failed:', error);
        return false;
    }
}

// Load all data when the page loads
window.onload = async function() {
    console.log('🚀 Initializing EduScore Predictor...');
    
    // Check server health
    const isHealthy = await checkServerHealth();
    if (!isHealthy) {
        errorDiv.textContent = '⚠️ Backend server is not running. Please start the Flask server (app.py) on port 5000.';
        errorDiv.style.display = 'block';
    }
    
    // Load data
    await loadMetricsAndDrawGraphs();
    await loadDatasetStats();
    
    console.log('✅ EduScore Predictor initialized successfully!');
};