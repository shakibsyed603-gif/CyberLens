# 🔐 CyberLens: Explainable AI for Network Security

An intelligent network intrusion detection system powered by Isolation Forest and SHAP (SHapley Additive exPlanations) for explainable AI-driven threat analysis.

## 📋 Overview

CyberLens uses unsupervised machine learning to detect network anomalies and provides human-interpretable explanations for each detected threat. Built with the KDD Cup dataset, it demonstrates how AI can enhance cybersecurity while maintaining transparency.

## ✨ Features

- **Anomaly Detection**: Isolation Forest algorithm for unsupervised threat detection
- **Explainable AI**: SHAP values provide feature-level explanations for each detection
- **Interactive Dashboard**: Real-time visualization of threats and security metrics
- **Threat Analysis**: Detailed breakdown of contributing factors for each anomaly
- **Visual Explanations**: Force plots showing how features influence predictions

## 🗂️ Project Structure

```
CyberLens/
├── data/
│   ├── raw/                    # KDD Cup dataset files (KDDTrain+.txt, KDDTest+.txt)
│   └── processed/              # Processed data (auto-generated)
├── models/
│   ├── isolation_forest_model.pkl  # Trained model (auto-generated)
│   └── scaler.pkl                  # Feature scaler (auto-generated)
├── src/
│   ├── __init__.py
│   ├── data_processor.py       # Data loading and preprocessing
│   ├── model_trainer.py        # Model training and evaluation
│   └── explainer.py            # SHAP-based explanations
├── app.py                      # Streamlit web application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd CyberLens
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify dataset**
   - Ensure KDD Cup dataset files are in `data/raw/`
   - Files should be: `KDDTrain+.txt` and `KDDTest+.txt`

## 📊 Dataset Information

This project uses the **KDD Cup 1999** dataset:
- **Features**: 42 network traffic features
- **Labels**: Normal traffic vs. various attack types (DoS, Probe, R2L, U2R)
- **Format**: Comma-separated text files without headers

### Dataset Features Include:
- Connection duration, protocol type, service
- Bytes transferred, error rates
- Connection counts and patterns
- Host-based features

## 🔧 Usage

### Step 1: Process the Data

```bash
python src/data_processor.py
```

This will:
- Load KDD Cup dataset files
- Clean and preprocess the data
- Encode categorical features
- Scale numerical features
- Save processed data to `data/processed/`

**Note**: By default, it processes 50,000 samples for faster execution. To use the full dataset, modify the `sample_size` parameter in `data_processor.py`.

### Step 2: Train the Model

```bash
python src/model_trainer.py
```

This will:
- Load processed data
- Split into training and test sets
- Train Isolation Forest on normal traffic
- Evaluate model performance
- Save trained model to `models/`

Expected output includes:
- Classification report
- Confusion matrix
- Precision, Recall, F1-Score
- ROC AUC Score

### Step 3: Run the Web Application

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 🎯 Using the Dashboard

### Main Features:

1. **Security Dashboard**
   - View total connections analyzed
   - Monitor anomalies detected
   - Track high-priority threats
   - Check overall threat level

2. **Threat Analysis**
   - Pie chart of threat level distribution
   - Bar chart of attack types
   - Timeline of threat detections

3. **Threat Investigation**
   - Select specific threats for detailed analysis
   - View SHAP explanations showing which features contributed to detection
   - Interactive force plots visualizing feature impacts

### Filters:
- **Threat Level**: Filter by High/Medium/Low severity
- **Attack Type**: Filter by specific attack categories

## 🧠 How It Works

### 1. Anomaly Detection (Isolation Forest)
- Trains only on normal network traffic
- Isolates anomalies based on feature space partitioning
- Assigns anomaly scores (lower = more anomalous)

### 2. Explainability (SHAP)
- Calculates Shapley values for each feature
- Shows positive/negative contributions to anomaly score
- Provides local explanations for individual predictions

### 3. Threat Classification
- **High**: Anomaly score < -0.5
- **Medium**: Anomaly score between -0.5 and -0.2
- **Low**: Anomaly score > -0.2

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model parameters
MODEL_CONFIG = {
    'contamination': 0.1,      # Expected anomaly proportion
    'n_estimators': 100,       # Number of trees
    'random_state': 42,
    'max_samples': 'auto',
    'max_features': 1.0
}

# UI settings
UI_CONFIG = {
    'max_anomalies_display': 100  # Max threats to display
}
```

## 📈 Performance Metrics

The model is evaluated using:
- **Precision**: Accuracy of anomaly predictions
- **Recall**: Percentage of actual anomalies detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Overall classification performance

## ⚠️ Important Notes

### Performance Considerations:
- **SHAP Explanations**: Can be slow (30-60 seconds per instance)
- **Background Size**: Reduce in `explainer.py` for faster explanations
- **Sample Size**: Use data sampling for quicker prototyping

### Known Limitations:
- SHAP KernelExplainer is computationally expensive
- Large datasets may require significant processing time
- Explainer initialization can take 1-2 minutes

## 🛠️ Troubleshooting

### "No data files found"
- Ensure KDD dataset files are in `data/raw/`
- Files must be named `KDDTrain+.txt` or `KDDTest+.txt`

### "Model not found"
- Run `python src/data_processor.py` first
- Then run `python src/model_trainer.py`

### "SHAP explanations too slow"
- Reduce `background_size` in `explainer.py` (line 72)
- Use smaller data samples
- Consider using TreeExplainer alternatives

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)

## 🔬 Technical Stack

- **Machine Learning**: scikit-learn (Isolation Forest)
- **Explainability**: SHAP
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, Seaborn, Matplotlib

## 📝 Future Enhancements

- [ ] Support for real-time network traffic monitoring
- [ ] Additional ML models (One-Class SVM, Autoencoders)
- [ ] Alert system for high-priority threats
- [ ] Historical threat tracking and reporting
- [ ] Model retraining pipeline
- [ ] Export threat reports to PDF/CSV

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Performance optimization for SHAP
- Additional visualization features
- Support for other datasets (CIC-IDS2017, NSL-KDD)
- Real-time data ingestion

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- KDD Cup 1999 dataset providers
- SHAP library developers
- Streamlit community

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ for transparent and explainable cybersecurity**
