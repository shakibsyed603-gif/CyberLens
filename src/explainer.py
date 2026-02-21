import shap
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_PATH, SCALER_PATH, PROCESSED_DATA_DIR

class ThreatExplainer:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.scaler = None
        self.feature_columns = None
        self.X_sample = None
        self.y_sample = None
        
    def load_model_and_data(self):
        """Load the trained model and data"""
        print("Loading model and data...")
        
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
        
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Please process data first.")
        
        # Load model
        self.model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        
        # Load scaler
        self.scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded from {SCALER_PATH}")
        
        # Load processed data
        processed_data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at {processed_data_path}")
            
        processed_data = joblib.load(processed_data_path)
        self.feature_columns = processed_data['feature_columns']
        
        # Get a sample of data for explanation
        X, y = processed_data['X'], processed_data['y']
        
        # Get predictions to find anomalies
        predictions = self.model.predict(X)
        anomaly_indices = np.where(predictions == -1)[0]
        
        print(f"Found {len(anomaly_indices)} anomalies in the dataset")
        
        # Take a sample of anomalies for explanation
        sample_size = min(1000, len(anomaly_indices))
        if len(anomaly_indices) > 0:
            sample_indices = np.random.choice(anomaly_indices, sample_size, replace=False)
            self.X_sample = X.iloc[sample_indices].reset_index(drop=True)
            self.y_sample = y.iloc[sample_indices].reset_index(drop=True)
        else:
            # If no anomalies, use random sample
            print("No anomalies found, using random sample")
            self.X_sample = X.sample(min(1000, len(X))).reset_index(drop=True)
            self.y_sample = y.loc[self.X_sample.index].reset_index(drop=True)
        
        print(f"Sample data shape: {self.X_sample.shape}")
        
        return self.X_sample, self.y_sample
    
    def initialize_explainer(self, background_size=100):
        """Initialize SHAP explainer"""
        print("Initializing SHAP explainer...")
        
        # Use a subset of data as background
        background_data = self.X_sample.sample(min(background_size, len(self.X_sample)), random_state=42)
        
        # Use KernelExplainer for Isolation Forest
        # This may take some time
        self.explainer = shap.KernelExplainer(
            self.model.decision_function, 
            background_data
        )
        
        print("SHAP explainer initialized!")
        
    def explain_instance(self, instance_index):
        """Explain a specific anomaly instance"""
        if instance_index >= len(self.X_sample):
            raise ValueError(f"Instance index {instance_index} out of range (max: {len(self.X_sample)-1})")
        
        # Get the instance
        instance = self.X_sample.iloc[instance_index:instance_index+1]
        
        # Calculate SHAP values
        print(f"Calculating SHAP values for instance {instance_index}...")
        shap_values = self.explainer.shap_values(instance)
        
        # Get feature contributions
        feature_contributions = pd.DataFrame({
            'feature': self.feature_columns,
            'value': instance.iloc[0].values,
            'shap_value': shap_values[0]
        })
        
        # Sort by absolute SHAP value
        feature_contributions['abs_shap'] = np.abs(feature_contributions['shap_value'])
        feature_contributions = feature_contributions.sort_values('abs_shap', ascending=False)
        
        # Get model prediction
        prediction = self.model.predict(instance)[0]
        anomaly_score = self.model.decision_function(instance)[0]
        
        return {
            'instance': instance,
            'prediction': prediction,
            'anomaly_score': anomaly_score,
            'feature_contributions': feature_contributions,
            'shap_values': shap_values,
            'expected_value': self.explainer.expected_value
        }
    
    def get_top_contributing_features(self, instance_index, top_n=10):
        """Get top contributing features for an instance"""
        explanation = self.explain_instance(instance_index)
        
        top_features = explanation['feature_contributions'].head(top_n)
        
        return top_features
    
    def generate_force_plot_html(self, instance_index):
        """Generate HTML for SHAP force plot"""
        explanation = self.explain_instance(instance_index)
        
        # Create force plot
        force_plot = shap.force_plot(
            explanation['expected_value'],
            explanation['shap_values'][0],
            explanation['instance'].iloc[0],
            feature_names=self.feature_columns,
            matplotlib=False
        )
        
        # Convert to HTML
        shap.initjs()
        force_plot_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        
        return force_plot_html

if __name__ == "__main__":
    explainer = ThreatExplainer()
    explainer.load_model_and_data()
    explainer.initialize_explainer()
    
    # Test explanation
    print("\nTesting explanation for first instance...")
    explanation = explainer.explain_instance(0)
    print("\nTop contributing features:")
    print(explanation['feature_contributions'].head(10))
    print(f"\nAnomaly score: {explanation['anomaly_score']:.4f}")
    print(f"Prediction: {'Anomaly' if explanation['prediction'] == -1 else 'Normal'}")
