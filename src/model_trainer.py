import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import joblib
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_CONFIG, MODEL_PATH, PROCESSED_DATA_DIR

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_processed_data(self):
        """Load the processed data"""
        data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed data not found at {data_path}. Please run data_processor.py first.")
        
        processed_data = joblib.load(data_path)
        print(f"Loaded processed data with shape: {processed_data['X'].shape}")
        
        return processed_data['X'], processed_data['y'], processed_data['feature_columns']
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets"""
        print("Splitting data...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set - Normal: {(self.y_train == 0).sum()}, Anomaly: {(self.y_train == 1).sum()}")
        print(f"Test set - Normal: {(self.y_test == 0).sum()}, Anomaly: {(self.y_test == 1).sum()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_isolation_forest(self):
        """Train the Isolation Forest model"""
        print("Training Isolation Forest model...")
        print(f"Model configuration: {MODEL_CONFIG}")
        
        # Initialize model
        self.model = IsolationForest(**MODEL_CONFIG)
        
        # Train on normal data only (unsupervised anomaly detection)
        normal_data = self.X_train[self.y_train == 0]
        print(f"Training on {len(normal_data)} normal samples...")
        
        self.model.fit(normal_data)
        
        print("Model training completed!")
        
        # Save the model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model performance"""
        print("\n" + "="*50)
        print("EVALUATING MODEL")
        print("="*50)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Convert Isolation Forest output (-1, 1) to (1, 0) for anomaly detection
        # -1 = anomaly, 1 = normal
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        # Get anomaly scores (lower scores = more anomalous)
        anomaly_scores = self.model.decision_function(self.X_test)
        
        # Print evaluation metrics
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_binary, target_names=['Normal', 'Anomaly']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred_binary)
        print(cm)
        print(f"\nTrue Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred_binary, average='binary')
        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # ROC AUC Score (negative because lower scores indicate anomalies)
        try:
            roc_auc = roc_auc_score(self.y_test, -anomaly_scores)
            print(f"ROC AUC Score: {roc_auc:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
        
        print("="*50 + "\n")
        
        return y_pred_binary, anomaly_scores
    
    def train_full_pipeline(self):
        """Run the complete training pipeline"""
        # Load processed data
        X, y, feature_columns = self.load_processed_data()
        
        # Split data
        self.split_data(X, y)
        
        # Train model
        self.train_isolation_forest()
        
        # Evaluate model
        predictions, scores = self.evaluate_model()
        
        print("Training pipeline completed successfully!")
        
        return self.model, self.X_test, self.y_test, predictions, scores

if __name__ == "__main__":
    trainer = ModelTrainer()
    model, X_test, y_test, predictions, scores = trainer.train_full_pipeline()
