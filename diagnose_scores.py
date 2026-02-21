import sys
sys.path.insert(0, '.')

import numpy as np

try:
    # Try to import with try/except for missing packages
    import joblib
    import os
    
    # Load processed data
    processed_data_path = 'data/processed/processed_data.pkl'
    if os.path.exists(processed_data_path):
        data = joblib.load(processed_data_path)
        X = data['X']
        print(f'Dataset shape: {X.shape}')
        
        # Load model
        model_path = 'models/isolation_forest_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Get predictions and scores
            predictions = model.predict(X)
            scores = model.decision_function(X)
            
            print(f'\nPredictions summary:')
            print(f'Normal (1): {(predictions == 1).sum()}')
            print(f'Anomaly (-1): {(predictions == -1).sum()}')
            
            # Separate scores for normal vs anomalies
            normal_scores = scores[predictions == 1]
            anomaly_scores = scores[predictions == -1]
            
            print(f'\n=== NORMAL DATA SCORES ===')
            print(f'Count: {len(normal_scores)}')
            print(f'Min: {normal_scores.min():.6f}')
            print(f'Max: {normal_scores.max():.6f}')
            print(f'Mean: {normal_scores.mean():.6f}')
            print(f'Median: {np.median(normal_scores):.6f}')
            
            print(f'\n=== ANOMALY DATA SCORES ===')
            print(f'Count: {len(anomaly_scores)}')
            print(f'Min: {anomaly_scores.min():.6f}')
            print(f'Max: {anomaly_scores.max():.6f}')
            print(f'Mean: {anomaly_scores.mean():.6f}')
            print(f'Median: {np.median(anomaly_scores):.6f}')
            
            print(f'\n=== ANOMALY SCORE PERCENTILES ===')
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                print(f'{p}%: {np.percentile(anomaly_scores, p):.6f}')
            
            print(f'\n=== FIRST 50 ANOMALY SCORES ===')
            print(anomaly_scores[:50])
            
            # Test new thresholds on anomalies
            print(f'\n=== NEW THRESHOLDS ON ANOMALIES ===')
            high = (anomaly_scores < -0.7).sum()
            medium = ((anomaly_scores >= -0.7) & (anomaly_scores < -0.3)).sum()
            low = (anomaly_scores >= -0.3).sum()
            
            print(f'High (< -0.7): {high} ({high/len(anomaly_scores)*100:.2f}%)')
            print(f'Medium ([-0.7, -0.3)): {medium} ({medium/len(anomaly_scores)*100:.2f}%)')
            print(f'Low (>= -0.3): {low} ({low/len(anomaly_scores)*100:.2f}%)')
            
        else:
            print('Model not found at:', model_path)
    else:
        print('Processed data not found at:', processed_data_path)
        
except ImportError as e:
    print(f'Missing module: {e}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
