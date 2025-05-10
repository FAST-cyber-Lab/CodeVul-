
import os
import joblib
import numpy as np
import pandas as pd

def load_saved_models(models_dir="saved_models/saved_models"):

    models = {}
    model_files = {
        'svm': 'svm_model.pkl',
        'xgb': 'xgb_model.pkl',
        'extra_tree': 'extra_tree_model.pkl',
        'log_reg': 'log_reg_model.pkl',
        'nn': 'stacked_nn_model.pkl'
    }
    
    for model_name, file_name in model_files.items():
        model_path = os.path.join(models_dir, file_name)
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
            
        else:
            print(f"Warning: Model file {file_name} not found")
    
    return models

def stacked_prediction(X_new, models):

    svm_probs = models['svm'].predict_proba(X_new)
    xgb_probs = models['xgb'].predict_proba(X_new)
    extra_tree_probs = models['extra_tree'].predict_proba(X_new)
    log_reg_probs = models['log_reg'].predict_proba(X_new)
    
    stacked_probs = np.hstack((svm_probs, xgb_probs, extra_tree_probs, log_reg_probs))
    return models['nn'].predict(stacked_probs), models['nn'].predict_proba(stacked_probs)

def predict_vulnerabilities(embeddings_path, models_dir="saved_models/saved_models", output_path=None):

    print(f"Loading embeddings from {embeddings_path}")
    embeddings_df = pd.read_csv(embeddings_path)

    models = load_saved_models(models_dir)
    if not models or len(models) < 5:
        print("Error: Could not load all required models")
        return None

    print("Making vulnerability predictions...")
    X = embeddings_df.values
    predictions, probabilities = stacked_prediction(X, models)
    
    results_df = embeddings_df.copy()
    results_df['prediction'] = predictions
    

    for i in range(5):  
        results_df[f'probability_class_{i}'] = probabilities[:, i]

    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Prediction results saved to {output_path}")
    
    return results_df
