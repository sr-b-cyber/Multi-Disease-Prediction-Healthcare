"""
Completely silent model loader that suppresses all errors
"""
import os
import sys
import warnings
import contextlib
import io

# Suppress all warnings
warnings.filterwarnings('ignore')

def silent_load_model(model_path):
    """
    Load model with complete silence - no errors, no warnings, no output
    """
    if not os.path.exists(model_path):
        return None
    
    # Capture all output and errors
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            import joblib
            return joblib.load(model_path)
        except Exception:
            return None

def silent_load_all_models(working_dir):
    """
    Load all models silently and return a dictionary
    """
    models = {}
    
    # List of model files to load
    model_files = [
        'Diabetes_model.pkl',
        'Asthma_model.pkl', 
        'BP_model.pkl',
        'Typhoid_model.pkl',
        'Diabetes_preprocessor.pkl',
        'Asthma_preprocessor.pkl',
        'BP_preprocessor.pkl'
    ]
    
    for model_file in model_files:
        model_path = os.path.join(working_dir, 'models', model_file)
        model_name = model_file.replace('.pkl', '')
        
        # Load silently
        model = silent_load_model(model_path)
        models[model_name] = model if model is not None else "unavailable"
    
    return models
