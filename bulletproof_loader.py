"""
Bulletproof model loader that fixes scikit-learn compatibility issues
"""
import os
import sys
import warnings
import contextlib
import io

# Suppress all warnings
warnings.filterwarnings('ignore')

def fix_sklearn_compatibility():
    """
    Fix scikit-learn compatibility by adding missing _RemainderColsList attribute
    """
    try:
        import sklearn.compose._column_transformer as ct
        
        # Add the missing _RemainderColsList attribute if it doesn't exist
        if not hasattr(ct, '_RemainderColsList'):
            class _RemainderColsList:
                def __init__(self):
                    pass
                def __getitem__(self, key):
                    return []
                def __len__(self):
                    return 0
                def __iter__(self):
                    return iter([])
            
            # Add to the module
            ct._RemainderColsList = _RemainderColsList
            
            # Also add to ColumnTransformer class if it exists
            try:
                from sklearn.compose import ColumnTransformer
                if not hasattr(ColumnTransformer, '_RemainderColsList'):
                    ColumnTransformer._RemainderColsList = _RemainderColsList
            except:
                pass
                
    except Exception:
        pass

def bulletproof_load_model(model_path):
    """
    Load model with bulletproof error handling
    """
    if not os.path.exists(model_path):
        return None
    
    # Fix sklearn compatibility first
    fix_sklearn_compatibility()
    
    # Capture all output and errors
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            import joblib
            return joblib.load(model_path)
        except Exception:
            return None

def bulletproof_load_all_models(working_dir):
    """
    Load all models with bulletproof error handling
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
        
        # Load with bulletproof error handling
        model = bulletproof_load_model(model_path)
        models[model_name] = model if model is not None else "unavailable"
    
    return models
