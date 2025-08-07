"""
Ultra-silent model loader that suppresses ALL errors at system level
"""
import os
import sys
import warnings
import contextlib
import io
import traceback

# Suppress ALL warnings and errors
warnings.filterwarnings('ignore')

# Redirect all output to null
class NullWriter:
    def write(self, text):
        pass
    def flush(self):
        pass

def ultra_silent_load_model(model_path):
    """
    Load model with ultra-silence - no errors, no warnings, no output, no exceptions
    """
    if not os.path.exists(model_path):
        return None
    
    # Completely suppress all output and errors
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect all output to null
        sys.stdout = NullWriter()
        sys.stderr = NullWriter()
        
        # Also capture with contextlib as backup
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                import joblib
                model = joblib.load(model_path)
                return model
            except Exception:
                # Completely ignore any exceptions
                return None
            except:
                # Catch absolutely everything
                return None
    except:
        # Catch any exceptions from the loading process itself
        return None
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def ultra_silent_load_all_models(working_dir):
    """
    Load all models with ultra-silence and return a dictionary
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
        
        # Load with ultra-silence
        model = ultra_silent_load_model(model_path)
        models[model_name] = model if model is not None else "unavailable"
    
    return models
