"""
Alternative model loader that handles scikit-learn compatibility issues
"""

import warnings
import os
import pickle
import io

# Suppress all warnings
warnings.filterwarnings('ignore')

def safe_load_model(model_path):
    """
    Safely load a model with compatibility fixes for scikit-learn version mismatches.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        The loaded model or None if loading fails
    """
    
    if not os.path.exists(model_path):
        return None
    
    try:
        # First, try to patch the ColumnTransformer class
        try:
            import sklearn.compose._column_transformer as ct
            from sklearn.compose import ColumnTransformer
            
            # Add the missing attribute if it doesn't exist
            if not hasattr(ColumnTransformer, '_RemainderColsList'):
                def _get_remainder_cols_list(self):
                    return []
                
                ColumnTransformer._RemainderColsList = property(_get_remainder_cols_list)
                ct.ColumnTransformer._RemainderColsList = property(_get_remainder_cols_list)
        except Exception:
            pass
        
        # Try to load the model
        try:
            import joblib
            model = joblib.load(model_path)
            return model
            
        except AttributeError as e:
            if '_RemainderColsList' in str(e):
                return _load_with_custom_unpickler(model_path)
            else:
                raise e
                
    except Exception:
        return None

def _load_with_custom_unpickler(model_path):
    """
    Load model using a custom unpickler that handles missing attributes.
    """
    try:
        # Read the raw pickle data
        with open(model_path, 'rb') as f:
            raw_data = f.read()
        
        # Create a custom unpickler
        class SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle missing _RemainderColsList attribute
                if module == 'sklearn.compose._column_transformer' and name == '_RemainderColsList':
                    # Return a simple class that can be instantiated
                    class DummyRemainderColsList:
                        def __init__(self):
                            pass
                        def __getitem__(self, key):
                            return []
                        def __len__(self):
                            return 0
                    return DummyRemainderColsList
                
                # For other missing attributes, try to return a dummy class
                if 'sklearn' in module and name.startswith('_'):
                    class DummyAttribute:
                        def __init__(self):
                            pass
                        def __getitem__(self, key):
                            return []
                        def __len__(self):
                            return 0
                    return DummyAttribute
                
                return super().find_class(module, name)
        
        # Load with custom unpickler
        with io.BytesIO(raw_data) as f:
            unpickler = SafeUnpickler(f)
            model = unpickler.load()
        
        return model
        
    except Exception:
        return None
