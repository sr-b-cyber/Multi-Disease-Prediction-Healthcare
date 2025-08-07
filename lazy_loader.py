"""
Lazy model loader that only loads models when needed
"""
import os
import warnings
import contextlib
import io

# Suppress all warnings
warnings.filterwarnings('ignore')

class LazyModelLoader:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self._models = {}
        self._loaded = False
    
    def _load_all_models(self):
        """Load all models silently when first accessed"""
        if self._loaded:
            return
        
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
            model_path = os.path.join(self.working_dir, 'models', model_file)
            model_name = model_file.replace('.pkl', '')
            
            # Load silently
            model = self._silent_load(model_path)
            self._models[model_name] = model if model is not None else "unavailable"
        
        self._loaded = True
    
    def _silent_load(self, model_path):
        """Load a single model with complete silence"""
        if not os.path.exists(model_path):
            return None
        
        # Capture all output and errors
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                import joblib
                return joblib.load(model_path)
            except Exception:
                return None
    
    def get_model(self, model_name):
        """Get a model, loading all models if not already loaded"""
        if not self._loaded:
            self._load_all_models()
        return self._models.get(model_name, "unavailable")
    
    def get_all_models(self):
        """Get all models, loading them if not already loaded"""
        if not self._loaded:
            self._load_all_models()
        return self._models.copy()
