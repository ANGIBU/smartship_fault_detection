# config.py

import os
import numpy as np
from pathlib import Path

class Config:
    # Path settings
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    
    # File paths
    TRAIN_FILE = DATA_DIR / "train.csv"
    TEST_FILE = DATA_DIR / "test.csv"
    SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"
    
    # Output files
    RESULT_FILE = BASE_DIR / "submission.csv"
    MODEL_FILE = MODEL_DIR / "best_model.pkl"
    SCALER_FILE = MODEL_DIR / "scaler.pkl"
    SELECTOR_FILE = MODEL_DIR / "selector.pkl"
    CV_RESULTS_FILE = MODEL_DIR / "cv_results.csv"
    
    # Data settings
    FEATURE_COLUMNS = [f'X_{i:02d}' for i in range(1, 53)]
    TARGET_COLUMN = 'target'
    ID_COLUMN = 'ID'
    
    # Model settings
    N_CLASSES = 21
    RANDOM_STATE = 42
    N_JOBS = 6
    
    # Cross-validation settings
    CV_FOLDS = 7
    VALIDATION_SIZE = 0.2
    
    # Feature selection settings
    FEATURE_SELECTION_METHODS = ['mutual_info', 'f_classif', 'chi2']
    TARGET_FEATURES = 35
    
    # Class weight settings
    USE_CLASS_WEIGHTS = True
    FOCAL_LOSS_ALPHA = 1.0
    FOCAL_LOSS_GAMMA = 2.0
    
    # LightGBM parameters
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'min_split_gain': 0.02,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'max_depth': 6,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 500,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced'
    }
    
    # XGBoost parameters
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': N_CLASSES,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'gamma': 0.1,
        'min_child_weight': 1,
        'random_state': RANDOM_STATE,
        'n_estimators': 500,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    # CatBoost parameters
    CAT_PARAMS = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'border_count': 128,
        'thread_count': N_JOBS,
        'random_state': RANDOM_STATE,
        'verbose': False,
        'loss_function': 'MultiClass',
        'classes_count': N_CLASSES,
        'auto_class_weights': 'Balanced'
    }
    
    # Random Forest parameters
    RF_PARAMS = {
        'n_estimators': 300,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced'
    }
    
    # Extra Trees parameters
    ET_PARAMS = {
        'n_estimators': 300,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced'
    }
    
    # Hyperparameter tuning settings
    OPTUNA_TRIALS = 100
    OPTUNA_TIMEOUT = 3600
    
    # Ensemble settings
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.3,
        'xgboost': 0.25,
        'catboost': 0.25,
        'random_forest': 0.1,
        'extra_trees': 0.1
    }
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Memory settings
    MEMORY_EFFICIENT = True
    CHUNK_SIZE = 10000
    
    # Validation strategy settings
    VALIDATION_STRATEGY = 'stratified'
    
    # Class performance threshold
    CLASS_PERFORMANCE_THRESHOLD = 0.60
    
    # Scaling method
    SCALING_METHOD = 'robust'
    
    @classmethod
    def create_directories(cls):
        """Create directories"""
        for directory in [cls.MODEL_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_params(cls, model_name):
        """Return model-specific parameters"""
        params_map = {
            'lightgbm': cls.LGBM_PARAMS,
            'xgboost': cls.XGB_PARAMS,
            'catboost': cls.CAT_PARAMS,
            'random_forest': cls.RF_PARAMS,
            'extra_trees': cls.ET_PARAMS
        }
        return params_map.get(model_name, {}).copy()
    
    @classmethod
    def validate_config(cls):
        """Validate configuration values"""
        errors = []
        
        if not cls.DATA_DIR.exists():
            errors.append(f"Data directory not found: {cls.DATA_DIR}")
        
        required_files = [cls.TRAIN_FILE, cls.TEST_FILE]
        for file_path in required_files:
            if not file_path.exists():
                errors.append(f"Required file not found: {file_path}")
        
        if cls.N_CLASSES <= 0:
            errors.append("N_CLASSES must be positive")
        
        if cls.CV_FOLDS < 2:
            errors.append("CV_FOLDS must be 2 or greater")
        
        if cls.TARGET_FEATURES <= 0:
            errors.append("TARGET_FEATURES must be positive")
        
        return errors
    
    @classmethod
    def update_for_hardware(cls, available_memory_gb, cpu_cores):
        """Adjust settings for hardware specifications"""
        if available_memory_gb >= 32:
            cls.N_JOBS = min(cpu_cores, 8)
            cls.CHUNK_SIZE = 15000
            cls.OPTUNA_TRIALS = 150
        elif available_memory_gb >= 16:
            cls.N_JOBS = min(cpu_cores, 6)
            cls.CHUNK_SIZE = 10000
            cls.OPTUNA_TRIALS = 100
        else:
            cls.N_JOBS = min(cpu_cores, 4)
            cls.CHUNK_SIZE = 5000
            cls.OPTUNA_TRIALS = 50
        
        # Update model parameters
        for params in [cls.LGBM_PARAMS, cls.XGB_PARAMS, cls.CAT_PARAMS, 
                      cls.RF_PARAMS, cls.ET_PARAMS]:
            if 'n_jobs' in params:
                params['n_jobs'] = cls.N_JOBS
            if 'thread_count' in params:
                params['thread_count'] = cls.N_JOBS