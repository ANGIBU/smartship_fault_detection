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
    N_JOBS = 8
    
    # Cross-validation settings
    CV_FOLDS = 7
    VALIDATION_SIZE = 0.2
    
    # Feature selection settings
    FEATURE_SELECTION_METHODS = ['mutual_info', 'f_classif', 'chi2']
    TARGET_FEATURES = 42
    
    # Class weight settings
    USE_CLASS_WEIGHTS = True
    FOCAL_LOSS_ALPHA = 1.2
    FOCAL_LOSS_GAMMA = 2.5
    
    # Quick mode settings
    QUICK_MODE = False
    QUICK_SAMPLE_SIZE = 1000
    QUICK_FEATURE_COUNT = 15
    QUICK_N_ESTIMATORS = 50
    
    # LightGBM parameters
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 45,
        'learning_rate': 0.045,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 3,
        'min_child_samples': 15,
        'min_child_weight': 0.0008,
        'min_split_gain': 0.015,
        'reg_alpha': 0.12,
        'reg_lambda': 0.08,
        'max_depth': 7,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 600,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced',
        'subsample': 0.88,
        'colsample_bytree': 0.9
    }
    
    # Quick LightGBM parameters
    QUICK_LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'min_child_samples': 10,
        'max_depth': 4,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': QUICK_N_ESTIMATORS,
        'n_jobs': 1,
        'class_weight': 'balanced'
    }
    
    # XGBoost parameters
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': N_CLASSES,
        'learning_rate': 0.042,
        'max_depth': 7,
        'subsample': 0.88,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.08,
        'reg_lambda': 0.12,
        'gamma': 0.08,
        'min_child_weight': 0.8,
        'random_state': RANDOM_STATE,
        'n_estimators': 650,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
        'verbosity': 0,
        'eval_metric': 'mlogloss',
        'scale_pos_weight': 1.2
    }
    
    # CatBoost parameters
    CAT_PARAMS = {
        'iterations': 550,
        'learning_rate': 0.048,
        'depth': 7,
        'l2_leaf_reg': 2.5,
        'border_count': 150,
        'thread_count': N_JOBS,
        'random_state': RANDOM_STATE,
        'verbose': False,
        'loss_function': 'MultiClass',
        'classes_count': N_CLASSES,
        'auto_class_weights': 'Balanced',
        'bootstrap_type': 'MVS'
    }
    
    # Random Forest parameters
    RF_PARAMS = {
        'n_estimators': 400,
        'max_depth': 12,
        'min_samples_split': 8,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced',
        'bootstrap': True,
        'max_samples': 0.9
    }
    
    # Extra Trees parameters
    ET_PARAMS = {
        'n_estimators': 350,
        'max_depth': 11,
        'min_samples_split': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced',
        'bootstrap': False
    }
    
    # Hyperparameter tuning settings
    OPTUNA_TRIALS = 35
    OPTUNA_TIMEOUT = 2400
    OPTUNA_CV_FOLDS = 3
    
    # Ensemble settings
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.32,
        'xgboost': 0.35,
        'catboost': 0.33
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
    CLASS_PERFORMANCE_THRESHOLD = 0.65
    
    # Scaling method
    SCALING_METHOD = 'robust'
    
    # Feature engineering settings
    CREATE_INTERACTION_FEATURES = True
    CREATE_POLYNOMIAL_FEATURES = True
    POLYNOMIAL_DEGREE = 2
    INTERACTION_TOP_N = 10
    
    # Statistical feature settings
    STATISTICAL_FEATURES = [
        'mean', 'std', 'median', 'min', 'max', 'range',
        'skew', 'kurtosis', 'q25', 'q75', 'iqr', 'cv',
        'outlier_count', 'outlier_ratio', 'zero_count',
        'negative_count', 'positive_count'
    ]
    
    @classmethod
    def create_directories(cls):
        """Create directories"""
        for directory in [cls.MODEL_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_quick_mode(cls):
        """Setup quick mode configuration"""
        cls.QUICK_MODE = True
        cls.TARGET_FEATURES = cls.QUICK_FEATURE_COUNT
        cls.CV_FOLDS = 2
        cls.VALIDATION_SIZE = 0.15
        cls.USE_CLASS_WEIGHTS = False
        cls.N_JOBS = 1
        cls.OPTUNA_TRIALS = 10
        cls.OPTUNA_TIMEOUT = 600
        
        print(f"Quick mode activated:")
        print(f"  Sample size: {cls.QUICK_SAMPLE_SIZE}")
        print(f"  Feature count: {cls.QUICK_FEATURE_COUNT}")
        print(f"  Estimators: {cls.QUICK_N_ESTIMATORS}")
    
    @classmethod
    def get_model_params(cls, model_name):
        """Return model-specific parameters"""
        params_map = {
            'lightgbm': cls.LGBM_PARAMS,
            'xgboost': cls.XGB_PARAMS,
            'catboost': cls.CAT_PARAMS,
            'random_forest': cls.RF_PARAMS,
            'extra_trees': cls.ET_PARAMS,
            'quick_lightgbm': cls.QUICK_LGBM_PARAMS
        }
        return params_map.get(model_name, {}).copy()
    
    @classmethod
    def get_tuning_space(cls, model_name):
        """Return hyperparameter tuning space for each model"""
        tuning_spaces = {
            'lightgbm': {
                'num_leaves': (35, 80),
                'learning_rate': (0.03, 0.08),
                'feature_fraction': (0.75, 0.95),
                'bagging_fraction': (0.75, 0.95),
                'min_child_samples': (10, 30),
                'reg_alpha': (0.05, 0.25),
                'reg_lambda': (0.05, 0.25),
                'max_depth': (6, 9),
                'n_estimators': (450, 750),
                'subsample': (0.8, 0.95),
                'colsample_bytree': (0.8, 0.95)
            },
            'xgboost': {
                'learning_rate': (0.03, 0.07),
                'max_depth': (6, 9),
                'subsample': (0.8, 0.95),
                'colsample_bytree': (0.75, 0.95),
                'reg_alpha': (0.05, 0.2),
                'reg_lambda': (0.05, 0.25),
                'gamma': (0.05, 0.15),
                'min_child_weight': (0.5, 2.0),
                'n_estimators': (500, 800),
                'scale_pos_weight': (1.0, 1.5)
            },
            'catboost': {
                'iterations': (400, 700),
                'learning_rate': (0.03, 0.08),
                'depth': (6, 9),
                'l2_leaf_reg': (1.5, 4.0),
                'border_count': (100, 200),
                'bagging_temperature': (0.5, 1.2)
            }
        }
        return tuning_spaces.get(model_name, {})
    
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
        if cls.QUICK_MODE:
            cls.N_JOBS = 1
            cls.CHUNK_SIZE = cls.QUICK_SAMPLE_SIZE
            return
        
        if available_memory_gb >= 32:
            cls.N_JOBS = min(cpu_cores, 8)
            cls.CHUNK_SIZE = 15000
            cls.OPTUNA_TRIALS = 40
            cls.OPTUNA_TIMEOUT = 3000
        elif available_memory_gb >= 16:
            cls.N_JOBS = min(cpu_cores, 6)
            cls.CHUNK_SIZE = 10000
            cls.OPTUNA_TRIALS = 35
            cls.OPTUNA_TIMEOUT = 2400
        else:
            cls.N_JOBS = min(cpu_cores, 4)
            cls.CHUNK_SIZE = 5000
            cls.OPTUNA_TRIALS = 25
            cls.OPTUNA_TIMEOUT = 1800
        
        # Update model parameters
        for params in [cls.LGBM_PARAMS, cls.XGB_PARAMS, cls.CAT_PARAMS, 
                      cls.RF_PARAMS, cls.ET_PARAMS]:
            if 'n_jobs' in params:
                params['n_jobs'] = cls.N_JOBS
            if 'thread_count' in params:
                params['thread_count'] = cls.N_JOBS