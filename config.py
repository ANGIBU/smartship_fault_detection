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
    VALIDATION_SIZE = 0.15
    
    # Feature selection settings
    FEATURE_SELECTION_METHODS = ['mutual_info', 'f_classif', 'recursive']
    TARGET_FEATURES = 38
    
    # Class balance settings
    USE_CLASS_BALANCED_LOSS = True
    EFFECTIVE_NUMBER_BETA = 0.9999
    FOCAL_LOSS_ALPHA = 2.0
    FOCAL_LOSS_GAMMA = 3.0
    ISOLATION_FOREST_CONTAMINATION = 0.1
    
    # Temperature scaling settings
    USE_TEMPERATURE_SCALING = True
    TEMPERATURE_INIT = 1.5
    CALIBRATION_EPOCHS = 50
    
    # Quick mode settings
    QUICK_MODE = False
    QUICK_SAMPLE_SIZE = 1000
    QUICK_FEATURE_COUNT = 15
    QUICK_N_ESTIMATORS = 50
    
    # Memory optimization settings
    MEMORY_EFFICIENT = True
    CHUNK_SIZE = 5000
    USE_STREAMING = True
    DTYPE_OPTIMIZATION = True
    
    # LightGBM parameters
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 48,
        'learning_rate': 0.035,
        'feature_fraction': 0.82,
        'bagging_fraction': 0.85,
        'bagging_freq': 3,
        'min_child_samples': 15,
        'min_child_weight': 0.001,
        'min_split_gain': 0.01,
        'reg_alpha': 0.08,
        'reg_lambda': 0.15,
        'max_depth': 7,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 1200,
        'n_jobs': N_JOBS,
        'subsample': 0.88,
        'colsample_bytree': 0.82,
        'min_data_in_leaf': 12,
        'lambda_l1': 0.05,
        'lambda_l2': 0.12,
        'extra_trees': False,
        'max_bin': 255,
        'is_unbalance': True
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
        'is_unbalance': True
    }
    
    # XGBoost parameters
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': N_CLASSES,
        'learning_rate': 0.032,
        'max_depth': 6,
        'subsample': 0.88,
        'colsample_bytree': 0.82,
        'reg_alpha': 0.05,
        'reg_lambda': 0.12,
        'gamma': 0.05,
        'min_child_weight': 1.2,
        'random_state': RANDOM_STATE,
        'n_estimators': 1200,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
        'verbosity': 0,
        'eval_metric': 'mlogloss',
        'scale_pos_weight': 1.0,
        'max_delta_step': 2,
        'colsample_bylevel': 0.85,
        'colsample_bynode': 0.85,
        'grow_policy': 'depthwise'
    }
    
    # CatBoost parameters
    CAT_PARAMS = {
        'iterations': 1200,
        'learning_rate': 0.032,
        'depth': 6,
        'l2_leaf_reg': 4.0,
        'border_count': 200,
        'thread_count': N_JOBS,
        'random_state': RANDOM_STATE,
        'verbose': False,
        'loss_function': 'MultiClass',
        'classes_count': N_CLASSES,
        'auto_class_weights': 'Balanced',
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 0.8,
        'rsm': 0.82,
        'random_strength': 0.8,
        'leaf_estimation_iterations': 8,
        'od_type': 'Iter',
        'od_wait': 100
    }
    
    # Random Forest parameters
    RF_PARAMS = {
        'n_estimators': 600,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced_subsample',
        'bootstrap': True,
        'max_samples': 0.88,
        'min_weight_fraction_leaf': 0.0,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'criterion': 'gini',
        'oob_score': True
    }
    
    # Extra Trees parameters
    ET_PARAMS = {
        'n_estimators': 500,
        'max_depth': 12,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced_subsample',
        'bootstrap': False,
        'criterion': 'gini',
        'min_weight_fraction_leaf': 0.0,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0
    }
    
    # Isolation Forest parameters for each class
    ISOLATION_FOREST_PARAMS = {
        'n_estimators': 200,
        'contamination': ISOLATION_FOREST_CONTAMINATION,
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'max_samples': 'auto',
        'max_features': 1.0,
        'bootstrap': False,
        'warm_start': False
    }
    
    # Hyperparameter tuning settings
    OPTUNA_TRIALS = 50
    OPTUNA_TIMEOUT = 3600
    OPTUNA_CV_FOLDS = 5
    
    # Ensemble settings
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.32,
        'xgboost': 0.35,
        'catboost': 0.28,
        'random_forest': 0.05
    }
    
    # Selective ensemble settings
    USE_SELECTIVE_ENSEMBLE = True
    ENSEMBLE_DIVERSITY_THRESHOLD = 0.15
    ENSEMBLE_ACCURACY_THRESHOLD = 0.75
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Scaling method
    SCALING_METHOD = 'sensor_specific'
    
    # Sensor-specific settings
    SENSOR_TYPES = {
        'temperature': ['X_01', 'X_02', 'X_03', 'X_04', 'X_05'],
        'pressure': ['X_06', 'X_07', 'X_08', 'X_09', 'X_10'],
        'vibration': ['X_11', 'X_12', 'X_13', 'X_14', 'X_15'],
        'flow': ['X_16', 'X_17', 'X_18', 'X_19', 'X_20'],
        'electrical': ['X_21', 'X_22', 'X_23', 'X_24', 'X_25'],
        'mechanical': ['X_26', 'X_27', 'X_28', 'X_29', 'X_30'],
        'chemical': ['X_31', 'X_32', 'X_33', 'X_34', 'X_35'],
        'thermal': ['X_36', 'X_37', 'X_38', 'X_39', 'X_40'],
        'optical': ['X_41', 'X_42', 'X_43', 'X_44', 'X_45'],
        'acoustic': ['X_46', 'X_47', 'X_48', 'X_49', 'X_50'],
        'other': ['X_51', 'X_52']
    }
    
    # Feature engineering settings
    CREATE_INTERACTION_FEATURES = True
    CREATE_POLYNOMIAL_FEATURES = False
    POLYNOMIAL_DEGREE = 2
    INTERACTION_TOP_N = 8
    
    # Statistical feature settings
    STATISTICAL_FEATURES = [
        'mean', 'std', 'median', 'min', 'max', 'range',
        'skew', 'kurtosis', 'q25', 'q75', 'iqr', 'rms', 'crest_factor'
    ]
    
    # Domain-specific feature settings
    DOMAIN_FEATURES_ENABLED = True
    TIME_SERIES_FEATURES_ENABLED = False
    PCA_COMPONENTS = 6
    
    # Class balancing methods
    CLASS_BALANCING_METHODS = ['class_balanced_loss', 'isolation_forest', 'focal_loss']
    DEFAULT_BALANCING_METHOD = 'class_balanced_loss'
    
    # Neural network parameters
    NN_PARAMS = {
        'hidden_layer_sizes': (256, 128, 64),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 'auto',
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 500,
        'random_state': RANDOM_STATE,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-08,
        'n_iter_no_change': 15
    }
    
    # Gradient Boosting parameters
    GB_PARAMS = {
        'n_estimators': 400,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.85,
        'random_state': RANDOM_STATE,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'min_weight_fraction_leaf': 0.0,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'validation_fraction': 0.1,
        'n_iter_no_change': 12,
        'tol': 1e-4
    }
    
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
        cls.USE_CLASS_BALANCED_LOSS = False
        cls.N_JOBS = 1
        cls.OPTUNA_TRIALS = 10
        cls.OPTUNA_TIMEOUT = 600
        cls.DOMAIN_FEATURES_ENABLED = False
        cls.TIME_SERIES_FEATURES_ENABLED = False
        cls.PCA_COMPONENTS = 3
        
        print(f"Quick mode activated:")
        print(f"  Sample size: {cls.QUICK_SAMPLE_SIZE}")
        print(f"  Feature count: {cls.QUICK_FEATURE_COUNT}")
        print(f"  Estimators: {cls.QUICK_N_ESTIMATORS}")
    
    @classmethod
    def get_effective_number_weights(cls, samples_per_class):
        """Calculate effective number based class weights"""
        beta = cls.EFFECTIVE_NUMBER_BETA
        effective_nums = [(1 - np.power(beta, n)) / (1 - beta) for n in samples_per_class]
        weights = [(1 - beta) / en for en in effective_nums]
        normalized_weights = np.array(weights) / np.sum(weights) * len(weights)
        return normalized_weights
    
    @classmethod
    def get_sensor_specific_config(cls, sensor_name):
        """Get sensor-specific configuration"""
        for sensor_type, sensors in cls.SENSOR_TYPES.items():
            if sensor_name in sensors:
                if sensor_type == 'temperature':
                    return {'scaler': 'robust', 'outlier_method': 'iqr'}
                elif sensor_type == 'vibration':
                    return {'scaler': 'standard', 'outlier_method': 'zscore'}
                elif sensor_type == 'pressure':
                    return {'scaler': 'minmax', 'outlier_method': 'isolation'}
                else:
                    return {'scaler': 'standard', 'outlier_method': 'iqr'}
        return {'scaler': 'standard', 'outlier_method': 'iqr'}
    
    @classmethod
    def get_model_params(cls, model_name):
        """Return model-specific parameters"""
        params_map = {
            'lightgbm': cls.LGBM_PARAMS,
            'xgboost': cls.XGB_PARAMS,
            'catboost': cls.CAT_PARAMS,
            'random_forest': cls.RF_PARAMS,
            'extra_trees': cls.ET_PARAMS,
            'quick_lightgbm': cls.QUICK_LGBM_PARAMS,
            'neural_network': cls.NN_PARAMS,
            'gradient_boosting': cls.GB_PARAMS,
            'isolation_forest': cls.ISOLATION_FOREST_PARAMS
        }
        return params_map.get(model_name, {}).copy()
    
    @classmethod
    def get_tuning_space(cls, model_name):
        """Return hyperparameter tuning space for each model"""
        tuning_spaces = {
            'lightgbm': {
                'num_leaves': (30, 65),
                'learning_rate': (0.025, 0.06),
                'feature_fraction': (0.75, 0.9),
                'bagging_fraction': (0.8, 0.95),
                'min_child_samples': (10, 30),
                'reg_alpha': (0.01, 0.2),
                'reg_lambda': (0.05, 0.25),
                'max_depth': (5, 8),
                'n_estimators': (800, 1500),
                'subsample': (0.8, 0.95),
                'colsample_bytree': (0.75, 0.9),
                'min_data_in_leaf': (5, 20)
            },
            'xgboost': {
                'learning_rate': (0.02, 0.06),
                'max_depth': (5, 8),
                'subsample': (0.8, 0.95),
                'colsample_bytree': (0.75, 0.9),
                'reg_alpha': (0.01, 0.15),
                'reg_lambda': (0.05, 0.2),
                'gamma': (0.01, 0.1),
                'min_child_weight': (0.8, 2.5),
                'n_estimators': (800, 1500),
                'max_delta_step': (1, 3),
                'colsample_bylevel': (0.75, 0.95),
                'colsample_bynode': (0.75, 0.95)
            },
            'catboost': {
                'iterations': (800, 1500),
                'learning_rate': (0.02, 0.06),
                'depth': (5, 8),
                'l2_leaf_reg': (2.0, 6.0),
                'border_count': (128, 255),
                'bagging_temperature': (0.5, 1.5),
                'rsm': (0.75, 0.95),
                'random_strength': (0.5, 1.2)
            },
            'random_forest': {
                'n_estimators': (400, 800),
                'max_depth': (10, 20),
                'min_samples_split': (3, 8),
                'min_samples_leaf': (1, 5),
                'max_samples': (0.8, 0.95)
            },
            'gradient_boosting': {
                'n_estimators': (250, 500),
                'learning_rate': (0.03, 0.08),
                'max_depth': (5, 9),
                'subsample': (0.8, 0.95),
                'min_samples_split': (3, 8),
                'min_samples_leaf': (2, 6)
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
            cls.N_JOBS = min(cpu_cores, 6)
            cls.CHUNK_SIZE = 8000  # Reduced for memory efficiency
            cls.OPTUNA_TRIALS = 60
            cls.OPTUNA_TIMEOUT = 4500
            cls.TARGET_FEATURES = 38
            cls.PCA_COMPONENTS = 6
        elif available_memory_gb >= 16:
            cls.N_JOBS = min(cpu_cores, 6)
            cls.CHUNK_SIZE = 6000
            cls.OPTUNA_TRIALS = 50
            cls.OPTUNA_TIMEOUT = 3600
            cls.TARGET_FEATURES = 35
            cls.PCA_COMPONENTS = 5
        else:
            cls.N_JOBS = min(cpu_cores, 4)
            cls.CHUNK_SIZE = 4000
            cls.OPTUNA_TRIALS = 30
            cls.OPTUNA_TIMEOUT = 2400
            cls.TARGET_FEATURES = 30
            cls.PCA_COMPONENTS = 4
        
        # Update model parameters
        for params in [cls.LGBM_PARAMS, cls.XGB_PARAMS, cls.CAT_PARAMS, 
                      cls.RF_PARAMS, cls.ET_PARAMS]:
            if 'n_jobs' in params:
                params['n_jobs'] = cls.N_JOBS
            if 'thread_count' in params:
                params['thread_count'] = cls.N_JOBS
    
    @classmethod
    def get_performance_targets(cls):
        """Return performance targets and thresholds"""
        return {
            'target_macro_f1': 0.84,
            'deployment_threshold': 0.80,
            'good_performance': 0.75,
            'acceptable_performance': 0.70,
            'stability_weight': 0.8,
            'confidence_threshold': 0.7
        }