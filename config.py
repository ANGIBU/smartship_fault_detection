# config.py

import os
from pathlib import Path

class Config:
    # 경로 설정
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    
    # 파일 경로
    TRAIN_FILE = DATA_DIR / "train.csv"
    TEST_FILE = DATA_DIR / "test.csv"
    SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"
    
    # 출력 파일
    RESULT_FILE = BASE_DIR / "submission.csv"
    MODEL_FILE = MODEL_DIR / "best_model.pkl"
    SCALER_FILE = MODEL_DIR / "scaler.pkl"
    FEATURE_SELECTOR_FILE = MODEL_DIR / "feature_selector.pkl"
    PCA_FILE = MODEL_DIR / "pca.pkl"
    CV_RESULTS_FILE = MODEL_DIR / "cv_results.csv"
    FEATURE_IMPORTANCE_FILE = MODEL_DIR / "feature_importance.csv"
    
    # 데이터 설정
    FEATURE_COLUMNS = [f'X_{i:02d}' for i in range(1, 53)]
    TARGET_COLUMN = 'target'
    ID_COLUMN = 'ID'
    
    # 모델 설정
    N_CLASSES = 21
    RANDOM_STATE = 42
    N_JOBS = -1
    
    # 교차 검증 설정
    CV_FOLDS = 5
    VALIDATION_SIZE = 0.15
    
    # 피처 선택 설정
    FEATURE_SELECTION_K = 45
    PCA_COMPONENTS = 0.98
    
    # LightGBM 파라미터
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
        'min_child_samples': 50,
        'min_child_weight': 0.001,
        'min_split_gain': 0.02,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'max_depth': 6,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 500,
        'n_jobs': N_JOBS,
        'early_stopping_rounds': 50,
        'class_weight': 'balanced',
        'is_unbalance': True,
        'force_col_wise': True,
        'max_bin': 255,
        'min_data_in_leaf': 20,
    }
    
    # XGBoost 파라미터
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': N_CLASSES,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'gamma': 0.1,
        'min_child_weight': 10,
        'random_state': RANDOM_STATE,
        'n_estimators': 500,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
        'grow_policy': 'depthwise',
        'max_leaves': 0,
    }
    
    # Random Forest 파라미터
    RF_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'max_samples': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced',
        'bootstrap': True,
        'oob_score': True,
    }
    
    # Extra Trees 파라미터
    ET_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced',
        'bootstrap': False,
    }
    
    # CatBoost 파라미터
    CATBOOST_PARAMS = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1,
        'sampling_frequency': 'PerTree',
        'colsample_bylevel': 0.8,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'thread_count': N_JOBS if N_JOBS > 0 else None,
        'task_type': 'CPU',
        'early_stopping_rounds': 50,
        'use_best_model': True,
        'eval_metric': 'MultiClass',
    }
    
    # 하이퍼파라미터 튜닝 설정
    OPTUNA_TRIALS = 100
    OPTUNA_TIMEOUT = 1800
    
    # 앙상블 설정
    ENSEMBLE_METHODS = ['voting', 'stacking']
    STACKING_CV = 3
    
    # 신경망 파라미터
    NN_PARAMS = {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.01,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 300,
        'random_state': RANDOM_STATE,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20,
        'tol': 1e-4,
    }
    
    # SVM 파라미터
    SVM_PARAMS = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'probability': True,
        'cache_size': 2000,
        'max_iter': 1000,
        'tol': 1e-3,
    }
    
    # 로깅 설정
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 실행 모드별 설정
    FAST_MODE_MODELS = ['lightgbm', 'random_forest']
    FULL_MODE_MODELS = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'linear']
    EXTENDED_MODE_MODELS = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'linear', 'neural_network', 'additional']
    
    # 성능 임계값 설정
    MIN_CV_SCORE = 0.65
    ENSEMBLE_THRESHOLD = 0.70
    
    # 메모리 최적화 설정
    MEMORY_EFFICIENT = True
    CHUNK_SIZE = 1000
    
    @classmethod
    def create_directories(cls):
        """디렉터리 생성"""
        for directory in [cls.MODEL_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_params(cls, model_name):
        """모델별 파라미터 반환"""
        params_map = {
            'lightgbm': cls.LGBM_PARAMS,
            'xgboost': cls.XGB_PARAMS,
            'random_forest': cls.RF_PARAMS,
            'extra_trees': cls.ET_PARAMS,
            'catboost': cls.CATBOOST_PARAMS,
            'neural_network': cls.NN_PARAMS,
            'svm': cls.SVM_PARAMS
        }
        return params_map.get(model_name, {}).copy()
    
    @classmethod
    def update_params(cls, model_name, new_params):
        """모델 파라미터 수정"""
        if model_name == 'lightgbm':
            cls.LGBM_PARAMS.update(new_params)
        elif model_name == 'xgboost':
            cls.XGB_PARAMS.update(new_params)
        elif model_name == 'random_forest':
            cls.RF_PARAMS.update(new_params)
        elif model_name == 'extra_trees':
            cls.ET_PARAMS.update(new_params)
        elif model_name == 'catboost':
            cls.CATBOOST_PARAMS.update(new_params)
        elif model_name == 'neural_network':
            cls.NN_PARAMS.update(new_params)
        elif model_name == 'svm':
            cls.SVM_PARAMS.update(new_params)
    
    @classmethod
    def get_fast_mode_config(cls):
        """빠른 실행 모드 설정 반환"""
        fast_config = {
            'models': cls.FAST_MODE_MODELS,
            'optuna_trials': 30,
            'cv_folds': 3,
            'feature_selection_k': 30,
            'n_estimators': 100,
            'validation_size': 0.1,
            'timeout': 600
        }
        return fast_config
    
    @classmethod
    def get_full_mode_config(cls):
        """전체 실행 모드 설정 반환"""
        full_config = {
            'models': cls.FULL_MODE_MODELS,
            'optuna_trials': cls.OPTUNA_TRIALS,
            'cv_folds': cls.CV_FOLDS,
            'feature_selection_k': cls.FEATURE_SELECTION_K,
            'n_estimators': 500,
            'validation_size': cls.VALIDATION_SIZE,
            'timeout': cls.OPTUNA_TIMEOUT
        }
        return full_config
    
    @classmethod
    def get_extended_mode_config(cls):
        """확장 모드 설정 반환"""
        extended_config = {
            'models': cls.EXTENDED_MODE_MODELS,
            'optuna_trials': cls.OPTUNA_TRIALS,
            'cv_folds': cls.CV_FOLDS,
            'feature_selection_k': cls.FEATURE_SELECTION_K,
            'n_estimators': 500,
            'validation_size': cls.VALIDATION_SIZE,
            'timeout': cls.OPTUNA_TIMEOUT,
            'use_calibration': True,
            'use_stacking': True
        }
        return extended_config
    
    @classmethod
    def get_performance_config(cls):
        """성능 중심 설정 반환"""
        performance_config = {
            'models': ['lightgbm', 'xgboost', 'catboost'],
            'optuna_trials': 150,
            'cv_folds': 7,
            'feature_selection_k': 40,
            'n_estimators': 800,
            'validation_size': 0.1,
            'timeout': 2400,
            'use_pca': False,
            'scaling_method': 'robust'
        }
        return performance_config
    
    @classmethod
    def adjust_for_data_size(cls, n_samples, n_features):
        """데이터 크기에 따른 파라미터 조정"""
        adjustments = {}
        
        # 샘플 수에 따른 조정
        if n_samples > 50000:
            adjustments.update({
                'cv_folds': 3,
                'n_estimators': 300,
                'optuna_trials': 50,
                'timeout': 1200
            })
        elif n_samples < 10000:
            adjustments.update({
                'cv_folds': 7,
                'n_estimators': 800,
                'optuna_trials': 200,
                'timeout': 3600
            })
        
        # 피처 수에 따른 조정
        if n_features > 100:
            adjustments.update({
                'feature_selection_k': min(80, n_features // 2),
                'max_depth': 5
            })
        elif n_features < 30:
            adjustments.update({
                'feature_selection_k': n_features,
                'max_depth': 8
            })
        
        return adjustments
    
    @classmethod
    def get_memory_config(cls):
        """메모리 최적화 설정"""
        return {
            'memory_efficient': cls.MEMORY_EFFICIENT,
            'chunk_size': cls.CHUNK_SIZE,
            'dtype_optimization': True,
            'garbage_collection': True
        }