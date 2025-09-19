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
    VALIDATION_SIZE = 0.2
    
    # 피처 선택 설정
    FEATURE_SELECTION_K = 60
    PCA_COMPONENTS = 0.95
    
    # 모델 파라미터
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 120,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 15,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 600,
        'n_jobs': N_JOBS,
        'early_stopping_rounds': 50,
        'class_weight': 'balanced',
        'is_unbalance': True
    }
    
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': N_CLASSES,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.1,
        'max_depth': 9,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'gamma': 0.2,
        'random_state': RANDOM_STATE,
        'n_estimators': 600,
        'n_jobs': N_JOBS,
        'tree_method': 'hist'
    }
    
    RF_PARAMS = {
        'n_estimators': 400,
        'max_depth': 25,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced'
    }
    
    ET_PARAMS = {
        'n_estimators': 400,
        'max_depth': 25,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced'
    }
    
    # CatBoost 파라미터 - subsample 제거
    CATBOOST_PARAMS = {
        'iterations': 600,
        'learning_rate': 0.1,
        'depth': 8,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1,
        'sampling_frequency': 'PerTree',
        'colsample_bylevel': 0.8,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'thread_count': N_JOBS if N_JOBS > 0 else None,
        'task_type': 'CPU'
    }
    
    # 하이퍼파라미터 튜닝 설정
    OPTUNA_TRIALS = 300
    OPTUNA_TIMEOUT = 1800
    
    # 앙상블 설정
    ENSEMBLE_METHODS = ['voting']
    STACKING_CV = 3
    
    # 로깅 설정
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 실행 모드 설정
    FAST_MODE_MODELS = ['lightgbm', 'xgboost']
    FULL_MODE_MODELS = ['lightgbm', 'xgboost', 'catboost', 'random_forest']
    
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
            'catboost': cls.CATBOOST_PARAMS
        }
        return params_map.get(model_name, {})
    
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
    
    @classmethod
    def get_fast_mode_config(cls):
        """빠른 실행 모드 설정 반환"""
        fast_config = {
            'models': cls.FAST_MODE_MODELS,
            'optuna_trials': 100,
            'cv_folds': 3,
            'feature_selection_k': 40,
            'n_estimators': 300
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
            'n_estimators': 600
        }
        return full_config