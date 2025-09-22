# config.py

import os
import numpy as np
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
    SELECTOR_FILE = MODEL_DIR / "selector.pkl"
    CV_RESULTS_FILE = MODEL_DIR / "cv_results.csv"
    
    # 데이터 설정
    FEATURE_COLUMNS = [f'X_{i:02d}' for i in range(1, 53)]
    TARGET_COLUMN = 'target'
    ID_COLUMN = 'ID'
    
    # 모델 설정
    N_CLASSES = 21
    RANDOM_STATE = 42
    N_JOBS = 6
    
    # 교차 검증 설정
    CV_FOLDS = 7
    VALIDATION_SIZE = 0.2
    
    # 피처 선택 설정
    FEATURE_SELECTION_METHODS = ['mutual_info', 'f_classif', 'chi2']
    TARGET_FEATURES = 35
    
    # 클래스 가중치 설정
    USE_CLASS_WEIGHTS = True
    FOCAL_LOSS_ALPHA = 1.0
    FOCAL_LOSS_GAMMA = 2.0
    
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
    
    # XGBoost 파라미터
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
    
    # CatBoost 파라미터
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
    
    # Random Forest 파라미터
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
    
    # Extra Trees 파라미터
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
    
    # 하이퍼파라미터 튜닝 설정
    OPTUNA_TRIALS = 100
    OPTUNA_TIMEOUT = 3600
    
    # 앙상블 설정
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.3,
        'xgboost': 0.25,
        'catboost': 0.25,
        'random_forest': 0.1,
        'extra_trees': 0.1
    }
    
    # 로깅 설정
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 메모리 설정
    MEMORY_EFFICIENT = True
    CHUNK_SIZE = 10000
    
    # 검증 전략 설정
    VALIDATION_STRATEGY = 'stratified'
    
    # 클래스별 성능 임계값
    CLASS_PERFORMANCE_THRESHOLD = 0.60
    
    # 스케일링 방법
    SCALING_METHOD = 'robust'
    
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
            'catboost': cls.CAT_PARAMS,
            'random_forest': cls.RF_PARAMS,
            'extra_trees': cls.ET_PARAMS
        }
        return params_map.get(model_name, {}).copy()
    
    @classmethod
    def validate_config(cls):
        """설정값 검증"""
        errors = []
        
        if not cls.DATA_DIR.exists():
            errors.append(f"데이터 디렉터리가 없습니다: {cls.DATA_DIR}")
        
        required_files = [cls.TRAIN_FILE, cls.TEST_FILE]
        for file_path in required_files:
            if not file_path.exists():
                errors.append(f"필수 파일이 없습니다: {file_path}")
        
        if cls.N_CLASSES <= 0:
            errors.append("N_CLASSES는 양수여야 합니다")
        
        if cls.CV_FOLDS < 2:
            errors.append("CV_FOLDS는 2 이상이어야 합니다")
        
        if cls.TARGET_FEATURES <= 0:
            errors.append("TARGET_FEATURES는 양수여야 합니다")
        
        return errors
    
    @classmethod
    def update_for_hardware(cls, available_memory_gb, cpu_cores):
        """하드웨어 사양에 맞춰 설정 조정"""
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
        
        # 모델 파라미터 업데이트
        for params in [cls.LGBM_PARAMS, cls.XGB_PARAMS, cls.CAT_PARAMS, 
                      cls.RF_PARAMS, cls.ET_PARAMS]:
            if 'n_jobs' in params:
                params['n_jobs'] = cls.N_JOBS
            if 'thread_count' in params:
                params['thread_count'] = cls.N_JOBS