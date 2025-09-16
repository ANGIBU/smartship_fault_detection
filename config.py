# config.py

import os
from pathlib import Path

class Config:
    # 경로 설정
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    
    # 파일 경로
    TRAIN_FILE = DATA_DIR / "train.csv"
    TEST_FILE = DATA_DIR / "test.csv"
    SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"
    
    # 출력 파일
    RESULT_FILE = BASE_DIR / "submission.csv"
    MODEL_FILE = MODEL_DIR / "best_model.pkl"
    SCALER_FILE = MODEL_DIR / "scaler.pkl"
    
    # 데이터 설정
    FEATURE_COLUMNS = [f'X_{i:02d}' for i in range(1, 53)]
    TARGET_COLUMN = 'target'
    ID_COLUMN = 'ID'
    
    # 모델 설정
    N_CLASSES = 21
    RANDOM_STATE = 42
    
    # 교차 검증 설정
    CV_FOLDS = 5
    
    # 피처 엔지니어링 설정
    FEATURE_SELECTION_METHODS = ['mutual_info', 'f_classif', 'random_forest', 'recursive', 'lasso']
    SCALING_METHODS = ['standard', 'robust', 'minmax', 'quantile', 'power']
    PCA_COMPONENTS = [0.90, 0.95, 0.99]
    
    # 앙상블 설정
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.35,
        'xgboost': 0.35,
        'random_forest': 0.15,
        'extra_trees': 0.15
    }
    
    # 예측 설정
    TTA_AUGMENTATIONS = 10
    TTA_NOISE_STD = 0.005
    CONFIDENCE_THRESHOLD = 0.75
    
    # 성능 모니터링
    MEMORY_LIMIT_GB = 32
    TIME_LIMIT_HOURS = 6
    
    # 검증 설정
    VALIDATION_SIZE = 0.2
    TEST_SIZE = 0.1
    STRATIFY_SPLITS = True
    
    # 모델 파라미터
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 2000,
        'early_stopping_rounds': 150,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 20,
        'min_child_weight': 1e-3,
        'subsample_freq': 1,
        'colsample_bytree': 0.8
    }
    
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': N_CLASSES,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_estimators': 2000,
        'early_stopping_rounds': 150,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'gamma': 0,
        'min_child_weight': 1,
        'max_delta_step': 0,
        'scale_pos_weight': 1,
        'tree_method': 'hist'
    }
    
    RF_PARAMS = {
        'n_estimators': 800,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'bootstrap': True,
        'oob_score': True,
        'warm_start': False,
        'class_weight': 'balanced'
    }
    
    ET_PARAMS = {
        'n_estimators': 800,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'bootstrap': False,
        'class_weight': 'balanced'
    }
    
    # 최적화 파라미터
    OPTUNA_PARAMS = {
        'n_trials': 100,
        'timeout': 3600,
        'n_jobs': -1,
        'show_progress_bar': True
    }
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉터리 생성"""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)