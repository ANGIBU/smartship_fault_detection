# config.py

import os
from pathlib import Path

class Config:
    # 경로 설정
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    MODEL_DIR = BASE_DIR / "models"
    
    # 파일 경로
    TRAIN_FILE = DATA_DIR / "train.csv"
    TEST_FILE = DATA_DIR / "test.csv"
    SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"
    
    # 출력 파일
    RESULT_FILE = OUTPUT_DIR / "submission.csv"
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
        'n_estimators': 1000,
        'early_stopping_rounds': 100
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
        'n_estimators': 1000,
        'early_stopping_rounds': 100
    }
    
    RF_PARAMS = {
        'n_estimators': 500,
        'max_depth': 12,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉터리 생성"""
        for directory in [cls.OUTPUT_DIR, cls.MODEL_DIR]:
            directory.mkdir(parents=True, exist_ok=True)