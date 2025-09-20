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
    FEATURE_SELECTOR_FILE = MODEL_DIR / "feature_selector.pkl"
    CV_RESULTS_FILE = MODEL_DIR / "cv_results.csv"
    FEATURE_IMPORTANCE_FILE = MODEL_DIR / "feature_importance.csv"
    
    # 데이터 설정
    FEATURE_COLUMNS = [f'X_{i:02d}' for i in range(1, 53)]
    TARGET_COLUMN = 'target'
    ID_COLUMN = 'ID'
    
    # 모델 설정
    N_CLASSES = 21
    RANDOM_STATE = 42
    N_JOBS = 4
    
    # 교차 검증 설정
    CV_FOLDS = 5
    VALIDATION_SIZE = 0.20
    
    # 피처 선택 설정
    FEATURE_SELECTION_K = 35
    
    # 센서 그룹 정의
    SENSOR_GROUPS = {
        'vibration': ['X_01', 'X_02', 'X_03', 'X_04', 'X_05', 'X_06', 'X_07', 'X_08'],
        'temperature': ['X_09', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14'],
        'pressure': ['X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_20', 'X_21'],
        'flow': ['X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27'],
        'power': ['X_28', 'X_29', 'X_30', 'X_31', 'X_32', 'X_33'],
        'position': ['X_34', 'X_35', 'X_36', 'X_37', 'X_38'],
        'speed': ['X_39', 'X_40', 'X_41', 'X_42', 'X_43'],
        'acoustic': ['X_44', 'X_45', 'X_46', 'X_47'],
        'electrical': ['X_48', 'X_49', 'X_50', 'X_51', 'X_52']
    }
    
    # 피처 엔지니어링 설정
    STATISTICAL_FEATURES = True
    
    # 클래스 가중치 설정
    CLASS_WEIGHTS = 'balanced'
    
    # LightGBM 파라미터
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.08,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'min_child_samples': 10,
        'min_child_weight': 0.001,
        'min_split_gain': 0.001,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'max_depth': 10,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 500,
        'n_jobs': N_JOBS,
        'is_unbalance': True
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
        'tree_method': 'hist'
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
        'class_weight': CLASS_WEIGHTS,
        'bootstrap': True
    }
    
    # Gradient Boosting 파라미터
    GB_PARAMS = {
        'n_estimators': 300,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'subsample': 0.8,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE
    }
    
    # 하이퍼파라미터 튜닝 설정
    OPTUNA_TRIALS = 15
    OPTUNA_TIMEOUT = 1800
    
    # 앙상블 설정
    MIN_CV_SCORE = 0.70
    
    # 로깅 설정
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 실행 모드별 설정
    CORE_MODELS = ['lightgbm', 'xgboost', 'random_forest', 'gradient_boosting']
    
    # 메모리 설정
    MEMORY_EFFICIENT = True
    DTYPE_OPTIMIZATION = True
    
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
            'gradient_boosting': cls.GB_PARAMS
        }
        return params_map.get(model_name, {}).copy()
    
    @classmethod
    def get_sensor_group_features(cls, group_name):
        """센서 그룹별 피처 반환"""
        return cls.SENSOR_GROUPS.get(group_name, [])
    
    @classmethod
    def get_all_sensor_groups(cls):
        """모든 센서 그룹 반환"""
        return list(cls.SENSOR_GROUPS.keys())
    
    @classmethod
    def adjust_for_data_size(cls, n_samples, n_features):
        """데이터 크기에 따른 파라미터 조정"""
        adjustments = {}
        
        if n_samples > 50000:
            adjustments.update({
                'cv_folds': 3,
                'n_estimators': 300,
                'optuna_trials': 10
            })
        elif n_samples < 10000:
            adjustments.update({
                'cv_folds': 7,
                'n_estimators': 700,
                'optuna_trials': 20
            })
        
        if n_features > 100:
            adjustments.update({
                'feature_selection_k': min(50, n_features // 2),
                'max_depth': 5
            })
        elif n_features < 30:
            adjustments.update({
                'feature_selection_k': n_features,
                'max_depth': 8
            })
        
        return adjustments