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
    N_JOBS = 2
    
    # 교차 검증 설정
    CV_FOLDS = 5
    VALIDATION_SIZE = 0.20
    
    # 피처 선택 설정
    FEATURE_SELECTION_K = 40
    
    # 센서 그룹 정의 (도메인 지식 기반)
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
    
    # 피처 중요도 기반 그룹 우선순위
    SENSOR_PRIORITY = {
        'vibration': 1,   # 진동은 이상 감지에 가장 중요
        'power': 2,       # 전력 이상은 설비 문제의 주요 지표
        'temperature': 3,  # 온도 변화는 시스템 상태 반영
        'pressure': 4,     # 압력 변화는 공정 이상 감지
        'flow': 5,        # 유량은 공정 효율성 지표
        'speed': 6,       # 속도는 동작 상태 지표
        'acoustic': 7,    # 음향은 기계적 이상 감지
        'electrical': 8,  # 전기적 특성
        'position': 9     # 위치는 보조 지표
    }
    
    # 피처 엔지니어링 설정
    STATISTICAL_FEATURES = True
    SIGNAL_FEATURES = True
    INTERACTION_FEATURES = True
    
    # 클래스 가중치 설정
    CLASS_WEIGHTS = 'balanced'
    
    # LightGBM 파라미터 (안정성 중심)
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.01,
        'min_split_gain': 0.01,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'max_depth': 6,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 300,
        'n_jobs': N_JOBS,
        'is_unbalance': True,
        'force_col_wise': True
    }
    
    # XGBoost 파라미터 (안정성 중심)
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': N_CLASSES,
        'learning_rate': 0.03,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'gamma': 0.1,
        'min_child_weight': 15,
        'random_state': RANDOM_STATE,
        'n_estimators': 200,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    # Random Forest 파라미터 (안정성 중심)
    RF_PARAMS = {
        'n_estimators': 150,
        'max_depth': 8,
        'min_samples_split': 15,
        'min_samples_leaf': 8,
        'max_features': 0.7,
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': CLASS_WEIGHTS,
        'bootstrap': True,
        'max_samples': 0.8
    }
    
    # Gradient Boosting 파라미터 (안정성 중심)
    GB_PARAMS = {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 15,
        'min_samples_leaf': 8,
        'subsample': 0.7,
        'max_features': 0.7,
        'random_state': RANDOM_STATE
    }
    
    # Extra Trees 파라미터 (추가 모델)
    ET_PARAMS = {
        'n_estimators': 150,
        'max_depth': 8,
        'min_samples_split': 15,
        'min_samples_leaf': 8,
        'max_features': 0.7,
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': CLASS_WEIGHTS,
        'bootstrap': False
    }
    
    # 하이퍼파라미터 튜닝 설정
    OPTUNA_TRIALS = 10
    OPTUNA_TIMEOUT = 1200
    
    # 앙상블 설정
    MIN_CV_SCORE = 0.55
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.3,
        'xgboost': 0.3,
        'random_forest': 0.2,
        'gradient_boosting': 0.2
    }
    
    # 로깅 설정
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 실행 모드별 설정
    CORE_MODELS = ['lightgbm', 'xgboost', 'random_forest', 'gradient_boosting', 'extra_trees']
    
    # 메모리 설정
    MEMORY_EFFICIENT = True
    DTYPE_OPTIMIZATION = True
    CHUNK_SIZE = 5000
    
    # 검증 전략 설정
    VALIDATION_STRATEGIES = {
        'time_based': {
            'train_ratio': 0.6,
            'val_ratio': 0.2,
            'holdout_ratio': 0.2
        },
        'stratified': {
            'test_size': 0.25,
            'random_state': RANDOM_STATE
        },
        'time_series_cv': {
            'n_splits': 5,
            'test_size': 0.2
        }
    }
    
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
            'gradient_boosting': cls.GB_PARAMS,
            'extra_trees': cls.ET_PARAMS
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
    def get_priority_sensors(cls, top_n=3):
        """우선순위 상위 센서 그룹 반환"""
        sorted_groups = sorted(cls.SENSOR_PRIORITY.items(), key=lambda x: x[1])
        return [group for group, _ in sorted_groups[:top_n]]
    
    @classmethod
    def adjust_for_data_size(cls, n_samples, n_features):
        """데이터 크기에 따른 파라미터 조정"""
        adjustments = {}
        
        # 메모리 사용량 기반 조정
        if n_samples > 50000:
            adjustments.update({
                'cv_folds': 3,
                'n_estimators': 100,
                'optuna_trials': 6,
                'n_jobs': 1
            })
        elif n_samples < 10000:
            adjustments.update({
                'cv_folds': 7,
                'n_estimators': 250,
                'optuna_trials': 12
            })
        
        if n_features > 100:
            adjustments.update({
                'feature_selection_k': min(50, n_features // 2),
                'max_depth': 4
            })
        elif n_features < 30:
            adjustments.update({
                'feature_selection_k': n_features,
                'max_depth': 8
            })
        
        # 시스템 리소스 기반 조정
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            if available_memory_gb < 8:
                adjustments.update({
                    'n_jobs': 1,
                    'chunk_size': 2000,
                    'n_estimators': 80
                })
            elif available_memory_gb > 32:
                adjustments.update({
                    'n_jobs': min(4, cls.N_JOBS * 2),
                    'chunk_size': 10000
                })
        except ImportError:
            pass
        
        return adjustments
    
    @classmethod
    def get_stable_params(cls):
        """안정적인 파라미터 반환"""
        return {
            'lgbm': {
                **cls.LGBM_PARAMS,
                'n_estimators': 200,
                'num_leaves': 31,
                'learning_rate': 0.03,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.6,
                'reg_alpha': 0.2,
                'reg_lambda': 0.2
            },
            'xgb': {
                **cls.XGB_PARAMS,
                'n_estimators': 150,
                'max_depth': 3,
                'learning_rate': 0.02,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3
            },
            'rf': {
                **cls.RF_PARAMS,
                'n_estimators': 100,
                'max_depth': 6,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 0.6
            },
            'gb': {
                **cls.GB_PARAMS,
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.03,
                'subsample': 0.6,
                'max_features': 0.6
            }
        }
    
    @classmethod
    def validate_config(cls):
        """설정값 검증"""
        errors = []
        
        # 필수 디렉터리 확인
        if not cls.DATA_DIR.exists():
            errors.append(f"데이터 디렉터리가 없습니다: {cls.DATA_DIR}")
        
        # 필수 파일 확인
        required_files = [cls.TRAIN_FILE, cls.TEST_FILE]
        for file_path in required_files:
            if not file_path.exists():
                errors.append(f"필수 파일이 없습니다: {file_path}")
        
        # 파라미터 검증
        if cls.N_CLASSES <= 0:
            errors.append("N_CLASSES는 양수여야 합니다")
        
        if cls.CV_FOLDS < 2:
            errors.append("CV_FOLDS는 2 이상이어야 합니다")
        
        if cls.FEATURE_SELECTION_K <= 0:
            errors.append("FEATURE_SELECTION_K는 양수여야 합니다")
        
        # 센서 그룹 검증
        all_sensors = []
        for sensors in cls.SENSOR_GROUPS.values():
            all_sensors.extend(sensors)
        
        if len(set(all_sensors)) != len(all_sensors):
            errors.append("센서 그룹에 중복된 센서가 있습니다")
        
        expected_sensors = set(cls.FEATURE_COLUMNS)
        actual_sensors = set(all_sensors)
        
        if expected_sensors != actual_sensors:
            missing = expected_sensors - actual_sensors
            extra = actual_sensors - expected_sensors
            if missing:
                errors.append(f"누락된 센서: {missing}")
            if extra:
                errors.append(f"추가된 센서: {extra}")
        
        return errors
    
    @classmethod
    def get_cross_validation_strategy(cls, strategy_name='time_based'):
        """교차 검증 전략 반환"""
        return cls.VALIDATION_STRATEGIES.get(strategy_name, cls.VALIDATION_STRATEGIES['time_based'])
    
    @classmethod
    def get_ensemble_config(cls):
        """앙상블 설정 반환"""
        return {
            'models': cls.CORE_MODELS,
            'weights': cls.ENSEMBLE_WEIGHTS,
            'min_score': cls.MIN_CV_SCORE,
            'voting': 'soft'
        }