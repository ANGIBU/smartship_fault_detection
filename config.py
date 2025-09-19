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
    
    # 교차 검증 설정 - 정교화
    CV_FOLDS = 7  # 21개 클래스에서 안정적 검증을 위해 증가
    VALIDATION_SIZE = 0.15  # 검증 세트 크기 축소로 훈련 데이터 확보
    
    # 피처 선택 설정 - 조정
    FEATURE_SELECTION_K = 45  # 과도한 피처 방지
    PCA_COMPONENTS = 0.98  # 정보 손실 최소화
    
    # LightGBM 파라미터 - 성능 중심 조정
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 80,  # 과적합 방지를 위해 감소
        'learning_rate': 0.08,  # 안정적 학습을 위해 감소
        'feature_fraction': 0.7,  # 피처 샘플링 비율 조정
        'bagging_fraction': 0.75,
        'bagging_freq': 3,
        'min_child_samples': 25,  # 과적합 방지 증가
        'min_child_weight': 0.01,
        'reg_alpha': 0.5,  # 정규화 강화
        'reg_lambda': 0.5,
        'max_depth': 8,  # 깊이 제한
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 800,
        'n_jobs': N_JOBS,
        'early_stopping_rounds': 100,
        'class_weight': 'balanced',
        'is_unbalance': True,
        'force_col_wise': True,  # 메모리 효율성
    }
    
    # XGBoost 파라미터 - 정밀 조정
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': N_CLASSES,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.06,  # 안정적 학습
        'max_depth': 7,  # 깊이 제한으로 과적합 방지
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,  # 추가 정규화
        'reg_alpha': 1.0,  # L1 정규화 강화
        'reg_lambda': 1.0,  # L2 정규화 강화
        'gamma': 0.5,  # 최소 분할 손실
        'min_child_weight': 5,  # 과적합 방지
        'random_state': RANDOM_STATE,
        'n_estimators': 800,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
        'early_stopping_rounds': 100,
        'grow_policy': 'depthwise',  # 안정적 성장
        'max_leaves': 0,  # max_depth로 제어
    }
    
    # Random Forest 파라미터 - 균형 조정
    RF_PARAMS = {
        'n_estimators': 500,  # 트리 수 증가
        'max_depth': 20,  # 깊이 제한
        'min_samples_split': 5,  # 분할 최소 샘플 증가
        'min_samples_leaf': 2,  # 리프 최소 샘플 증가
        'max_features': 'sqrt',
        'max_samples': 0.8,  # 배깅 샘플링 비율
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced',
        'bootstrap': True,
        'oob_score': True,  # OOB 점수 계산
    }
    
    # Extra Trees 파라미터
    ET_PARAMS = {
        'n_estimators': 500,
        'max_depth': 18,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': 'balanced',
        'bootstrap': False,  # Extra Trees 특성
    }
    
    # CatBoost 파라미터 - 안정성 중심
    CATBOOST_PARAMS = {
        'iterations': 800,
        'learning_rate': 0.05,  # 안정적 학습률
        'depth': 7,  # 깊이 제한
        'l2_leaf_reg': 5,  # 정규화 강화
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 0.8,
        'sampling_frequency': 'PerTree',
        'colsample_bylevel': 0.8,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'thread_count': N_JOBS if N_JOBS > 0 else None,
        'task_type': 'CPU',
        'early_stopping_rounds': 100,
        'use_best_model': True,
        'eval_metric': 'MultiClass',
    }
    
    # 하이퍼파라미터 튜닝 설정 - 확장
    OPTUNA_TRIALS = 500  # 시도 횟수 증가
    OPTUNA_TIMEOUT = 3600  # 1시간 제한
    
    # 앙상블 설정
    ENSEMBLE_METHODS = ['voting', 'stacking']
    STACKING_CV = 5  # 스태킹 CV 폴드 증가
    
    # 신경망 파라미터
    NN_PARAMS = {
        'hidden_layer_sizes': (128, 64, 32),  # 더 깊은 네트워크
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,  # 정규화
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20,
        'tol': 1e-6,
    }
    
    # SVM 파라미터
    SVM_PARAMS = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'probability': True,
        'cache_size': 2000,  # 메모리 할당량
        'max_iter': 1000,
        'tol': 1e-4,
    }
    
    # 로깅 설정
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 실행 모드 설정
    FAST_MODE_MODELS = ['lightgbm', 'xgboost']
    FULL_MODE_MODELS = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'linear']
    EXTENDED_MODE_MODELS = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'linear', 'neural_network', 'additional']
    
    # 성능 임계값 설정
    MIN_CV_SCORE = 0.73  # 최소 교차 검증 점수
    ENSEMBLE_THRESHOLD = 0.75  # 앙상블 포함 임계값
    
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
        """모델 파라미터 동적 수정"""
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
            'optuna_trials': 200,
            'cv_folds': 5,
            'feature_selection_k': 35,
            'n_estimators': 400,
            'validation_size': 0.1
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
            'n_estimators': 800,
            'validation_size': cls.VALIDATION_SIZE
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
            'n_estimators': 800,
            'validation_size': cls.VALIDATION_SIZE,
            'use_calibration': True,
            'use_stacking': True
        }
        return extended_config
    
    @classmethod
    def get_performance_config(cls):
        """성능 중심 설정 반환"""
        performance_config = {
            'models': ['lightgbm', 'xgboost', 'catboost'],
            'optuna_trials': 1000,
            'cv_folds': 10,
            'feature_selection_k': 40,
            'n_estimators': 1200,
            'validation_size': 0.1,
            'use_pca': False,
            'scaling_method': 'robust'
        }
        return performance_config
    
    @classmethod
    def adjust_for_data_size(cls, n_samples, n_features):
        """데이터 크기에 따른 파라미터 동적 조정"""
        adjustments = {}
        
        # 샘플 수에 따른 조정
        if n_samples > 50000:
            adjustments.update({
                'cv_folds': 5,
                'n_estimators': 600,
                'optuna_trials': 300
            })
        elif n_samples < 10000:
            adjustments.update({
                'cv_folds': 10,
                'n_estimators': 1000,
                'optuna_trials': 500
            })
        
        # 피처 수에 따른 조정
        if n_features > 100:
            adjustments.update({
                'feature_selection_k': min(80, n_features // 2),
                'max_depth': 6
            })
        elif n_features < 30:
            adjustments.update({
                'feature_selection_k': n_features,
                'max_depth': 10
            })
        
        return adjustments