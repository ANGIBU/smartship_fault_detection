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
    CV_FOLDS = 7
    VALIDATION_SIZE = 0.12
    
    # 피처 선택 설정
    FEATURE_SELECTION_K = 48
    PCA_COMPONENTS = 0.99
    
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
    INTERACTION_FEATURES = True
    FREQUENCY_FEATURES = True
    ROLLING_WINDOW_SIZES = [3, 5, 7, 10]
    POLYNOMIAL_DEGREE = 2
    
    # 클래스 가중치 설정
    CLASS_WEIGHTS = 'balanced'
    SAMPLING_STRATEGY = 'auto'
    
    # LightGBM 파라미터
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 7,
        'min_child_samples': 25,
        'min_child_weight': 0.001,
        'min_split_gain': 0.01,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'max_depth': 8,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 800,
        'n_jobs': N_JOBS,
        'early_stopping_rounds': 100,
        'class_weight': CLASS_WEIGHTS,
        'is_unbalance': True,
        'force_col_wise': True,
        'max_bin': 511,
        'min_data_in_leaf': 15,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'cat_smooth': 20,
        'cat_l2': 10
    }
    
    # XGBoost 파라미터
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': N_CLASSES,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.03,
        'max_depth': 8,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'colsample_bylevel': 0.85,
        'colsample_bynode': 0.85,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'gamma': 0.1,
        'min_child_weight': 15,
        'random_state': RANDOM_STATE,
        'n_estimators': 800,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
        'early_stopping_rounds': 100,
        'grow_policy': 'depthwise',
        'max_leaves': 0,
        'scale_pos_weight': 1
    }
    
    # CatBoost 파라미터
    CATBOOST_PARAMS = {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 5,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 0.8,
        'sampling_frequency': 'PerTreeLevel',
        'colsample_bylevel': 0.85,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'thread_count': N_JOBS if N_JOBS > 0 else None,
        'task_type': 'CPU',
        'early_stopping_rounds': 100,
        'use_best_model': True,
        'eval_metric': 'MultiClass',
        'od_type': 'Iter',
        'od_wait': 50
    }
    
    # Random Forest 파라미터
    RF_PARAMS = {
        'n_estimators': 400,
        'max_depth': 12,
        'min_samples_split': 8,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'max_samples': 0.85,
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': CLASS_WEIGHTS,
        'bootstrap': True,
        'oob_score': True,
        'criterion': 'gini',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0
    }
    
    # Extra Trees 파라미터
    ET_PARAMS = {
        'n_estimators': 400,
        'max_depth': 12,
        'min_samples_split': 8,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'class_weight': CLASS_WEIGHTS,
        'bootstrap': False,
        'criterion': 'gini',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0
    }
    
    # Gradient Boosting 파라미터
    GB_PARAMS = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'subsample': 0.85,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
        'tol': 1e-4
    }
    
    # 신경망 파라미터
    NN_PARAMS = {
        'hidden_layer_sizes': (128, 64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.01,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 500,
        'random_state': RANDOM_STATE,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'n_iter_no_change': 30,
        'tol': 1e-4,
        'batch_size': 'auto'
    }
    
    # SVM 파라미터
    SVM_PARAMS = {
        'kernel': 'rbf',
        'C': 2.0,
        'gamma': 'scale',
        'class_weight': CLASS_WEIGHTS,
        'random_state': RANDOM_STATE,
        'probability': True,
        'cache_size': 2000,
        'max_iter': 2000,
        'tol': 1e-3,
        'decision_function_shape': 'ovr'
    }
    
    # KNN 파라미터
    KNN_PARAMS = {
        'n_neighbors': 9,
        'weights': 'distance',
        'algorithm': 'auto',
        'leaf_size': 30,
        'p': 2,
        'n_jobs': N_JOBS
    }
    
    # AdaBoost 파라미터
    ADABOOST_PARAMS = {
        'n_estimators': 200,
        'learning_rate': 0.8,
        'algorithm': 'SAMME.R',
        'random_state': RANDOM_STATE
    }
    
    # Voting Classifier 파라미터
    VOTING_PARAMS = {
        'voting': 'soft',
        'n_jobs': N_JOBS
    }
    
    # 하이퍼파라미터 튜닝 설정
    OPTUNA_TRIALS = 150
    OPTUNA_TIMEOUT = 3600
    OPTUNA_PRUNING = True
    
    # 앙상블 설정
    ENSEMBLE_METHODS = ['voting', 'stacking', 'blending']
    STACKING_CV = 5
    BLEND_RATIO = 0.7
    
    # 로깅 설정
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 실행 모드별 설정
    FAST_MODE_MODELS = ['lightgbm', 'random_forest', 'extra_trees']
    FULL_MODE_MODELS = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'extra_trees', 'gradient_boosting']
    EXTENDED_MODE_MODELS = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'extra_trees', 'gradient_boosting', 'neural_network', 'knn', 'adaboost']
    
    # 성능 임계값 설정
    MIN_CV_SCORE = 0.72
    ENSEMBLE_THRESHOLD = 0.76
    CALIBRATION_THRESHOLD = 0.78
    
    # 메모리 설정
    MEMORY_EFFICIENT = True
    CHUNK_SIZE = 2000
    DTYPE_OPTIMIZATION = True
    
    # 불균형 처리 설정
    IMBALANCE_METHODS = ['smote', 'borderline_smote', 'adasyn', 'random_over', 'random_under']
    SMOTE_K_NEIGHBORS = 5
    SMOTE_SAMPLING_STRATEGY = 'auto'
    
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
            'catboost': cls.CATBOOST_PARAMS,
            'random_forest': cls.RF_PARAMS,
            'extra_trees': cls.ET_PARAMS,
            'gradient_boosting': cls.GB_PARAMS,
            'neural_network': cls.NN_PARAMS,
            'svm': cls.SVM_PARAMS,
            'knn': cls.KNN_PARAMS,
            'adaboost': cls.ADABOOST_PARAMS
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
    def update_params(cls, model_name, new_params):
        """모델 파라미터 수정"""
        if model_name == 'lightgbm':
            cls.LGBM_PARAMS.update(new_params)
        elif model_name == 'xgboost':
            cls.XGB_PARAMS.update(new_params)
        elif model_name == 'catboost':
            cls.CATBOOST_PARAMS.update(new_params)
        elif model_name == 'random_forest':
            cls.RF_PARAMS.update(new_params)
        elif model_name == 'extra_trees':
            cls.ET_PARAMS.update(new_params)
        elif model_name == 'gradient_boosting':
            cls.GB_PARAMS.update(new_params)
        elif model_name == 'neural_network':
            cls.NN_PARAMS.update(new_params)
        elif model_name == 'svm':
            cls.SVM_PARAMS.update(new_params)
        elif model_name == 'knn':
            cls.KNN_PARAMS.update(new_params)
        elif model_name == 'adaboost':
            cls.ADABOOST_PARAMS.update(new_params)
    
    @classmethod
    def get_fast_mode_config(cls):
        """빠른 실행 모드 설정 반환"""
        return {
            'models': cls.FAST_MODE_MODELS,
            'optuna_trials': 50,
            'cv_folds': 5,
            'feature_selection_k': 35,
            'n_estimators': 300,
            'validation_size': 0.15,
            'timeout': 1200
        }
    
    @classmethod
    def get_full_mode_config(cls):
        """전체 실행 모드 설정 반환"""
        return {
            'models': cls.FULL_MODE_MODELS,
            'optuna_trials': cls.OPTUNA_TRIALS,
            'cv_folds': cls.CV_FOLDS,
            'feature_selection_k': cls.FEATURE_SELECTION_K,
            'n_estimators': 800,
            'validation_size': cls.VALIDATION_SIZE,
            'timeout': cls.OPTUNA_TIMEOUT
        }
    
    @classmethod
    def get_extended_mode_config(cls):
        """확장 모드 설정 반환"""
        return {
            'models': cls.EXTENDED_MODE_MODELS,
            'optuna_trials': cls.OPTUNA_TRIALS,
            'cv_folds': cls.CV_FOLDS,
            'feature_selection_k': cls.FEATURE_SELECTION_K,
            'n_estimators': 1000,
            'validation_size': cls.VALIDATION_SIZE,
            'timeout': cls.OPTUNA_TIMEOUT,
            'use_calibration': True,
            'use_stacking': True,
            'use_blending': True
        }
    
    @classmethod
    def get_performance_config(cls):
        """성능 중심 설정 반환"""
        return {
            'models': ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'extra_trees'],
            'optuna_trials': 200,
            'cv_folds': 10,
            'feature_selection_k': 50,
            'n_estimators': 1200,
            'validation_size': 0.08,
            'timeout': 5400,
            'use_pca': False,
            'scaling_method': 'quantile',
            'imbalance_method': 'borderline_smote'
        }
    
    @classmethod
    def adjust_for_data_size(cls, n_samples, n_features):
        """데이터 크기에 따른 파라미터 조정"""
        adjustments = {}
        
        # 샘플 수에 따른 조정
        if n_samples > 100000:
            adjustments.update({
                'cv_folds': 5,
                'n_estimators': 500,
                'optuna_trials': 80,
                'timeout': 2400
            })
        elif n_samples < 20000:
            adjustments.update({
                'cv_folds': 10,
                'n_estimators': 1200,
                'optuna_trials': 200,
                'timeout': 5400
            })
        
        # 피처 수에 따른 조정
        if n_features > 200:
            adjustments.update({
                'feature_selection_k': min(100, n_features // 3),
                'max_depth': 6
            })
        elif n_features < 50:
            adjustments.update({
                'feature_selection_k': n_features,
                'max_depth': 10
            })
        
        return adjustments
    
    @classmethod
    def get_memory_config(cls):
        """메모리 설정"""
        return {
            'memory_efficient': cls.MEMORY_EFFICIENT,
            'chunk_size': cls.CHUNK_SIZE,
            'dtype_optimization': cls.DTYPE_OPTIMIZATION,
            'garbage_collection': True
        }
    
    @classmethod
    def get_imbalance_config(cls):
        """불균형 처리 설정"""
        return {
            'methods': cls.IMBALANCE_METHODS,
            'k_neighbors': cls.SMOTE_K_NEIGHBORS,
            'sampling_strategy': cls.SMOTE_SAMPLING_STRATEGY,
            'class_weights': cls.CLASS_WEIGHTS
        }