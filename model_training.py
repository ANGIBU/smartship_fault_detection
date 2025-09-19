# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import f1_score, make_scorer
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_macro_f1, print_classification_metrics, save_model, save_joblib, setup_logging

class ModelTraining:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.cv_scores = {}
        self.logger = setup_logging()
        self.scorer = make_scorer(f1_score, average='macro')
        self.calibrated_models = {}
        self.resampled_data = {}
        
    def _create_cv_strategy(self, X, y, cv_type='stratified'):
        """교차 검증 전략 생성"""
        if cv_type == 'stratified':
            return StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        elif cv_type == 'time_series':
            return TimeSeriesSplit(n_splits=Config.CV_FOLDS)
        else:
            return StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    def _handle_class_imbalance(self, X_train, y_train, method='smote'):
        """클래스 불균형 처리"""
        print(f"클래스 불균형 처리 중: {method}")
        
        # 원본 분포 확인
        original_dist = np.bincount(y_train)
        print(f"원본 클래스 분포: {original_dist}")
        
        try:
            if method == 'smote':
                sampler = SMOTE(
                    random_state=Config.RANDOM_STATE,
                    k_neighbors=min(5, min(original_dist[original_dist > 0]) - 1),
                    sampling_strategy='auto'
                )
            elif method == 'borderline_smote':
                sampler = BorderlineSMOTE(
                    random_state=Config.RANDOM_STATE,
                    k_neighbors=min(5, min(original_dist[original_dist > 0]) - 1),
                    sampling_strategy='auto'
                )
            elif method == 'adasyn':
                sampler = ADASYN(
                    random_state=Config.RANDOM_STATE,
                    n_neighbors=min(5, min(original_dist[original_dist > 0]) - 1),
                    sampling_strategy='auto'
                )
            elif method == 'smote_tomek':
                sampler = SMOTETomek(
                    random_state=Config.RANDOM_STATE,
                    sampling_strategy='auto'
                )
            elif method == 'smote_enn':
                sampler = SMOTEENN(
                    random_state=Config.RANDOM_STATE,
                    sampling_strategy='auto'
                )
            else:
                # 기본값: SMOTE
                sampler = SMOTE(
                    random_state=Config.RANDOM_STATE,
                    k_neighbors=min(5, min(original_dist[original_dist > 0]) - 1)
                )
            
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            
            # 리샘플링 후 분포 확인
            resampled_dist = np.bincount(y_resampled)
            print(f"리샘플링 후 분포: {resampled_dist}")
            print(f"데이터 크기 변화: {X_train.shape} -> {X_resampled.shape}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"리샘플링 실패 ({method}): {e}")
            return X_train, y_train
    
    @timer
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None, use_resampling=True):
        """LightGBM 모델 훈련"""
        print("=== LightGBM 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.LGBM_PARAMS.copy()
        else:
            params = params.copy()
        
        # 클래스 불균형 처리
        if use_resampling:
            X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'smote')
        else:
            X_train_res, y_train_res = X_train, y_train
        
        try:
            # 파라미터 설정
            params['class_weight'] = 'balanced'
            params['is_unbalance'] = True
            params['verbose'] = -1
            
            model = lgb.LGBMClassifier(**{k: v for k, v in params.items() 
                                        if k not in ['early_stopping_rounds']})
            
            # Early stopping 처리
            if X_val is not None and y_val is not None:
                early_stopping_rounds = params.get('early_stopping_rounds', 100)
                
                fit_params = {
                    'eval_set': [(X_val, y_val)],
                    'eval_names': ['validation'],
                    'callbacks': [
                        lgb.early_stopping(early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                }
                
                model.fit(X_train_res, y_train_res, **fit_params)
            else:
                model.fit(X_train_res, y_train_res)
            
            self.models['lightgbm'] = model
            self.logger.info("LightGBM 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"LightGBM 훈련 중 오류 발생: {e}")
            self.logger.error(f"LightGBM 훈련 실패: {e}")
            
            # 기본 파라미터로 재시도
            simple_params = {
                'objective': 'multiclass',
                'num_class': Config.N_CLASSES,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'random_state': Config.RANDOM_STATE,
                'verbose': -1,
                'n_jobs': 1
            }
            
            try:
                model = lgb.LGBMClassifier(**simple_params)
                model.fit(X_train_res, y_train_res)
                self.models['lightgbm'] = model
                print("기본 파라미터로 LightGBM 훈련 완료")
                return model
            except Exception as e2:
                print(f"LightGBM 재시도 실패: {e2}")
                return None
    
    @timer
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None, use_resampling=True):
        """XGBoost 모델 훈련"""
        print("=== XGBoost 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.XGB_PARAMS.copy()
        else:
            params = params.copy()
        
        # 클래스 불균형 처리
        if use_resampling:
            X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'borderline_smote')
        else:
            X_train_res, y_train_res = X_train, y_train
        
        try:
            # 클래스 가중치 계산
            sample_weights = compute_sample_weight('balanced', y=y_train_res)
            
            early_stopping_rounds = params.pop('early_stopping_rounds', None)
            
            model = xgb.XGBClassifier(**params)
            
            if X_val is not None and y_val is not None and early_stopping_rounds:
                model.fit(
                    X_train_res, y_train_res, 
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            else:
                model.fit(X_train_res, y_train_res, sample_weight=sample_weights)
            
            self.models['xgboost'] = model
            self.logger.info("XGBoost 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"XGBoost 훈련 중 오류 발생: {e}")
            self.logger.error(f"XGBoost 훈련 실패: {e}")
            return None
    
    @timer
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None, use_resampling=True):
        """CatBoost 모델 훈련"""
        print("=== CatBoost 모델 훈련 시작 ===")
        
        try:
            from catboost import CatBoostClassifier
            
            # 클래스 불균형 처리
            if use_resampling:
                X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'adasyn')
            else:
                X_train_res, y_train_res = X_train, y_train
            
            params = Config.CATBOOST_PARAMS.copy()
            
            model = CatBoostClassifier(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train_res, y_train_res,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=100,
                    use_best_model=True,
                    verbose=False
                )
            else:
                model.fit(X_train_res, y_train_res, verbose=False)
            
            self.models['catboost'] = model
            self.logger.info("CatBoost 모델 훈련 완료")
            
            return model
        
        except ImportError:
            print("CatBoost 라이브러리가 설치되지 않음. 건너뛰기")
            return None
        except Exception as e:
            print(f"CatBoost 훈련 중 오류 발생: {e}")
            self.logger.error(f"CatBoost 훈련 실패: {e}")
            return None
    
    @timer
    def train_random_forest(self, X_train, y_train, params=None, use_resampling=True):
        """Random Forest 모델 훈련"""
        print("=== Random Forest 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.RF_PARAMS.copy()
        
        # 클래스 불균형 처리
        if use_resampling:
            X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'smote')
        else:
            X_train_res, y_train_res = X_train, y_train
        
        try:
            params['class_weight'] = 'balanced'
            
            model = RandomForestClassifier(**params)
            model.fit(X_train_res, y_train_res)
            
            self.models['random_forest'] = model
            self.logger.info("Random Forest 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"Random Forest 훈련 중 오류 발생: {e}")
            self.logger.error(f"Random Forest 훈련 실패: {e}")
            return None
    
    @timer
    def train_extra_trees(self, X_train, y_train, use_resampling=True):
        """Extra Trees 모델 훈련"""
        print("=== Extra Trees 모델 훈련 시작 ===")
        
        # 클래스 불균형 처리
        if use_resampling:
            X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'smote')
        else:
            X_train_res, y_train_res = X_train, y_train
        
        try:
            params = Config.ET_PARAMS.copy()
            params['class_weight'] = 'balanced'
            
            model = ExtraTreesClassifier(**params)
            model.fit(X_train_res, y_train_res)
            
            self.models['extra_trees'] = model
            self.logger.info("Extra Trees 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"Extra Trees 훈련 중 오류 발생: {e}")
            self.logger.error(f"Extra Trees 훈련 실패: {e}")
            return None
    
    @timer
    def train_gradient_boosting(self, X_train, y_train, use_resampling=True):
        """Gradient Boosting 모델 훈련"""
        print("=== Gradient Boosting 모델 훈련 시작 ===")
        
        # 클래스 불균형 처리
        if use_resampling:
            X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'borderline_smote')
        else:
            X_train_res, y_train_res = X_train, y_train
        
        try:
            params = Config.GB_PARAMS.copy()
            
            model = GradientBoostingClassifier(**params)
            model.fit(X_train_res, y_train_res)
            
            self.models['gradient_boosting'] = model
            self.logger.info("Gradient Boosting 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"Gradient Boosting 훈련 중 오류 발생: {e}")
            self.logger.error(f"Gradient Boosting 훈련 실패: {e}")
            return None
    
    @timer
    def train_linear_models(self, X_train, y_train, use_resampling=True):
        """선형 모델들 훈련"""
        print("=== 선형 모델들 훈련 시작 ===")
        
        # 클래스 불균형 처리
        if use_resampling:
            X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'smote_tomek')
        else:
            X_train_res, y_train_res = X_train, y_train
        
        try:
            # Logistic Regression
            lr_params = {
                'multi_class': 'ovr',
                'class_weight': 'balanced',
                'random_state': Config.RANDOM_STATE,
                'max_iter': 2000,
                'n_jobs': Config.N_JOBS,
                'solver': 'liblinear',
                'C': 1.0,
                'tol': 1e-4
            }
            
            lr_model = LogisticRegression(**lr_params)
            lr_model.fit(X_train_res, y_train_res)
            self.models['logistic_regression'] = lr_model
            
            # Ridge Classifier
            ridge_params = {
                'class_weight': 'balanced',
                'random_state': Config.RANDOM_STATE,
                'alpha': 1.0,
                'solver': 'auto',
                'tol': 1e-4
            }
            
            ridge_model = RidgeClassifier(**ridge_params)
            ridge_model.fit(X_train_res, y_train_res)
            self.models['ridge'] = ridge_model
            
            print("선형 모델들 훈련 완료")
            return lr_model, ridge_model
            
        except Exception as e:
            print(f"선형 모델 훈련 중 오류 발생: {e}")
            self.logger.error(f"선형 모델 훈련 실패: {e}")
            return None, None
    
    @timer
    def train_svm(self, X_train, y_train, use_resampling=True):
        """SVM 모델 훈련"""
        print("=== SVM 모델 훈련 시작 ===")
        
        # 데이터 크기 확인
        if X_train.shape[0] > 20000:
            print("SVM 건너뜀: 데이터 크기가 너무 큼")
            return None
        
        # 클래스 불균형 처리
        if use_resampling:
            X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'smote')
            # SVM은 메모리 이슈로 인해 샘플 수 제한
            if X_train_res.shape[0] > 15000:
                from sklearn.utils import resample
                X_train_res, y_train_res = resample(
                    X_train_res, y_train_res, n_samples=15000, 
                    random_state=Config.RANDOM_STATE, stratify=y_train_res
                )
        else:
            X_train_res, y_train_res = X_train, y_train
        
        try:
            params = Config.SVM_PARAMS.copy()
            
            # 데이터 크기에 따른 파라미터 조정
            if X_train_res.shape[0] > 10000:
                params['C'] = 0.1
                params['gamma'] = 'scale'
            else:
                params['C'] = 2.0
                params['gamma'] = 'scale'
            
            model = SVC(**params)
            model.fit(X_train_res, y_train_res)
            
            self.models['svm'] = model
            self.logger.info("SVM 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"SVM 훈련 중 오류 발생: {e}")
            self.logger.error(f"SVM 훈련 실패: {e}")
            return None
    
    @timer
    def train_neural_network(self, X_train, y_train, use_resampling=True):
        """신경망 모델 훈련"""
        print("=== 신경망 모델 훈련 시작 ===")
        
        # 클래스 불균형 처리
        if use_resampling:
            X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'adasyn')
        else:
            X_train_res, y_train_res = X_train, y_train
        
        try:
            params = Config.NN_PARAMS.copy()
            
            model = MLPClassifier(**params)
            model.fit(X_train_res, y_train_res)
            
            self.models['neural_network'] = model
            self.logger.info("신경망 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"신경망 훈련 중 오류 발생: {e}")
            self.logger.error(f"신경망 훈련 실패: {e}")
            return None
    
    @timer
    def train_knn(self, X_train, y_train, use_resampling=True):
        """KNN 모델 훈련"""
        print("=== KNN 모델 훈련 시작 ===")
        
        # 클래스 불균형 처리
        if use_resampling:
            X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'smote')
            # KNN은 메모리 이슈로 인해 샘플 수 제한
            if X_train_res.shape[0] > 25000:
                from sklearn.utils import resample
                X_train_res, y_train_res = resample(
                    X_train_res, y_train_res, n_samples=25000, 
                    random_state=Config.RANDOM_STATE, stratify=y_train_res
                )
        else:
            X_train_res, y_train_res = X_train, y_train
        
        try:
            params = Config.KNN_PARAMS.copy()
            
            model = KNeighborsClassifier(**params)
            model.fit(X_train_res, y_train_res)
            
            self.models['knn'] = model
            self.logger.info("KNN 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"KNN 훈련 중 오류 발생: {e}")
            self.logger.error(f"KNN 훈련 실패: {e}")
            return None
    
    @timer
    def train_adaboost(self, X_train, y_train, use_resampling=True):
        """AdaBoost 모델 훈련"""
        print("=== AdaBoost 모델 훈련 시작 ===")
        
        # 클래스 불균형 처리
        if use_resampling:
            X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'borderline_smote')
        else:
            X_train_res, y_train_res = X_train, y_train
        
        try:
            params = Config.ADABOOST_PARAMS.copy()
            
            model = AdaBoostClassifier(**params)
            model.fit(X_train_res, y_train_res)
            
            self.models['adaboost'] = model
            self.logger.info("AdaBoost 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"AdaBoost 훈련 중 오류 발생: {e}")
            self.logger.error(f"AdaBoost 훈련 실패: {e}")
            return None
    
    @timer
    def hyperparameter_optimization_optuna(self, X_train, y_train, model_type='lightgbm', n_trials=100):
        """하이퍼파라미터 튜닝"""
        print(f"=== {model_type} 하이퍼파라미터 튜닝 시작 ===")
        
        # 클래스 불균형 처리
        X_train_res, y_train_res = self._handle_class_imbalance(X_train, y_train, 'smote')
        
        def objective(trial):
            try:
                if model_type == 'lightgbm':
                    params = {
                        'objective': 'multiclass',
                        'num_class': Config.N_CLASSES,
                        'metric': 'multi_logloss',
                        'boosting_type': 'gbdt',
                        'class_weight': 'balanced',
                        'is_unbalance': True,
                        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.95),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.95),
                        'bagging_freq': trial.suggest_int('bagging_freq', 3, 10),
                        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
                        'max_depth': trial.suggest_int('max_depth', 5, 12),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 25),
                        'verbose': -1,
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                        'n_jobs': 1
                    }
                    model = lgb.LGBMClassifier(**params)
                    
                elif model_type == 'xgboost':
                    params = {
                        'objective': 'multi:softprob',
                        'num_class': Config.N_CLASSES,
                        'eval_metric': 'mlogloss',
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                        'max_depth': trial.suggest_int('max_depth', 5, 12),
                        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.95),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
                        'gamma': trial.suggest_float('gamma', 0, 0.5),
                        'min_child_weight': trial.suggest_int('min_child_weight', 5, 25),
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                        'n_jobs': 1,
                        'tree_method': 'hist'
                    }
                    model = xgb.XGBClassifier(**params)
                
                elif model_type == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
                        'max_depth': trial.suggest_int('max_depth', 8, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 5, 15),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                        'max_samples': trial.suggest_float('max_samples', 0.7, 0.95),
                        'class_weight': 'balanced',
                        'random_state': Config.RANDOM_STATE,
                        'n_jobs': 1
                    }
                    model = RandomForestClassifier(**params)
                
                # 교차 검증 (빠른 실행을 위해 3-fold)
                cv_scores = cross_val_score(
                    model, X_train_res, y_train_res, 
                    cv=3, scoring=self.scorer, n_jobs=1
                )
                return cv_scores.mean()
                
            except Exception as e:
                print(f"옵튜나 시행 중 오류: {e}")
                return 0.0
        
        # Optuna 설정
        sampler = TPESampler(seed=Config.RANDOM_STATE)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5) if Config.OPTUNA_PRUNING else None
        
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=Config.OPTUNA_TIMEOUT, show_progress_bar=False)
        
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 점수: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    @timer
    def cross_validation(self, X_train, y_train, cv_folds=None, cv_type='stratified'):
        """교차 검증 수행"""
        if cv_folds is None:
            cv_folds = Config.CV_FOLDS
        
        print("=== 교차 검증 시작 ===")
        
        cv_strategy = self._create_cv_strategy(X_train, y_train, cv_type)
        
        for model_name, model in self.models.items():
            if model is None:
                print(f"{model_name} 건너뜀: 모델이 None")
                continue
                
            print(f"\n{model_name} 교차 검증 중...")
            
            try:
                # 시간이 오래 걸리는 모델은 fold 수 축소
                if model_name in ['svm', 'neural_network', 'knn']:
                    cv_strategy_reduced = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_strategy_reduced, scoring=self.scorer, n_jobs=1
                    )
                else:
                    # 일반 교차 검증
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_strategy, scoring=self.scorer, n_jobs=1
                    )
                
                self.cv_scores[model_name] = {
                    'scores': cv_scores,
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std()
                }
                
                print(f"{model_name} CV 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"{model_name} 교차 검증 실패: {e}")
                continue
        
        # 최고 성능 모델 선택
        if self.cv_scores:
            best_model_name = max(self.cv_scores.keys(), key=lambda x: self.cv_scores[x]['mean'])
            if best_model_name in self.models and self.models[best_model_name] is not None:
                self.best_model = self.models[best_model_name]
                self.best_score = self.cv_scores[best_model_name]['mean']
                
                print(f"\n최고 성능 모델: {best_model_name}")
                print(f"최고 CV 점수: {self.best_score:.4f}")
        
        # 메모리 정리
        gc.collect()
        
        return self.cv_scores
    
    @timer
    def create_stacking_ensemble(self, X_train, y_train):
        """스태킹 앙상블 생성"""
        print("=== 스태킹 앙상블 생성 시작 ===")
        
        # 성능 기준 모델 선택
        good_models = []
        for name, model in self.models.items():
            if (model is not None and name in self.cv_scores and 
                self.cv_scores[name]['mean'] >= Config.MIN_CV_SCORE and
                hasattr(model, 'predict_proba')):
                good_models.append((name, model))
        
        if len(good_models) < 2:
            # 기준 완화하여 상위 모델 선택
            sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
            good_models = []
            for name, _ in sorted_models[:5]:
                if name in self.models and self.models[name] is not None:
                    model = self.models[name]
                    if hasattr(model, 'predict_proba'):
                        good_models.append((name, model))
        
        if len(good_models) >= 2:
            print(f"스태킹에 사용할 모델: {[name for name, _ in good_models]}")
            
            try:
                # 다양한 메타 모델 시도
                meta_models = [
                    LogisticRegression(
                        multi_class='ovr',
                        class_weight='balanced',
                        random_state=Config.RANDOM_STATE,
                        max_iter=1000,
                        solver='liblinear',
                        C=0.1
                    ),
                    RidgeClassifier(
                        class_weight='balanced',
                        random_state=Config.RANDOM_STATE,
                        alpha=10.0
                    )
                ]
                
                best_stacking = None
                best_score = 0
                
                for i, meta_model in enumerate(meta_models):
                    try:
                        stacking_ensemble = StackingClassifier(
                            estimators=good_models[:min(7, len(good_models))],  # 최대 7개 모델
                            final_estimator=meta_model,
                            cv=Config.STACKING_CV,
                            n_jobs=1,
                            passthrough=False
                        )
                        
                        # 빠른 검증
                        cv_score = cross_val_score(
                            stacking_ensemble, X_train, y_train,
                            cv=3, scoring=self.scorer, n_jobs=1
                        ).mean()
                        
                        if cv_score > best_score:
                            best_score = cv_score
                            best_stacking = stacking_ensemble
                            
                        print(f"메타 모델 {i+1} 점수: {cv_score:.4f}")
                        
                    except Exception as e:
                        print(f"메타 모델 {i+1} 실패: {e}")
                        continue
                
                if best_stacking is not None:
                    best_stacking.fit(X_train, y_train)
                    self.models['stacking_ensemble'] = best_stacking
                    print("스태킹 앙상블 생성 완료")
                    return best_stacking
                
            except Exception as e:
                print(f"스태킹 앙상블 생성 실패: {e}")
                return None
        else:
            print("스태킹을 위한 충분한 모델이 없음")
            return None
    
    @timer
    def create_voting_ensemble(self, X_train, y_train):
        """보팅 앙상블 생성"""
        print("=== 보팅 앙상블 생성 시작 ===")
        
        # 성능 기준 모델 선택
        good_models = []
        for name, model in self.models.items():
            if (model is not None and name in self.cv_scores and 
                self.cv_scores[name]['mean'] >= Config.ENSEMBLE_THRESHOLD):
                good_models.append((name, model))
        
        if len(good_models) < 2:
            # 기준 완화하여 상위 모델 선택
            sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
            good_models = []
            for name, _ in sorted_models[:6]:
                if name in self.models and self.models[name] is not None:
                    good_models.append((name, self.models[name]))
        
        if len(good_models) >= 2:
            print(f"보팅에 사용할 모델: {[name for name, _ in good_models]}")
            
            try:
                # Soft voting 시도
                voting_ensemble = VotingClassifier(
                    estimators=good_models[:8],  # 최대 8개 모델
                    voting='soft',
                    n_jobs=1
                )
                voting_ensemble.fit(X_train, y_train)
                
                self.models['voting_ensemble'] = voting_ensemble
                print("소프트 보팅 앙상블 생성 완료")
                
                return voting_ensemble
            except Exception as e:
                print(f"소프트 보팅 실패: {e}")
                # Hard voting으로 재시도
                try:
                    voting_ensemble = VotingClassifier(
                        estimators=good_models[:8],
                        voting='hard',
                        n_jobs=1
                    )
                    voting_ensemble.fit(X_train, y_train)
                    self.models['voting_ensemble'] = voting_ensemble
                    print("하드 보팅 앙상블 생성 완료")
                    return voting_ensemble
                except Exception as e2:
                    print(f"하드 보팅도 실패: {e2}")
                    return None
        else:
            print("보팅을 위한 충분한 모델이 없음")
            return None
    
    @timer
    def calibrate_models(self, X_train, y_train):
        """모델 확률 보정"""
        print("=== 모델 확률 보정 시작 ===")
        
        calibration_methods = ['isotonic', 'sigmoid']
        
        for model_name, model in self.models.items():
            if (model is not None and hasattr(model, 'predict_proba') and 
                model_name not in ['voting_ensemble', 'stacking_ensemble'] and
                model_name in self.cv_scores and 
                self.cv_scores[model_name]['mean'] >= Config.CALIBRATION_THRESHOLD):
                
                try:
                    best_calibrated = None
                    best_method = None
                    best_score = 0
                    
                    for method in calibration_methods:
                        calibrated_model = CalibratedClassifierCV(
                            base_estimator=model,
                            method=method,
                            cv=3
                        )
                        
                        # 빠른 검증
                        cv_score = cross_val_score(
                            calibrated_model, X_train, y_train,
                            cv=3, scoring=self.scorer, n_jobs=1
                        ).mean()
                        
                        if cv_score > best_score:
                            best_score = cv_score
                            best_calibrated = calibrated_model
                            best_method = method
                    
                    if best_calibrated is not None and best_score > self.cv_scores[model_name]['mean']:
                        best_calibrated.fit(X_train, y_train)
                        calibrated_name = f'{model_name}_calibrated'
                        self.calibrated_models[calibrated_name] = best_calibrated
                        print(f"{model_name} 확률 보정 완료 ({best_method}): {best_score:.4f}")
                    
                except Exception as e:
                    print(f"{model_name} 확률 보정 실패: {e}")
                    continue
        
        print(f"보정된 모델 수: {len(self.calibrated_models)}")
        return self.calibrated_models
    
    @timer
    def feature_importance_analysis(self, model=None, feature_names=None):
        """피처 중요도 분석"""
        if model is None:
            model = self.best_model
        
        if model is None:
            print("분석할 모델이 없습니다.")
            return None
        
        print("=== 피처 중요도 분석 시작 ===")
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                print("피처 중요도를 추출할 수 없는 모델입니다.")
                return None
            
            if feature_names is None and hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            elif feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("상위 20개 중요 피처:")
            print(feature_importance_df.head(20))
            
            from utils import save_results
            save_results(feature_importance_df, Config.FEATURE_IMPORTANCE_FILE)
            
            return feature_importance_df
        except Exception as e:
            print(f"피처 중요도 분석 실패: {e}")
            return None
    
    @timer
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, use_optimization=False, model_list=None):
        """모든 모델 훈련"""
        print("=== 전체 모델 훈련 시작 ===")
        
        if model_list is None:
            model_list = Config.FULL_MODE_MODELS
        
        # 기본 모델들 훈련
        if 'lightgbm' in model_list:
            if use_optimization:
                print("LightGBM 하이퍼파라미터 튜닝 중...")
                try:
                    best_params, _ = self.hyperparameter_optimization_optuna(
                        X_train, y_train, 'lightgbm', n_trials=Config.OPTUNA_TRIALS
                    )
                    self.train_lightgbm(X_train, y_train, X_val, y_val, best_params)
                except Exception as e:
                    print(f"LightGBM 튜닝 실패, 기본 파라미터 사용: {e}")
                    self.train_lightgbm(X_train, y_train, X_val, y_val)
            else:
                self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        if 'xgboost' in model_list:
            if use_optimization:
                print("XGBoost 하이퍼파라미터 튜닝 중...")
                try:
                    best_params, _ = self.hyperparameter_optimization_optuna(
                        X_train, y_train, 'xgboost', n_trials=Config.OPTUNA_TRIALS
                    )
                    self.train_xgboost(X_train, y_train, X_val, y_val, best_params)
                except Exception as e:
                    print(f"XGBoost 튜닝 실패, 기본 파라미터 사용: {e}")
                    self.train_xgboost(X_train, y_train, X_val, y_val)
            else:
                self.train_xgboost(X_train, y_train, X_val, y_val)
        
        if 'catboost' in model_list:
            try:
                self.train_catboost(X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"CatBoost 훈련 건너뜀: {e}")
        
        if 'random_forest' in model_list:
            if use_optimization:
                try:
                    best_params, _ = self.hyperparameter_optimization_optuna(
                        X_train, y_train, 'random_forest', n_trials=Config.OPTUNA_TRIALS // 2
                    )
                    self.train_random_forest(X_train, y_train, best_params)
                except:
                    self.train_random_forest(X_train, y_train)
            else:
                self.train_random_forest(X_train, y_train)
        
        if 'extra_trees' in model_list:
            self.train_extra_trees(X_train, y_train)
        
        if 'gradient_boosting' in model_list:
            self.train_gradient_boosting(X_train, y_train)
        
        if 'linear' in model_list:
            self.train_linear_models(X_train, y_train)
        
        if 'svm' in model_list and X_train.shape[0] <= 20000:
            self.train_svm(X_train, y_train)
        
        if 'neural_network' in model_list:
            self.train_neural_network(X_train, y_train)
        
        if 'knn' in model_list:
            self.train_knn(X_train, y_train)
        
        if 'adaboost' in model_list:
            self.train_adaboost(X_train, y_train)
        
        # None인 모델 제거
        self.models = {k: v for k, v in self.models.items() if v is not None}
        
        # 교차 검증
        if self.models:
            self.cross_validation(X_train, y_train)
        
            # 확률 보정
            if use_optimization:
                self.calibrate_models(X_train, y_train)
            
            # 앙상블 생성
            voting_ensemble = self.create_voting_ensemble(X_train, y_train)
            stacking_ensemble = self.create_stacking_ensemble(X_train, y_train)
            
            # 앙상블 성능 검증
            ensembles = [
                ('voting_ensemble', voting_ensemble), 
                ('stacking_ensemble', stacking_ensemble)
            ]
            
            for ensemble_name, ensemble in ensembles:
                if ensemble is not None:
                    try:
                        ensemble_cv = cross_val_score(
                            ensemble, X_train, y_train, 
                            cv=3, scoring=self.scorer, n_jobs=1
                        )
                        
                        self.cv_scores[ensemble_name] = {
                            'scores': ensemble_cv,
                            'mean': ensemble_cv.mean(),
                            'std': ensemble_cv.std()
                        }
                        
                        print(f"{ensemble_name} CV 점수: {ensemble_cv.mean():.4f} (+/- {ensemble_cv.std() * 2:.4f})")
                        
                        # 앙상블이 더 좋으면 업데이트
                        if ensemble_cv.mean() > self.best_score:
                            self.best_model = ensemble
                            self.best_score = ensemble_cv.mean()
                            print(f"{ensemble_name}이 최고 성능 모델로 선택됨")
                            
                    except Exception as e:
                        print(f"{ensemble_name} 검증 실패: {e}")
        
        # 보정된 모델들도 고려
        for calibrated_name, calibrated_model in self.calibrated_models.items():
            try:
                calibrated_cv = cross_val_score(
                    calibrated_model, X_train, y_train,
                    cv=3, scoring=self.scorer, n_jobs=1
                )
                
                self.cv_scores[calibrated_name] = {
                    'scores': calibrated_cv,
                    'mean': calibrated_cv.mean(),
                    'std': calibrated_cv.std()
                }
                
                print(f"{calibrated_name} CV 점수: {calibrated_cv.mean():.4f}")
                
                if calibrated_cv.mean() > self.best_score:
                    self.best_model = calibrated_model
                    self.best_score = calibrated_cv.mean()
                    print(f"{calibrated_name}이 최고 성능 모델로 선택됨")
                    
            except Exception as e:
                print(f"{calibrated_name} 검증 실패: {e}")
        
        # 최고 모델 저장
        if self.best_model is not None:
            save_model(self.best_model, Config.MODEL_FILE)
            print(f"최고 모델 저장: {type(self.best_model).__name__}")
        
        # CV 결과 저장
        if self.cv_scores:
            cv_results_data = []
            for model_name, scores in self.cv_scores.items():
                cv_results_data.append({
                    'model': model_name,
                    'mean': scores['mean'],
                    'std': scores['std'],
                    'scores': scores['scores'].tolist() if hasattr(scores['scores'], 'tolist') else scores['scores']
                })
            
            cv_results_df = pd.DataFrame(cv_results_data)
            
            from utils import save_results
            save_results(cv_results_df, Config.CV_RESULTS_FILE)
        
        # 메모리 정리
        gc.collect()
        
        print("=== 전체 모델 훈련 완료 ===")
        print(f"훈련된 모델 수: {len(self.models)}")
        
        return self.models, self.best_model