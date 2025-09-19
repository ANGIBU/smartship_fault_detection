# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import f1_score, make_scorer
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
import optuna
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
        
    def _create_cv_strategy(self, X, y, cv_type='stratified'):
        """교차 검증 전략 생성"""
        if cv_type == 'stratified':
            return StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        elif cv_type == 'time_series':
            return TimeSeriesSplit(n_splits=Config.CV_FOLDS)
        else:
            return StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    @timer
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """LightGBM 모델 훈련"""
        print("=== LightGBM 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.LGBM_PARAMS.copy()
        else:
            params = params.copy()
        
        # 클래스 균형 조정
        params['class_weight'] = 'balanced'
        params['is_unbalance'] = True
        params['verbose'] = -1
        
        try:
            # 모델 생성
            model = lgb.LGBMClassifier(**{k: v for k, v in params.items() 
                                        if k not in ['early_stopping_rounds']})
            
            # Early stopping 처리
            if X_val is not None and y_val is not None:
                early_stopping_rounds = params.get('early_stopping_rounds', 50)
                
                # fit_params 설정
                fit_params = {
                    'eval_set': [(X_val, y_val)],
                    'eval_names': ['validation'],
                    'callbacks': [lgb.early_stopping(early_stopping_rounds, verbose=False),
                                 lgb.log_evaluation(0)]
                }
                
                model.fit(X_train, y_train, **fit_params)
            else:
                model.fit(X_train, y_train)
            
            self.models['lightgbm'] = model
            self.logger.info("LightGBM 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"LightGBM 훈련 중 오류 발생: {e}")
            self.logger.error(f"LightGBM 훈련 실패: {e}")
            
            # 단순한 파라미터로 재시도
            simple_params = {
                'objective': 'multiclass',
                'num_class': Config.N_CLASSES,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': Config.RANDOM_STATE,
                'verbose': -1,
                'n_jobs': 1
            }
            
            try:
                model = lgb.LGBMClassifier(**simple_params)
                model.fit(X_train, y_train)
                self.models['lightgbm'] = model
                print("단순 파라미터로 LightGBM 훈련 완료")
                return model
            except Exception as e2:
                print(f"LightGBM 재시도 실패: {e2}")
                return None
    
    @timer
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """XGBoost 모델 훈련"""
        print("=== XGBoost 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.XGB_PARAMS.copy()
        else:
            params = params.copy()
        
        try:
            # 클래스 가중치 계산
            classes = np.unique(y_train)
            sample_weights = compute_sample_weight('balanced', y=y_train)
            
            # early_stopping_rounds 추출
            early_stopping_rounds = params.pop('early_stopping_rounds', None)
            
            model = xgb.XGBClassifier(**params)
            
            if X_val is not None and y_val is not None and early_stopping_rounds:
                model.fit(
                    X_train, y_train, 
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['xgboost'] = model
            self.logger.info("XGBoost 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"XGBoost 훈련 중 오류 발생: {e}")
            self.logger.error(f"XGBoost 훈련 실패: {e}")
            return None
    
    @timer
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None):
        """CatBoost 모델 훈련"""
        print("=== CatBoost 모델 훈련 시작 ===")
        
        try:
            from catboost import CatBoostClassifier
            
            params = Config.CATBOOST_PARAMS.copy()
            
            model = CatBoostClassifier(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    use_best_model=True,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train, verbose=False)
            
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
    def train_random_forest(self, X_train, y_train, params=None):
        """Random Forest 모델 훈련"""
        print("=== Random Forest 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.RF_PARAMS.copy()
        
        try:
            params['class_weight'] = 'balanced'
            
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            self.models['random_forest'] = model
            self.logger.info("Random Forest 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"Random Forest 훈련 중 오류 발생: {e}")
            self.logger.error(f"Random Forest 훈련 실패: {e}")
            return None
    
    @timer
    def train_linear_models(self, X_train, y_train):
        """선형 모델들 훈련"""
        print("=== 선형 모델들 훈련 시작 ===")
        
        try:
            # Logistic Regression
            lr_params = {
                'multi_class': 'ovr',
                'class_weight': 'balanced',
                'random_state': Config.RANDOM_STATE,
                'max_iter': 1000,
                'n_jobs': Config.N_JOBS,
                'solver': 'liblinear'
            }
            
            lr_model = LogisticRegression(**lr_params)
            lr_model.fit(X_train, y_train)
            self.models['logistic_regression'] = lr_model
            
            # Ridge Classifier
            ridge_params = {
                'class_weight': 'balanced',
                'random_state': Config.RANDOM_STATE,
                'alpha': 1.0
            }
            
            ridge_model = RidgeClassifier(**ridge_params)
            ridge_model.fit(X_train, y_train)
            self.models['ridge'] = ridge_model
            
            print("선형 모델들 훈련 완료")
            return lr_model, ridge_model
            
        except Exception as e:
            print(f"선형 모델 훈련 중 오류 발생: {e}")
            self.logger.error(f"선형 모델 훈련 실패: {e}")
            return None, None
    
    @timer
    def train_svm(self, X_train, y_train):
        """SVM 모델 훈련"""
        print("=== SVM 모델 훈련 시작 ===")
        
        try:
            svm_params = {
                'kernel': 'rbf',
                'class_weight': 'balanced',
                'random_state': Config.RANDOM_STATE,
                'probability': True,
                'cache_size': 1000
            }
            
            # 데이터 크기에 따라 C 값 조정
            if X_train.shape[0] > 10000:
                svm_params['C'] = 0.1
            else:
                svm_params['C'] = 1.0
            
            model = SVC(**svm_params)
            model.fit(X_train, y_train)
            
            self.models['svm'] = model
            self.logger.info("SVM 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"SVM 훈련 중 오류 발생: {e}")
            self.logger.error(f"SVM 훈련 실패: {e}")
            return None
    
    @timer
    def train_neural_network(self, X_train, y_train):
        """신경망 모델 훈련"""
        print("=== 신경망 모델 훈련 시작 ===")
        
        try:
            nn_params = Config.NN_PARAMS.copy()
            
            model = MLPClassifier(**nn_params)
            model.fit(X_train, y_train)
            
            self.models['neural_network'] = model
            self.logger.info("신경망 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"신경망 훈련 중 오류 발생: {e}")
            self.logger.error(f"신경망 훈련 실패: {e}")
            return None
    
    @timer
    def train_additional_models(self, X_train, y_train):
        """추가 모델들 훈련"""
        print("=== 추가 모델들 훈련 시작 ===")
        
        try:
            # K-Nearest Neighbors
            knn_params = {
                'n_neighbors': 7,
                'weights': 'distance',
                'n_jobs': Config.N_JOBS
            }
            
            knn_model = KNeighborsClassifier(**knn_params)
            knn_model.fit(X_train, y_train)
            self.models['knn'] = knn_model
            
            # Gaussian Naive Bayes
            nb_model = GaussianNB()
            nb_model.fit(X_train, y_train)
            self.models['naive_bayes'] = nb_model
            
            print("추가 모델들 훈련 완료")
            return knn_model, nb_model
            
        except Exception as e:
            print(f"추가 모델 훈련 중 오류 발생: {e}")
            self.logger.error(f"추가 모델 훈련 실패: {e}")
            return None, None
    
    @timer
    def hyperparameter_optimization_optuna(self, X_train, y_train, model_type='lightgbm', n_trials=100):
        """하이퍼파라미터 튜닝"""
        print(f"=== {model_type} 하이퍼파라미터 튜닝 시작 ===")
        
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
                        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                        'max_depth': trial.suggest_int('max_depth', 3, 8),
                        'verbose': -1,
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                        'n_jobs': 1
                    }
                    model = lgb.LGBMClassifier(**params)
                    
                elif model_type == 'xgboost':
                    params = {
                        'objective': 'multi:softprob',
                        'num_class': Config.N_CLASSES,
                        'eval_metric': 'mlogloss',
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'max_depth': trial.suggest_int('max_depth', 3, 8),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                        'n_jobs': 1,
                        'tree_method': 'hist'
                    }
                    
                    model = xgb.XGBClassifier(**params)
                
                # 교차 검증
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=3, scoring=self.scorer, n_jobs=1
                )
                return cv_scores.mean()
                
            except Exception as e:
                print(f"옵튜나 시행 중 오류: {e}")
                return 0.0
        
        # Optuna 최적화
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=1800, show_progress_bar=False)
        
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
                # XGBoost는 가중치 적용한 별도 검증
                if 'xgb' in str(type(model)).lower():
                    cv_scores = []
                    sample_weights = compute_sample_weight('balanced', y=y_train)
                    
                    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        sw_tr = sample_weights[train_idx]
                        
                        # 모델 복사 및 훈련
                        model_copy = xgb.XGBClassifier(**model.get_params())
                        model_copy.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
                        
                        y_pred = model_copy.predict(X_val)
                        score = f1_score(y_val, y_pred, average='macro')
                        cv_scores.append(score)
                    
                    cv_scores = np.array(cv_scores)
                    
                # SVM이나 NN은 시간이 오래 걸리므로 3-fold로 축소
                elif model_name in ['svm', 'neural_network']:
                    cv_strategy_small = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_strategy_small, scoring=self.scorer, n_jobs=1
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
            if model is not None and name in self.cv_scores and self.cv_scores[name]['mean'] >= Config.MIN_CV_SCORE:
                if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
                    good_models.append((name, model))
        
        if len(good_models) < 2:
            # 기준 완화하여 상위 3개 모델 선택
            sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
            good_models = []
            for name, _ in sorted_models[:3]:
                if name in self.models and self.models[name] is not None:
                    model = self.models[name]
                    if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
                        good_models.append((name, model))
        
        if len(good_models) >= 2:
            print(f"스태킹에 사용할 모델: {[name for name, _ in good_models]}")
            
            try:
                # 메타 모델로 LogisticRegression 사용
                meta_model = LogisticRegression(
                    multi_class='ovr',
                    class_weight='balanced',
                    random_state=Config.RANDOM_STATE,
                    max_iter=1000,
                    solver='liblinear'
                )
                
                stacking_ensemble = StackingClassifier(
                    estimators=good_models,
                    final_estimator=meta_model,
                    cv=3,
                    n_jobs=1,
                    passthrough=False
                )
                
                stacking_ensemble.fit(X_train, y_train)
                self.models['stacking_ensemble'] = stacking_ensemble
                
                print("스태킹 앙상블 생성 완료")
                return stacking_ensemble
                
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
            if model is not None and name in self.cv_scores and self.cv_scores[name]['mean'] >= Config.ENSEMBLE_THRESHOLD:
                good_models.append((name, model))
        
        if len(good_models) < 2:
            # 기준 완화하여 상위 3개 모델 선택
            sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
            good_models = []
            for name, _ in sorted_models[:3]:
                if name in self.models and self.models[name] is not None:
                    good_models.append((name, self.models[name]))
        
        if len(good_models) >= 2:
            print(f"보팅에 사용할 모델: {[name for name, _ in good_models]}")
            
            try:
                # Soft voting 앙상블
                voting_ensemble = VotingClassifier(
                    estimators=good_models, 
                    voting='soft',
                    n_jobs=1
                )
                voting_ensemble.fit(X_train, y_train)
                
                self.models['voting_ensemble'] = voting_ensemble
                print("소프트 보팅 앙상블 생성 완료")
                
                return voting_ensemble
            except Exception as e:
                print(f"보팅 앙상블 생성 실패: {e}")
                # Hard voting으로 재시도
                try:
                    voting_ensemble = VotingClassifier(
                        estimators=good_models, 
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
            if model is not None and hasattr(model, 'predict_proba') and model_name not in ['voting_ensemble', 'stacking_ensemble']:
                try:
                    # 두 가지 보정 방법 모두 시도
                    for method in calibration_methods:
                        calibrated_model = CalibratedClassifierCV(
                            base_estimator=model,
                            method=method,
                            cv=3
                        )
                        
                        calibrated_model.fit(X_train, y_train)
                        calibrated_name = f'{model_name}_calibrated_{method}'
                        self.calibrated_models[calibrated_name] = calibrated_model
                        
                    print(f"{model_name} 확률 보정 완료")
                    
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
            model_list = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'linear', 'svm', 'neural_network', 'additional']
        
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
                self.logger.warning(f"CatBoost 훈련 실패: {e}")
        
        if 'random_forest' in model_list:
            self.train_random_forest(X_train, y_train)
        
        if 'linear' in model_list:
            self.train_linear_models(X_train, y_train)
        
        if 'svm' in model_list:
            # SVM은 데이터 크기가 클 때 건너뜀
            if X_train.shape[0] <= 15000:
                self.train_svm(X_train, y_train)
            else:
                print("SVM 건너뜀: 데이터 크기가 너무 큼")
        
        if 'neural_network' in model_list:
            self.train_neural_network(X_train, y_train)
        
        if 'additional' in model_list:
            self.train_additional_models(X_train, y_train)
        
        # None인 모델 제거
        self.models = {k: v for k, v in self.models.items() if v is not None}
        
        # 교차 검증
        if self.models:
            self.cross_validation(X_train, y_train)
        
            # 확률 보정
            self.calibrate_models(X_train, y_train)
            
            # 앙상블 생성
            voting_ensemble = self.create_voting_ensemble(X_train, y_train)
            stacking_ensemble = self.create_stacking_ensemble(X_train, y_train)
            
            # 앙상블 성능 검증
            for ensemble_name, ensemble in [('voting_ensemble', voting_ensemble), ('stacking_ensemble', stacking_ensemble)]:
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