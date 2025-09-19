# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import xgboost as xgb
import optuna
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
        
        # Early stopping 처리
        callbacks = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            eval_names = ['validation']
            early_stopping_rounds = params.pop('early_stopping_rounds', 50)
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=0)]
        else:
            eval_set = None
            eval_names = None
            params.pop('early_stopping_rounds', None)
        
        model = lgb.LGBMClassifier(**params)
        
        if eval_set:
            model.fit(
                X_train, y_train, 
                eval_set=eval_set, 
                eval_names=eval_names, 
                callbacks=callbacks
            )
        else:
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        self.logger.info("LightGBM 모델 훈련 완료")
        
        return model
    
    @timer
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """XGBoost 모델 훈련"""
        print("=== XGBoost 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.XGB_PARAMS.copy()
        else:
            params = params.copy()
        
        # 클래스 가중치 계산
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[y] for y in y_train])
        
        # early_stopping_rounds 추출
        early_stopping_rounds = params.pop('early_stopping_rounds', None)
        
        model = xgb.XGBClassifier(**params)
        
        if X_val is not None and y_val is not None and early_stopping_rounds:
            # Validation 데이터가 있는 경우 early stopping 적용
            model.fit(
                X_train, y_train, 
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            # Validation 데이터가 없는 경우 일반 훈련
            model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self.models['xgboost'] = model
        self.logger.info("XGBoost 모델 훈련 완료")
        
        return model
    
    @timer
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None):
        """CatBoost 모델 훈련"""
        print("=== CatBoost 모델 훈련 시작 ===")
        
        try:
            from catboost import CatBoostClassifier
            
            # 안전한 파라미터 설정
            params = {
                'iterations': 600,
                'learning_rate': 0.08,
                'depth': 8,
                'l2_leaf_reg': 3,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1,
                'sampling_frequency': 'PerTree',
                'colsample_bylevel': 0.8,
                'random_seed': Config.RANDOM_STATE,
                'verbose': False,
                'auto_class_weights': 'Balanced',
                'thread_count': Config.N_JOBS if Config.N_JOBS > 0 else None,
                'task_type': 'CPU'
            }
            
            model = CatBoostClassifier(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    use_best_model=True
                )
            else:
                model.fit(X_train, y_train)
            
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
        
        params['class_weight'] = 'balanced'
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        self.logger.info("Random Forest 모델 훈련 완료")
        
        return model
    
    @timer
    def hyperparameter_optimization_optuna(self, X_train, y_train, model_type='lightgbm', n_trials=300):
        """하이퍼파라미터 튜닝"""
        print(f"=== {model_type} 하이퍼파라미터 튜닝 시작 ===")
        
        def objective(trial):
            if model_type == 'lightgbm':
                params = {
                    'objective': 'multiclass',
                    'num_class': Config.N_CLASSES,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'class_weight': 'balanced',
                    'is_unbalance': True,
                    'num_leaves': trial.suggest_int('num_leaves', 30, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'verbose': -1,
                    'random_state': Config.RANDOM_STATE,
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                    'n_jobs': Config.N_JOBS
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_type == 'xgboost':
                params = {
                    'objective': 'multi:softprob',
                    'num_class': Config.N_CLASSES,
                    'eval_metric': 'mlogloss',
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'random_state': Config.RANDOM_STATE,
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                    'n_jobs': Config.N_JOBS,
                    'tree_method': 'hist'
                }
                
                # 클래스 가중치 적용
                classes = np.unique(y_train)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                weight_dict = dict(zip(classes, class_weights))
                sample_weights = np.array([weight_dict[y] for y in y_train])
                
                model = xgb.XGBClassifier(**params)
                
                # 가중치 적용한 교차 검증
                cv_scores = []
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
                
                for train_idx, val_idx in skf.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    sw_tr = sample_weights[train_idx]
                    
                    model_copy = xgb.XGBClassifier(**params)
                    model_copy.fit(X_tr, y_tr, sample_weight=sw_tr)
                    y_pred = model_copy.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro')
                    cv_scores.append(score)
                
                return np.mean(cv_scores)
            
            # LightGBM 교차 검증
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=3, scoring=self.scorer, n_jobs=1
            )
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=1800, show_progress_bar=False)
        
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 점수: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    @timer
    def cross_validation(self, X_train, y_train, cv_folds=None):
        """교차 검증 수행"""
        if cv_folds is None:
            cv_folds = Config.CV_FOLDS
        
        print("=== 교차 검증 시작 ===")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name} 교차 검증 중...")
            
            # XGBoost는 가중치 적용한 별도 검증
            if 'xgb' in str(type(model)).lower():
                cv_scores = []
                classes = np.unique(y_train)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                weight_dict = dict(zip(classes, class_weights))
                
                for train_idx, val_idx in skf.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    sample_weights = np.array([weight_dict[y] for y in y_tr])
                    
                    # 모델 복사 및 훈련
                    model_copy = xgb.XGBClassifier(**model.get_params())
                    model_copy.fit(X_tr, y_tr, sample_weight=sample_weights)
                    
                    y_pred = model_copy.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro')
                    cv_scores.append(score)
                
                cv_scores = np.array(cv_scores)
            else:
                # 일반 교차 검증
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=skf, scoring=self.scorer, n_jobs=1
                )
            
            self.cv_scores[model_name] = {
                'scores': cv_scores,
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            print(f"{model_name} CV 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 최고 성능 모델 선택
        if self.cv_scores:
            best_model_name = max(self.cv_scores.keys(), key=lambda x: self.cv_scores[x]['mean'])
            self.best_model = self.models[best_model_name]
            self.best_score = self.cv_scores[best_model_name]['mean']
            
            print(f"\n최고 성능 모델: {best_model_name}")
            print(f"최고 CV 점수: {self.best_score:.4f}")
        
        return self.cv_scores
    
    @timer
    def create_simple_ensemble(self, X_train, y_train):
        """간단한 앙상블 생성"""
        print("=== 간단한 앙상블 생성 시작 ===")
        
        # 성능 기준 모델 선택 (CV 점수 0.72 이상)
        good_models = []
        for name, model in self.models.items():
            if name in self.cv_scores and self.cv_scores[name]['mean'] >= 0.72:
                good_models.append((name, model))
        
        if len(good_models) < 2:
            # 기준 완화하여 상위 3개 모델 선택
            sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
            good_models = [(name, self.models[name]) for name, _ in sorted_models[:3] if name in self.models]
        
        if len(good_models) >= 2:
            print(f"앙상블에 사용할 모델: {[name for name, _ in good_models]}")
            
            try:
                # Soft voting 앙상블
                voting_ensemble = VotingClassifier(
                    estimators=good_models, 
                    voting='soft',
                    n_jobs=Config.N_JOBS
                )
                voting_ensemble.fit(X_train, y_train)
                
                self.models['ensemble'] = voting_ensemble
                print("Soft voting 앙상블 생성 완료")
                
                return voting_ensemble
            except Exception as e:
                print(f"앙상블 생성 실패: {e}")
                return None
        else:
            print("앙상블을 위한 충분한 모델이 없음")
            return None
    
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
            
            print("상위 15개 중요 피처:")
            print(feature_importance_df.head(15))
            
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
            model_list = ['lightgbm', 'xgboost', 'catboost', 'random_forest']
        
        # 기본 모델들 훈련
        if 'lightgbm' in model_list:
            if use_optimization:
                print("LightGBM 하이퍼파라미터 튜닝 중...")
                best_params, _ = self.hyperparameter_optimization_optuna(
                    X_train, y_train, 'lightgbm', n_trials=300
                )
                self.train_lightgbm(X_train, y_train, X_val, y_val, best_params)
            else:
                self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        if 'xgboost' in model_list:
            if use_optimization:
                print("XGBoost 하이퍼파라미터 튜닝 중...")
                best_params, _ = self.hyperparameter_optimization_optuna(
                    X_train, y_train, 'xgboost', n_trials=300
                )
                self.train_xgboost(X_train, y_train, X_val, y_val, best_params)
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
        
        # 교차 검증
        self.cross_validation(X_train, y_train)
        
        # 간단한 앙상블 생성
        ensemble = self.create_simple_ensemble(X_train, y_train)
        if ensemble:
            # 앙상블 성능 검증
            ensemble_cv = cross_val_score(
                ensemble, X_train, y_train, 
                cv=3, scoring=self.scorer, n_jobs=1
            )
            
            self.cv_scores['ensemble'] = {
                'scores': ensemble_cv,
                'mean': ensemble_cv.mean(),
                'std': ensemble_cv.std()
            }
            
            print(f"앙상블 CV 점수: {ensemble_cv.mean():.4f} (+/- {ensemble_cv.std() * 2:.4f})")
            
            # 앙상블이 더 좋으면 업데이트
            if ensemble_cv.mean() > self.best_score:
                self.best_model = ensemble
                self.best_score = ensemble_cv.mean()
                print("앙상블이 최고 성능 모델로 선택됨")
        
        # 최고 모델 저장
        if self.best_model is not None:
            save_model(self.best_model, Config.MODEL_FILE)
            print(f"최고 모델 저장: {type(self.best_model).__name__}")
        
        # CV 결과 저장
        cv_results_df = pd.DataFrame(self.cv_scores).T
        cv_results_df.reset_index(inplace=True)
        cv_results_df.rename(columns={'index': 'model'}, inplace=True)
        
        from utils import save_results
        save_results(cv_results_df, Config.CV_RESULTS_FILE)
        
        print("=== 전체 모델 훈련 완료 ===")
        print(f"훈련된 모델 수: {len(self.models)}")
        
        return self.models, self.best_model