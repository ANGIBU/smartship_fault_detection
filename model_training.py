# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
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
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            eval_names = ['validation']
            early_stopping_rounds = params.pop('early_stopping_rounds', 100)
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train, 
                eval_set=eval_set, 
                eval_names=eval_names, 
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=0)]
            )
        else:
            params.pop('early_stopping_rounds', None)
            model = lgb.LGBMClassifier(**params)
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
        
        # 클래스 균형 조정
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[y] for y in y_train])
        
        early_stopping_rounds = params.pop('early_stopping_rounds', None)
        
        if X_val is not None and y_val is not None and early_stopping_rounds:
            eval_set = [(X_val, y_val)]
            callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, verbose=False)]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train, 
                sample_weight=sample_weights,
                eval_set=eval_set,
                callbacks=callbacks,
                verbose=False
            )
        else:
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self.models['xgboost'] = model
        self.logger.info("XGBoost 모델 훈련 완료")
        
        return model
    
    @timer
    def train_random_forest(self, X_train, y_train, params=None):
        """Random Forest 모델 훈련"""
        print("=== Random Forest 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.RF_PARAMS.copy()
        
        # 클래스 균형 조정
        params['class_weight'] = 'balanced'
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        self.logger.info("Random Forest 모델 훈련 완료")
        
        return model
    
    @timer
    def train_extra_trees(self, X_train, y_train, params=None):
        """Extra Trees 모델 훈련"""
        print("=== Extra Trees 모델 훈련 시작 ===")
        
        if params is None:
            params = Config.ET_PARAMS.copy()
        
        # 클래스 균형 조정
        params['class_weight'] = 'balanced'
        
        model = ExtraTreesClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['extra_trees'] = model
        self.logger.info("Extra Trees 모델 훈련 완료")
        
        return model
    
    @timer
    def hyperparameter_optimization_optuna(self, X_train, y_train, model_type='lightgbm', n_trials=200):
        """Optuna를 사용한 하이퍼파라미터 튜닝"""
        print(f"=== {model_type} 하이퍼파라미터 튜닝 시작 ===")
        
        def objective(trial):
            if model_type == 'lightgbm':
                params = {
                    'objective': 'multiclass',
                    'num_class': Config.N_CLASSES,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'class_weight': 'balanced',
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'verbose': -1,
                    'random_state': Config.RANDOM_STATE,
                    'n_estimators': 500,
                    'n_jobs': Config.N_JOBS
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_type == 'xgboost':
                params = {
                    'objective': 'multi:softprob',
                    'num_class': Config.N_CLASSES,
                    'eval_metric': 'mlogloss',
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'random_state': Config.RANDOM_STATE,
                    'n_estimators': 500,
                    'n_jobs': Config.N_JOBS,
                    'tree_method': 'hist'
                }
                
                # 클래스 가중치 계산
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_train)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                weight_dict = dict(zip(classes, class_weights))
                sample_weights = np.array([weight_dict[y] for y in y_train])
                
                model = xgb.XGBClassifier(**params)
                
                # 교차 검증에서 가중치 적용
                cv_scores = []
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
                
                for train_idx, val_idx in skf.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    sw_tr = sample_weights[train_idx]
                    
                    model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro')
                    cv_scores.append(score)
                
                return np.mean(cv_scores)
            
            # LightGBM의 경우 일반적인 교차 검증
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=3, scoring=self.scorer, n_jobs=1
            )
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=1800)
        
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
            
            # XGBoost의 경우 별도 처리
            if 'xgb' in str(type(model)).lower():
                cv_scores = []
                from sklearn.utils.class_weight import compute_class_weight
                
                classes = np.unique(y_train)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                weight_dict = dict(zip(classes, class_weights))
                
                for train_idx, val_idx in skf.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    sample_weights = np.array([weight_dict[y] for y in y_tr])
                    
                    # 모델 복사 및 훈련
                    model_copy = xgb.XGBClassifier(**model.get_params())
                    model_copy.fit(X_tr, y_tr, sample_weight=sample_weights, verbose=False)
                    
                    y_pred = model_copy.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro')
                    cv_scores.append(score)
                
                cv_scores = np.array(cv_scores)
            else:
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
    def create_voting_ensemble(self, X_train, y_train):
        """투표 기반 앙상블 생성"""
        print("=== 투표 앙상블 생성 시작 ===")
        
        if len(self.models) < 2:
            print("앙상블을 위해 최소 2개의 모델이 필요합니다.")
            return None
        
        # 상위 성능 모델만 선택 (CV 점수 0.75 이상)
        good_models = []
        for name, model in self.models.items():
            if name in self.cv_scores and self.cv_scores[name]['mean'] >= 0.75:
                good_models.append((name, model))
        
        if len(good_models) < 2:
            # 성능 기준 완화
            good_models = [(name, model) for name, model in self.models.items()][:3]
        
        print(f"앙상블에 사용할 모델: {[name for name, _ in good_models]}")
        
        try:
            # Soft voting 앙상블
            voting_ensemble = VotingClassifier(
                estimators=good_models, 
                voting='soft',
                n_jobs=Config.N_JOBS
            )
            voting_ensemble.fit(X_train, y_train)
            
            self.models['voting_ensemble'] = voting_ensemble
            print("Soft voting 앙상블 생성 완료")
            
            return voting_ensemble
        except Exception as e:
            print(f"앙상블 생성 실패: {e}")
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
    
    @timer
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, use_optimization=False, model_list=None):
        """모든 모델 훈련"""
        print("=== 전체 모델 훈련 시작 ===")
        
        if model_list is None:
            model_list = ['lightgbm', 'xgboost', 'random_forest', 'extra_trees']
        
        # 기본 모델들 훈련
        if 'lightgbm' in model_list:
            if use_optimization:
                print("LightGBM 하이퍼파라미터 튜닝 중...")
                best_params, _ = self.hyperparameter_optimization_optuna(
                    X_train, y_train, 'lightgbm', n_trials=200
                )
                self.train_lightgbm(X_train, y_train, X_val, y_val, best_params)
            else:
                self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        if 'xgboost' in model_list:
            if use_optimization:
                print("XGBoost 하이퍼파라미터 튜닝 중...")
                best_params, _ = self.hyperparameter_optimization_optuna(
                    X_train, y_train, 'xgboost', n_trials=200
                )
                self.train_xgboost(X_train, y_train, X_val, y_val, best_params)
            else:
                self.train_xgboost(X_train, y_train, X_val, y_val)
        
        if 'random_forest' in model_list:
            self.train_random_forest(X_train, y_train)
        
        if 'extra_trees' in model_list:
            self.train_extra_trees(X_train, y_train)
        
        # 교차 검증
        self.cross_validation(X_train, y_train)
        
        # 앙상블 생성 (성능이 좋을 때만)
        if len(self.models) >= 2:
            self.create_voting_ensemble(X_train, y_train)
            
            # 앙상블 성능도 검증
            if 'voting_ensemble' in self.models:
                ensemble_cv = cross_val_score(
                    self.models['voting_ensemble'], X_train, y_train, 
                    cv=3, scoring=self.scorer, n_jobs=1
                )
                
                self.cv_scores['voting_ensemble'] = {
                    'scores': ensemble_cv,
                    'mean': ensemble_cv.mean(),
                    'std': ensemble_cv.std()
                }
                
                print(f"앙상블 CV 점수: {ensemble_cv.mean():.4f} (+/- {ensemble_cv.std() * 2:.4f})")
                
                # 앙상블이 더 좋으면 최고 모델로 업데이트
                if ensemble_cv.mean() > self.best_score:
                    self.best_model = self.models['voting_ensemble']
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