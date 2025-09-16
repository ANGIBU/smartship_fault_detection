# model_trainer.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_macro_f1, print_classification_metrics, save_model, setup_logging

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.cv_scores = {}
        self.logger = setup_logging()
        
    @timer
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """LightGBM 모델 훈련"""
        print("=== LightGBM 모델 훈련 시작 ===")
        
        if X_val is not None:
            eval_set = [(X_val, y_val)]
            eval_names = ['validation']
        else:
            eval_set = None
            eval_names = None
        
        model = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        
        if eval_set:
            model.fit(X_train, y_train, eval_set=eval_set, eval_names=eval_names, verbose=False)
        else:
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        self.logger.info("LightGBM 모델 훈련 완료")
        
        return model
    
    @timer
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """XGBoost 모델 훈련"""
        print("=== XGBoost 모델 훈련 시작 ===")
        
        if X_val is not None:
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None
        
        model = xgb.XGBClassifier(**Config.XGB_PARAMS)
        
        if eval_set:
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        self.logger.info("XGBoost 모델 훈련 완료")
        
        return model
    
    @timer
    def train_random_forest(self, X_train, y_train):
        """Random Forest 모델 훈련"""
        print("=== Random Forest 모델 훈련 시작 ===")
        
        model = RandomForestClassifier(**Config.RF_PARAMS)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        self.logger.info("Random Forest 모델 훈련 완료")
        
        return model
    
    @timer
    def hyperparameter_optimization(self, X_train, y_train, model_type='lightgbm', n_trials=50):
        """하이퍼파라미터 최적화"""
        print(f"=== {model_type} 하이퍼파라미터 최적화 시작 ===")
        
        def objective(trial):
            if model_type == 'lightgbm':
                params = {
                    'objective': 'multiclass',
                    'num_class': Config.N_CLASSES,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'verbose': -1,
                    'random_state': Config.RANDOM_STATE,
                    'n_estimators': 500
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_type == 'xgboost':
                params = {
                    'objective': 'multi:softprob',
                    'num_class': Config.N_CLASSES,
                    'eval_metric': 'mlogloss',
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': Config.RANDOM_STATE,
                    'n_estimators': 500
                }
                model = xgb.XGBClassifier(**params)
            
            # 교차 검증
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=3, scoring='f1_macro', n_jobs=-1
            )
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 점수: {study.best_value:.4f}")
        
        return study.best_params
    
    @timer
    def cross_validation(self, X_train, y_train, cv_folds=5):
        """교차 검증 수행"""
        print("=== 교차 검증 시작 ===")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name} 교차 검증 중...")
            
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=skf, scoring='f1_macro', n_jobs=-1
            )
            
            self.cv_scores[model_name] = {
                'scores': cv_scores,
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            print(f"{model_name} CV 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 최고 성능 모델 선택
        best_model_name = max(self.cv_scores.keys(), key=lambda x: self.cv_scores[x]['mean'])
        self.best_model = self.models[best_model_name]
        
        print(f"\n최고 성능 모델: {best_model_name}")
        print(f"최고 CV 점수: {self.cv_scores[best_model_name]['mean']:.4f}")
        
        return self.cv_scores
    
    @timer
    def create_ensemble(self, X_train, y_train):
        """앙상블 모델 생성"""
        print("=== 앙상블 모델 생성 시작 ===")
        
        if len(self.models) < 2:
            print("앙상블을 위해 최소 2개의 모델이 필요합니다.")
            return None
        
        # 투표 기반 앙상블
        estimators = [(name, model) for name, model in self.models.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        self.models['ensemble'] = ensemble
        print("앙상블 모델 생성 완료")
        
        return ensemble
    
    @timer
    def train_stacking_ensemble(self, X_train, y_train):
        """스태킹 앙상블 훈련"""
        print("=== 스태킹 앙상블 훈련 시작 ===")
        
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        if len(self.models) < 2:
            print("스태킹을 위해 최소 2개의 모델이 필요합니다.")
            return None
        
        estimators = [(name, model) for name, model in self.models.items() if name != 'ensemble']
        meta_classifier = LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=1000)
        
        stacking_ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=3,
            n_jobs=-1
        )
        
        stacking_ensemble.fit(X_train, y_train)
        self.models['stacking'] = stacking_ensemble
        
        print("스태킹 앙상블 훈련 완료")
        return stacking_ensemble
    
    @timer
    def evaluate_model(self, model, X_val, y_val, model_name="Model"):
        """모델 평가"""
        print(f"=== {model_name} 평가 시작 ===")
        
        y_pred = model.predict(X_val)
        macro_f1 = print_classification_metrics(y_val, y_pred)
        
        return macro_f1
    
    @timer
    def feature_importance_analysis(self, model, feature_names):
        """피처 중요도 분석"""
        print("=== 피처 중요도 분석 시작 ===")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            print("피처 중요도를 추출할 수 없는 모델입니다.")
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("상위 20개 중요 피처:")
        print(feature_importance_df.head(20))
        
        return feature_importance_df
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, use_optimization=False):
        """모든 모델 훈련"""
        print("=== 전체 모델 훈련 시작 ===")
        
        # 개별 모델 훈련
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train)
        
        # 하이퍼파라미터 최적화 (선택사항)
        if use_optimization:
            print("하이퍼파라미터 최적화 수행 중...")
            best_lgb_params = self.hyperparameter_optimization(X_train, y_train, 'lightgbm', n_trials=20)
            
            # 최적 파라미터로 재훈련
            optimized_params = Config.LGBM_PARAMS.copy()
            optimized_params.update(best_lgb_params)
            
            model = lgb.LGBMClassifier(**optimized_params)
            model.fit(X_train, y_train)
            self.models['lightgbm_optimized'] = model
        
        # 앙상블 모델 생성
        self.create_ensemble(X_train, y_train)
        self.train_stacking_ensemble(X_train, y_train)
        
        # 교차 검증
        self.cross_validation(X_train, y_train)
        
        # 최고 모델 저장
        save_model(self.best_model, Config.MODEL_FILE)
        
        print("=== 전체 모델 훈련 완료 ===")
        
        return self.models, self.best_model