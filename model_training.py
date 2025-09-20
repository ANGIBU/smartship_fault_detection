# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_macro_f1, save_model, save_joblib, setup_logging

class ModelTraining:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.cv_scores = {}
        self.logger = setup_logging()
        self.scorer = make_scorer(f1_score, average='macro')
        
    def _create_cv_strategy(self, X, y):
        """교차 검증 전략 생성"""
        return StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    @timer
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """LightGBM 모델 훈련"""
        print("LightGBM 모델 훈련 시작")
        
        if params is None:
            params = Config.LGBM_PARAMS.copy()
        else:
            params = params.copy()
        
        try:
            # 클래스 가중치를 파라미터에서 제거하고 별도 처리
            params.pop('class_weight', None)
            
            # LightGBM에서 클래스 가중치 계산
            classes = np.unique(y_train)
            class_weights = compute_class_weight(
                'balanced', 
                classes=classes, 
                y=y_train
            )
            class_weight_dict = dict(zip(classes, class_weights))
            
            # 샘플 가중치 계산
            sample_weights = np.array([class_weight_dict[y] for y in y_train])
            
            model = lgb.LGBMClassifier(**params)
            
            # Early stopping 처리
            if X_val is not None and y_val is not None:
                fit_params = {
                    'eval_set': [(X_val, y_val)],
                    'eval_names': ['validation'],
                    'callbacks': [
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(0)
                    ],
                    'sample_weight': sample_weights
                }
                
                model.fit(X_train, y_train, **fit_params)
            else:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['lightgbm'] = model
            self.logger.info("LightGBM 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"LightGBM 훈련 중 오류 발생: {e}")
            self.logger.error(f"LightGBM 훈련 실패: {e}")
            return None
    
    @timer
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """XGBoost 모델 훈련"""
        print("XGBoost 모델 훈련 시작")
        
        if params is None:
            params = Config.XGB_PARAMS.copy()
        else:
            params = params.copy()
        
        try:
            # 클래스 가중치 계산
            classes = np.unique(y_train)
            class_weights = compute_class_weight(
                'balanced', 
                classes=classes, 
                y=y_train
            )
            class_weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([class_weight_dict[y] for y in y_train])
            
            model = xgb.XGBClassifier(**params)
            
            # XGBoost 버전 호환성 처리
            if X_val is not None and y_val is not None:
                try:
                    # 새로운 방식 시도
                    model.fit(
                        X_train, y_train, 
                        sample_weight=sample_weights,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                except TypeError:
                    # 구버전 XGBoost 호환
                    model.fit(
                        X_train, y_train, 
                        sample_weight=sample_weights,
                        eval_set=[(X_val, y_val)],
                        eval_metric='mlogloss',
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
            
            # 기본 파라미터로 재시도
            try:
                print("기본 설정으로 XGBoost 재시도")
                basic_params = {
                    'objective': 'multi:softprob',
                    'num_class': Config.N_CLASSES,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': Config.RANDOM_STATE,
                    'n_estimators': 100,
                    'n_jobs': 1
                }
                
                model = xgb.XGBClassifier(**basic_params)
                model.fit(X_train, y_train, sample_weight=sample_weights)
                
                self.models['xgboost'] = model
                self.logger.info("XGBoost 기본 설정으로 훈련 완료")
                
                return model
                
            except Exception as e2:
                print(f"XGBoost 기본 설정도 실패: {e2}")
                self.logger.error(f"XGBoost 완전 실패: {e2}")
                return None
    
    @timer
    def train_random_forest(self, X_train, y_train, params=None):
        """Random Forest 모델 훈련"""
        print("Random Forest 모델 훈련 시작")
        
        if params is None:
            params = Config.RF_PARAMS.copy()
        
        try:
            # 클래스 가중치 적용
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
    def train_gradient_boosting(self, X_train, y_train):
        """Gradient Boosting 모델 훈련"""
        print("Gradient Boosting 모델 훈련 시작")
        
        try:
            params = Config.GB_PARAMS.copy()
            
            model = GradientBoostingClassifier(**params)
            model.fit(X_train, y_train)
            
            self.models['gradient_boosting'] = model
            self.logger.info("Gradient Boosting 모델 훈련 완료")
            
            return model
            
        except Exception as e:
            print(f"Gradient Boosting 훈련 중 오류 발생: {e}")
            self.logger.error(f"Gradient Boosting 훈련 실패: {e}")
            return None
    
    @timer
    def hyperparameter_optimization(self, X_train, y_train, model_type='lightgbm', n_trials=15):
        """하이퍼파라미터 튜닝"""
        print(f"{model_type} 하이퍼파라미터 튜닝 시작")
        
        def objective(trial):
            try:
                if model_type == 'lightgbm':
                    params = {
                        'objective': 'multiclass',
                        'num_class': Config.N_CLASSES,
                        'metric': 'multi_logloss',
                        'boosting_type': 'gbdt',
                        'is_unbalance': True,
                        'num_leaves': trial.suggest_int('num_leaves', 31, 100),
                        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.9),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
                        'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 20),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 0.1),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.1),
                        'max_depth': trial.suggest_int('max_depth', 6, 12),
                        'verbose': -1,
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 300, 600),
                        'n_jobs': 1
                    }
                    
                    # 클래스 가중치 계산
                    classes = np.unique(y_train)
                    class_weights = compute_class_weight(
                        'balanced', 
                        classes=classes, 
                        y=y_train
                    )
                    class_weight_dict = dict(zip(classes, class_weights))
                    sample_weights = np.array([class_weight_dict[y] for y in y_train])
                    
                    model = lgb.LGBMClassifier(**params)
                    
                elif model_type == 'xgboost':
                    params = {
                        'objective': 'multi:softprob',
                        'num_class': Config.N_CLASSES,
                        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
                        'max_depth': trial.suggest_int('max_depth', 4, 10),
                        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 0.1),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.1),
                        'gamma': trial.suggest_float('gamma', 0, 0.3),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 300, 600),
                        'n_jobs': 1,
                        'tree_method': 'hist'
                    }
                    
                    # 클래스 가중치 계산
                    classes = np.unique(y_train)
                    class_weights = compute_class_weight(
                        'balanced', 
                        classes=classes, 
                        y=y_train
                    )
                    class_weight_dict = dict(zip(classes, class_weights))
                    sample_weights = np.array([class_weight_dict[y] for y in y_train])
                    
                    model = xgb.XGBClassifier(**params)
                
                # 3-fold 교차 검증
                cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
                cv_scores = []
                
                for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # 가중치도 분할
                    if model_type in ['lightgbm', 'xgboost']:
                        sw_tr = sample_weights[train_idx]
                        model.fit(X_tr, y_tr, sample_weight=sw_tr)
                    else:
                        model.fit(X_tr, y_tr)
                    
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro')
                    cv_scores.append(score)
                
                return np.mean(cv_scores)
                
            except Exception as e:
                print(f"옵튜나 시행 중 오류: {e}")
                return 0.0
        
        # Optuna 설정
        sampler = TPESampler(seed=Config.RANDOM_STATE)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=Config.OPTUNA_TIMEOUT, show_progress_bar=False)
        
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 점수: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    @timer
    def cross_validation(self, X_train, y_train):
        """교차 검증 수행"""
        print("교차 검증 시작")
        
        cv_strategy = self._create_cv_strategy(X_train, y_train)
        
        for model_name, model in self.models.items():
            if model is None:
                print(f"{model_name} 건너뜀: 모델이 None")
                continue
                
            print(f"{model_name} 교차 검증 중")
            
            try:
                # 단순 교차 검증 (가중치 없이)
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
                
                print(f"최고 성능 모델: {best_model_name}")
                print(f"최고 CV 점수: {self.best_score:.4f}")
        
        # 메모리 정리
        gc.collect()
        
        return self.cv_scores
    
    @timer
    def create_voting_ensemble(self, X_train, y_train):
        """보팅 앙상블 생성"""
        print("보팅 앙상블 생성 시작")
        
        # 성능 기준 모델 선택
        good_models = []
        for name, model in self.models.items():
            if (model is not None and name in self.cv_scores and 
                self.cv_scores[name]['mean'] >= Config.MIN_CV_SCORE):
                good_models.append((name, model))
        
        if len(good_models) < 2:
            # 기준 완화하여 상위 모델 선택
            if self.cv_scores:
                sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
                good_models = []
                for name, _ in sorted_models[:3]:
                    if name in self.models and self.models[name] is not None:
                        good_models.append((name, self.models[name]))
        
        if len(good_models) >= 2:
            print(f"보팅에 사용할 모델: {[name for name, _ in good_models]}")
            
            try:
                voting_ensemble = VotingClassifier(
                    estimators=good_models,
                    voting='soft',
                    n_jobs=1
                )
                
                voting_ensemble.fit(X_train, y_train)
                
                self.models['voting_ensemble'] = voting_ensemble
                print("보팅 앙상블 생성 완료")
                
                return voting_ensemble
            except Exception as e:
                print(f"소프트 보팅 실패: {e}")
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
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, use_optimization=False):
        """모든 모델 훈련"""
        print("전체 모델 훈련 시작")
        
        # 기본 모델들 훈련
        if use_optimization:
            print("LightGBM 하이퍼파라미터 튜닝 중")
            try:
                best_params, _ = self.hyperparameter_optimization(
                    X_train, y_train, 'lightgbm', n_trials=Config.OPTUNA_TRIALS
                )
                self.train_lightgbm(X_train, y_train, X_val, y_val, best_params)
            except Exception as e:
                print(f"LightGBM 튜닝 실패, 기본 파라미터 사용: {e}")
                self.train_lightgbm(X_train, y_train, X_val, y_val)
        else:
            self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        if use_optimization:
            print("XGBoost 하이퍼파라미터 튜닝 중")
            try:
                best_params, _ = self.hyperparameter_optimization(
                    X_train, y_train, 'xgboost', n_trials=Config.OPTUNA_TRIALS
                )
                self.train_xgboost(X_train, y_train, X_val, y_val, best_params)
            except Exception as e:
                print(f"XGBoost 튜닝 실패, 기본 파라미터 사용: {e}")
                self.train_xgboost(X_train, y_train, X_val, y_val)
        else:
            self.train_xgboost(X_train, y_train, X_val, y_val)
        
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # None인 모델 제거
        self.models = {k: v for k, v in self.models.items() if v is not None}
        
        # 교차 검증
        if self.models:
            self.cross_validation(X_train, y_train)
        
            # 앙상블 생성
            voting_ensemble = self.create_voting_ensemble(X_train, y_train)
            
            # 앙상블 성능 검증
            if voting_ensemble is not None:
                try:
                    ensemble_cv = cross_val_score(
                        voting_ensemble, X_train, y_train, 
                        cv=3, scoring=self.scorer, n_jobs=1
                    )
                    
                    self.cv_scores['voting_ensemble'] = {
                        'scores': ensemble_cv,
                        'mean': ensemble_cv.mean(),
                        'std': ensemble_cv.std()
                    }
                    
                    print(f"voting_ensemble CV 점수: {ensemble_cv.mean():.4f} (+/- {ensemble_cv.std() * 2:.4f})")
                    
                    # 앙상블이 더 좋으면 업데이트
                    if ensemble_cv.mean() > self.best_score:
                        self.best_model = voting_ensemble
                        self.best_score = ensemble_cv.mean()
                        print("voting_ensemble이 최고 성능 모델로 선택됨")
                        
                except Exception as e:
                    print(f"voting_ensemble 검증 실패: {e}")
        
        # 최고 모델이 없으면 기본 모델 선택
        if self.best_model is None and self.models:
            # 첫 번째 유효한 모델을 선택
            for model_name, model in self.models.items():
                if model is not None:
                    self.best_model = model
                    print(f"기본 모델로 {model_name} 선택")
                    break
        
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
        
        print("전체 모델 훈련 완료")
        print(f"훈련된 모델 수: {len(self.models)}")
        
        return self.models, self.best_model