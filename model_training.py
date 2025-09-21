# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, TimeSeriesSplit
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
        self.ensemble_models = {}
        self.logger = setup_logging()
        self.scorer = make_scorer(f1_score, average='macro')
        
    def _create_cv_strategy(self, X, y, strategy='stratified'):
        """교차 검증 전략 생성"""
        if strategy == 'stratified':
            return StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        elif strategy == 'time_series':
            return TimeSeriesSplit(n_splits=Config.CV_FOLDS)
        else:
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
            params.pop('class_weight', None)
            
            # 클래스 가중치 계산
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([class_weight_dict[y] for y in y_train])
            
            model = lgb.LGBMClassifier(**params)
            
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
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([class_weight_dict[y] for y in y_train])
            
            model = xgb.XGBClassifier(**params)
            
            if X_val is not None and y_val is not None:
                try:
                    model.fit(
                        X_train, y_train, 
                        sample_weight=sample_weights,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                except (TypeError, ValueError) as e:
                    print(f"XGBoost 새 방식 실패, 이전 방식 시도: {e}")
                    try:
                        model.fit(
                            X_train, y_train, 
                            sample_weight=sample_weights,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    except Exception as e2:
                        print(f"XGBoost 검증 세트 사용 실패, 단순 훈련: {e2}")
                        model.fit(X_train, y_train, sample_weight=sample_weights)
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
                    'n_jobs': 1,
                    'verbosity': 0
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
    def train_gradient_boosting(self, X_train, y_train, params=None):
        """Gradient Boosting 모델 훈련"""
        print("Gradient Boosting 모델 훈련 시작")
        
        try:
            if params is None:
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
    def train_extra_trees(self, X_train, y_train, params=None):
        """Extra Trees 모델 훈련"""
        print("Extra Trees 모델 훈련 시작")
        
        try:
            if params is None:
                params = Config.ET_PARAMS.copy()
            
            params['class_weight'] = 'balanced'
            model = ExtraTreesClassifier(**params)
            model.fit(X_train, y_train)
            
            self.models['extra_trees'] = model
            self.logger.info("Extra Trees 모델 훈련 완료")
            return model
            
        except Exception as e:
            print(f"Extra Trees 훈련 중 오류 발생: {e}")
            self.logger.error(f"Extra Trees 훈련 실패: {e}")
            return None
    
    @timer
    def hyperparameter_optimization(self, X_train, y_train, model_type='lightgbm', n_trials=10):
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
                        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.8),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.8),
                        'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.05, 0.3),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.05, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 4, 8),
                        'verbose': -1,
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 200, 400),
                        'n_jobs': 1
                    }
                    
                    classes = np.unique(y_train)
                    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                    class_weight_dict = dict(zip(classes, class_weights))
                    sample_weights = np.array([class_weight_dict[y] for y in y_train])
                    
                    model = lgb.LGBMClassifier(**params)
                    
                elif model_type == 'xgboost':
                    params = {
                        'objective': 'multi:softprob',
                        'num_class': Config.N_CLASSES,
                        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
                        'max_depth': trial.suggest_int('max_depth', 3, 6),
                        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.4),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 0.4),
                        'gamma': trial.suggest_float('gamma', 0, 0.3),
                        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 150, 300),
                        'n_jobs': 1,
                        'tree_method': 'hist',
                        'verbosity': 0
                    }
                    
                    classes = np.unique(y_train)
                    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                    class_weight_dict = dict(zip(classes, class_weights))
                    sample_weights = np.array([class_weight_dict[y] for y in y_train])
                    
                    model = xgb.XGBClassifier(**params)
                
                # 3-fold 교차 검증
                cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
                cv_scores = []
                
                for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
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
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
        
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=Config.OPTUNA_TIMEOUT, show_progress_bar=False)
        
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 점수: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    @timer
    def stable_cross_validation(self, X_train, y_train):
        """안정적인 교차 검증 수행"""
        print("안정적인 교차 검증 시작")
        
        # 시간 기반과 계층화 검증 모두 수행
        strategies = {
            'stratified': StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE),
            'time_based': TimeSeriesSplit(n_splits=3)
        }
        
        for model_name, model in self.models.items():
            if model is None:
                print(f"{model_name} 건너뜀: 모델이 None")
                continue
                
            print(f"{model_name} 안정적인 교차 검증 중")
            
            model_scores = {}
            
            for strategy_name, cv_strategy in strategies.items():
                try:
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_strategy, scoring=self.scorer, n_jobs=1
                    )
                    
                    model_scores[strategy_name] = {
                        'scores': cv_scores,
                        'mean': cv_scores.mean(),
                        'std': cv_scores.std()
                    }
                    
                    print(f"  {strategy_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                    
                except Exception as e:
                    print(f"  {strategy_name} 검증 실패: {e}")
                    continue
            
            if model_scores:
                # 안정성을 고려한 점수 계산 (표준편차 페널티 적용)
                stability_scores = []
                for strategy_scores in model_scores.values():
                    mean_score = strategy_scores['mean']
                    std_score = strategy_scores['std']
                    # 표준편차가 클수록 페널티
                    stability_score = mean_score - (std_score * 2)
                    stability_scores.append(stability_score)
                
                final_score = np.mean(stability_scores)
                overall_std = np.std([scores['mean'] for scores in model_scores.values()])
                
                self.cv_scores[model_name] = {
                    'scores': model_scores,
                    'mean': final_score,
                    'std': overall_std,
                    'stability_score': final_score
                }
                
                print(f"{model_name} 최종 안정성 점수: {final_score:.4f}")
        
        # 최고 성능 모델 선택 (안정성 고려)
        if self.cv_scores:
            best_model_name = max(self.cv_scores.keys(), key=lambda x: self.cv_scores[x]['stability_score'])
            if best_model_name in self.models and self.models[best_model_name] is not None:
                self.best_model = self.models[best_model_name]
                self.best_score = self.cv_scores[best_model_name]['stability_score']
                
                print(f"최고 안정성 모델: {best_model_name}")
                print(f"최고 안정성 점수: {self.best_score:.4f}")
        
        gc.collect()
        return self.cv_scores
    
    @timer
    def create_weighted_ensemble(self, X_train, y_train):
        """가중치 기반 앙상블 생성"""
        print("가중치 기반 앙상블 생성 시작")
        
        # 성능 기준 모델 선택
        good_models = []
        model_weights = []
        
        for name, model in self.models.items():
            if (model is not None and name in self.cv_scores and 
                self.cv_scores[name]['stability_score'] >= Config.MIN_CV_SCORE):
                
                score = self.cv_scores[name]['stability_score']
                weight = score / sum(cv_info['stability_score'] for cv_info in self.cv_scores.values() 
                                   if cv_info['stability_score'] >= Config.MIN_CV_SCORE)
                
                good_models.append((name, model))
                model_weights.append(weight)
                
                print(f"{name}: 점수 {score:.4f}, 가중치 {weight:.3f}")
        
        if len(good_models) >= 2:
            print(f"앙상블에 사용할 모델: {[name for name, _ in good_models]}")
            
            try:
                # 가중치 기반 소프트 보팅
                class WeightedVotingClassifier(VotingClassifier):
                    def __init__(self, estimators, voting='soft', weights=None):
                        super().__init__(estimators, voting=voting, weights=weights)
                        self.model_weights = weights
                    
                    def predict_proba(self, X):
                        if self.voting == 'hard':
                            raise AttributeError("predict_proba is not available when voting='hard'")
                        
                        avg = np.average(self._collect_probas(X), axis=0, weights=self.model_weights)
                        return avg
                
                weighted_ensemble = WeightedVotingClassifier(
                    estimators=good_models,
                    voting='soft',
                    weights=model_weights
                )
                
                weighted_ensemble.fit(X_train, y_train)
                
                self.models['weighted_ensemble'] = weighted_ensemble
                self.ensemble_models['weighted'] = weighted_ensemble
                print("가중치 기반 앙상블 생성 완료")
                
                return weighted_ensemble
                
            except Exception as e:
                print(f"가중치 앙상블 실패: {e}")
                
                # 기본 보팅 앙상블로 대체
                try:
                    voting_ensemble = VotingClassifier(
                        estimators=good_models,
                        voting='soft',
                        n_jobs=1
                    )
                    voting_ensemble.fit(X_train, y_train)
                    
                    self.models['voting_ensemble'] = voting_ensemble
                    print("기본 보팅 앙상블 생성 완료")
                    return voting_ensemble
                    
                except Exception as e2:
                    print(f"기본 보팅도 실패: {e2}")
                    return None
        else:
            print("앙상블을 위한 충분한 모델이 없음")
            return None
    
    @timer
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, use_optimization=False):
        """모든 모델 훈련"""
        print("전체 모델 훈련 시작")
        
        # 기본 모델들 훈련
        if use_optimization:
            print("하이퍼파라미터 튜닝 적용")
            
            # LightGBM 튜닝
            try:
                best_params, _ = self.hyperparameter_optimization(
                    X_train, y_train, 'lightgbm', n_trials=Config.OPTUNA_TRIALS
                )
                self.train_lightgbm(X_train, y_train, X_val, y_val, best_params)
            except Exception as e:
                print(f"LightGBM 튜닝 실패, 기본 파라미터 사용: {e}")
                self.train_lightgbm(X_train, y_train, X_val, y_val)
            
            # XGBoost 튜닝
            try:
                best_params, _ = self.hyperparameter_optimization(
                    X_train, y_train, 'xgboost', n_trials=Config.OPTUNA_TRIALS
                )
                self.train_xgboost(X_train, y_train, X_val, y_val, best_params)
            except Exception as e:
                print(f"XGBoost 튜닝 실패, 기본 파라미터 사용: {e}")
                self.train_xgboost(X_train, y_train, X_val, y_val)
        else:
            self.train_lightgbm(X_train, y_train, X_val, y_val)
            self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # 기타 모델들 훈련
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        self.train_extra_trees(X_train, y_train)
        
        # None인 모델 제거
        self.models = {k: v for k, v in self.models.items() if v is not None}
        
        # 안정적인 교차 검증
        if self.models:
            self.stable_cross_validation(X_train, y_train)
            
            # 가중치 기반 앙상블 생성
            ensemble = self.create_weighted_ensemble(X_train, y_train)
            
            # 앙상블 성능 검증
            if ensemble is not None:
                try:
                    ensemble_cv = cross_val_score(
                        ensemble, X_train, y_train, 
                        cv=3, scoring=self.scorer, n_jobs=1
                    )
                    
                    ensemble_score = ensemble_cv.mean()
                    ensemble_std = ensemble_cv.std()
                    stability_score = ensemble_score - (ensemble_std * 2)
                    
                    ensemble_name = 'weighted_ensemble' if 'weighted_ensemble' in self.models else 'voting_ensemble'
                    
                    self.cv_scores[ensemble_name] = {
                        'scores': {'ensemble': {'scores': ensemble_cv, 'mean': ensemble_score, 'std': ensemble_std}},
                        'mean': ensemble_score,
                        'std': ensemble_std,
                        'stability_score': stability_score
                    }
                    
                    print(f"{ensemble_name} 안정성 점수: {stability_score:.4f}")
                    
                    # 앙상블이 더 안정적이면 업데이트
                    if stability_score > self.best_score:
                        self.best_model = ensemble
                        self.best_score = stability_score
                        print(f"{ensemble_name}이 최고 안정성 모델로 선택됨")
                        
                except Exception as e:
                    print(f"앙상블 검증 실패: {e}")
        
        # 최고 모델이 없으면 기본 모델 선택
        if self.best_model is None and self.models:
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
                    'stability_score': scores.get('stability_score', scores['mean'])
                })
            
            cv_results_df = pd.DataFrame(cv_results_data)
            
            from utils import save_results
            save_results(cv_results_df, Config.CV_RESULTS_FILE)
        
        gc.collect()
        
        print("전체 모델 훈련 완료")
        print(f"훈련된 모델 수: {len(self.models)}")
        
        return self.models, self.best_model
    
    @timer
    def train_stable_models(self, X_train, y_train):
        """안정적인 모델 훈련"""
        print("안정적인 모델 훈련 시작")
        
        # 안정적인 파라미터
        stable_params = Config.get_stable_params()
        
        stable_models = {}
        
        # LightGBM
        try:
            lgbm_model = self.train_lightgbm(X_train, y_train, params=stable_params['lgbm'])
            if lgbm_model is not None:
                stable_models['stable_lightgbm'] = lgbm_model
        except Exception as e:
            print(f"안정적인 LightGBM 훈련 실패: {e}")
        
        # XGBoost
        try:
            xgb_model = self.train_xgboost(X_train, y_train, params=stable_params['xgb'])
            if xgb_model is not None:
                stable_models['stable_xgboost'] = xgb_model
        except Exception as e:
            print(f"안정적인 XGBoost 훈련 실패: {e}")
        
        # Random Forest
        try:
            rf_model = self.train_random_forest(X_train, y_train, params=stable_params['rf'])
            if rf_model is not None:
                stable_models['stable_random_forest'] = rf_model
        except Exception as e:
            print(f"안정적인 Random Forest 훈련 실패: {e}")
        
        # Gradient Boosting
        try:
            gb_model = self.train_gradient_boosting(X_train, y_train, params=stable_params['gb'])
            if gb_model is not None:
                stable_models['stable_gradient_boosting'] = gb_model
        except Exception as e:
            print(f"안정적인 Gradient Boosting 훈련 실패: {e}")
        
        # 교차 검증으로 최고 모델 선택
        if stable_models:
            print(f"안정적인 모델 교차 검증 시작")
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
            
            best_score = 0
            best_stable_model = None
            
            for model_name, model in stable_models.items():
                try:
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_strategy, scoring=self.scorer, n_jobs=1
                    )
                    
                    mean_score = cv_scores.mean()
                    std_score = cv_scores.std()
                    stability_score = mean_score - (std_score * 2)
                    
                    print(f"{model_name} 안정성 점수: {stability_score:.4f} (평균: {mean_score:.4f}, 표준편차: {std_score:.4f})")
                    
                    if stability_score > best_score:
                        best_score = stability_score
                        best_stable_model = model
                        
                except Exception as e:
                    print(f"{model_name} 교차 검증 실패: {e}")
                    continue
            
            if best_stable_model is not None:
                print(f"최고 안정적인 모델 선택 완료, 안정성 점수: {best_score:.4f}")
                save_model(best_stable_model, Config.MODEL_FILE)
                return stable_models, best_stable_model
        
        print("안정적인 모델 훈련 실패, 기존 모델 사용")
        return {}, self.best_model