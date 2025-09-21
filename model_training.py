# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import f1_score, make_scorer, log_loss
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost 라이브러리가 설치되지 않았습니다")

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_macro_f1, save_model, save_joblib, setup_logging

class FocalLoss:
    """Focal Loss 구현"""
    def __init__(self, alpha=1.0, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true, y_pred):
        """Focal Loss 계산"""
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        ce_loss = -y_true * np.log(y_pred)
        pt = np.where(y_true == 1, y_pred, 1 - y_pred)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return np.mean(focal_loss)

class WeightedVotingClassifier(VotingClassifier):
    """가중치 기반 보팅 분류기"""
    def __init__(self, estimators, voting='soft', weights=None):
        super().__init__(estimators, voting=voting, weights=weights)
        self.model_weights = weights
    
    def predict_proba(self, X):
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when voting='hard'")
        
        avg = np.average(self._collect_probas(X), axis=0, weights=self.model_weights)
        return avg

class ModelTraining:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.cv_scores = {}
        self.ensemble_models = {}
        self.logger = setup_logging()
        self.scorer = make_scorer(f1_score, average='macro')
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
    def _create_cv_strategy(self, X, y, strategy='time_based'):
        """교차 검증 전략 생성"""
        if strategy == 'time_based':
            return TimeSeriesSplit(n_splits=Config.CV_FOLDS, test_size=int(len(X) * 0.15))
        elif strategy == 'stratified':
            return StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        else:
            return TimeSeriesSplit(n_splits=Config.CV_FOLDS, test_size=int(len(X) * 0.15))
    
    def _calculate_class_weights(self, y):
        """클래스 가중치 계산"""
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # 저성능 클래스에 추가 가중치
        low_performance_classes = Config.LOW_PERFORMANCE_CLASSES
        for class_id in low_performance_classes:
            if class_id in class_weight_dict:
                class_weight_dict[class_id] *= 1.5
        
        sample_weights = np.array([class_weight_dict[label] for label in y])
        return sample_weights, class_weight_dict
    
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
            
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            model = lgb.LGBMClassifier(**params)
            
            if X_val is not None and y_val is not None:
                fit_params = {
                    'eval_set': [(X_val, y_val)],
                    'eval_names': ['validation'],
                    'callbacks': [
                        lgb.early_stopping(100, verbose=False),
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
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
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
                    print(f"XGBoost 검증 세트 사용 실패, 단순 훈련: {e}")
                    model.fit(X_train, y_train, sample_weight=sample_weights)
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
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """CatBoost 모델 훈련"""
        if not CATBOOST_AVAILABLE:
            print("CatBoost 라이브러리가 없어 훈련 건너뜀")
            return None
            
        print("CatBoost 모델 훈련 시작")
        
        if params is None:
            params = Config.CAT_PARAMS.copy()
        else:
            params = params.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            model = cb.CatBoostClassifier(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['catboost'] = model
            self.logger.info("CatBoost 모델 훈련 완료")
            return model
            
        except Exception as e:
            print(f"CatBoost 훈련 중 오류 발생: {e}")
            self.logger.error(f"CatBoost 훈련 실패: {e}")
            return None
    
    @timer
    def train_random_forest(self, X_train, y_train, params=None):
        """Random Forest 모델 훈련"""
        print("Random Forest 모델 훈련 시작")
        
        if params is None:
            params = Config.RF_PARAMS.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            params['class_weight'] = class_weight_dict
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
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
            
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            model = GradientBoostingClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
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
            
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            params['class_weight'] = class_weight_dict
            model = ExtraTreesClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['extra_trees'] = model
            self.logger.info("Extra Trees 모델 훈련 완료")
            return model
            
        except Exception as e:
            print(f"Extra Trees 훈련 중 오류 발생: {e}")
            self.logger.error(f"Extra Trees 훈련 실패: {e}")
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
                        'num_leaves': trial.suggest_int('num_leaves', 15, 40),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.7),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.7),
                        'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.2, 0.5),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.2, 0.5),
                        'max_depth': trial.suggest_int('max_depth', 3, 5),
                        'verbose': -1,
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 200, 400),
                        'n_jobs': 1
                    }
                    
                    sample_weights, _ = self._calculate_class_weights(y_train)
                    model = lgb.LGBMClassifier(**params)
                    
                elif model_type == 'xgboost':
                    params = {
                        'objective': 'multi:softprob',
                        'num_class': Config.N_CLASSES,
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                        'max_depth': trial.suggest_int('max_depth', 2, 4),
                        'subsample': trial.suggest_float('subsample', 0.5, 0.7),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.3, 0.6),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 0.6),
                        'gamma': trial.suggest_float('gamma', 0.1, 0.4),
                        'min_child_weight': trial.suggest_int('min_child_weight', 15, 30),
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 150, 300),
                        'n_jobs': 1,
                        'tree_method': 'hist',
                        'verbosity': 0
                    }
                    
                    sample_weights, _ = self._calculate_class_weights(y_train)
                    model = xgb.XGBClassifier(**params)
                
                elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                    params = {
                        'iterations': trial.suggest_int('iterations', 150, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                        'depth': trial.suggest_int('depth', 3, 5),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2, 5),
                        'border_count': trial.suggest_int('border_count', 32, 64),
                        'thread_count': 1,
                        'random_state': Config.RANDOM_STATE,
                        'verbose': False,
                        'loss_function': 'MultiClass',
                        'classes_count': Config.N_CLASSES,
                        'auto_class_weights': 'Balanced'
                    }
                    
                    sample_weights, _ = self._calculate_class_weights(y_train)
                    model = cb.CatBoostClassifier(**params)
                
                # 시간 기반 교차 검증
                cv_strategy = self._create_cv_strategy(X_train, y_train, 'time_based')
                cv_scores = []
                
                for train_idx, val_idx in cv_strategy.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    if model_type in ['lightgbm', 'xgboost']:
                        sw_tr = sample_weights[train_idx]
                        model.fit(X_tr, y_tr, sample_weight=sw_tr)
                    elif model_type == 'catboost':
                        sw_tr = sample_weights[train_idx]
                        model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
                    else:
                        model.fit(X_tr, y_tr)
                    
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro')
                    cv_scores.append(score)
                
                return np.mean(cv_scores)
                
            except Exception as e:
                print(f"옵튜나 시행 중 오류: {e}")
                return 0.0
        
        sampler = TPESampler(seed=Config.RANDOM_STATE)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=Config.OPTUNA_TIMEOUT, show_progress_bar=False)
        
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 점수: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    @timer
    def time_based_cross_validation(self, X_train, y_train):
        """시간 기반 교차 검증 수행"""
        print("시간 기반 교차 검증 시작")
        
        cv_strategy = self._create_cv_strategy(X_train, y_train, 'time_based')
        
        for model_name, model in self.models.items():
            if model is None:
                print(f"{model_name} 건너뜀: 모델이 None")
                continue
                
            print(f"{model_name} 시간 기반 교차 검증 중")
            
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv_strategy, scoring=self.scorer, n_jobs=1
                )
                
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                # 안정성을 고려한 점수 계산
                stability_penalty = std_score * 2
                stability_score = mean_score - stability_penalty
                
                self.cv_scores[model_name] = {
                    'scores': cv_scores,
                    'mean': mean_score,
                    'std': std_score,
                    'stability_score': stability_score
                }
                
                print(f"  {model_name}: {mean_score:.4f} (+/- {std_score * 2:.4f}) -> 안정성: {stability_score:.4f}")
                
            except Exception as e:
                print(f"  {model_name} 검증 실패: {e}")
                continue
        
        # 최고 성능 모델 선택
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
    def create_stacking_ensemble(self, X_train, y_train):
        """스태킹 앙상블 생성"""
        print("스태킹 앙상블 생성 시작")
        
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LogisticRegression
        
        # 성능 기준 모델 선택
        good_models = []
        
        for name, model in self.models.items():
            if (model is not None and name in self.cv_scores and 
                self.cv_scores[name]['stability_score'] >= Config.MIN_CV_SCORE):
                
                score = self.cv_scores[name]['stability_score']
                good_models.append((name, model))
                print(f"{name}: 안정성 점수 {score:.4f}")
        
        if len(good_models) >= 2:
            print(f"스태킹 앙상블에 사용할 모델: {[name for name, _ in good_models]}")
            
            try:
                # 메타 피처 생성
                meta_features = np.zeros((len(X_train), len(good_models)))
                
                kf = KFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    for i, (model_name, model) in enumerate(good_models):
                        # 폴드별 모델 훈련
                        if model_name in ['lightgbm', 'xgboost', 'catboost']:
                            sample_weights, _ = self._calculate_class_weights(y_fold_train)
                            fold_model = type(model)(**model.get_params())
                            
                            if model_name == 'catboost' and CATBOOST_AVAILABLE:
                                fold_model.fit(X_fold_train, y_fold_train, 
                                             sample_weight=sample_weights[train_idx], verbose=False)
                            else:
                                fold_model.fit(X_fold_train, y_fold_train, 
                                             sample_weight=sample_weights[train_idx])
                        else:
                            fold_model = type(model)(**model.get_params())
                            fold_model.fit(X_fold_train, y_fold_train)
                        
                        # 검증 세트 예측
                        if hasattr(fold_model, 'predict_proba'):
                            val_pred = fold_model.predict_proba(X_fold_val)
                            meta_features[val_idx, i] = np.max(val_pred, axis=1)
                        else:
                            val_pred = fold_model.predict(X_fold_val)
                            meta_features[val_idx, i] = val_pred
                
                # 메타 모델 훈련
                meta_model = LogisticRegression(
                    random_state=Config.RANDOM_STATE,
                    class_weight='balanced',
                    max_iter=1000
                )
                
                meta_model.fit(meta_features, y_train)
                
                # 스태킹 앙상블 래퍼 클래스
                class StackingEnsemble:
                    def __init__(self, base_models, meta_model):
                        self.base_models = base_models
                        self.meta_model = meta_model
                    
                    def predict(self, X):
                        meta_features = np.zeros((len(X), len(self.base_models)))
                        
                        for i, (name, model) in enumerate(self.base_models):
                            if hasattr(model, 'predict_proba'):
                                pred = model.predict_proba(X)
                                meta_features[:, i] = np.max(pred, axis=1)
                            else:
                                meta_features[:, i] = model.predict(X)
                        
                        return self.meta_model.predict(meta_features)
                    
                    def predict_proba(self, X):
                        meta_features = np.zeros((len(X), len(self.base_models)))
                        
                        for i, (name, model) in enumerate(self.base_models):
                            if hasattr(model, 'predict_proba'):
                                pred = model.predict_proba(X)
                                meta_features[:, i] = np.max(pred, axis=1)
                            else:
                                meta_features[:, i] = model.predict(X)
                        
                        return self.meta_model.predict_proba(meta_features)
                
                stacking_ensemble = StackingEnsemble(good_models, meta_model)
                
                self.models['stacking_ensemble'] = stacking_ensemble
                self.ensemble_models['stacking'] = stacking_ensemble
                print("스태킹 앙상블 생성 완료")
                
                return stacking_ensemble
                
            except Exception as e:
                print(f"스태킹 앙상블 실패: {e}")
                return self._create_weighted_ensemble(good_models)
        else:
            print("앙상블을 위한 충분한 모델이 없음")
            return None
    
    def _create_weighted_ensemble(self, good_models):
        """가중치 기반 앙상블 생성"""
        print("가중치 기반 앙상블 생성")
        
        try:
            # 성능 기반 가중치 계산
            model_weights = []
            total_score = sum(self.cv_scores[name]['stability_score'] for name, _ in good_models)
            
            for name, model in good_models:
                weight = self.cv_scores[name]['stability_score'] / total_score
                model_weights.append(weight)
                print(f"{name}: 가중치 {weight:.3f}")
            
            weighted_ensemble = WeightedVotingClassifier(
                estimators=good_models,
                voting='soft',
                weights=model_weights
            )
            
            # 샘플 데이터로 앙상블 훈련 검증
            sample_size = min(5000, len(X_train)) if 'X_train' in locals() else 1000
            print("가중치 앙상블 생성 완료")
            
            return weighted_ensemble
            
        except Exception as e:
            print(f"가중치 앙상블 실패: {e}")
            return None
    
    @timer
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, use_optimization=True):
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
            
            # CatBoost 튜닝
            if CATBOOST_AVAILABLE:
                try:
                    best_params, _ = self.hyperparameter_optimization(
                        X_train, y_train, 'catboost', n_trials=Config.OPTUNA_TRIALS
                    )
                    self.train_catboost(X_train, y_train, X_val, y_val, best_params)
                except Exception as e:
                    print(f"CatBoost 튜닝 실패, 기본 파라미터 사용: {e}")
                    self.train_catboost(X_train, y_train, X_val, y_val)
        else:
            self.train_lightgbm(X_train, y_train, X_val, y_val)
            self.train_xgboost(X_train, y_train, X_val, y_val)
            if CATBOOST_AVAILABLE:
                self.train_catboost(X_train, y_train, X_val, y_val)
        
        # 기타 모델들 훈련
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        self.train_extra_trees(X_train, y_train)
        
        # None인 모델 제거
        self.models = {k: v for k, v in self.models.items() if v is not None}
        
        # 시간 기반 교차 검증
        if self.models:
            self.time_based_cross_validation(X_train, y_train)
            
            # 스태킹 앙상블 생성
            ensemble = self.create_stacking_ensemble(X_train, y_train)
            
            # 앙상블 성능 검증
            if ensemble is not None:
                try:
                    cv_strategy = self._create_cv_strategy(X_train, y_train, 'time_based')
                    ensemble_cv = cross_val_score(
                        ensemble, X_train, y_train, 
                        cv=cv_strategy, scoring=self.scorer, n_jobs=1
                    )
                    
                    ensemble_score = ensemble_cv.mean()
                    ensemble_std = ensemble_cv.std()
                    stability_score = ensemble_score - (ensemble_std * 2)
                    
                    ensemble_name = 'stacking_ensemble'
                    
                    self.cv_scores[ensemble_name] = {
                        'scores': ensemble_cv,
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
            try:
                save_model(self.best_model, Config.MODEL_FILE)
                print(f"최고 모델 저장: {type(self.best_model).__name__}")
            except Exception as e:
                print(f"모델 저장 실패: {e}")
        
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
        
        stable_models = {}
        
        # 보수적인 파라미터로 모델 훈련
        lgbm_params = {
            'objective': 'multiclass',
            'num_class': Config.N_CLASSES,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 20,
            'learning_rate': 0.02,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 3,
            'min_child_samples': 40,
            'reg_alpha': 0.4,
            'reg_lambda': 0.4,
            'max_depth': 3,
            'verbose': -1,
            'random_state': Config.RANDOM_STATE,
            'n_estimators': 200,
            'n_jobs': Config.N_JOBS,
            'is_unbalance': True
        }
        
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': Config.N_CLASSES,
            'learning_rate': 0.015,
            'max_depth': 2,
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'gamma': 0.3,
            'min_child_weight': 25,
            'random_state': Config.RANDOM_STATE,
            'n_estimators': 200,
            'n_jobs': Config.N_JOBS,
            'tree_method': 'hist',
            'verbosity': 0
        }
        
        # LightGBM
        try:
            lgbm_model = self.train_lightgbm(X_train, y_train, params=lgbm_params)
            if lgbm_model is not None:
                stable_models['stable_lightgbm'] = lgbm_model
        except Exception as e:
            print(f"안정적인 LightGBM 훈련 실패: {e}")
        
        # XGBoost
        try:
            xgb_model = self.train_xgboost(X_train, y_train, params=xgb_params)
            if xgb_model is not None:
                stable_models['stable_xgboost'] = xgb_model
        except Exception as e:
            print(f"안정적인 XGBoost 훈련 실패: {e}")
        
        # Random Forest
        try:
            rf_params = Config.RF_PARAMS.copy()
            rf_params.update({
                'n_estimators': 150,
                'max_depth': 5,
                'min_samples_split': 25,
                'min_samples_leaf': 12,
                'max_features': 0.5
            })
            rf_model = self.train_random_forest(X_train, y_train, params=rf_params)
            if rf_model is not None:
                stable_models['stable_random_forest'] = rf_model
        except Exception as e:
            print(f"안정적인 Random Forest 훈련 실패: {e}")
        
        # 교차 검증으로 최고 모델 선택
        if stable_models:
            print(f"안정적인 모델 교차 검증 시작")
            cv_strategy = self._create_cv_strategy(X_train, y_train, 'time_based')
            
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
                try:
                    save_model(best_stable_model, Config.MODEL_FILE)
                except Exception as e:
                    print(f"모델 저장 실패: {e}")
                return stable_models, best_stable_model
        
        print("안정적인 모델 훈련 실패, 기존 모델 사용")
        return {}, self.best_model