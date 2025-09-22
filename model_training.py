# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer, classification_report
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

class MacroF1Scorer:
    """Macro F1 직접 최적화를 위한 커스텀 스코어러"""
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
    
    def __call__(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

class WeightCalculator:
    """동적 클래스 가중치 계산"""
    
    @staticmethod
    def compute_focal_weights(y, alpha=1.0, gamma=2.0):
        """Focal Loss 기반 가중치 계산"""
        class_counts = np.bincount(y, minlength=Config.N_CLASSES)
        total_samples = len(y)
        
        # 빈도 기반 가중치
        freq_weights = total_samples / (Config.N_CLASSES * class_counts + 1e-8)
        
        # Focal 가중치 적용
        focal_weights = alpha * np.power(1 - (class_counts / total_samples), gamma)
        
        # 결합된 가중치
        combined_weights = freq_weights * focal_weights
        
        # 정규화
        combined_weights = combined_weights / np.mean(combined_weights)
        
        return dict(enumerate(combined_weights))
    
    @staticmethod
    def compute_balanced_weights(y):
        """균형 가중치 계산"""
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, class_weights))

class ModelTraining:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.cv_scores = {}
        self.ensemble_models = {}
        self.logger = setup_logging()
        self.scorer = MacroF1Scorer(Config.N_CLASSES)
        self.weight_calculator = WeightCalculator()
        
    def _create_cv_strategy(self, X, y):
        """Stratified K-Fold 교차 검증 전략 생성"""
        return StratifiedKFold(
            n_splits=Config.CV_FOLDS, 
            shuffle=True, 
            random_state=Config.RANDOM_STATE
        )
    
    def _calculate_class_weights(self, y, method='focal'):
        """클래스 가중치 계산"""
        if method == 'focal':
            class_weight_dict = self.weight_calculator.compute_focal_weights(
                y, Config.FOCAL_LOSS_ALPHA, Config.FOCAL_LOSS_GAMMA
            )
        else:
            class_weight_dict = self.weight_calculator.compute_balanced_weights(y)
        
        sample_weights = np.array([class_weight_dict[label] for label in y])
        return sample_weights, class_weight_dict
    
    @timer
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """LightGBM 모델 훈련"""
        print("LightGBM 모델 훈련 시작")
        
        if params is None:
            params = Config.LGBM_PARAMS.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            # 클래스 가중치를 파라미터에 적용
            params['class_weight'] = class_weight_dict
            
            model = lgb.LGBMClassifier(**params)
            
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=eval_set,
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                )
            else:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['lightgbm'] = model
            self.logger.info("LightGBM 모델 훈련 완료")
            return model
            
        except Exception as e:
            print(f"LightGBM 훈련 중 오류: {e}")
            self.logger.error(f"LightGBM 훈련 실패: {e}")
            return None
    
    @timer
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """XGBoost 모델 훈련"""
        print("XGBoost 모델 훈련 시작")
        
        if params is None:
            params = Config.XGB_PARAMS.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            model = xgb.XGBClassifier(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['xgboost'] = model
            self.logger.info("XGBoost 모델 훈련 완료")
            return model
            
        except Exception as e:
            print(f"XGBoost 훈련 중 오류: {e}")
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
                model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            
            self.models['catboost'] = model
            self.logger.info("CatBoost 모델 훈련 완료")
            return model
            
        except Exception as e:
            print(f"CatBoost 훈련 중 오류: {e}")
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
            print(f"Random Forest 훈련 중 오류: {e}")
            self.logger.error(f"Random Forest 훈련 실패: {e}")
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
            print(f"Extra Trees 훈련 중 오류: {e}")
            self.logger.error(f"Extra Trees 훈련 실패: {e}")
            return None
    
    @timer
    def hyperparameter_optimization(self, X_train, y_train, model_type='lightgbm', n_trials=100):
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
                        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'verbose': -1,
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                        'n_jobs': 1
                    }
                    
                    sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
                    params['class_weight'] = class_weight_dict
                    model = lgb.LGBMClassifier(**params)
                    
                elif model_type == 'xgboost':
                    params = {
                        'objective': 'multi:softprob',
                        'num_class': Config.N_CLASSES,
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                        'n_jobs': 1,
                        'tree_method': 'hist',
                        'verbosity': 0
                    }
                    
                    sample_weights, _ = self._calculate_class_weights(y_train)
                    model = xgb.XGBClassifier(**params)
                
                elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                    params = {
                        'iterations': trial.suggest_int('iterations', 200, 800),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'depth': trial.suggest_int('depth', 3, 10),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                        'border_count': trial.suggest_int('border_count', 32, 255),
                        'thread_count': 1,
                        'random_state': Config.RANDOM_STATE,
                        'verbose': False,
                        'loss_function': 'MultiClass',
                        'classes_count': Config.N_CLASSES,
                        'auto_class_weights': 'Balanced'
                    }
                    
                    sample_weights, _ = self._calculate_class_weights(y_train)
                    model = cb.CatBoostClassifier(**params)
                
                # Stratified K-Fold 교차 검증
                cv_strategy = self._create_cv_strategy(X_train, y_train)
                cv_scores = []
                
                for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    if model_type in ['lightgbm', 'xgboost']:
                        sw_tr, _ = self._calculate_class_weights(y_tr)
                        model.fit(X_tr, y_tr, sample_weight=sw_tr)
                    elif model_type == 'catboost':
                        sw_tr, _ = self._calculate_class_weights(y_tr)
                        model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
                    else:
                        model.fit(X_tr, y_tr)
                    
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro', zero_division=0)
                    cv_scores.append(score)
                
                return np.mean(cv_scores)
                
            except Exception as e:
                print(f"튜닝 시행 중 오류: {e}")
                return 0.0
        
        sampler = TPESampler(seed=Config.RANDOM_STATE)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=Config.OPTUNA_TIMEOUT, show_progress_bar=False)
        
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 점수: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    @timer
    def cross_validate_models(self, X_train, y_train):
        """모델 교차 검증 수행"""
        print("모델 교차 검증 시작")
        
        cv_strategy = self._create_cv_strategy(X_train, y_train)
        
        for model_name, model in self.models.items():
            if model is None:
                print(f"{model_name} 건너뜀: 모델이 None")
                continue
                
            print(f"{model_name} 교차 검증 중")
            
            try:
                # 커스텀 스코어링 함수 사용
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_strategy,
                    scoring=make_scorer(self.scorer),
                    n_jobs=1
                )
                
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                self.cv_scores[model_name] = {
                    'scores': cv_scores,
                    'mean': mean_score,
                    'std': std_score,
                    'stability': mean_score - std_score  # 안정성 지표
                }
                
                print(f"  {model_name}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
                
            except Exception as e:
                print(f"  {model_name} 검증 실패: {e}")
                continue
        
        # 최고 성능 모델 선택
        if self.cv_scores:
            best_model_name = max(self.cv_scores.keys(), key=lambda x: self.cv_scores[x]['stability'])
            if best_model_name in self.models and self.models[best_model_name] is not None:
                self.best_model = self.models[best_model_name]
                self.best_score = self.cv_scores[best_model_name]['stability']
                
                print(f"최고 안정성 모델: {best_model_name}")
                print(f"안정성 점수: {self.best_score:.4f}")
        
        return self.cv_scores
    
    @timer
    def create_ensemble(self, X_train, y_train):
        """앙상블 모델 생성"""
        print("앙상블 모델 생성 시작")
        
        # 성능 기준으로 모델 선택
        good_models = []
        min_score = 0.55  # 최소 성능 기준
        
        for name, model in self.models.items():
            if (model is not None and name in self.cv_scores and 
                self.cv_scores[name]['stability'] >= min_score):
                
                score = self.cv_scores[name]['stability']
                good_models.append((name, model, score))
                print(f"{name}: 안정성 점수 {score:.4f}")
        
        if len(good_models) >= 2:
            print(f"앙상블에 사용할 모델: {[name for name, _, _ in good_models]}")
            
            try:
                # 성능 기반 가중치 계산
                total_score = sum(score for _, _, score in good_models)
                model_weights = [score / total_score for _, _, score in good_models]
                
                # VotingClassifier 생성
                estimators = [(name, model) for name, model, _ in good_models]
                
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=model_weights
                )
                
                # 앙상블 훈련
                sample_weights, _ = self._calculate_class_weights(y_train)
                ensemble.fit(X_train, y_train, sample_weight=sample_weights)
                
                self.models['ensemble'] = ensemble
                self.ensemble_models['voting'] = ensemble
                
                print("앙상블 모델 생성 완료")
                return ensemble
                
            except Exception as e:
                print(f"앙상블 생성 실패: {e}")
                return None
        else:
            print("앙상블을 위한 충분한 모델이 없음")
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
                print(f"LightGBM 튜닝 실패: {e}")
                self.train_lightgbm(X_train, y_train, X_val, y_val)
            
            # XGBoost 튜닝
            try:
                best_params, _ = self.hyperparameter_optimization(
                    X_train, y_train, 'xgboost', n_trials=Config.OPTUNA_TRIALS
                )
                self.train_xgboost(X_train, y_train, X_val, y_val, best_params)
            except Exception as e:
                print(f"XGBoost 튜닝 실패: {e}")
                self.train_xgboost(X_train, y_train, X_val, y_val)
            
            # CatBoost 튜닝
            if CATBOOST_AVAILABLE:
                try:
                    best_params, _ = self.hyperparameter_optimization(
                        X_train, y_train, 'catboost', n_trials=Config.OPTUNA_TRIALS
                    )
                    self.train_catboost(X_train, y_train, X_val, y_val, best_params)
                except Exception as e:
                    print(f"CatBoost 튜닝 실패: {e}")
                    self.train_catboost(X_train, y_train, X_val, y_val)
        else:
            self.train_lightgbm(X_train, y_train, X_val, y_val)
            self.train_xgboost(X_train, y_train, X_val, y_val)
            if CATBOOST_AVAILABLE:
                self.train_catboost(X_train, y_train, X_val, y_val)
        
        # 추가 모델들 훈련
        self.train_random_forest(X_train, y_train)
        self.train_extra_trees(X_train, y_train)
        
        # None인 모델 제거
        self.models = {k: v for k, v in self.models.items() if v is not None}
        
        # 교차 검증
        if self.models:
            self.cross_validate_models(X_train, y_train)
            
            # 앙상블 생성
            ensemble = self.create_ensemble(X_train, y_train)
            
            # 앙상블 성능 검증
            if ensemble is not None:
                try:
                    cv_strategy = self._create_cv_strategy(X_train, y_train)
                    ensemble_cv = cross_val_score(
                        ensemble, X_train, y_train,
                        cv=cv_strategy,
                        scoring=make_scorer(self.scorer),
                        n_jobs=1
                    )
                    
                    ensemble_score = ensemble_cv.mean()
                    ensemble_std = ensemble_cv.std()
                    stability_score = ensemble_score - ensemble_std
                    
                    self.cv_scores['ensemble'] = {
                        'scores': ensemble_cv,
                        'mean': ensemble_score,
                        'std': ensemble_std,
                        'stability': stability_score
                    }
                    
                    print(f"앙상블 안정성 점수: {stability_score:.4f}")
                    
                    # 앙상블이 더 안정적이면 업데이트
                    if stability_score > self.best_score:
                        self.best_model = ensemble
                        self.best_score = stability_score
                        print("앙상블이 최고 모델로 선택됨")
                        
                except Exception as e:
                    print(f"앙상블 검증 실패: {e}")
        
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
                    'stability': scores['stability']
                })
            
            cv_results_df = pd.DataFrame(cv_results_data)
            
            try:
                cv_results_df.to_csv(Config.CV_RESULTS_FILE, index=False)
                print(f"CV 결과 저장: {Config.CV_RESULTS_FILE}")
            except Exception as e:
                print(f"CV 결과 저장 실패: {e}")
        
        gc.collect()
        
        print("전체 모델 훈련 완료")
        print(f"훈련된 모델 수: {len(self.models)}")
        
        return self.models, self.best_model
    
    def get_feature_importance(self):
        """피처 중요도 반환"""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if model is None:
                continue
                
            try:
                if hasattr(model, 'feature_importances_'):
                    importance_dict[model_name] = model.feature_importances_
                elif hasattr(model, 'get_feature_importance'):
                    importance_dict[model_name] = model.get_feature_importance()
            except Exception as e:
                print(f"{model_name} 피처 중요도 추출 실패: {e}")
        
        return importance_dict