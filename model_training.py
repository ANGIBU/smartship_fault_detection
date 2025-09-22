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
    print("CatBoost library not installed")

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_macro_f1, save_model, save_joblib, setup_logging

class MacroF1Scorer:
    """Custom scorer for direct Macro F1 optimization"""
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
    
    def __call__(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

class WeightCalculator:
    """Dynamic class weight calculation"""
    
    @staticmethod
    def compute_focal_weights(y, alpha=1.0, gamma=2.0):
        """Calculate Focal Loss based weights"""
        class_counts = np.bincount(y, minlength=Config.N_CLASSES)
        total_samples = len(y)
        
        # Frequency-based weights
        freq_weights = total_samples / (Config.N_CLASSES * class_counts + 1e-8)
        
        # Apply focal weights
        focal_weights = alpha * np.power(1 - (class_counts / total_samples), gamma)
        
        # Combined weights
        combined_weights = freq_weights * focal_weights
        
        # Normalize
        combined_weights = combined_weights / np.mean(combined_weights)
        
        return dict(enumerate(combined_weights))
    
    @staticmethod
    def compute_balanced_weights(y):
        """Calculate balanced weights"""
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
        """Create Stratified K-Fold cross-validation strategy"""
        return StratifiedKFold(
            n_splits=Config.OPTUNA_CV_FOLDS, 
            shuffle=True, 
            random_state=Config.RANDOM_STATE
        )
    
    def _calculate_class_weights(self, y, method='focal'):
        """Calculate class weights"""
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
        """Train LightGBM model"""
        print("Starting LightGBM model training")
        
        if params is None:
            params = Config.LGBM_PARAMS.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            # Apply class weights to parameters
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
            self.logger.info("LightGBM model training completed")
            return model
            
        except Exception as e:
            print(f"Error during LightGBM training: {e}")
            self.logger.error(f"LightGBM training failed: {e}")
            return None
    
    @timer
    def train_quick_lightgbm(self, X_train, y_train):
        """Train LightGBM model for quick mode"""
        print("Starting quick LightGBM model training")
        
        try:
            params = Config.get_model_params('quick_lightgbm')
            
            # Simple balanced weights for quick mode
            if Config.USE_CLASS_WEIGHTS:
                sample_weights, class_weight_dict = self._calculate_class_weights(y_train, method='balanced')
                params['class_weight'] = class_weight_dict
            else:
                sample_weights = None
            
            model = lgb.LGBMClassifier(**params)
            
            if sample_weights is not None:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
            
            self.models['quick_lightgbm'] = model
            self.best_model = model
            
            print("Quick LightGBM model training completed")
            return model
            
        except Exception as e:
            print(f"Error during quick LightGBM training: {e}")
            return None
    
    @timer
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """Train XGBoost model"""
        print("Starting XGBoost model training")
        
        if params is None:
            params = Config.XGB_PARAMS.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            model = xgb.XGBClassifier(**params)
            
            if X_val is not None and y_val is not None:
                # Updated XGBoost early stopping usage
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['xgboost'] = model
            self.logger.info("XGBoost model training completed")
            return model
            
        except Exception as e:
            print(f"Error during XGBoost training: {e}")
            self.logger.error(f"XGBoost training failed: {e}")
            return None
    
    @timer
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """Train CatBoost model"""
        if not CATBOOST_AVAILABLE:
            print("CatBoost library not available, skipping training")
            return None
            
        print("Starting CatBoost model training")
        
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
            self.logger.info("CatBoost model training completed")
            return model
            
        except Exception as e:
            print(f"Error during CatBoost training: {e}")
            self.logger.error(f"CatBoost training failed: {e}")
            return None
    
    @timer
    def train_random_forest(self, X_train, y_train, params=None):
        """Train Random Forest model"""
        print("Starting Random Forest model training")
        
        if params is None:
            params = Config.RF_PARAMS.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            params['class_weight'] = class_weight_dict
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['random_forest'] = model
            self.logger.info("Random Forest model training completed")
            return model
            
        except Exception as e:
            print(f"Error during Random Forest training: {e}")
            self.logger.error(f"Random Forest training failed: {e}")
            return None
    
    @timer
    def train_extra_trees(self, X_train, y_train, params=None):
        """Train Extra Trees model"""
        print("Starting Extra Trees model training")
        
        try:
            if params is None:
                params = Config.ET_PARAMS.copy()
            
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
            
            params['class_weight'] = class_weight_dict
            model = ExtraTreesClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['extra_trees'] = model
            self.logger.info("Extra Trees model training completed")
            return model
            
        except Exception as e:
            print(f"Error during Extra Trees training: {e}")
            self.logger.error(f"Extra Trees training failed: {e}")
            return None
    
    @timer
    def hyperparameter_optimization(self, X_train, y_train, model_type='lightgbm', n_trials=25):
        """Hyperparameter tuning"""
        print(f"Starting {model_type} hyperparameter tuning")
        
        def objective(trial):
            try:
                if model_type == 'lightgbm':
                    params = {
                        'objective': 'multiclass',
                        'num_class': Config.N_CLASSES,
                        'metric': 'multi_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': trial.suggest_int('num_leaves', 31, 70),
                        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.9),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 3),
                        'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
                        'max_depth': trial.suggest_int('max_depth', 4, 8),
                        'verbose': -1,
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 300, 700),
                        'n_jobs': 1
                    }
                    
                    sample_weights, class_weight_dict = self._calculate_class_weights(y_train)
                    params['class_weight'] = class_weight_dict
                    model = lgb.LGBMClassifier(**params)
                    
                elif model_type == 'xgboost':
                    params = {
                        'objective': 'multi:softprob',
                        'num_class': Config.N_CLASSES,
                        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
                        'max_depth': trial.suggest_int('max_depth', 4, 8),
                        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
                        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                        'random_state': Config.RANDOM_STATE,
                        'n_estimators': trial.suggest_int('n_estimators', 300, 700),
                        'n_jobs': 1,
                        'tree_method': 'hist',
                        'verbosity': 0,
                        'eval_metric': 'mlogloss'
                    }
                    
                    sample_weights, _ = self._calculate_class_weights(y_train)
                    model = xgb.XGBClassifier(**params)
                
                elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                    params = {
                        'iterations': trial.suggest_int('iterations', 300, 600),
                        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
                        'depth': trial.suggest_int('depth', 4, 8),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 8),
                        'border_count': trial.suggest_int('border_count', 64, 200),
                        'thread_count': 1,
                        'random_state': Config.RANDOM_STATE,
                        'verbose': False,
                        'loss_function': 'MultiClass',
                        'classes_count': Config.N_CLASSES,
                        'auto_class_weights': 'Balanced'
                    }
                    
                    sample_weights, _ = self._calculate_class_weights(y_train)
                    model = cb.CatBoostClassifier(**params)
                
                # Stratified K-Fold cross-validation
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
                print(f"Error during tuning trial: {e}")
                return 0.0
        
        sampler = TPESampler(seed=Config.RANDOM_STATE)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=Config.OPTUNA_TIMEOUT, show_progress_bar=False)
        
        print(f"Best parameters: {study.best_params}")
        print(f"Best score: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    @timer
    def cross_validate_models(self, X_train, y_train):
        """Perform model cross-validation"""
        print("Starting model cross-validation")
        
        cv_strategy = StratifiedKFold(
            n_splits=Config.CV_FOLDS, 
            shuffle=True, 
            random_state=Config.RANDOM_STATE
        )
        
        for model_name, model in self.models.items():
            if model is None:
                print(f"Skipping {model_name}: model is None")
                continue
                
            print(f"Cross-validating {model_name}")
            
            try:
                # Use custom scoring function
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
                    'stability': mean_score - std_score  # Stability metric
                }
                
                print(f"  {model_name}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
                
            except Exception as e:
                print(f"  {model_name} validation failed: {e}")
                continue
        
        # Select best performing model
        if self.cv_scores:
            best_model_name = max(self.cv_scores.keys(), key=lambda x: self.cv_scores[x]['stability'])
            if best_model_name in self.models and self.models[best_model_name] is not None:
                self.best_model = self.models[best_model_name]
                self.best_score = self.cv_scores[best_model_name]['stability']
                
                print(f"Best stability model: {best_model_name}")
                print(f"Stability score: {self.best_score:.4f}")
        
        return self.cv_scores
    
    @timer
    def create_ensemble(self, X_train, y_train):
        """Create ensemble model"""
        print("Starting ensemble model creation")
        
        # Select models based on performance
        good_models = []
        min_score = 0.55  # Minimum performance threshold
        
        for name, model in self.models.items():
            if (model is not None and name in self.cv_scores and 
                self.cv_scores[name]['stability'] >= min_score):
                
                score = self.cv_scores[name]['stability']
                good_models.append((name, model, score))
                print(f"{name}: stability score {score:.4f}")
        
        if len(good_models) >= 2:
            print(f"Models for ensemble: {[name for name, _, _ in good_models]}")
            
            try:
                # Calculate performance-based weights
                total_score = sum(score for _, _, score in good_models)
                model_weights = [score / total_score for _, _, score in good_models]
                
                # Create VotingClassifier
                estimators = [(name, model) for name, model, _ in good_models]
                
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=model_weights
                )
                
                # Train ensemble
                sample_weights, _ = self._calculate_class_weights(y_train)
                ensemble.fit(X_train, y_train, sample_weight=sample_weights)
                
                self.models['ensemble'] = ensemble
                self.ensemble_models['voting'] = ensemble
                
                print("Ensemble model creation completed")
                return ensemble
                
            except Exception as e:
                print(f"Ensemble creation failed: {e}")
                return None
        else:
            print("Insufficient models for ensemble")
            return None
    
    @timer
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, use_optimization=True):
        """Train all models"""
        print("Starting comprehensive model training")
        
        # Train basic models
        if use_optimization:
            print("Applying hyperparameter tuning")
            
            # LightGBM tuning
            try:
                best_params, _ = self.hyperparameter_optimization(
                    X_train, y_train, 'lightgbm', n_trials=Config.OPTUNA_TRIALS
                )
                self.train_lightgbm(X_train, y_train, X_val, y_val, best_params)
            except Exception as e:
                print(f"LightGBM tuning failed: {e}")
                self.train_lightgbm(X_train, y_train, X_val, y_val)
            
            # XGBoost tuning
            try:
                best_params, _ = self.hyperparameter_optimization(
                    X_train, y_train, 'xgboost', n_trials=Config.OPTUNA_TRIALS
                )
                self.train_xgboost(X_train, y_train, X_val, y_val, best_params)
            except Exception as e:
                print(f"XGBoost tuning failed: {e}")
                self.train_xgboost(X_train, y_train, X_val, y_val)
            
            # CatBoost tuning
            if CATBOOST_AVAILABLE:
                try:
                    best_params, _ = self.hyperparameter_optimization(
                        X_train, y_train, 'catboost', n_trials=Config.OPTUNA_TRIALS
                    )
                    self.train_catboost(X_train, y_train, X_val, y_val, best_params)
                except Exception as e:
                    print(f"CatBoost tuning failed: {e}")
                    self.train_catboost(X_train, y_train, X_val, y_val)
        else:
            self.train_lightgbm(X_train, y_train, X_val, y_val)
            self.train_xgboost(X_train, y_train, X_val, y_val)
            if CATBOOST_AVAILABLE:
                self.train_catboost(X_train, y_train, X_val, y_val)
        
        # Train additional models
        self.train_random_forest(X_train, y_train)
        self.train_extra_trees(X_train, y_train)
        
        # Remove None models
        self.models = {k: v for k, v in self.models.items() if v is not None}
        
        # Cross-validation
        if self.models:
            self.cross_validate_models(X_train, y_train)
            
            # Create ensemble
            ensemble = self.create_ensemble(X_train, y_train)
            
            # Validate ensemble performance
            if ensemble is not None:
                try:
                    cv_strategy = StratifiedKFold(
                        n_splits=Config.CV_FOLDS, 
                        shuffle=True, 
                        random_state=Config.RANDOM_STATE
                    )
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
                    
                    print(f"Ensemble stability score: {stability_score:.4f}")
                    
                    # Update if ensemble is more stable
                    if stability_score > self.best_score:
                        self.best_model = ensemble
                        self.best_score = stability_score
                        print("Ensemble selected as best model")
                        
                except Exception as e:
                    print(f"Ensemble validation failed: {e}")
        
        # Save best model
        if self.best_model is not None:
            try:
                save_model(self.best_model, Config.MODEL_FILE)
                print(f"Best model saved: {type(self.best_model).__name__}")
            except Exception as e:
                print(f"Model save failed: {e}")
        
        # Save CV results
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
                print(f"CV results saved: {Config.CV_RESULTS_FILE}")
            except Exception as e:
                print(f"CV results save failed: {e}")
        
        gc.collect()
        
        print("Comprehensive model training completed")
        print(f"Number of trained models: {len(self.models)}")
        
        return self.models, self.best_model
    
    def get_feature_importance(self):
        """Return feature importance"""
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
                print(f"Feature importance extraction failed for {model_name}: {e}")
        
        return importance_dict