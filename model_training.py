# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost library not installed")

import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
import gc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_macro_f1, save_model, save_joblib, setup_logging

def macro_f1_score(y_true, y_pred):
    """Simple function for macro F1 scoring"""
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
    
    @staticmethod
    def compute_log_weights(y):
        """Calculate log-based weights"""
        class_counts = np.bincount(y, minlength=Config.N_CLASSES)
        total_samples = len(y)
        
        log_weights = np.log(total_samples / (class_counts + 1))
        log_weights = log_weights / np.mean(log_weights)
        
        return dict(enumerate(log_weights))

class ModelTraining:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.cv_scores = {}
        self.ensemble_models = {}
        self.logger = setup_logging()
        self.weight_calculator = WeightCalculator()
        self.calibrated_models = {}
        
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
        elif method == 'log':
            class_weight_dict = self.weight_calculator.compute_log_weights(y)
        else:
            class_weight_dict = self.weight_calculator.compute_balanced_weights(y)
        
        sample_weights = np.array([class_weight_dict.get(label, 1.0) for label in y])
        return sample_weights, class_weight_dict
    
    @timer
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """Train LightGBM model"""
        print("Starting LightGBM model training")
        
        if params is None:
            params = Config.LGBM_PARAMS.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train, 'focal')
            
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
                        lgb.early_stopping(80, verbose=False),
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
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train, 'focal')
            
            model = xgb.XGBClassifier(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train, 
                    sample_weight=sample_weights,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
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
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train, 'focal')
            
            # Remove conflicting parameters for bayesian bootstrap
            if 'bootstrap_type' in params and params['bootstrap_type'] == 'Bayesian':
                if 'subsample' in params:
                    del params['subsample']
            
            model = cb.CatBoostClassifier(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=80,
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
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train, 'log')
            
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
            
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train, 'log')
            
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
    def hyperparameter_optimization(self, X_train, y_train, model_type='lightgbm', n_trials=35):
        """Hyperparameter tuning"""
        print(f"Starting {model_type} hyperparameter tuning")
        
        tuning_space = Config.get_tuning_space(model_type)
        if not tuning_space:
            print(f"No tuning space defined for {model_type}")
            return {}, 0.0
        
        def objective(trial):
            try:
                if model_type == 'lightgbm':
                    params = {
                        'objective': 'multiclass',
                        'num_class': Config.N_CLASSES,
                        'metric': 'multi_logloss',
                        'boosting_type': 'gbdt',
                        'verbose': -1,
                        'random_state': Config.RANDOM_STATE,
                        'n_jobs': 1
                    }
                    
                    for param, (min_val, max_val) in tuning_space.items():
                        if param == 'n_estimators':
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val), step=50)
                        elif param in ['num_leaves']:
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val))
                        elif param in ['min_child_samples']:
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val))
                        else:
                            params[param] = trial.suggest_float(param, min_val, max_val)
                    
                    sample_weights, class_weight_dict = self._calculate_class_weights(y_train, 'focal')
                    params['class_weight'] = class_weight_dict
                    model = lgb.LGBMClassifier(**params)
                    
                elif model_type == 'xgboost':
                    params = {
                        'objective': 'multi:softprob',
                        'num_class': Config.N_CLASSES,
                        'random_state': Config.RANDOM_STATE,
                        'n_jobs': 1,
                        'tree_method': 'hist',
                        'verbosity': 0,
                        'eval_metric': 'mlogloss'
                    }
                    
                    for param, (min_val, max_val) in tuning_space.items():
                        if param == 'n_estimators':
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val), step=50)
                        elif param in ['max_depth']:
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val))
                        else:
                            params[param] = trial.suggest_float(param, min_val, max_val)
                    
                    sample_weights, _ = self._calculate_class_weights(y_train, 'focal')
                    model = xgb.XGBClassifier(**params)
                
                elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                    params = {
                        'thread_count': 1,
                        'random_state': Config.RANDOM_STATE,
                        'verbose': False,
                        'loss_function': 'MultiClass',
                        'classes_count': Config.N_CLASSES,
                        'auto_class_weights': 'Balanced'
                    }
                    
                    for param, (min_val, max_val) in tuning_space.items():
                        if param == 'iterations':
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val), step=50)
                        elif param in ['depth', 'border_count']:
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val))
                        else:
                            params[param] = trial.suggest_float(param, min_val, max_val)
                    
                    sample_weights, _ = self._calculate_class_weights(y_train, 'focal')
                    model = cb.CatBoostClassifier(**params)
                
                # Stratified K-Fold cross-validation
                cv_strategy = self._create_cv_strategy(X_train, y_train)
                cv_scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    if model_type in ['lightgbm', 'xgboost']:
                        sw_tr, _ = self._calculate_class_weights(y_tr, 'focal')
                        model.fit(X_tr, y_tr, sample_weight=sw_tr)
                    elif model_type == 'catboost':
                        sw_tr, _ = self._calculate_class_weights(y_tr, 'focal')
                        model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
                    else:
                        model.fit(X_tr, y_tr)
                    
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro', zero_division=0)
                    cv_scores.append(score)
                    
                    # Early stopping if performance is poor
                    if fold_idx >= 1 and np.mean(cv_scores) < 0.65:
                        break
                
                return np.mean(cv_scores)
                
            except Exception as e:
                print(f"Error during tuning trial: {e}")
                return 0.0
        
        sampler = TPESampler(seed=Config.RANDOM_STATE)
        pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=5)
        
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
                # Use macro_f1_score function directly
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_strategy,
                    scoring=make_scorer(macro_f1_score),
                    n_jobs=1
                )
                
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                # Calculate stability score with bias towards higher mean
                stability_score = mean_score - (0.8 * std_score)
                
                self.cv_scores[model_name] = {
                    'scores': cv_scores,
                    'mean': mean_score,
                    'std': std_score,
                    'stability': stability_score,
                    'confidence_interval': (mean_score - 1.96 * std_score / np.sqrt(len(cv_scores)),
                                          mean_score + 1.96 * std_score / np.sqrt(len(cv_scores)))
                }
                
                print(f"  {model_name}: {mean_score:.4f} (+/- {std_score * 2:.4f}) [stability: {stability_score:.4f}]")
                
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
        
        # Select only top 3 performing models
        if not self.cv_scores:
            print("No CV scores available for ensemble creation")
            return None
        
        # Sort models by stability score
        sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1]['stability'], reverse=True)
        top_models = sorted_models[:3]  # Top 3 models only
        
        good_models = []
        min_score = 0.70  # Threshold
        
        for name, scores in top_models:
            if name in self.models and self.models[name] is not None and scores['stability'] >= min_score:
                score = scores['stability']
                good_models.append((name, self.models[name], score))
                print(f"{name}: stability score {score:.4f}")
        
        if len(good_models) >= 2:
            print(f"Models for ensemble: {[name for name, _, _ in good_models]}")
            
            try:
                # Calculate performance-based weights with higher emphasis on stability
                total_score = sum(score**2 for _, _, score in good_models)  # Squared weights
                model_weights = [score**2 / total_score for _, _, score in good_models]
                
                # Create VotingClassifier
                estimators = [(name, model) for name, model, _ in good_models]
                
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=model_weights
                )
                
                # Train ensemble
                sample_weights, _ = self._calculate_class_weights(y_train, 'focal')
                ensemble.fit(X_train, y_train, sample_weight=sample_weights)
                
                self.models['ensemble'] = ensemble
                self.ensemble_models['voting'] = ensemble
                
                print("Ensemble model creation completed")
                print(f"Ensemble weights: {dict(zip([name for name, _, _ in good_models], model_weights))}")
                
                return ensemble
                
            except Exception as e:
                print(f"Ensemble creation failed: {e}")
                return None
        else:
            print("Insufficient models for ensemble")
            return None
    
    @timer
    def calibrate_models(self, X_train, y_train):
        """Calibrate model probabilities"""
        print("Starting model probability calibration")
        
        for model_name, model in self.models.items():
            if model is None or model_name == 'ensemble':
                continue
                
            try:
                calibrated_model = CalibratedClassifierCV(
                    model, method='isotonic', cv=3
                )
                calibrated_model.fit(X_train, y_train)
                
                self.calibrated_models[f'{model_name}_calibrated'] = calibrated_model
                print(f"Calibrated {model_name}")
                
            except Exception as e:
                print(f"Calibration failed for {model_name}: {e}")
        
        print(f"Calibrated {len(self.calibrated_models)} models")
    
    @timer
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, use_optimization=True):
        """Train all models"""
        print("Starting comprehensive model training")
        
        # Train basic models
        if use_optimization:
            print("Applying hyperparameter tuning")
            
            # LightGBM tuning
            try:
                best_params, best_score = self.hyperparameter_optimization(
                    X_train, y_train, 'lightgbm', n_trials=Config.OPTUNA_TRIALS
                )
                if best_score > 0.6:  # Only use optimal params if they're decent
                    self.train_lightgbm(X_train, y_train, X_val, y_val, best_params)
                else:
                    self.train_lightgbm(X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"LightGBM tuning failed: {e}")
                self.train_lightgbm(X_train, y_train, X_val, y_val)
            
            # XGBoost tuning
            try:
                best_params, best_score = self.hyperparameter_optimization(
                    X_train, y_train, 'xgboost', n_trials=Config.OPTUNA_TRIALS
                )
                if best_score > 0.6:
                    self.train_xgboost(X_train, y_train, X_val, y_val, best_params)
                else:
                    self.train_xgboost(X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"XGBoost tuning failed: {e}")
                self.train_xgboost(X_train, y_train, X_val, y_val)
            
            # CatBoost tuning
            if CATBOOST_AVAILABLE:
                try:
                    best_params, best_score = self.hyperparameter_optimization(
                        X_train, y_train, 'catboost', n_trials=Config.OPTUNA_TRIALS
                    )
                    if best_score > 0.6:
                        self.train_catboost(X_train, y_train, X_val, y_val, best_params)
                    else:
                        self.train_catboost(X_train, y_train, X_val, y_val)
                except Exception as e:
                    print(f"CatBoost tuning failed: {e}")
                    self.train_catboost(X_train, y_train, X_val, y_val)
        else:
            self.train_lightgbm(X_train, y_train, X_val, y_val)
            self.train_xgboost(X_train, y_train, X_val, y_val)
            if CATBOOST_AVAILABLE:
                self.train_catboost(X_train, y_train, X_val, y_val)
        
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
                        scoring=make_scorer(macro_f1_score),
                        n_jobs=1
                    )
                    
                    ensemble_score = ensemble_cv.mean()
                    ensemble_std = ensemble_cv.std()
                    stability_score = ensemble_score - (0.8 * ensemble_std)
                    
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
            
            # Calibrate models
            try:
                self.calibrate_models(X_train, y_train)
            except Exception as e:
                print(f"Model calibration failed: {e}")
        
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
    
    def get_best_single_model(self):
        """Get best performing single model (non-ensemble)"""
        if not self.cv_scores:
            return None
        
        single_model_scores = {k: v for k, v in self.cv_scores.items() if k != 'ensemble'}
        
        if not single_model_scores:
            return None
        
        best_model_name = max(single_model_scores.keys(), key=lambda x: single_model_scores[x]['stability'])
        return self.models.get(best_model_name)