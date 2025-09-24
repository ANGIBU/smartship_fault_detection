# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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
    """Class weight calculation with focal loss strategy"""
    
    @staticmethod
    def compute_focal_weights(y, alpha=2.0, gamma=3.0):
        """Calculate Focal Loss based weights"""
        class_counts = np.bincount(y, minlength=Config.N_CLASSES)
        total_samples = len(y)
        
        # Frequency-based weights
        freq_weights = total_samples / (Config.N_CLASSES * class_counts + 1e-8)
        
        # Apply focal weights with stronger penalty for minority classes
        class_frequencies = class_counts / total_samples
        focal_weights = alpha * np.power(1 - class_frequencies, gamma)
        
        # Combined weights
        combined_weights = freq_weights * focal_weights
        
        # Apply square root to reduce extreme values
        combined_weights = np.sqrt(combined_weights)
        
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
        self.weight_calculator = WeightCalculator()
        self.calibrated_models = {}
        
    def _create_cv_strategy(self, X, y):
        """Create Stratified K-Fold cross-validation strategy with better stratification"""
        return StratifiedKFold(
            n_splits=Config.CV_FOLDS, 
            shuffle=True, 
            random_state=Config.RANDOM_STATE
        )
    
    def _calculate_class_weights(self, y, method='focal'):
        """Calculate class weights using focal loss strategy"""
        if method == 'focal':
            class_weight_dict = self.weight_calculator.compute_focal_weights(
                y, Config.FOCAL_LOSS_ALPHA, Config.FOCAL_LOSS_GAMMA
            )
        else:
            class_weight_dict = self.weight_calculator.compute_balanced_weights(y)
        
        sample_weights = np.array([class_weight_dict.get(label, 1.0) for label in y])
        return sample_weights, class_weight_dict
    
    @timer
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """Train LightGBM model with stability improvements"""
        print("Starting LightGBM model training")
        
        if params is None:
            params = Config.LGBM_PARAMS.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train, 'focal')
            
            # Apply class weights to parameters
            params['class_weight'] = class_weight_dict
            
            # Add stability parameters
            params['force_col_wise'] = True
            params['deterministic'] = True
            params['feature_pre_filter'] = False
            
            model = lgb.LGBMClassifier(**params)
            
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=eval_set,
                    callbacks=[
                        lgb.early_stopping(120, verbose=False),
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
        """Train XGBoost model with focal weights"""
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
        """Train CatBoost model with focal weights"""
        if not CATBOOST_AVAILABLE:
            print("CatBoost library not available, skipping training")
            return None
            
        print("Starting CatBoost model training")
        
        if params is None:
            params = Config.CAT_PARAMS.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train, 'focal')
            
            model = cb.CatBoostClassifier(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=120,
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
        """Train Random Forest model with focal weights"""
        print("Starting Random Forest model training")
        
        if params is None:
            params = Config.RF_PARAMS.copy()
        
        try:
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train, 'focal')
            
            # Use class_weight parameter instead of sample_weight for Random Forest
            params['class_weight'] = class_weight_dict
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            self.models['random_forest'] = model
            self.logger.info("Random Forest model training completed")
            return model
            
        except Exception as e:
            print(f"Error during Random Forest training: {e}")
            self.logger.error(f"Random Forest training failed: {e}")
            return None
    
    @timer
    def train_extra_trees(self, X_train, y_train, params=None):
        """Train Extra Trees model with focal weights"""
        print("Starting Extra Trees model training")
        
        try:
            if params is None:
                params = Config.ET_PARAMS.copy()
            
            sample_weights, class_weight_dict = self._calculate_class_weights(y_train, 'focal')
            
            params['class_weight'] = class_weight_dict
            model = ExtraTreesClassifier(**params)
            model.fit(X_train, y_train)
            
            self.models['extra_trees'] = model
            self.logger.info("Extra Trees model training completed")
            return model
            
        except Exception as e:
            print(f"Error during Extra Trees training: {e}")
            self.logger.error(f"Extra Trees training failed: {e}")
            return None
    
    @timer
    def train_gradient_boosting(self, X_train, y_train, params=None):
        """Train Gradient Boosting model with focal weights"""
        print("Starting Gradient Boosting model training")
        
        try:
            if params is None:
                params = Config.GB_PARAMS.copy()
            
            sample_weights, _ = self._calculate_class_weights(y_train, 'focal')
            
            model = GradientBoostingClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['gradient_boosting'] = model
            self.logger.info("Gradient Boosting model training completed")
            return model
            
        except Exception as e:
            print(f"Error during Gradient Boosting training: {e}")
            self.logger.error(f"Gradient Boosting training failed: {e}")
            return None
    
    @timer
    def train_neural_network(self, X_train, y_train, params=None):
        """Train Neural Network model"""
        print("Starting Neural Network model training")
        
        try:
            if params is None:
                params = Config.NN_PARAMS.copy()
            
            model = MLPClassifier(**params)
            model.fit(X_train, y_train)
            
            self.models['neural_network'] = model
            self.logger.info("Neural Network model training completed")
            return model
            
        except Exception as e:
            print(f"Error during Neural Network training: {e}")
            self.logger.error(f"Neural Network training failed: {e}")
            return None
    
    @timer
    def hyperparameter_optimization(self, X_train, y_train, model_type='lightgbm', n_trials=50):
        """Hyperparameter tuning with Optuna"""
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
                        'n_jobs': 1,
                        'force_col_wise': True,
                        'deterministic': True
                    }
                    
                    for param, (min_val, max_val) in tuning_space.items():
                        if param == 'n_estimators':
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val), step=100)
                        elif param in ['num_leaves', 'min_child_samples', 'max_depth', 'min_data_in_leaf']:
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
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val), step=100)
                        elif param in ['max_depth', 'max_delta_step']:
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
                        'auto_class_weights': 'Balanced',
                        'bootstrap_type': 'Bayesian'
                    }
                    
                    for param, (min_val, max_val) in tuning_space.items():
                        if param == 'iterations':
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val), step=100)
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
                    if fold_idx >= 2 and np.mean(cv_scores) < 0.60:
                        break
                    
                    # Pruning based on intermediate results
                    if fold_idx >= 3:
                        trial.report(np.mean(cv_scores), fold_idx)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                
                return np.mean(cv_scores)
                
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                print(f"Error during tuning trial: {e}")
                return 0.0
        
        sampler = TPESampler(seed=Config.RANDOM_STATE, n_startup_trials=15, n_ei_candidates=50)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=3, interval_steps=2)
        
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=Config.OPTUNA_TIMEOUT, show_progress_bar=False)
        
        print(f"Best parameters: {study.best_params}")
        print(f"Best score: {study.best_value:.4f}")
        
        return study.best_params, study.best_value
    
    @timer
    def cross_validate_models(self, X_train, y_train):
        """Perform model cross-validation with stability calculation"""
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
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_strategy,
                    scoring=make_scorer(macro_f1_score),
                    n_jobs=1
                )
                
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                # Stability score calculation
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
    def create_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Create ensemble model with diversity consideration"""
        print("Starting ensemble model creation")
        
        if not self.cv_scores:
            print("No CV scores available for ensemble creation")
            return None
        
        # Select top performing models with diversity
        sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1]['stability'], reverse=True)
        top_models = sorted_models[:5]  # Top 5 models
        
        good_models = []
        min_score = 0.70  # Minimum threshold
        
        for name, scores in top_models:
            if name in self.models and self.models[name] is not None and scores['stability'] >= min_score:
                score = scores['stability']
                good_models.append((name, self.models[name], score))
                print(f"{name}: stability score {score:.4f}")
        
        if len(good_models) >= 2:
            print(f"Models for ensemble: {[name for name, _, _ in good_models]}")
            
            try:
                # Weighted Voting Classifier with stability-based weights
                stability_weights = []
                total_weight = 0
                
                for _, _, score in good_models:
                    # Use exponential weighting to emphasize better models
                    weight = np.exp(score * 3)  # Stability-based weighting
                    stability_weights.append(weight)
                    total_weight += weight
                
                # Normalize weights
                stability_weights = [w / total_weight for w in stability_weights]
                
                # Create VotingClassifier
                estimators = [(name, model) for name, model, _ in good_models]
                
                voting_ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=stability_weights
                )
                
                # Train ensemble
                sample_weights, _ = self._calculate_class_weights(y_train, 'focal')
                voting_ensemble.fit(X_train, y_train, sample_weight=sample_weights)
                
                self.models['voting_ensemble'] = voting_ensemble
                self.ensemble_models['voting'] = voting_ensemble
                
                print("Voting ensemble creation completed")
                print(f"Voting weights: {dict(zip([name for name, _, _ in good_models], stability_weights))}")
                
                return voting_ensemble
                
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
            if model is None or 'ensemble' in model_name:
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
        """Train all models with strategy"""
        print("Starting comprehensive model training")
        
        # Train basic tree-based models
        if use_optimization:
            print("Applying hyperparameter tuning")
            
            # LightGBM tuning
            try:
                best_params, best_score = self.hyperparameter_optimization(
                    X_train, y_train, 'lightgbm', n_trials=Config.OPTUNA_TRIALS
                )
                if best_score > 0.65:
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
                if best_score > 0.65:
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
                    if best_score > 0.65:
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
        
        # Train additional models for diversity
        self.train_random_forest(X_train, y_train)
        self.train_extra_trees(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # Train neural network if data size is sufficient
        if len(X_train) > 8000:
            self.train_neural_network(X_train, y_train)
        
        # Remove None models
        self.models = {k: v for k, v in self.models.items() if v is not None}
        
        # Cross-validation
        if self.models:
            self.cross_validate_models(X_train, y_train)
            
            # Create ensemble
            ensemble = self.create_ensemble(X_train, y_train, X_val, y_val)
            
            # Validate ensemble performance
            if ensemble is not None:
                try:
                    cv_strategy = StratifiedKFold(
                        n_splits=Config.CV_FOLDS, 
                        shuffle=True, 
                        random_state=Config.RANDOM_STATE
                    )
                    
                    # Validate voting ensemble
                    if 'voting_ensemble' in self.models:
                        ensemble_model = self.models['voting_ensemble']
                        
                        ensemble_cv = cross_val_score(
                            ensemble_model, X_train, y_train,
                            cv=cv_strategy,
                            scoring=make_scorer(macro_f1_score),
                            n_jobs=1
                        )
                        
                        ensemble_score = ensemble_cv.mean()
                        ensemble_std = ensemble_cv.std()
                        stability_score = ensemble_score - (0.8 * ensemble_std)
                        
                        self.cv_scores['voting_ensemble'] = {
                            'scores': ensemble_cv,
                            'mean': ensemble_score,
                            'std': ensemble_std,
                            'stability': stability_score
                        }
                        
                        print(f"voting_ensemble stability score: {stability_score:.4f}")
                        
                        # Update best model if ensemble is better
                        if stability_score > self.best_score:
                            self.best_model = ensemble_model
                            self.best_score = stability_score
                            print(f"voting_ensemble selected as best model")
                    
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
        """Return feature importance for all models"""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if model is None or 'ensemble' in model_name:
                continue
                
            try:
                if hasattr(model, 'feature_importances_'):
                    importance_dict[model_name] = model.feature_importances_
                elif hasattr(model, 'get_feature_importance'):
                    importance_dict[model_name] = model.get_feature_importance()
                elif hasattr(model, 'coef_') and len(model.coef_.shape) == 2:
                    # For linear models, use mean absolute coefficients
                    importance_dict[model_name] = np.mean(np.abs(model.coef_), axis=0)
            except Exception as e:
                print(f"Feature importance extraction failed for {model_name}: {e}")
        
        return importance_dict
    
    def get_best_single_model(self):
        """Get best performing single model (non-ensemble)"""
        if not self.cv_scores:
            return None
        
        single_model_scores = {k: v for k, v in self.cv_scores.items() if 'ensemble' not in k}
        
        if not single_model_scores:
            return None
        
        best_model_name = max(single_model_scores.keys(), key=lambda x: single_model_scores[x]['stability'])
        return self.models.get(best_model_name)
    
    def get_ensemble_performance_summary(self):
        """Get performance summary of all ensemble methods"""
        ensemble_summary = {}
        
        for ensemble_name in ['voting_ensemble', 'stacking_ensemble', 'dynamic_ensemble']:
            if ensemble_name in self.cv_scores:
                scores = self.cv_scores[ensemble_name]
                ensemble_summary[ensemble_name] = {
                    'mean_f1': scores['mean'],
                    'stability': scores['stability'],
                    'std': scores['std']
                }
        
        return ensemble_summary