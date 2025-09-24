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
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_macro_f1, save_model, save_joblib, setup_logging

def macro_f1_score(y_true, y_pred):
    """Simple function for macro F1 scoring"""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

class ClassBalancedLoss:
    """Class-balanced loss implementation using effective number of samples"""
    
    def __init__(self, samples_per_class, beta=0.9999, loss_type='focal'):
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.loss_type = loss_type
        self.class_weights = self._calculate_weights()
    
    def _calculate_weights(self):
        """Calculate class weights based on effective number of samples"""
        effective_nums = [(1 - np.power(self.beta, n)) / (1 - self.beta) 
                         for n in self.samples_per_class]
        weights = [(1 - self.beta) / en for en in effective_nums]
        normalized_weights = np.array(weights) / np.sum(weights) * len(weights)
        return normalized_weights
    
    def focal_loss(self, y_true, y_pred, alpha=1.0, gamma=2.0):
        """Focal loss implementation"""
        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def get_weights_dict(self):
        """Return weights as dictionary for sklearn compatibility"""
        return {i: weight for i, weight in enumerate(self.class_weights)}

class TemperatureScaling(nn.Module):
    """Temperature scaling for model calibration"""
    
    def __init__(self, model=None):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * Config.TEMPERATURE_INIT)
    
    def temperature_scale(self, logits):
        """Apply temperature scaling to logits"""
        return logits / self.temperature
    
    def forward(self, logits):
        """Forward pass with temperature scaling"""
        return self.temperature_scale(logits)
    
    def calibrate(self, val_loader, criterion=nn.CrossEntropyLoss()):
        """Calibrate temperature parameter"""
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            loss = 0
            for logits, labels in val_loader:
                logits = self.temperature_scale(logits)
                loss += criterion(logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        return self.temperature.item()

class SelectiveEnsemble:
    """Selective ensemble based on diversity and accuracy"""
    
    def __init__(self, diversity_threshold=0.15, accuracy_threshold=0.75):
        self.diversity_threshold = diversity_threshold
        self.accuracy_threshold = accuracy_threshold
        self.selected_models = {}
        self.weights = {}
    
    def calculate_diversity(self, predictions_dict):
        """Calculate diversity between model predictions"""
        models = list(predictions_dict.keys())
        diversities = {}
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                pred1 = predictions_dict[model1]
                pred2 = predictions_dict[model2]
                
                # Calculate disagreement rate
                disagreement = np.mean(pred1 != pred2)
                diversities[(model1, model2)] = disagreement
        
        return diversities
    
    def select_models(self, models_dict, X_val, y_val):
        """Select models based on accuracy and diversity criteria"""
        # Calculate individual model accuracies
        accuracies = {}
        predictions = {}
        
        for name, model in models_dict.items():
            try:
                pred = model.predict(X_val)
                acc = f1_score(y_val, pred, average='macro', zero_division=0)
                accuracies[name] = acc
                predictions[name] = pred
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue
        
        # Filter models by accuracy threshold
        good_models = {name: acc for name, acc in accuracies.items() 
                      if acc >= self.accuracy_threshold}
        
        if len(good_models) < 2:
            print("Insufficient models meeting accuracy threshold")
            return {}
        
        # Calculate diversities
        diversities = self.calculate_diversity({name: predictions[name] 
                                             for name in good_models.keys()})
        
        # Select diverse models using greedy approach
        selected = [max(good_models.keys(), key=good_models.get)]  # Start with best model
        
        while len(selected) < min(5, len(good_models)):  # Max 5 models
            best_candidate = None
            best_diversity = 0
            
            for candidate in good_models.keys():
                if candidate in selected:
                    continue
                
                # Calculate average diversity with selected models
                avg_diversity = np.mean([diversities.get((selected_model, candidate), 
                                                       diversities.get((candidate, selected_model), 0))
                                       for selected_model in selected])
                
                if avg_diversity > best_diversity and avg_diversity > self.diversity_threshold:
                    best_diversity = avg_diversity
                    best_candidate = candidate
            
            if best_candidate is None:
                break
            
            selected.append(best_candidate)
        
        # Calculate weights based on accuracy and diversity
        self.selected_models = {name: models_dict[name] for name in selected}
        
        # Weight calculation: higher accuracy and diversity get more weight
        total_weight = 0
        self.weights = {}
        
        for name in selected:
            acc_weight = good_models[name]
            diversity_weight = np.mean([diversities.get((name, other), 
                                                      diversities.get((other, name), 0))
                                      for other in selected if other != name])
            
            combined_weight = acc_weight * (1 + diversity_weight)
            self.weights[name] = combined_weight
            total_weight += combined_weight
        
        # Normalize weights
        self.weights = {name: weight/total_weight for name, weight in self.weights.items()}
        
        print(f"Selected {len(selected)} models for ensemble:")
        for name in selected:
            print(f"  {name}: accuracy {good_models[name]:.4f}, weight {self.weights[name]:.4f}")
        
        return self.selected_models

class ModelTraining:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.cv_scores = {}
        self.ensemble_models = {}
        self.logger = setup_logging()
        self.calibrated_models = {}
        self.class_balanced_loss = None
        self.temperature_scaling = {}
        self.selective_ensemble = None
        
    def _create_cv_strategy(self, X, y):
        """Create Stratified K-Fold cross-validation strategy"""
        return StratifiedKFold(
            n_splits=Config.CV_FOLDS, 
            shuffle=True, 
            random_state=Config.RANDOM_STATE
        )
    
    def _setup_class_balanced_loss(self, y_train):
        """Setup class-balanced loss"""
        if not Config.USE_CLASS_BALANCED_LOSS:
            return None
        
        samples_per_class = []
        for class_id in range(Config.N_CLASSES):
            count = np.sum(y_train == class_id)
            samples_per_class.append(count)
        
        self.class_balanced_loss = ClassBalancedLoss(
            samples_per_class, 
            beta=Config.EFFECTIVE_NUMBER_BETA
        )
        
        return self.class_balanced_loss.get_weights_dict()
    
    @timer
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None, sample_weights=None):
        """Train LightGBM model with class-balanced loss"""
        print("Starting LightGBM model training with class balancing")
        
        if params is None:
            params = Config.LGBM_PARAMS.copy()
        
        try:
            # Setup class weights
            if Config.USE_CLASS_BALANCED_LOSS and sample_weights is not None:
                # Use sample weights directly
                pass
            else:
                class_weight_dict = self._setup_class_balanced_loss(y_train)
                if class_weight_dict:
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
    def train_quick_lightgbm(self, X_train, y_train, sample_weights=None):
        """Train LightGBM model for quick mode"""
        print("Starting quick LightGBM model training")
        
        try:
            params = Config.get_model_params('quick_lightgbm')
            
            # Simple class weights for quick mode
            if Config.USE_CLASS_BALANCED_LOSS and sample_weights is not None:
                pass
            else:
                class_weight_dict = self._setup_class_balanced_loss(y_train)
                if class_weight_dict:
                    params['class_weight'] = class_weight_dict
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['quick_lightgbm'] = model
            self.best_model = model
            
            print("Quick LightGBM model training completed")
            return model
            
        except Exception as e:
            print(f"Error during quick LightGBM training: {e}")
            return None
    
    @timer
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None, sample_weights=None):
        """Train XGBoost model with class-balanced loss"""
        print("Starting XGBoost model training with class balancing")
        
        if params is None:
            params = Config.XGB_PARAMS.copy()
        
        try:
            # Setup class weights using sample_weight parameter
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
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None, params=None, sample_weights=None):
        """Train CatBoost model with class-balanced loss"""
        if not CATBOOST_AVAILABLE:
            print("CatBoost library not available, skipping training")
            return None
            
        print("Starting CatBoost model training with class balancing")
        
        if params is None:
            params = Config.CAT_PARAMS.copy()
        
        try:
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
    def train_random_forest(self, X_train, y_train, params=None, sample_weights=None):
        """Train Random Forest model with class-balanced loss"""
        print("Starting Random Forest model training with class balancing")
        
        if params is None:
            params = Config.RF_PARAMS.copy()
        
        try:
            # Use class weights in the model parameters
            class_weight_dict = self._setup_class_balanced_loss(y_train)
            if class_weight_dict:
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
    def train_extra_trees(self, X_train, y_train, params=None, sample_weights=None):
        """Train Extra Trees model with class-balanced loss"""
        print("Starting Extra Trees model training with class balancing")
        
        try:
            if params is None:
                params = Config.ET_PARAMS.copy()
            
            class_weight_dict = self._setup_class_balanced_loss(y_train)
            if class_weight_dict:
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
    def train_gradient_boosting(self, X_train, y_train, params=None, sample_weights=None):
        """Train Gradient Boosting model with class-balanced loss"""
        print("Starting Gradient Boosting model training with class balancing")
        
        try:
            if params is None:
                params = Config.GB_PARAMS.copy()
            
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
    def train_neural_network(self, X_train, y_train, params=None, sample_weights=None):
        """Train Neural Network model with class balancing"""
        print("Starting Neural Network model training with class balancing")
        
        try:
            if params is None:
                params = Config.NN_PARAMS.copy()
            
            model = MLPClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            self.models['neural_network'] = model
            self.logger.info("Neural Network model training completed")
            return model
            
        except Exception as e:
            print(f"Error during Neural Network training: {e}")
            self.logger.error(f"Neural Network training failed: {e}")
            return None
    
    @timer
    def apply_temperature_scaling(self, model, model_name, X_val, y_val):
        """Apply temperature scaling to model for calibration"""
        if not Config.USE_TEMPERATURE_SCALING:
            return model
        
        print(f"Applying temperature scaling to {model_name}")
        
        try:
            # Get model predictions as logits/probabilities
            if hasattr(model, 'predict_proba'):
                val_probs = model.predict_proba(X_val)
                val_logits = torch.tensor(np.log(val_probs + 1e-8), dtype=torch.float32)
            elif hasattr(model, 'decision_function'):
                val_scores = model.decision_function(X_val)
                val_logits = torch.tensor(val_scores, dtype=torch.float32)
            else:
                print(f"Model {model_name} doesn't support probability prediction")
                return model
            
            val_labels = torch.tensor(y_val.values, dtype=torch.long)
            
            # Create temperature scaling module
            temp_scaling = TemperatureScaling()
            
            # Simple temperature optimization
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.LBFGS([temp_scaling.temperature], lr=0.01, max_iter=50)
            
            def closure():
                optimizer.zero_grad()
                scaled_logits = temp_scaling(val_logits)
                loss = criterion(scaled_logits, val_labels)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
            final_temperature = temp_scaling.temperature.item()
            self.temperature_scaling[model_name] = final_temperature
            
            print(f"Temperature scaling for {model_name}: T = {final_temperature:.3f}")
            
            return model
            
        except Exception as e:
            print(f"Temperature scaling failed for {model_name}: {e}")
            return model
    
    @timer
    def hyperparameter_optimization(self, X_train, y_train, model_type='lightgbm', n_trials=50, sample_weights=None):
        """Hyperparameter tuning with Optuna"""
        print(f"Starting {model_type} hyperparameter tuning with class balancing")
        
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
                        'deterministic': True,
                        'is_unbalance': True
                    }
                    
                    for param, (min_val, max_val) in tuning_space.items():
                        if param == 'n_estimators':
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val), step=100)
                        elif param in ['num_leaves', 'min_child_samples', 'max_depth', 'min_data_in_leaf']:
                            params[param] = trial.suggest_int(param, int(min_val), int(max_val))
                        else:
                            params[param] = trial.suggest_float(param, min_val, max_val)
                    
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
                    
                    model = cb.CatBoostClassifier(**params)
                
                # Stratified K-Fold cross-validation with sample weights
                cv_strategy = self._create_cv_strategy(X_train, y_train)
                cv_scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Apply sample weights if available
                    sw_tr = sample_weights[train_idx] if sample_weights is not None else None
                    
                    if model_type in ['lightgbm', 'xgboost']:
                        model.fit(X_tr, y_tr, sample_weight=sw_tr)
                    elif model_type == 'catboost':
                        model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
                    else:
                        model.fit(X_tr, y_tr)
                    
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro', zero_division=0)
                    cv_scores.append(score)
                    
                    # Early stopping if performance is poor
                    if fold_idx >= 2 and np.mean(cv_scores) < 0.65:
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
    def cross_validate_models(self, X_train, y_train, sample_weights=None):
        """Perform model cross-validation with class balancing"""
        print("Starting model cross-validation with class balancing")
        
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
                # Custom cross-validation to handle sample weights
                cv_scores = []
                
                for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Clone model for this fold
                    from sklearn.base import clone
                    fold_model = clone(model)
                    
                    # Apply sample weights if available
                    sw_tr = sample_weights[train_idx] if sample_weights is not None else None
                    
                    # Train and evaluate
                    if hasattr(fold_model, 'fit') and 'sample_weight' in fold_model.fit.__code__.co_varnames:
                        fold_model.fit(X_tr, y_tr, sample_weight=sw_tr)
                    else:
                        fold_model.fit(X_tr, y_tr)
                    
                    y_pred = fold_model.predict(X_val)
                    score = macro_f1_score(y_val, y_pred)
                    cv_scores.append(score)
                
                cv_scores = np.array(cv_scores)
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
    def create_selective_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Create selective ensemble model with diversity consideration"""
        print("Starting selective ensemble model creation")
        
        if not Config.USE_SELECTIVE_ENSEMBLE:
            print("Selective ensemble disabled")
            return None
        
        if not self.cv_scores:
            print("No CV scores available for ensemble creation")
            return None
        
        # Use validation set or create one
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_val, _, y_val, _ = train_test_split(
                X_train, y_train, test_size=0.2, 
                random_state=Config.RANDOM_STATE, stratify=y_train
            )
        
        # Initialize selective ensemble
        self.selective_ensemble = SelectiveEnsemble(
            diversity_threshold=Config.ENSEMBLE_DIVERSITY_THRESHOLD,
            accuracy_threshold=Config.ENSEMBLE_ACCURACY_THRESHOLD
        )
        
        # Select models based on diversity and accuracy
        selected_models = self.selective_ensemble.select_models(self.models, X_val, y_val)
        
        if len(selected_models) >= 2:
            print(f"Created selective ensemble with {len(selected_models)} models")
            
            try:
                # Create voting ensemble with selected models and weights
                estimators = [(name, model) for name, model in selected_models.items()]
                weights = [self.selective_ensemble.weights[name] for name, _ in estimators]
                
                voting_ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=weights
                )
                
                # Train ensemble
                voting_ensemble.fit(X_train, y_train)
                
                self.models['selective_ensemble'] = voting_ensemble
                self.ensemble_models['selective'] = voting_ensemble
                
                print("Selective ensemble creation completed")
                return voting_ensemble
                
            except Exception as e:
                print(f"Selective ensemble creation failed: {e}")
                return None
        else:
            print("Insufficient models for selective ensemble")
            return None
    
    @timer
    def calibrate_models(self, X_train, y_train):
        """Calibrate model probabilities with temperature scaling"""
        print("Starting model probability calibration")
        
        # Create calibration validation set
        from sklearn.model_selection import train_test_split
        X_cal_train, X_cal_val, y_cal_train, y_cal_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=Config.RANDOM_STATE, stratify=y_train
        )
        
        for model_name, model in self.models.items():
            if model is None or 'ensemble' in model_name:
                continue
                
            try:
                # Apply temperature scaling
                calibrated_model = self.apply_temperature_scaling(
                    model, model_name, X_cal_val, y_cal_val
                )
                
                # Traditional Platt scaling as backup
                platt_calibrated = CalibratedClassifierCV(
                    model, method='sigmoid', cv=3
                )
                platt_calibrated.fit(X_cal_train, y_cal_train)
                
                self.calibrated_models[f'{model_name}_calibrated'] = platt_calibrated
                print(f"Calibrated {model_name}")
                
            except Exception as e:
                print(f"Calibration failed for {model_name}: {e}")
        
        print(f"Calibrated {len(self.calibrated_models)} models")
    
    @timer
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, use_optimization=True, sample_weights=None):
        """Train all models with class balancing strategy"""
        print("Starting comprehensive model training with class balancing")
        
        # Setup class-balanced loss if needed
        if Config.USE_CLASS_BALANCED_LOSS:
            print("Using class-balanced loss approach")
        
        # Train basic tree-based models
        if use_optimization:
            print("Applying hyperparameter tuning")
            
            # LightGBM tuning
            try:
                best_params, best_score = self.hyperparameter_optimization(
                    X_train, y_train, 'lightgbm', n_trials=Config.OPTUNA_TRIALS, sample_weights=sample_weights
                )
                if best_score > 0.70:
                    self.train_lightgbm(X_train, y_train, X_val, y_val, best_params, sample_weights)
                else:
                    self.train_lightgbm(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
            except Exception as e:
                print(f"LightGBM tuning failed: {e}")
                self.train_lightgbm(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
            
            # XGBoost tuning
            try:
                best_params, best_score = self.hyperparameter_optimization(
                    X_train, y_train, 'xgboost', n_trials=Config.OPTUNA_TRIALS, sample_weights=sample_weights
                )
                if best_score > 0.70:
                    self.train_xgboost(X_train, y_train, X_val, y_val, best_params, sample_weights)
                else:
                    self.train_xgboost(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
            except Exception as e:
                print(f"XGBoost tuning failed: {e}")
                self.train_xgboost(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
            
            # CatBoost tuning
            if CATBOOST_AVAILABLE:
                try:
                    best_params, best_score = self.hyperparameter_optimization(
                        X_train, y_train, 'catboost', n_trials=Config.OPTUNA_TRIALS, sample_weights=sample_weights
                    )
                    if best_score > 0.70:
                        self.train_catboost(X_train, y_train, X_val, y_val, best_params, sample_weights)
                    else:
                        self.train_catboost(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
                except Exception as e:
                    print(f"CatBoost tuning failed: {e}")
                    self.train_catboost(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
        else:
            self.train_lightgbm(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
            self.train_xgboost(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
            if CATBOOST_AVAILABLE:
                self.train_catboost(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
        
        # Train additional models for diversity
        self.train_random_forest(X_train, y_train, sample_weights=sample_weights)
        self.train_extra_trees(X_train, y_train, sample_weights=sample_weights)
        self.train_gradient_boosting(X_train, y_train, sample_weights=sample_weights)
        
        # Train neural network if data size is sufficient
        if len(X_train) > 8000:
            self.train_neural_network(X_train, y_train, sample_weights=sample_weights)
        
        # Remove None models
        self.models = {k: v for k, v in self.models.items() if v is not None}
        
        # Cross-validation with class balancing
        if self.models:
            self.cross_validate_models(X_train, y_train, sample_weights)
            
            # Create selective ensemble
            ensemble = self.create_selective_ensemble(X_train, y_train, X_val, y_val)
            
            # Validate ensemble performance
            if ensemble is not None:
                try:
                    cv_strategy = StratifiedKFold(
                        n_splits=Config.CV_FOLDS, 
                        shuffle=True, 
                        random_state=Config.RANDOM_STATE
                    )
                    
                    # Custom CV for ensemble with sample weights
                    ensemble_cv = []
                    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                        X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        
                        ensemble.fit(X_tr, y_tr)
                        y_pred = ensemble.predict(X_val_fold)
                        score = macro_f1_score(y_val_fold, y_pred)
                        ensemble_cv.append(score)
                    
                    ensemble_cv = np.array(ensemble_cv)
                    ensemble_score = ensemble_cv.mean()
                    ensemble_std = ensemble_cv.std()
                    stability_score = ensemble_score - (0.8 * ensemble_std)
                    
                    self.cv_scores['selective_ensemble'] = {
                        'scores': ensemble_cv,
                        'mean': ensemble_score,
                        'std': ensemble_std,
                        'stability': stability_score
                    }
                    
                    print(f"selective_ensemble stability score: {stability_score:.4f}")
                    
                    # Update best model if ensemble is better
                    if stability_score > self.best_score:
                        self.best_model = ensemble
                        self.best_score = stability_score
                        print(f"selective_ensemble selected as best model")
                    
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
    
    def get_class_balanced_loss(self):
        """Return class balanced loss object"""
        return self.class_balanced_loss
    
    def get_temperature_scaling(self):
        """Return temperature scaling parameters"""
        return self.temperature_scaling
    
    def get_best_single_model(self):
        """Get best performing single model (non-ensemble)"""
        if not self.cv_scores:
            return None
        
        single_model_scores = {k: v for k, v in self.cv_scores.items() if 'ensemble' not in k}
        
        if not single_model_scores:
            return None
        
        best_model_name = max(single_model_scores.keys(), key=lambda x: single_model_scores[x]['stability'])
        return self.models.get(best_model_name)