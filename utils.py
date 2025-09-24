# utils.py

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import time
import logging
import warnings
import psutil
from pathlib import Path
import gc
import os
from scipy.stats import zscore
from scipy import stats

warnings.filterwarnings('ignore')

def setup_logging(log_file=None, level='INFO'):
    """Setup logging configuration"""
    if log_file is None:
        log_file = Path('logs') / 'training.log'
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)

def load_data(file_path, chunk_size=None):
    """Memory-efficient data loading with optimization"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if chunk_size and file_size_mb > 200:
            chunks = []
            chunk_count = 0
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunk = optimize_dataframe_memory(chunk)
                chunks.append(chunk)
                chunk_count += 1
                
                if chunk_count % 10 == 0:
                    gc.collect()
            
            data = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
        else:
            data = pd.read_csv(file_path)
            data = optimize_dataframe_memory(data)
        
        print(f"Data loaded: {data.shape}, memory: {data.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")
        return data
        
    except Exception as e:
        raise Exception(f"Data loading failed: {e}")

def save_model(model, file_path):
    """Save model with compression"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(model, file_path, compress=9)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"Model saved successfully (joblib): {file_path} ({file_size_mb:.1f}MB)")
            return
        except Exception as joblib_error:
            print(f"joblib save failed: {joblib_error}")
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"Model saved successfully (pickle): {file_path} ({file_size_mb:.1f}MB)")
            
        except Exception as pickle_error:
            print(f"pickle save failed: {pickle_error}")
            raise Exception(f"Model save failed: joblib={joblib_error}, pickle={pickle_error}")
        
    except Exception as e:
        print(f"Model save failed: {e}")
        raise

def load_model(file_path):
    """Load model with error handling"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            model = joblib.load(file_path)
            print(f"Model loaded successfully (joblib): {file_path}")
            return model
        except:
            pass
        
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully (pickle): {file_path}")
            return model
        except:
            pass
        
        raise Exception("All loading methods failed")
        
    except Exception as e:
        raise Exception(f"Model loading failed: {e}")

def save_joblib(obj, file_path, compress=9):
    """Save object using joblib with high compression"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path, compress=compress)
        
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"Object saved successfully: {file_path} ({file_size_mb:.1f}MB)")
        else:
            print(f"Object saved successfully: {file_path}")
            
    except Exception as e:
        raise Exception(f"Object save failed: {e}")

def load_joblib(file_path):
    """Load object using joblib"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        obj = joblib.load(file_path)
        print(f"Object loaded successfully: {file_path}")
        return obj
    except Exception as e:
        raise Exception(f"Object loading failed: {e}")

def calculate_macro_f1(y_true, y_pred):
    """Calculate Macro F1 score with zero division handling"""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def calculate_weighted_f1(y_true, y_pred):
    """Calculate Weighted F1 score"""
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

def calculate_class_balanced_f1(y_true, y_pred, class_weights=None):
    """Calculate class-balanced F1 score"""
    try:
        class_f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        if class_weights is not None:
            # Weight the F1 scores by class importance
            weighted_scores = []
            for class_id, f1 in enumerate(class_f1_scores):
                weight = class_weights.get(class_id, 1.0)
                weighted_scores.append(f1 * weight)
            
            return np.mean(weighted_scores)
        else:
            return np.mean(class_f1_scores)
    except Exception as e:
        print(f"Class-balanced F1 calculation failed: {e}")
        return 0.0

def calculate_effective_number_metrics(y_true, y_pred, beta=0.9999):
    """Calculate metrics using effective number of samples"""
    try:
        # Calculate effective numbers for each class
        class_counts = Counter(y_true)
        effective_numbers = {}
        class_weights = {}
        
        for class_id, count in class_counts.items():
            effective_num = (1 - np.power(beta, count)) / (1 - beta)
            effective_numbers[class_id] = effective_num
            class_weights[class_id] = (1 - beta) / effective_num
        
        # Normalize weights
        total_weight = sum(class_weights.values())
        for class_id in class_weights:
            class_weights[class_id] = class_weights[class_id] / total_weight * len(class_weights)
        
        # Calculate weighted F1
        weighted_f1 = calculate_class_balanced_f1(y_true, y_pred, class_weights)
        
        return {
            'effective_numbers': effective_numbers,
            'class_weights': class_weights,
            'weighted_f1': weighted_f1
        }
    except Exception as e:
        print(f"Effective number metrics calculation failed: {e}")
        return None

def calculate_all_metrics(y_true, y_pred):
    """Calculate all evaluation metrics with class imbalance considerations"""
    from sklearn.metrics import precision_score, recall_score
    
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Class-specific metrics
        class_f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        valid_scores = class_f1_scores[class_f1_scores > 0]
        
        if len(valid_scores) > 0:
            metrics['f1_std'] = np.std(valid_scores)
            metrics['f1_min'] = np.min(valid_scores)
            metrics['f1_max'] = np.max(valid_scores)
            metrics['f1_range'] = metrics['f1_max'] - metrics['f1_min']
            metrics['low_performance_classes'] = np.sum(class_f1_scores < 0.5)
            
            # Stability metrics
            metrics['f1_cv'] = metrics['f1_std'] / metrics['macro_f1'] if metrics['macro_f1'] > 0 else 0
            metrics['balanced_accuracy'] = np.mean(class_f1_scores)
            
            # Class imbalance metrics
            class_distribution = Counter(y_true)
            max_count = max(class_distribution.values())
            min_count = min([v for v in class_distribution.values() if v > 0])
            metrics['imbalance_ratio'] = max_count / min_count if min_count > 0 else float('inf')
            
            # Effective number based metrics
            effective_metrics = calculate_effective_number_metrics(y_true, y_pred)
            if effective_metrics:
                metrics['effective_weighted_f1'] = effective_metrics['weighted_f1']
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics = {
            'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0, 'micro_f1': 0.0,
            'macro_precision': 0.0, 'macro_recall': 0.0, 'weighted_precision': 0.0, 
            'weighted_recall': 0.0, 'f1_std': 0.0, 'f1_min': 0.0, 'f1_max': 0.0,
            'f1_range': 0.0, 'low_performance_classes': 0, 'f1_cv': 0.0, 
            'balanced_accuracy': 0.0, 'imbalance_ratio': 0.0
        }
    
    return metrics

def analyze_class_imbalance(y_data, n_classes=21):
    """Analyze class imbalance with multiple metrics"""
    try:
        class_counts = Counter(y_data)
        total_samples = len(y_data)
        
        # Basic imbalance metrics
        max_count = max(class_counts.values())
        min_count = min([v for v in class_counts.values() if v > 0])
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Calculate class frequencies
        class_frequencies = {}
        for class_id in range(n_classes):
            count = class_counts.get(class_id, 0)
            frequency = count / total_samples
            class_frequencies[class_id] = {
                'count': count,
                'frequency': frequency,
                'is_minority': count < (total_samples / n_classes) * 0.5
            }
        
        # Shannon entropy for distribution uniformity
        frequencies = [class_counts.get(i, 0) / total_samples for i in range(n_classes)]
        entropy = -np.sum([f * np.log(f + 1e-10) for f in frequencies if f > 0])
        max_entropy = np.log(n_classes)
        normalized_entropy = entropy / max_entropy
        
        # Gini coefficient for inequality
        sorted_counts = sorted([class_counts.get(i, 0) for i in range(n_classes)])
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        # Missing classes
        missing_classes = [i for i in range(n_classes) if class_counts.get(i, 0) == 0]
        
        # Effective number of classes (Simpson's diversity index)
        probs = np.array([class_counts.get(i, 0) for i in range(n_classes)]) / total_samples
        effective_classes = 1 / np.sum(probs ** 2) if np.sum(probs ** 2) > 0 else 0
        
        analysis_results = {
            'total_samples': total_samples,
            'n_classes': n_classes,
            'imbalance_ratio': imbalance_ratio,
            'normalized_entropy': normalized_entropy,
            'gini_coefficient': gini,
            'missing_classes': missing_classes,
            'effective_classes': effective_classes,
            'class_frequencies': class_frequencies,
            'minority_classes': [cid for cid, info in class_frequencies.items() 
                               if info['is_minority']],
            'is_severely_imbalanced': imbalance_ratio > 50,
            'distribution_quality': normalized_entropy * (1 - gini)
        }
        
        return analysis_results
        
    except Exception as e:
        print(f"Class imbalance analysis failed: {e}")
        return None

def calculate_sensor_data_quality(X_data, feature_columns):
    """Calculate sensor data quality metrics"""
    try:
        quality_metrics = {}
        
        for col in feature_columns:
            if col in X_data.columns:
                col_data = X_data[col]
                
                # Basic statistics
                col_stats = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min(),
                    'missing_rate': col_data.isnull().sum() / len(col_data),
                    'zero_rate': (col_data == 0).sum() / len(col_data),
                    'unique_values': col_data.nunique()
                }
                
                # Signal quality metrics
                if col_stats['std'] > 0:
                    # Signal-to-noise ratio approximation
                    col_stats['snr'] = abs(col_stats['mean']) / col_stats['std']
                    
                    # Coefficient of variation
                    col_stats['cv'] = col_stats['std'] / abs(col_stats['mean']) if col_stats['mean'] != 0 else float('inf')
                else:
                    col_stats['snr'] = 0
                    col_stats['cv'] = 0
                
                # Outlier detection
                z_scores = np.abs(zscore(col_data.dropna()))
                col_stats['outlier_rate'] = (z_scores > 3).sum() / len(z_scores) if len(z_scores) > 0 else 0
                
                # Sensor reliability score
                reliability_score = (1 - col_stats['missing_rate']) * (1 - col_stats['outlier_rate'])
                if col_stats['unique_values'] > 1:
                    reliability_score *= min(1.0, col_stats['snr'] / 10)
                
                col_stats['reliability_score'] = reliability_score
                quality_metrics[col] = col_stats
        
        # Overall data quality
        if quality_metrics:
            reliability_scores = [metrics['reliability_score'] for metrics in quality_metrics.values()]
            overall_quality = {
                'mean_reliability': np.mean(reliability_scores),
                'min_reliability': np.min(reliability_scores),
                'sensors_high_quality': sum(1 for score in reliability_scores if score > 0.8),
                'sensors_low_quality': sum(1 for score in reliability_scores if score < 0.5),
                'total_sensors': len(reliability_scores)
            }
            
            quality_metrics['overall_quality'] = overall_quality
        
        return quality_metrics
        
    except Exception as e:
        print(f"Sensor data quality analysis failed: {e}")
        return None

def calculate_cross_sensor_correlations(X_data, feature_columns, threshold=0.95):
    """Calculate cross-sensor correlations and identify redundant sensors"""
    try:
        correlation_analysis = {}
        
        # Calculate correlation matrix
        corr_matrix = X_data[feature_columns].corr()
        
        # Find highly correlated pairs
        high_correlations = []
        redundant_sensors = set()
        
        for i in range(len(feature_columns)):
            for j in range(i+1, len(feature_columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_correlations.append({
                        'sensor1': feature_columns[i],
                        'sensor2': feature_columns[j],
                        'correlation': corr_val
                    })
                    redundant_sensors.add(feature_columns[j])  # Mark second sensor as redundant
        
        # Sensor importance based on correlations
        sensor_importance = {}
        for col in feature_columns:
            # Calculate average correlation with other sensors
            col_corrs = [abs(corr_matrix.loc[col, other]) for other in feature_columns if other != col]
            avg_correlation = np.mean(col_corrs) if col_corrs else 0
            
            # Higher average correlation might indicate redundancy
            sensor_importance[col] = {
                'avg_correlation': avg_correlation,
                'max_correlation': max(col_corrs) if col_corrs else 0,
                'is_redundant': col in redundant_sensors,
                'unique_information': 1 - avg_correlation  # Approximation
            }
        
        correlation_analysis = {
            'high_correlations': high_correlations,
            'redundant_sensors': list(redundant_sensors),
            'sensor_importance': sensor_importance,
            'correlation_threshold': threshold,
            'total_high_correlations': len(high_correlations)
        }
        
        return correlation_analysis
        
    except Exception as e:
        print(f"Cross-sensor correlation analysis failed: {e}")
        return None

def calculate_uncertainty_metrics(probabilities, predictions):
    """Calculate various uncertainty metrics for predictions"""
    try:
        uncertainties = {}
        
        # Shannon entropy
        entropies = []
        for probs in probabilities:
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        
        entropies = np.array(entropies)
        max_entropy = np.log(len(probabilities[0]))
        normalized_entropies = entropies / max_entropy
        
        # Confidence margin
        margins = []
        for probs in probabilities:
            sorted_probs = sorted(probs, reverse=True)
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            margins.append(margin)
        
        margins = np.array(margins)
        
        # Mutual information approximation
        mutual_info_scores = 1 - normalized_entropies
        
        uncertainties = {
            'shannon_entropy': {
                'values': entropies,
                'normalized': normalized_entropies,
                'mean': np.mean(normalized_entropies),
                'std': np.std(normalized_entropies)
            },
            'confidence_margin': {
                'values': margins,
                'mean': np.mean(margins),
                'std': np.std(margins)
            },
            'mutual_information': {
                'values': mutual_info_scores,
                'mean': np.mean(mutual_info_scores)
            },
            'high_uncertainty_indices': np.where(normalized_entropies > 0.8)[0].tolist(),
            'low_uncertainty_indices': np.where(normalized_entropies < 0.2)[0].tolist(),
            'uncertain_prediction_count': np.sum(normalized_entropies > 0.7)
        }
        
        return uncertainties
        
    except Exception as e:
        print(f"Uncertainty metrics calculation failed: {e}")
        return None

def optimize_temperature_scaling(probabilities, y_true):
    """Optimize temperature parameter for probability calibration"""
    try:
        from scipy.optimize import minimize_scalar
        from scipy.special import softmax
        
        def calibration_loss(temperature):
            # Apply temperature scaling
            scaled_logits = np.log(probabilities + 1e-8) / temperature
            scaled_probs = softmax(scaled_logits, axis=1)
            
            # Calculate negative log-likelihood
            nll = 0
            for i, true_class in enumerate(y_true):
                nll -= np.log(scaled_probs[i, true_class] + 1e-8)
            
            return nll / len(y_true)
        
        # Find optimal temperature
        result = minimize_scalar(
            calibration_loss,
            bounds=(0.1, 5.0),
            method='bounded'
        )
        
        if result.success:
            optimal_temp = result.x
            
            # Calculate calibration metrics
            scaled_logits = np.log(probabilities + 1e-8) / optimal_temp
            calibrated_probs = softmax(scaled_logits, axis=1)
            
            # Expected calibration error approximation
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Get predictions in this confidence bin
                max_probs = np.max(calibrated_probs, axis=1)
                in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
                
                if np.sum(in_bin) > 0:
                    bin_accuracy = np.mean(np.argmax(calibrated_probs[in_bin], axis=1) == y_true[in_bin])
                    bin_confidence = np.mean(max_probs[in_bin])
                    bin_size = np.sum(in_bin) / len(y_true)
                    
                    ece += bin_size * abs(bin_accuracy - bin_confidence)
            
            return {
                'optimal_temperature': optimal_temp,
                'calibrated_probabilities': calibrated_probs,
                'expected_calibration_error': ece,
                'optimization_success': True
            }
        else:
            return {
                'optimal_temperature': 1.0,
                'calibrated_probabilities': probabilities,
                'expected_calibration_error': 1.0,
                'optimization_success': False
            }
    
    except Exception as e:
        print(f"Temperature scaling optimization failed: {e}")
        return None

def calculate_class_metrics(y_true, y_pred, labels=None):
    """Calculate class-wise metrics with support analysis"""
    try:
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        
        class_metrics = []
        for i, label in enumerate(labels):
            class_metrics.append({
                'class': label,
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1_score': f1[i] if i < len(f1) else 0.0,
                'support': support[i] if i < len(support) else 0
            })
        
        return class_metrics
    except Exception as e:
        print(f"Error calculating class metrics: {e}")
        return []

def calculate_stability_metrics(cv_scores_dict):
    """Calculate stability metrics for cross-validation results"""
    stability_metrics = {}
    
    for model_name, scores in cv_scores_dict.items():
        if 'scores' in scores:
            cv_scores = scores['scores']
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # Coefficient of variation (normalized stability)
            cv_stability = std_score / mean_score if mean_score > 0 else float('inf')
            
            # Confidence interval
            confidence_interval = stats.t.interval(
                0.95, len(cv_scores)-1, 
                loc=mean_score, 
                scale=stats.sem(cv_scores)
            )
            
            # Range of scores
            score_range = np.max(cv_scores) - np.min(cv_scores)
            
            # Stability index (lower is more stable)
            stability_index = cv_stability + (score_range / mean_score if mean_score > 0 else 0)
            
            stability_metrics[model_name] = {
                'cv_stability': cv_stability,
                'confidence_interval': confidence_interval,
                'score_range': score_range,
                'confidence_width': confidence_interval[1] - confidence_interval[0],
                'stability_index': stability_index
            }
    
    return stability_metrics

def print_classification_metrics(y_true, y_pred, class_names=None, target_names=None):
    """Print classification performance metrics with detailed analysis"""
    try:
        metrics = calculate_all_metrics(y_true, y_pred)
        
        print("Classification Performance Metrics")
        print("-" * 40)
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric_name:20s}: {value:.4f}")
        
        # Class imbalance analysis
        class_counts = Counter(y_true)
        pred_counts = Counter(y_pred)
        
        print(f"\nClass Distribution Analysis:")
        print(f"True classes: {len(class_counts)}, Predicted classes: {len(pred_counts)}")
        
        # Missing predictions
        missing_classes = set(class_counts.keys()) - set(pred_counts.keys())
        if missing_classes:
            print(f"Classes not predicted: {sorted(missing_classes)}")
        
        if target_names is None and class_names is not None:
            target_names = [str(c) for c in class_names]
        elif target_names is None:
            unique_classes = sorted(list(set(y_true) | set(y_pred)))
            target_names = [str(c) for c in unique_classes]
        
        try:
            report = classification_report(
                y_true, y_pred, 
                target_names=target_names,
                zero_division=0,
                digits=4
            )
            print("\nClassification Report:")
            print(report)
        except Exception as e:
            print(f"Classification report generation failed: {e}")
        
        return metrics['macro_f1']
    except Exception as e:
        print(f"Error printing performance metrics: {e}")
        return 0.0

def create_cv_folds(X, y, n_splits=5, random_state=42):
    """Create stratified cross-validation folds with balance check"""
    try:
        # Check class distribution
        class_counts = Counter(y)
        min_samples = min(class_counts.values())
        
        if min_samples < n_splits:
            print(f"Warning: Minimum class samples ({min_samples}) < n_splits ({n_splits})")
            n_splits = max(2, min_samples)
            print(f"Adjusted n_splits to {n_splits}")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(skf.split(X, y))
        
        # Validate fold quality
        for i, (train_idx, val_idx) in enumerate(folds):
            train_classes = len(set(y[train_idx]))
            val_classes = len(set(y[val_idx]))
            print(f"Fold {i+1}: train classes={train_classes}, val classes={val_classes}")
        
        return folds
        
    except Exception as e:
        print(f"CV fold creation failed: {e}")
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(X))
        folds = []
        for i in range(n_splits):
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, random_state=random_state + i, stratify=y
            )
            folds.append((train_idx, val_idx))
        return folds

def calculate_class_weights(y, method='effective_number', beta=0.9999):
    """Calculate class weights with multiple methods"""
    try:
        if method == 'balanced':
            unique_classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
            return dict(zip(unique_classes, class_weights))
        
        elif method == 'effective_number':
            class_counts = Counter(y)
            class_weights = {}
            
            for class_label, count in class_counts.items():
                effective_num = (1 - np.power(beta, count)) / (1 - beta)
                class_weights[class_label] = (1 - beta) / effective_num
            
            # Normalize weights
            total_weight = sum(class_weights.values())
            for class_label in class_weights:
                class_weights[class_label] = class_weights[class_label] / total_weight * len(class_weights)
            
            return class_weights
        
        elif method == 'inverse_freq':
            class_counts = Counter(y)
            total_samples = len(y)
            weights = {}
            for class_label, count in class_counts.items():
                weights[class_label] = total_samples / (len(class_counts) * count)
            return weights
        
        elif method == 'sqrt_inv_freq':
            class_counts = Counter(y)
            total_samples = len(y)
            weights = {}
            for class_label, count in class_counts.items():
                weights[class_label] = np.sqrt(total_samples / count)
            return weights
        
        elif method == 'log_inv_freq':
            class_counts = Counter(y)
            total_samples = len(y)
            weights = {}
            for class_label, count in class_counts.items():
                weights[class_label] = np.log(total_samples / count + 1)
            return weights
        
        else:
            return None
            
    except Exception as e:
        print(f"Class weight calculation failed: {e}")
        return None

def detect_outliers_iqr(data, multiplier=1.5):
    """Detect outliers using IQR method with robust handling"""
    try:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        if IQR == 0:  # Handle case where IQR is 0
            return np.zeros(len(data), dtype=bool)
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers
    except Exception as e:
        print(f"IQR outlier detection failed: {e}")
        return np.zeros(len(data), dtype=bool)

def detect_outliers_zscore(data, threshold=3):
    """Detect outliers using Z-score method with robust handling"""
    try:
        if np.std(data) == 0:  # Handle constant data
            return np.zeros(len(data), dtype=bool)
            
        z_scores = np.abs(zscore(data, nan_policy='omit'))
        outliers = z_scores > threshold
        
        # Handle NaN values
        outliers = np.nan_to_num(outliers, nan=False)
        return outliers.astype(bool)
    except Exception as e:
        print(f"Z-score outlier detection failed: {e}")
        return np.zeros(len(data), dtype=bool)

def detect_outliers_isolation_forest(data, contamination=0.1):
    """Detect outliers using Isolation Forest"""
    try:
        from sklearn.ensemble import IsolationForest
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(data) == -1
        
        return outliers
    except Exception as e:
        print(f"Isolation forest outlier detection failed: {e}")
        return np.zeros(len(data), dtype=bool)

def timer(func):
    """Function execution time measurement decorator with memory tracking"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        memory_before = memory_usage_check()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            memory_after = memory_usage_check()
            elapsed = end_time - start_time
            memory_increase = memory_after - memory_before
            
            if elapsed < 60:
                print(f"{func.__name__} execution time: {elapsed:.2f} seconds")
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                print(f"{func.__name__} execution time: {minutes}m {seconds:.2f}s")
            
            if memory_increase > 100:  # Display if increase > 100MB
                print(f"Memory increase: {memory_increase:.1f}MB")
            
            # Automatic garbage collection for large memory usage
            if memory_increase > 500:
                gc.collect()
                memory_final = memory_usage_check()
                freed = memory_after - memory_final
                if freed > 10:
                    print(f"Memory freed by GC: {freed:.1f}MB")
            
            return result
            
        except Exception as e:
            print(f"Error during {func.__name__} execution: {e}")
            raise
            
    return wrapper

def check_data_quality(df, feature_columns):
    """Data quality check with sensor-specific validation"""
    try:
        print("Data quality check")
        print(f"Data shape: {df.shape}")
        
        # Check missing values
        missing_info = df[feature_columns].isnull().sum()
        total_missing = missing_info.sum()
        print(f"Total missing values: {total_missing}")
        
        if total_missing > 0:
            print("Columns with missing values (top 5):")
            missing_cols = missing_info[missing_info > 0].sort_values(ascending=False)
            for col in missing_cols.head(5).index:
                print(f"  {col}: {missing_info[col]} ({missing_info[col]/len(df)*100:.2f}%)")
        
        # Check infinite values
        numeric_cols = df[feature_columns].select_dtypes(include=[np.number]).columns
        inf_counts = {}
        total_inf = 0
        
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
                total_inf += inf_count
        
        print(f"Infinite values: {total_inf}")
        
        if total_inf > 0:
            print("Columns with infinite values (top 5):")
            sorted_inf = sorted(inf_counts.items(), key=lambda x: x[1], reverse=True)
            for col, count in sorted_inf[:5]:
                print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
        
        # Sensor-specific quality checks
        sensor_quality_issues = 0
        
        # Check for sensors with zero variance
        zero_var_sensors = []
        for col in numeric_cols:
            if df[col].std() == 0:
                zero_var_sensors.append(col)
                sensor_quality_issues += 1
        
        if zero_var_sensors:
            print(f"Zero variance sensors: {len(zero_var_sensors)}")
        
        # Check for highly correlated sensor pairs
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > 0.98:
                            high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
                            sensor_quality_issues += 1
                
                if high_corr_pairs:
                    print(f"Highly correlated sensor pairs (>0.98): {len(high_corr_pairs)}")
                    
            except Exception as e:
                print(f"Correlation analysis failed: {e}")
        
        # Check data types
        print(f"\nData types:")
        dtype_counts = df[feature_columns].dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Basic statistics
        if len(numeric_cols) > 0:
            print(f"\nBasic statistics ({len(numeric_cols)} numeric columns):")
            stats_df = df[numeric_cols].describe()
            print(f"  Mean range: {stats_df.loc['mean'].min():.4f} ~ {stats_df.loc['mean'].max():.4f}")
            print(f"  Std range: {stats_df.loc['std'].min():.4f} ~ {stats_df.loc['std'].max():.4f}")
            print(f"  Min value: {stats_df.loc['min'].min():.4f}")
            print(f"  Max value: {stats_df.loc['max'].max():.4f}")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"Memory usage: {memory_mb:.2f} MB")
        
        # Overall quality score
        quality_score = 1.0
        if total_missing > 0:
            quality_score -= min(0.3, total_missing / (len(df) * len(feature_columns)))
        if total_inf > 0:
            quality_score -= min(0.2, total_inf / (len(df) * len(feature_columns)))
        if sensor_quality_issues > 0:
            quality_score -= min(0.2, sensor_quality_issues / len(feature_columns))
        
        print(f"Data quality score: {quality_score:.3f}")
        
        return total_missing == 0 and total_inf == 0 and sensor_quality_issues < 5
        
    except Exception as e:
        print(f"Error during data quality check: {e}")
        return False

def memory_usage_check():
    """Check memory usage with system information"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / (1024 ** 3)
        
        if memory_mb > 2000:  # > 2GB
            print(f"High memory usage: {memory_mb:.1f}MB (Available: {available_gb:.1f}GB)")
        
        return memory_mb
    except Exception:
        return 0

def save_results(results, file_path):
    """Save results to CSV file with error handling"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(results, dict):
            df = pd.DataFrame([results])
        elif isinstance(results, list):
            df = pd.DataFrame(results)
        else:
            df = results
        
        # Optimize before saving
        df = optimize_dataframe_memory(df)
        
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Results saved successfully: {file_path}")
        
    except Exception as e:
        print(f"Results save failed: {e}")

def validate_predictions(y_pred, n_classes, sample_ids=None):
    """Validate prediction results with detailed analysis"""
    try:
        print("Prediction validation")
        
        if sample_ids is not None:
            print(f"Sample count: {len(sample_ids)}")
            print(f"Prediction count: {len(y_pred)}")
            
            if len(sample_ids) != len(y_pred):
                print("Warning: Sample count and prediction count do not match")
        
        # Check prediction value range
        min_pred = np.min(y_pred)
        max_pred = np.max(y_pred)
        unique_pred = len(np.unique(y_pred))
        
        print(f"Prediction range: {min_pred} ~ {max_pred}")
        print(f"Unique prediction count: {unique_pred}")
        print(f"Total class count: {n_classes}")
        
        # Validity check
        is_valid = True
        if min_pred < 0 or max_pred >= n_classes:
            print(f"Warning: Predictions outside valid range (0 ~ {n_classes-1})")
            is_valid = False
        
        # Distribution analysis
        unique, counts = np.unique(y_pred, return_counts=True)
        print(f"\nPrediction distribution (top 15):")
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        display_count = min(15, len(unique))
        
        for i in range(display_count):
            idx = sorted_indices[i]
            class_id = unique[idx]
            count = counts[idx]
            percentage = count / len(y_pred) * 100
            print(f"  Class {class_id:2d}: {count:4d} ({percentage:5.2f}%)")
        
        if len(unique) > 15:
            print(f"  ... (total {len(unique)} classes)")
        
        # Check missing classes
        all_classes = set(range(n_classes))
        predicted_classes = set(unique)
        missing_classes = all_classes - predicted_classes
        
        if missing_classes:
            missing_list = sorted(list(missing_classes))
            if len(missing_list) <= 10:
                print(f"Missing classes: {missing_list}")
            else:
                print(f"Missing classes: {missing_list[:10]} ... (total {len(missing_list)})")
            is_valid = False
        
        # Distribution metrics
        probs = counts / len(y_pred)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(n_classes)
        normalized_entropy = entropy / max_entropy
        
        print(f"Distribution entropy: {normalized_entropy:.3f} (1.0 = uniform)")
        
        # Class imbalance metrics
        imbalance_ratio = np.max(counts) / np.max([np.min(counts), 1])
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Quality assessment
        if normalized_entropy < 0.8:
            print("Warning: Low distribution entropy indicates potential bias")
            is_valid = False
            
        if imbalance_ratio > 50:
            print("Warning: Extreme class imbalance detected")
            is_valid = False
        
        return is_valid
        
    except Exception as e:
        print(f"Error during prediction validation: {e}")
        return False

def create_submission_template(test_ids, predictions, id_col='ID', target_col='target'):
    """Create submission file template with validation"""
    try:
        # Validate inputs
        if len(test_ids) != len(predictions):
            raise ValueError(f"Length mismatch: IDs({len(test_ids)}) vs Predictions({len(predictions)})")
        
        submission = pd.DataFrame({
            id_col: test_ids,
            target_col: predictions
        })
        
        # Data type optimization
        submission[target_col] = submission[target_col].astype('int16')
        
        # Validate submission format
        if submission[id_col].duplicated().any():
            print("Warning: Duplicate IDs found in submission")
        
        if submission[target_col].isnull().any():
            print("Warning: Null predictions found in submission")
            submission[target_col].fillna(0, inplace=True)
        
        print(f"Submission template created: {submission.shape}")
        return submission
        
    except Exception as e:
        print(f"Submission file template creation failed: {e}")
        return None

def analyze_class_distribution(y, class_names=None):
    """Analyze class distribution with statistical measures"""
    try:
        print("Class distribution analysis")
        
        unique, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        print(f"Total samples: {total_samples}")
        print(f"Class count: {len(unique)}")
        
        # Distribution statistics
        print(f"\nClass-wise distribution:")
        
        distribution_data = []
        for class_id, count in zip(unique, counts):
            percentage = count / total_samples * 100
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class_{class_id}"
            distribution_data.append({
                'class': class_id,
                'name': class_name,
                'count': count,
                'percentage': percentage
            })
            
            if class_id < 15:  # Display top 15 only
                print(f"  {class_name:>12}: {count:5d} ({percentage:5.2f}%)")
        
        # Statistical measures
        max_count = max(counts)
        min_count = min(counts[counts > 0]) if np.any(counts > 0) else 1
        imbalance_ratio = max_count / min_count
        
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        cv_count = std_count / mean_count if mean_count > 0 else float('inf')
        
        print(f"\nDistribution statistics:")
        print(f"  Max class size: {max_count}")
        print(f"  Min class size: {min_count}")
        print(f"  Mean class size: {mean_count:.1f}")
        print(f"  Std class size: {std_count:.1f}")
        print(f"  Coefficient of variation: {cv_count:.3f}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Effective number of classes (Simpson's diversity index)
        probs = counts / total_samples
        effective_classes = 1 / np.sum(probs ** 2)
        print(f"  Effective number of classes: {effective_classes:.1f}")
        
        # Shannon diversity index
        shannon_diversity = -np.sum(probs * np.log(probs + 1e-10))
        max_shannon = np.log(len(unique))
        normalized_shannon = shannon_diversity / max_shannon
        print(f"  Shannon diversity index: {normalized_shannon:.3f}")
        
        return distribution_data
        
    except Exception as e:
        print(f"Error during class distribution analysis: {e}")
        return []

def garbage_collect():
    """Perform garbage collection with memory reporting"""
    try:
        memory_before = memory_usage_check()
        collected = gc.collect()
        memory_after = memory_usage_check()
        memory_freed = memory_before - memory_after
        
        if collected > 0:
            print(f"Memory cleanup: {collected} objects released, {memory_freed:.1f}MB freed")
        return collected
    except Exception as e:
        print(f"Error during garbage collection: {e}")
        return 0

def format_time(seconds):
    """Convert seconds to readable format"""
    try:
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.2f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}h {remaining_minutes}m {remaining_seconds:.2f}s"
    except Exception:
        return f"{seconds} seconds"

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError, ValueError):
        return default

def check_system_resources():
    """Check system resources with recommendations"""
    try:
        # CPU information
        cpu_count = os.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        memory_usage_percent = memory.percent
        
        # Disk information
        disk = psutil.disk_usage('.')
        disk_total_gb = disk.total / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        
        print(f"System resource status:")
        print(f"  CPU: {cpu_count} cores, usage {cpu_usage}%")
        print(f"  Memory: {memory_available_gb:.1f}GB available / {memory_total_gb:.1f}GB total ({memory_usage_percent:.1f}% used)")
        print(f"  Disk: {disk_free_gb:.1f}GB free / {disk_total_gb:.1f}GB total")
        
        # Resource recommendations
        if memory_usage_percent > 85:
            print("  Warning: High memory usage detected")
        if cpu_usage > 90:
            print("  Warning: High CPU usage detected")
        if disk_free_gb < 5:
            print("  Warning: Low disk space available")
        
        return {
            'cpu_count': cpu_count,
            'cpu_usage': cpu_usage,
            'memory_total_gb': memory_total_gb,
            'memory_available_gb': memory_available_gb,
            'memory_usage_percent': memory_usage_percent,
            'disk_total_gb': disk_total_gb,
            'disk_free_gb': disk_free_gb
        }
        
    except Exception as e:
        print(f"System resource check failed: {e}")
        return None

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage with smart type conversion"""
    try:
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    # For float types, be more conservative with precision requirements
                    if col.startswith(('X_', 'sensor_', 'pca_')):  # Sensor data needs precision
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
                    else:
                        # Non-sensor data can use float32
                        df[col] = df[col].astype(np.float32)
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        memory_reduction = (initial_memory - final_memory) / initial_memory * 100
        
        if memory_reduction > 5:  # Only report significant reductions
            print(f"Memory usage reduction: {initial_memory:.1f}MB -> {final_memory:.1f}MB ({memory_reduction:.1f}% reduction)")
        
        return df
        
    except Exception as e:
        print(f"Memory optimization failed: {e}")
        return df

def validate_data_consistency(train_df, test_df, feature_columns):
    """Validate training and test data consistency with sensor analysis"""
    try:
        print("Data consistency validation")
        
        # Check feature columns
        train_features = set(train_df.columns) & set(feature_columns)
        test_features = set(test_df.columns) & set(feature_columns)
        
        missing_in_test = train_features - test_features
        missing_in_train = test_features - train_features
        
        if missing_in_test:
            print(f"Features missing in test data: {missing_in_test}")
        
        if missing_in_train:
            print(f"Features missing in train data: {missing_in_train}")
        
        # Check data type consistency
        common_features = train_features & test_features
        type_mismatches = []
        
        for feature in common_features:
            if train_df[feature].dtype != test_df[feature].dtype:
                type_mismatches.append({
                    'feature': feature,
                    'train_type': train_df[feature].dtype,
                    'test_type': test_df[feature].dtype
                })
        
        if type_mismatches:
            print("Data type mismatches:")
            for mismatch in type_mismatches:
                print(f"  {mismatch['feature']}: train={mismatch['train_type']}, test={mismatch['test_type']}")
        
        # Statistical distribution comparison (sensor-specific)
        distribution_differences = []
        
        for feature in list(common_features)[:15]:  # Check top 15 only
            train_mean = train_df[feature].mean()
            test_mean = test_df[feature].mean()
            train_std = train_df[feature].std()
            test_std = test_df[feature].std()
            
            mean_diff = abs(train_mean - test_mean) / (abs(train_mean) + 1e-8)
            std_diff = abs(train_std - test_std) / (abs(train_std) + 1e-8)
            
            # More lenient thresholds for sensor data
            if mean_diff > 0.2 or std_diff > 0.3:
                distribution_differences.append({
                    'feature': feature,
                    'mean_diff': mean_diff,
                    'std_diff': std_diff
                })
        
        if distribution_differences:
            print("Features with large distribution differences (top 5):")
            for diff in distribution_differences[:5]:
                print(f"  {diff['feature']}: mean diff {diff['mean_diff']:.3f}, std diff {diff['std_diff']:.3f}")
        
        # Sensor range consistency check
        sensor_range_issues = []
        for feature in common_features:
            train_range = train_df[feature].max() - train_df[feature].min()
            test_range = test_df[feature].max() - test_df[feature].min()
            
            if train_range > 0 and test_range > 0:
                range_ratio = abs(train_range - test_range) / train_range
                if range_ratio > 0.5:  # 50% difference in range
                    sensor_range_issues.append({
                        'feature': feature,
                        'train_range': train_range,
                        'test_range': test_range,
                        'ratio': range_ratio
                    })
        
        if sensor_range_issues:
            print(f"Sensor range inconsistencies: {len(sensor_range_issues)}")
            for issue in sensor_range_issues[:3]:
                print(f"  {issue['feature']}: train range {issue['train_range']:.3f}, test range {issue['test_range']:.3f}")
        
        is_consistent = (len(missing_in_test) == 0 and len(missing_in_train) == 0 and 
                        len(type_mismatches) == 0 and len(distribution_differences) < 8 and
                        len(sensor_range_issues) < 10)
        
        return is_consistent
        
    except Exception as e:
        print(f"Data consistency validation failed: {e}")
        return False

def calculate_performance_gain(baseline_score, improved_score):
    """Calculate performance gain metrics with target analysis"""
    try:
        absolute_gain = improved_score - baseline_score
        relative_gain = (improved_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0
        
        # Target achievement percentage
        target_score = 0.84  # Target macro F1
        target_achievement = (improved_score / target_score * 100) if target_score > 0 else 0
        
        # Gap to target
        gap_to_target = target_score - improved_score
        gap_percentage = gap_to_target / target_score * 100 if target_score > 0 else 0
        
        # Performance tier assessment
        if improved_score >= 0.84:
            tier = "TARGET_ACHIEVED"
        elif improved_score >= 0.75:
            tier = "GOOD"
        elif improved_score >= 0.65:
            tier = "ACCEPTABLE"
        else:
            tier = "NEEDS_WORK"
        
        return {
            'absolute_gain': absolute_gain,
            'relative_gain': relative_gain,
            'target_achievement': target_achievement,
            'gap_to_target': gap_to_target,
            'gap_percentage': gap_percentage,
            'performance_tier': tier,
            'needs_improvement': improved_score < target_score
        }
    except Exception as e:
        print(f"Performance gain calculation failed: {e}")
        return None

def calculate_sensor_stability_index(X_data, feature_columns, window_size=100):
    """Calculate sensor stability index over time windows"""
    try:
        stability_indices = {}
        
        for col in feature_columns:
            if col in X_data.columns:
                col_data = X_data[col].values
                
                if len(col_data) >= window_size * 2:
                    # Calculate variance across sliding windows
                    window_variances = []
                    for i in range(0, len(col_data) - window_size, window_size // 2):
                        window_data = col_data[i:i+window_size]
                        window_var = np.var(window_data)
                        window_variances.append(window_var)
                    
                    # Stability is inverse of variance of variances
                    if len(window_variances) > 1:
                        stability = 1 / (1 + np.var(window_variances))
                    else:
                        stability = 1.0
                else:
                    # For short sequences, use coefficient of variation
                    cv = np.std(col_data) / (abs(np.mean(col_data)) + 1e-8)
                    stability = 1 / (1 + cv)
                
                stability_indices[col] = stability
        
        # Overall system stability
        if stability_indices:
            overall_stability = np.mean(list(stability_indices.values()))
            stability_indices['overall'] = overall_stability
        
        return stability_indices
        
    except Exception as e:
        print(f"Sensor stability calculation failed: {e}")
        return None