# prediction.py

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
from scipy.special import softmax
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, load_model, calculate_macro_f1, validate_predictions, create_submission_template

class PredictionProcessor:
    def __init__(self, model=None):
        self.model = model
        self.predictions = None
        self.prediction_probabilities = None
        self.confidence_scores = None
        self.calibrated_model = None
        self.class_distribution_target = None
        self.temperature_parameter = 1.0
        self.class_isolation_forests = {}
        self.uncertainty_scores = None
        
    def load_trained_model(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = Config.MODEL_FILE
            
        try:
            self.model = load_model(model_path)
            print(f"Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise
    
    def calibrate_model(self, X_val, y_val):
        """Model probability calibration with temperature scaling"""
        if self.model is None:
            print("Model not loaded")
            return
        
        try:
            print("Starting model probability calibration")
            
            # Get uncalibrated probabilities
            if hasattr(self.model, 'predict_proba'):
                uncalibrated_probs = self.model.predict_proba(X_val)
            else:
                print("Model does not support probability prediction")
                return
            
            # Temperature scaling calibration
            self.temperature_parameter = self._optimize_temperature_scaling(
                uncalibrated_probs, y_val
            )
            
            # Store target distribution for balancing
            self.class_distribution_target = y_val.value_counts(normalize=True).sort_index()
            
            print(f"Model calibrated with temperature: {self.temperature_parameter:.3f}")
            
        except Exception as e:
            print(f"Model calibration failed: {e}")
            self.calibrated_model = None
    
    def _optimize_temperature_scaling(self, uncalibrated_probs, y_true):
        """Optimize temperature parameter for calibration"""
        from scipy.optimize import minimize_scalar
        
        def negative_log_likelihood(temperature):
            # Apply temperature scaling
            scaled_logits = np.log(uncalibrated_probs + 1e-8) / temperature
            scaled_probs = softmax(scaled_logits, axis=1)
            
            # Calculate negative log-likelihood
            nll = 0
            for i, true_class in enumerate(y_true):
                nll -= np.log(scaled_probs[i, true_class] + 1e-8)
            
            return nll / len(y_true)
        
        # Find optimal temperature
        result = minimize_scalar(
            negative_log_likelihood,
            bounds=(0.1, 5.0),
            method='bounded'
        )
        
        return result.x if result.success else 1.0
    
    def _calculate_uncertainty_metrics(self, probabilities):
        """Calculate prediction uncertainty using multiple metrics"""
        uncertainties = np.zeros(len(probabilities))
        
        for i, probs in enumerate(probabilities):
            # Shannon entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            # Predictive entropy normalized
            max_entropy = np.log(len(probs))
            normalized_entropy = entropy / max_entropy
            
            # Mutual information approximation
            top_2_probs = sorted(probs, reverse=True)[:2]
            margin = top_2_probs[0] - top_2_probs[1]
            margin_uncertainty = 1.0 - margin
            
            # Combined uncertainty score
            uncertainties[i] = 0.6 * normalized_entropy + 0.4 * margin_uncertainty
        
        return uncertainties
    
    def _apply_class_specific_correction(self, predictions, probabilities):
        """Apply class-specific prediction corrections"""
        corrected_predictions = predictions.copy()
        
        # Calculate class confidence thresholds
        class_thresholds = {}
        for class_id in range(Config.N_CLASSES):
            class_mask = predictions == class_id
            if np.sum(class_mask) > 10:
                class_confidences = np.max(probabilities[class_mask], axis=1)
                class_thresholds[class_id] = np.percentile(class_confidences, 25)
            else:
                class_thresholds[class_id] = 0.5
        
        # Apply corrections for low-confidence predictions
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            current_confidence = probs[pred]
            threshold = class_thresholds.get(pred, 0.5)
            
            if current_confidence < threshold:
                # Find alternative high-confidence class
                sorted_indices = np.argsort(probs)[::-1]
                for alt_class in sorted_indices[1:4]:  # Check top 3 alternatives
                    alt_confidence = probs[alt_class]
                    alt_threshold = class_thresholds.get(alt_class, 0.5)
                    
                    if alt_confidence > alt_threshold * 1.1:  # 10% higher threshold
                        corrected_predictions[i] = alt_class
                        break
        
        return corrected_predictions
    
    def _multi_class_isolation_detection(self, probabilities):
        """Multi-class isolation forest for anomaly detection"""
        anomaly_scores = np.zeros(len(probabilities))
        
        try:
            from sklearn.ensemble import IsolationForest
            
            # Create isolation forest for probability patterns
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=Config.RANDOM_STATE,
                n_estimators=100
            )
            
            # Fit on probability distributions
            iso_forest.fit(probabilities)
            
            # Get anomaly scores (lower scores = more anomalous)
            anomaly_scores = iso_forest.decision_function(probabilities)
            
        except Exception as e:
            print(f"Isolation forest detection failed: {e}")
            anomaly_scores = np.zeros(len(probabilities))
        
        return anomaly_scores
    
    @timer
    def predict(self, X_test, use_calibrated=True):
        """Model prediction with uncertainty quantification"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        print("Performing model prediction with uncertainty analysis")
        
        # Get raw probabilities
        if hasattr(self.model, 'predict_proba'):
            raw_probabilities = self.model.predict_proba(X_test)
            print(f"Raw prediction probability shape: {raw_probabilities.shape}")
            
            # Apply temperature scaling if calibrated
            if use_calibrated and self.temperature_parameter != 1.0:
                scaled_logits = np.log(raw_probabilities + 1e-8) / self.temperature_parameter
                self.prediction_probabilities = softmax(scaled_logits, axis=1)
                print(f"Applied temperature scaling: {self.temperature_parameter:.3f}")
            else:
                self.prediction_probabilities = raw_probabilities
            
        else:
            print("Probability prediction not available")
            self.prediction_probabilities = None
        
        # Initial predictions
        if self.prediction_probabilities is not None:
            initial_predictions = np.argmax(self.prediction_probabilities, axis=1)
        else:
            initial_predictions = self.model.predict(X_test)
        
        # Calculate uncertainty metrics
        if self.prediction_probabilities is not None:
            self.uncertainty_scores = self._calculate_uncertainty_metrics(
                self.prediction_probabilities
            )
            
            # Apply class-specific corrections
            self.predictions = self._apply_class_specific_correction(
                initial_predictions, self.prediction_probabilities
            )
            
            # Multi-class isolation detection
            anomaly_scores = self._multi_class_isolation_detection(
                self.prediction_probabilities
            )
            
            # Calculate final confidence scores
            max_probs = np.max(self.prediction_probabilities, axis=1)
            normalized_anomaly = (anomaly_scores - np.min(anomaly_scores)) / (
                np.max(anomaly_scores) - np.min(anomaly_scores) + 1e-8
            )
            
            # Combined confidence score
            self.confidence_scores = (0.5 * max_probs + 
                                    0.3 * (1 - self.uncertainty_scores) + 
                                    0.2 * normalized_anomaly)
        else:
            self.predictions = initial_predictions
            self.confidence_scores = np.ones(len(initial_predictions)) * 0.5
        
        print(f"Final prediction shape: {self.predictions.shape}")
        
        # Validate results
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    def _adaptive_distribution_balancing(self, predictions, probabilities, method='entropy_weighted'):
        """Balance prediction distribution with adaptive weights"""
        if probabilities is None:
            return predictions
        
        print(f"Starting adaptive distribution balancing ({method})")
        
        balanced_predictions = predictions.copy()
        total_samples = len(predictions)
        current_counts = Counter(predictions)
        
        # Calculate target distribution
        if self.class_distribution_target is not None:
            target_counts = {}
            for class_id in range(Config.N_CLASSES):
                if class_id in self.class_distribution_target.index:
                    target_prop = self.class_distribution_target[class_id]
                else:
                    target_prop = 1.0 / Config.N_CLASSES
                target_counts[class_id] = int(total_samples * target_prop)
        else:
            # Uniform distribution as fallback
            base_count = total_samples // Config.N_CLASSES
            remainder = total_samples % Config.N_CLASSES
            target_counts = {i: base_count + (1 if i < remainder else 0) 
                           for i in range(Config.N_CLASSES)}
        
        # Adaptive rebalancing with uncertainty consideration
        if method == 'entropy_weighted':
            return self._entropy_weighted_balancing(
                balanced_predictions, probabilities, current_counts, target_counts
            )
        else:
            return self._confidence_weighted_balancing(
                balanced_predictions, probabilities, current_counts, target_counts
            )
    
    def _entropy_weighted_balancing(self, predictions, probabilities, current_counts, target_counts):
        """Entropy-weighted balancing strategy"""
        uncertainty_indices = np.argsort(self.uncertainty_scores)[::-1]
        
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts.get(class_id, 0)
            target_count = target_counts[class_id]
            
            if current_count != target_count:
                diff = target_count - current_count
                
                if diff > 0:  # Need more predictions for this class
                    added = 0
                    for idx in uncertainty_indices:
                        if added >= diff:
                            break
                        
                        if predictions[idx] != class_id and probabilities[idx, class_id] > 0.1:
                            # Check if uncertainty allows change
                            if self.uncertainty_scores[idx] > 0.3:  # High uncertainty
                                old_class = predictions[idx]
                                predictions[idx] = class_id
                                
                                # Update counts
                                current_counts[class_id] = current_counts.get(class_id, 0) + 1
                                if old_class in current_counts:
                                    current_counts[old_class] = max(0, current_counts[old_class] - 1)
                                
                                added += 1
                
                elif diff < 0:  # Need fewer predictions for this class
                    removed = 0
                    class_indices = np.where(predictions == class_id)[0]
                    
                    # Sort by uncertainty and low probability
                    class_uncertainties = [(idx, self.uncertainty_scores[idx], 
                                          probabilities[idx, class_id]) 
                                         for idx in class_indices]
                    class_uncertainties.sort(key=lambda x: (x[1], -x[2]), reverse=True)
                    
                    for idx, uncertainty, prob in class_uncertainties:
                        if removed >= -diff:
                            break
                        
                        if uncertainty > 0.4:  # High uncertainty
                            # Find best alternative
                            alt_probs = probabilities[idx].copy()
                            alt_probs[class_id] = 0
                            best_alt = np.argmax(alt_probs)
                            
                            if alt_probs[best_alt] > 0.1:
                                predictions[idx] = best_alt
                                current_counts[class_id] -= 1
                                current_counts[best_alt] = current_counts.get(best_alt, 0) + 1
                                removed += 1
        
        return predictions
    
    def _confidence_weighted_balancing(self, predictions, probabilities, current_counts, target_counts):
        """Confidence-weighted balancing strategy"""
        confidence_indices = np.argsort(self.confidence_scores)
        
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts.get(class_id, 0)
            target_count = target_counts[class_id]
            
            if current_count < target_count:
                needed = target_count - current_count
                candidates = []
                
                # Find low-confidence predictions that could change
                for idx in confidence_indices:
                    if predictions[idx] != class_id and probabilities[idx, class_id] > 0.08:
                        if self.confidence_scores[idx] < 0.6:  # Low confidence
                            candidates.append((idx, probabilities[idx, class_id]))
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                for i, (idx, prob) in enumerate(candidates[:needed]):
                    old_class = predictions[idx]
                    predictions[idx] = class_id
                    
                    current_counts[class_id] = current_counts.get(class_id, 0) + 1
                    if old_class in current_counts:
                        current_counts[old_class] = max(0, current_counts[old_class] - 1)
        
        return predictions
    
    @timer
    def create_submission_file(self, test_ids, output_path=None, predictions=None, 
                             apply_balancing=True, confidence_threshold=0.6):
        """Create submission file with multi-stage prediction refinement"""
        if predictions is None:
            predictions = self.predictions
        
        if predictions is None:
            raise ValueError("No predictions available")
        
        if output_path is None:
            output_path = Config.RESULT_FILE
        
        print("Creating submission file with multi-stage refinement")
        
        # Stage 1: Confidence-based filtering
        if self.prediction_probabilities is not None and self.confidence_scores is not None:
            low_confidence_mask = self.confidence_scores < confidence_threshold
            low_confidence_count = np.sum(low_confidence_mask)
            
            print(f"Low confidence predictions (< {confidence_threshold}): {low_confidence_count}")
            
            if low_confidence_count > 0:
                predictions = self._refine_low_confidence_predictions(
                    predictions, low_confidence_mask
                )
        
        # Stage 2: Distribution balancing
        if apply_balancing and self.prediction_probabilities is not None:
            balancing_methods = ['entropy_weighted', 'confidence_weighted']
            best_predictions = predictions
            best_quality = 0
            
            for method in balancing_methods:
                try:
                    balanced_preds = self._adaptive_distribution_balancing(
                        predictions, self.prediction_probabilities, method
                    )
                    
                    # Evaluate quality
                    quality = self._evaluate_prediction_quality(balanced_preds)
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_predictions = balanced_preds
                        print(f"Best balancing method: {method} (quality: {quality:.3f})")
                        
                except Exception as e:
                    print(f"Balancing method {method} failed: {e}")
                    continue
            
            predictions = best_predictions
        
        # Validate final predictions
        if len(test_ids) != len(predictions):
            raise ValueError(f"ID count ({len(test_ids)}) != prediction count ({len(predictions)})")
        
        submission_df = create_submission_template(
            test_ids, predictions,
            Config.ID_COLUMN, Config.TARGET_COLUMN
        )
        
        # Data validation
        print(f"Submission shape: {submission_df.shape}")
        print(f"Prediction range: {submission_df[Config.TARGET_COLUMN].min()} ~ {submission_df[Config.TARGET_COLUMN].max()}")
        
        # Check for invalid predictions
        invalid_predictions = ((submission_df[Config.TARGET_COLUMN] < 0) | 
                             (submission_df[Config.TARGET_COLUMN] >= Config.N_CLASSES))
        
        if invalid_predictions.any():
            invalid_count = invalid_predictions.sum()
            print(f"Invalid predictions found: {invalid_count}")
            
            # Fix with balanced assignment
            valid_classes = list(range(Config.N_CLASSES))
            for idx in submission_df[invalid_predictions].index:
                submission_df.loc[idx, Config.TARGET_COLUMN] = np.random.choice(valid_classes)
            print("Fixed invalid predictions")
        
        # Save file
        try:
            submission_df.to_csv(output_path, index=False)
            print(f"Submission file saved: {output_path}")
        except Exception as e:
            print(f"File save failed: {e}")
            raise
        
        # Final analysis
        self.analyze_prediction_distribution(predictions)
        
        return submission_df
    
    def _refine_low_confidence_predictions(self, predictions, low_confidence_mask):
        """Refine predictions with low confidence scores"""
        refined_predictions = predictions.copy()
        
        low_conf_indices = np.where(low_confidence_mask)[0]
        
        for idx in low_conf_indices:
            if self.prediction_probabilities is not None:
                probs = self.prediction_probabilities[idx]
                uncertainty = self.uncertainty_scores[idx] if self.uncertainty_scores is not None else 0.5
                
                # If very uncertain, use ensemble voting
                if uncertainty > 0.7:
                    top_3_classes = np.argsort(probs)[-3:][::-1]
                    top_3_probs = probs[top_3_classes]
                    
                    # Weighted voting based on class frequency balance
                    weights = np.ones(3)
                    current_counts = Counter(refined_predictions)
                    
                    for i, class_id in enumerate(top_3_classes):
                        count = current_counts.get(class_id, 0)
                        expected = len(predictions) / Config.N_CLASSES
                        if count > expected * 1.5:  # Over-represented
                            weights[i] *= 0.5
                        elif count < expected * 0.5:  # Under-represented
                            weights[i] *= 2.0
                    
                    # Select class with highest weighted probability
                    weighted_probs = top_3_probs * weights
                    best_idx = np.argmax(weighted_probs)
                    refined_predictions[idx] = top_3_classes[best_idx]
        
        return refined_predictions
    
    def _evaluate_prediction_quality(self, predictions):
        """Evaluate prediction quality using multiple metrics"""
        try:
            # Distribution uniformity
            counts = Counter(predictions)
            class_counts = [counts.get(i, 0) for i in range(Config.N_CLASSES)]
            
            # Shannon entropy for uniformity
            total = len(predictions)
            probs = np.array(class_counts) / total
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(Config.N_CLASSES)
            uniformity_score = entropy / max_entropy
            
            # Confidence consistency
            confidence_score = 0.5
            if self.confidence_scores is not None:
                confidence_score = np.mean(self.confidence_scores)
            
            # Combined quality score
            quality = 0.6 * uniformity_score + 0.4 * confidence_score
            
            return quality
            
        except Exception as e:
            print(f"Quality evaluation failed: {e}")
            return 0.0
    
    def analyze_prediction_distribution(self, predictions=None):
        """Analyze prediction distribution with quality metrics"""
        if predictions is None:
            predictions = self.predictions
        
        if predictions is None:
            print("No predictions available")
            return None
        
        print("Prediction distribution analysis")
        
        unique, counts = np.unique(predictions, return_counts=True)
        total_predictions = len(predictions)
        
        print("Class distribution:")
        for class_id in range(min(15, Config.N_CLASSES)):
            count = counts[unique == class_id][0] if class_id in unique else 0
            percentage = (count / total_predictions) * 100
            print(f"Class {class_id:2d}: {count:4d} ({percentage:5.2f}%)")
        
        if Config.N_CLASSES > 15:
            print(f"... (total {Config.N_CLASSES} classes)")
        
        # Quality metrics
        expected_per_class = total_predictions / Config.N_CLASSES
        actual_counts = [counts[unique == i][0] if i in unique else 0 for i in range(Config.N_CLASSES)]
        
        print(f"\nDistribution metrics:")
        print(f"Total predictions: {total_predictions}")
        print(f"Expected per class: {expected_per_class:.1f}")
        print(f"Standard deviation: {np.std(actual_counts):.2f}")
        
        # Entropy analysis
        probs = np.array(actual_counts) / total_predictions
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(Config.N_CLASSES)
        normalized_entropy = entropy / max_entropy
        
        print(f"Distribution entropy: {normalized_entropy:.3f}")
        
        # Imbalance metrics
        max_count = max(actual_counts)
        min_count = min([c for c in actual_counts if c > 0]) if any(actual_counts) else 1
        imbalance_ratio = max_count / min_count
        
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Missing classes
        missing_classes = [i for i in range(Config.N_CLASSES) if i not in unique]
        if missing_classes:
            print(f"Missing classes: {missing_classes}")
        
        # Quality score
        gini = self._calculate_gini_coefficient(actual_counts)
        quality_score = (normalized_entropy * 0.4 + 
                        (1 - min(gini, 1.0)) * 0.3 + 
                        (1 - min(imbalance_ratio / 10, 1.0)) * 0.3)
        
        print(f"Quality score: {quality_score:.3f}")
        
        return {
            'distribution': dict(zip(range(Config.N_CLASSES), actual_counts)),
            'total_predictions': total_predictions,
            'entropy': normalized_entropy,
            'imbalance_ratio': imbalance_ratio,
            'missing_classes': missing_classes,
            'quality_score': quality_score,
            'gini_coefficient': gini
        }
    
    def _calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for inequality measurement"""
        try:
            values = np.array(values)
            if np.sum(values) == 0:
                return 0.0
            
            n = len(values)
            sorted_values = np.sort(values)
            cumsum = np.cumsum(sorted_values)
            
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        except:
            return 0.0
    
    def validate_predictions(self, y_true=None):
        """Validate prediction results with uncertainty analysis"""
        if self.predictions is None:
            print("No predictions available")
            return None
        
        print("Prediction validation with uncertainty analysis")
        
        validation_results = {
            'prediction_count': len(self.predictions),
            'unique_classes': len(np.unique(self.predictions)),
            'prediction_range': (self.predictions.min(), self.predictions.max())
        }
        
        # Confidence analysis
        if self.confidence_scores is not None:
            confidence_stats = {
                'mean_confidence': np.mean(self.confidence_scores),
                'median_confidence': np.median(self.confidence_scores),
                'std_confidence': np.std(self.confidence_scores)
            }
            
            validation_results['confidence_stats'] = confidence_stats
            
            print(f"Confidence statistics:")
            print(f"  Mean: {confidence_stats['mean_confidence']:.4f}")
            print(f"  Median: {confidence_stats['median_confidence']:.4f}")
            print(f"  Std: {confidence_stats['std_confidence']:.4f}")
        
        # Uncertainty analysis
        if self.uncertainty_scores is not None:
            uncertainty_stats = {
                'mean_uncertainty': np.mean(self.uncertainty_scores),
                'high_uncertainty_count': np.sum(self.uncertainty_scores > 0.7),
                'low_uncertainty_count': np.sum(self.uncertainty_scores < 0.3)
            }
            
            validation_results['uncertainty_stats'] = uncertainty_stats
            
            print(f"Uncertainty analysis:")
            print(f"  Mean uncertainty: {uncertainty_stats['mean_uncertainty']:.4f}")
            print(f"  High uncertainty (>0.7): {uncertainty_stats['high_uncertainty_count']}")
            print(f"  Low uncertainty (<0.3): {uncertainty_stats['low_uncertainty_count']}")
        
        # Performance analysis if true labels provided
        if y_true is not None:
            macro_f1 = calculate_macro_f1(y_true, self.predictions)
            validation_results['macro_f1'] = macro_f1
            
            print(f"Macro F1 Score: {macro_f1:.4f}")
            
            # Class-wise analysis
            try:
                report = classification_report(y_true, self.predictions, 
                                             output_dict=True, zero_division=0)
                
                class_metrics = []
                for class_id in range(Config.N_CLASSES):
                    class_key = str(class_id)
                    if class_key in report:
                        class_metrics.append({
                            'class': class_id,
                            'precision': report[class_key]['precision'],
                            'recall': report[class_key]['recall'],
                            'f1_score': report[class_key]['f1-score'],
                            'support': report[class_key]['support']
                        })
                
                validation_results['class_metrics'] = class_metrics
                
                # Performance tier classification
                if macro_f1 >= 0.84:
                    tier = "TARGET_ACHIEVED"
                elif macro_f1 >= 0.75:
                    tier = "GOOD"
                elif macro_f1 >= 0.65:
                    tier = "ACCEPTABLE"
                else:
                    tier = "NEEDS_WORK"
                
                validation_results['performance_tier'] = tier
                print(f"Performance tier: {tier}")
                
            except Exception as e:
                print(f"Detailed analysis failed: {e}")
        
        return validation_results
    
    def get_prediction_insights(self):
        """Get detailed prediction insights"""
        if self.prediction_probabilities is None:
            return None
        
        insights = {
            'temperature_scaling': self.temperature_parameter,
            'total_predictions': len(self.predictions),
            'calibration_applied': self.temperature_parameter != 1.0
        }
        
        # Probability distribution insights
        max_probs = np.max(self.prediction_probabilities, axis=1)
        insights['probability_stats'] = {
            'mean_max_prob': np.mean(max_probs),
            'median_max_prob': np.median(max_probs),
            'high_confidence_count': np.sum(max_probs > 0.8),
            'low_confidence_count': np.sum(max_probs < 0.5)
        }
        
        # Class prediction confidence
        class_confidences = {}
        for class_id in range(Config.N_CLASSES):
            class_mask = self.predictions == class_id
            if np.sum(class_mask) > 0:
                class_conf = np.mean(max_probs[class_mask])
                class_confidences[class_id] = class_conf
        
        insights['class_confidence'] = class_confidences
        
        return insights