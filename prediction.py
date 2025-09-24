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
        """Model probability calibration with improved method"""
        if self.model is None:
            print("Model not loaded")
            return
        
        try:
            print("Starting model probability calibration")
            
            # Use sigmoid calibration for better performance
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method='sigmoid',  # Changed from isotonic
                cv=5,  # Increased CV folds
                ensemble=False  # Use base estimator for speed
            )
            self.calibrated_model.fit(X_val, y_val)
            
            # Store target distribution for balancing
            self.class_distribution_target = y_val.value_counts(normalize=True).sort_index()
            
            print("Model probability calibration completed")
            
        except Exception as e:
            print(f"Model calibration failed: {e}")
            self.calibrated_model = None
    
    @timer
    def predict(self, X_test, use_calibrated=True):
        """Model prediction with confidence analysis"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        print("Performing model prediction")
        
        # Select model to use
        active_model = self.calibrated_model if (use_calibrated and self.calibrated_model is not None) else self.model
        
        # Prediction probabilities
        if hasattr(active_model, 'predict_proba'):
            self.prediction_probabilities = active_model.predict_proba(X_test)
            print(f"Prediction probability shape: {self.prediction_probabilities.shape}")
            
            # Apply temperature scaling for better calibration
            self.prediction_probabilities = self._apply_temperature_scaling(self.prediction_probabilities)
            
        else:
            print("Probability prediction not available")
            self.prediction_probabilities = None
        
        # Prediction classes
        if self.prediction_probabilities is not None:
            # Use probability-based prediction instead of direct predict
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)
        else:
            self.predictions = active_model.predict(X_test)
            
        print(f"Prediction result shape: {self.predictions.shape}")
        
        # Calculate confidence scores
        if self.prediction_probabilities is not None:
            self.confidence_scores = np.max(self.prediction_probabilities, axis=1)
            
            # Calculate entropy-based uncertainty
            entropy_scores = self._calculate_prediction_entropy(self.prediction_probabilities)
            
            # Combine max probability and entropy for better confidence
            normalized_entropy = entropy_scores / np.log(Config.N_CLASSES)
            self.confidence_scores = self.confidence_scores * (1 - normalized_entropy * 0.3)
        
        # Validate prediction results
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    def _apply_temperature_scaling(self, probabilities, temperature=1.2):
        """Apply temperature scaling to improve calibration"""
        try:
            # Apply temperature scaling
            scaled_logits = np.log(probabilities + 1e-8) / temperature
            scaled_probabilities = softmax(scaled_logits, axis=1)
            
            return scaled_probabilities
            
        except Exception as e:
            print(f"Temperature scaling failed: {e}")
            return probabilities
    
    def _calculate_prediction_entropy(self, probabilities):
        """Calculate prediction entropy for uncertainty measurement"""
        try:
            entropies = []
            for i in range(len(probabilities)):
                probs = probabilities[i]
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
            
            return np.array(entropies)
            
        except Exception as e:
            print(f"Entropy calculation failed: {e}")
            return np.zeros(len(probabilities))
    
    def _adjust_predictions_by_confidence(self, predictions, probabilities, threshold=0.7):
        """Adjust predictions based on confidence with class balance consideration"""
        if probabilities is None:
            return predictions
        
        adjusted_predictions = predictions.copy()
        max_probs = np.max(probabilities, axis=1)
        low_confidence_mask = max_probs < threshold
        
        print(f"Low confidence predictions (< {threshold}): {np.sum(low_confidence_mask)}")
        
        if np.sum(low_confidence_mask) > 0:
            low_conf_indices = np.where(low_confidence_mask)[0]
            
            # Get current prediction distribution
            current_pred_counts = Counter(adjusted_predictions)
            total_samples = len(predictions)
            expected_count_per_class = total_samples / Config.N_CLASSES
            
            for idx in low_conf_indices:
                probs = probabilities[idx]
                sorted_indices = np.argsort(probs)[::-1]
                
                # Consider top 3 classes instead of just 2
                top_classes = sorted_indices[:3]
                top_probs = probs[top_classes]
                
                # If top 2 probabilities are close, consider class balance
                if len(top_classes) >= 2 and top_probs[0] - top_probs[1] < 0.15:
                    class1, class2 = top_classes[0], top_classes[1]
                    
                    # Prefer class with fewer current predictions
                    count1 = current_pred_counts.get(class1, 0)
                    count2 = current_pred_counts.get(class2, 0)
                    
                    if count2 < count1 and count2 < expected_count_per_class * 1.2:
                        adjusted_predictions[idx] = class2
                        current_pred_counts[class2] = current_pred_counts.get(class2, 0) + 1
                        if class1 in current_pred_counts:
                            current_pred_counts[class1] -= 1
        
        return adjusted_predictions
    
    def _balance_prediction_distribution(self, predictions, probabilities=None, method='entropy_balance'):
        """Balance prediction distribution with multiple strategies"""
        if probabilities is None:
            print("No probability information available for distribution adjustment")
            return predictions
        
        print(f"Starting prediction distribution balancing ({method})")
        
        current_counts = Counter(predictions)
        total_samples = len(predictions)
        
        if method == 'entropy_balance':
            return self._entropy_based_balancing(predictions, probabilities, current_counts, total_samples)
        elif method == 'target_distribution':
            return self._target_distribution_balancing(predictions, probabilities, current_counts, total_samples)
        else:
            return self._uniform_balancing(predictions, probabilities, current_counts, total_samples)
    
    def _entropy_based_balancing(self, predictions, probabilities, current_counts, total_samples):
        """Entropy-based distribution balancing"""
        balanced_predictions = predictions.copy()
        expected_count_per_class = total_samples / Config.N_CLASSES
        
        # Calculate prediction entropy
        entropies = self._calculate_prediction_entropy(probabilities)
        
        # Sort predictions by entropy (high entropy = uncertain predictions)
        entropy_indices = np.argsort(entropies)[::-1]  # High entropy first
        
        # Adjust predictions for classes that are over/under-represented
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts.get(class_id, 0)
            target_range = (int(expected_count_per_class * 0.7), int(expected_count_per_class * 1.3))
            
            if current_count < target_range[0]:
                # Under-represented class - add predictions
                needed = target_range[0] - current_count
                candidates = []
                
                # Find high-entropy predictions that could be changed to this class
                for idx in entropy_indices:
                    if balanced_predictions[idx] != class_id and probabilities[idx, class_id] > 0.1:
                        candidates.append((idx, probabilities[idx, class_id]))
                
                # Sort by probability for this class and take top candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                for i, (idx, prob) in enumerate(candidates[:needed]):
                    old_class = balanced_predictions[idx]
                    balanced_predictions[idx] = class_id
                    
                    # Update counts
                    current_counts[class_id] = current_counts.get(class_id, 0) + 1
                    if old_class in current_counts:
                        current_counts[old_class] -= 1
            
            elif current_count > target_range[1]:
                # Over-represented class - remove predictions
                excess = current_count - target_range[1]
                class_indices = np.where(balanced_predictions == class_id)[0]
                
                # Sort by entropy and probability for this class
                class_entropies = [(idx, entropies[idx], probabilities[idx, class_id]) 
                                 for idx in class_indices]
                class_entropies.sort(key=lambda x: (x[1], -x[2]), reverse=True)
                
                # Remove high entropy, low probability predictions
                for i, (idx, entropy, prob) in enumerate(class_entropies[:excess]):
                    # Find best alternative class
                    probs = probabilities[idx]
                    probs_copy = probs.copy()
                    probs_copy[class_id] = 0  # Exclude current class
                    
                    # Prefer under-represented classes
                    for alt_class in np.argsort(probs_copy)[::-1]:
                        alt_count = current_counts.get(alt_class, 0)
                        if alt_count < expected_count_per_class * 1.2 and probs_copy[alt_class] > 0.05:
                            balanced_predictions[idx] = alt_class
                            current_counts[alt_class] = current_counts.get(alt_class, 0) + 1
                            current_counts[class_id] -= 1
                            break
        
        return balanced_predictions
    
    def _target_distribution_balancing(self, predictions, probabilities, current_counts, total_samples):
        """Balance using target distribution from training data"""
        if self.class_distribution_target is None:
            return self._uniform_balancing(predictions, probabilities, current_counts, total_samples)
        
        balanced_predictions = predictions.copy()
        
        # Calculate target counts based on training distribution
        target_counts = {}
        for class_id in range(Config.N_CLASSES):
            if class_id in self.class_distribution_target.index:
                target_proportion = self.class_distribution_target[class_id]
            else:
                target_proportion = 1.0 / Config.N_CLASSES
            target_counts[class_id] = int(total_samples * target_proportion)
        
        # Ensure total equals sample count
        total_target = sum(target_counts.values())
        if total_target != total_samples:
            diff = total_samples - total_target
            # Add remainder to most under-represented classes
            for class_id in sorted(target_counts.keys(), key=lambda x: current_counts.get(x, 0)):
                if diff == 0:
                    break
                target_counts[class_id] += 1
                diff -= 1
        
        return self._rebalance_to_targets(balanced_predictions, probabilities, current_counts, target_counts)
    
    def _uniform_balancing(self, predictions, probabilities, current_counts, total_samples):
        """Uniform distribution balancing"""
        target_count = total_samples // Config.N_CLASSES
        remainder = total_samples % Config.N_CLASSES
        
        target_counts = {}
        for class_id in range(Config.N_CLASSES):
            target_counts[class_id] = target_count + (1 if class_id < remainder else 0)
        
        return self._rebalance_to_targets(predictions.copy(), probabilities, current_counts, target_counts)
    
    def _rebalance_to_targets(self, predictions, probabilities, current_counts, target_counts):
        """Rebalance predictions to match target counts"""
        # Calculate entropy for uncertainty-based adjustments
        entropies = self._calculate_prediction_entropy(probabilities)
        
        for class_id in range(Config.N_CLASSES):
            current_count = current_counts.get(class_id, 0)
            target_count = target_counts[class_id]
            
            if current_count != target_count:
                diff = target_count - current_count
                
                if diff > 0:
                    # Need to add predictions for this class
                    self._add_predictions_for_class(predictions, probabilities, entropies, 
                                                  class_id, diff, current_counts)
                else:
                    # Need to remove predictions for this class
                    self._remove_predictions_for_class(predictions, probabilities, entropies, 
                                                     class_id, -diff, current_counts, target_counts)
        
        return predictions
    
    def _add_predictions_for_class(self, predictions, probabilities, entropies, target_class, needed, current_counts):
        """Add predictions for under-represented class"""
        # Find candidates: high entropy predictions with decent probability for target class
        candidates = []
        
        for idx in range(len(predictions)):
            if predictions[idx] != target_class and probabilities[idx, target_class] > 0.1:
                score = entropies[idx] * probabilities[idx, target_class]  # Entropy * probability
                candidates.append((idx, score, probabilities[idx, target_class]))
        
        # Sort by score (high entropy and high probability preferred)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Add predictions
        added = 0
        for idx, score, prob in candidates:
            if added >= needed:
                break
            
            old_class = predictions[idx]
            predictions[idx] = target_class
            
            # Update counts
            current_counts[target_class] = current_counts.get(target_class, 0) + 1
            if old_class in current_counts:
                current_counts[old_class] = max(0, current_counts[old_class] - 1)
            
            added += 1
    
    def _remove_predictions_for_class(self, predictions, probabilities, entropies, target_class, 
                                    excess, current_counts, target_counts):
        """Remove predictions for over-represented class"""
        # Find class predictions sorted by uncertainty
        class_indices = np.where(predictions == target_class)[0]
        
        # Create candidates with entropy and alternative class probability
        candidates = []
        for idx in class_indices:
            probs = probabilities[idx].copy()
            probs[target_class] = 0  # Exclude current class
            
            best_alt_class = np.argmax(probs)
            best_alt_prob = probs[best_alt_class]
            
            # Score: high entropy, low current class prob, high alternative prob
            current_class_prob = probabilities[idx, target_class]
            score = entropies[idx] * best_alt_prob / (current_class_prob + 1e-8)
            
            candidates.append((idx, score, best_alt_class, best_alt_prob))
        
        # Sort by score (prefer uncertain predictions with good alternatives)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Remove predictions
        removed = 0
        for idx, score, alt_class, alt_prob in candidates:
            if removed >= excess:
                break
            
            # Check if alternative class can accept more predictions
            alt_current = current_counts.get(alt_class, 0)
            alt_target = target_counts.get(alt_class, len(predictions) // Config.N_CLASSES)
            
            if alt_current < alt_target * 1.2 and alt_prob > 0.05:
                predictions[idx] = alt_class
                
                # Update counts
                current_counts[target_class] -= 1
                current_counts[alt_class] = current_counts.get(alt_class, 0) + 1
                
                removed += 1
    
    @timer
    def create_submission_file(self, test_ids, output_path=None, predictions=None, 
                             apply_balancing=True, confidence_threshold=0.7):
        """Create submission file with multiple balancing strategies"""
        if predictions is None:
            predictions = self.predictions
        
        if predictions is None:
            raise ValueError("No predictions available")
        
        if output_path is None:
            output_path = Config.RESULT_FILE
        
        print("Creating submission file")
        
        # Step 1: Confidence-based adjustment
        if self.prediction_probabilities is not None:
            predictions = self._adjust_predictions_by_confidence(
                predictions, self.prediction_probabilities, confidence_threshold
            )
        
        # Step 2: Distribution balancing
        if apply_balancing and self.prediction_probabilities is not None:
            # Try different balancing methods and choose best
            balancing_methods = ['entropy_balance', 'target_distribution', 'uniform']
            best_predictions = predictions
            best_entropy = float('inf')
            
            for method in balancing_methods:
                try:
                    balanced_preds = self._balance_prediction_distribution(
                        predictions, self.prediction_probabilities, method
                    )
                    
                    # Evaluate balance quality using distribution entropy
                    pred_counts = Counter(balanced_preds)
                    probs = np.array([pred_counts.get(i, 0) for i in range(Config.N_CLASSES)])
                    probs = probs / probs.sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    
                    if entropy > best_entropy:
                        best_entropy = entropy
                        best_predictions = balanced_preds
                        print(f"Best balancing method so far: {method} (entropy: {entropy:.3f})")
                        
                except Exception as e:
                    print(f"Balancing method {method} failed: {e}")
                    continue
            
            predictions = best_predictions
        
        # Basic validation
        if len(test_ids) != len(predictions):
            raise ValueError(f"ID count ({len(test_ids)}) and prediction count ({len(predictions)}) do not match")
        
        submission_df = create_submission_template(
            test_ids, predictions,
            Config.ID_COLUMN, Config.TARGET_COLUMN
        )
        
        # Data validation and cleanup
        print(f"Submission file shape: {submission_df.shape}")
        print(f"ID count: {len(submission_df[Config.ID_COLUMN].unique())}")
        print(f"Prediction range: {submission_df[Config.TARGET_COLUMN].min()} ~ {submission_df[Config.TARGET_COLUMN].max()}")
        
        # Check duplicate IDs
        if submission_df[Config.ID_COLUMN].duplicated().any():
            print("Warning: Duplicate IDs found")
        
        # Check prediction range
        invalid_predictions = (submission_df[Config.TARGET_COLUMN] < 0) | (submission_df[Config.TARGET_COLUMN] >= Config.N_CLASSES)
        if invalid_predictions.any():
            print("Warning: Predictions outside valid range")
            invalid_count = invalid_predictions.sum()
            print(f"Invalid prediction count: {invalid_count}")
            
            # Fix with balanced assignment
            valid_classes = list(range(Config.N_CLASSES))
            for idx in submission_df[invalid_predictions].index:
                submission_df.loc[idx, Config.TARGET_COLUMN] = np.random.choice(valid_classes)
            print(f"Fixed invalid predictions with random assignment")
        
        # Save file
        try:
            submission_df.to_csv(output_path, index=False)
            print(f"Submission file saved successfully: {output_path}")
        except Exception as e:
            print(f"File save failed: {e}")
            raise
        
        # Verify saved file
        try:
            saved_df = pd.read_csv(output_path)
            if saved_df.shape == submission_df.shape:
                print("File save verification completed")
            else:
                print(f"Warning: Saved file size differs")
        except Exception as e:
            print(f"File verification failed: {e}")
        
        # Analyze final prediction distribution
        self.analyze_prediction_distribution(predictions)
        
        return submission_df
    
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
        
        print("Class-wise prediction counts:")
        for class_id in range(min(15, Config.N_CLASSES)):
            count = counts[unique == class_id][0] if class_id in unique else 0
            percentage = (count / total_predictions) * 100
            print(f"Class {class_id:2d}: {count:4d} ({percentage:5.2f}%)")
        
        if Config.N_CLASSES > 15:
            print(f"... (total {Config.N_CLASSES} classes)")
        
        # Distribution quality metrics
        expected_per_class = total_predictions / Config.N_CLASSES
        actual_counts = [counts[unique == i][0] if i in unique else 0 for i in range(Config.N_CLASSES)]
        
        print(f"\nDistribution Quality Analysis:")
        print(f"Total predictions: {total_predictions}")
        print(f"Expected per class: {expected_per_class:.1f}")
        print(f"Actual distribution std: {np.std(actual_counts):.2f}")
        
        # Calculate distribution entropy
        probs = np.array(actual_counts) / total_predictions
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(Config.N_CLASSES)
        normalized_entropy = entropy / max_entropy
        
        print(f"Distribution entropy: {normalized_entropy:.3f} (1.0 = uniform)")
        
        # Calculate imbalance metrics
        max_count = max(actual_counts)
        min_count = min([c for c in actual_counts if c > 0]) if any(actual_counts) else 1
        imbalance_ratio = max_count / min_count
        
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Gini coefficient (inequality measure)
        sorted_counts = sorted(actual_counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        print(f"Gini coefficient: {gini:.3f} (0.0 = perfect equality)")
        
        # Missing classes
        missing_classes = [i for i in range(Config.N_CLASSES) if i not in unique]
        if missing_classes:
            print(f"Missing classes: {missing_classes}")
        
        # Quality assessment
        quality_score = (normalized_entropy * 0.4 + 
                        (1 - min(gini, 1.0)) * 0.3 + 
                        (1 - min(imbalance_ratio / 10, 1.0)) * 0.3)
        print(f"Distribution quality score: {quality_score:.3f} (1.0 = perfect)")
        
        return {
            'distribution': dict(zip(range(Config.N_CLASSES), actual_counts)),
            'total_predictions': total_predictions,
            'expected_per_class': expected_per_class,
            'imbalance_ratio': imbalance_ratio,
            'missing_classes': missing_classes,
            'std': np.std(actual_counts),
            'entropy': normalized_entropy,
            'gini_coefficient': gini,
            'quality_score': quality_score
        }
    
    def validate_predictions(self, y_true=None):
        """Validate prediction results with detailed metrics"""
        if self.predictions is None:
            print("No predictions available")
            return None
        
        print("Prediction result validation")
        
        print(f"Prediction count: {len(self.predictions)}")
        print(f"Unique class count: {len(np.unique(self.predictions))}")
        print(f"Prediction range: {self.predictions.min()} ~ {self.predictions.max()}")
        
        # Confidence analysis
        if self.confidence_scores is not None:
            print(f"Average confidence: {np.mean(self.confidence_scores):.4f}")
            print(f"Confidence std: {np.std(self.confidence_scores):.4f}")
            print(f"Confidence median: {np.median(self.confidence_scores):.4f}")
            
            # Confidence distribution
            high_conf = np.sum(self.confidence_scores > 0.8)
            medium_conf = np.sum((self.confidence_scores > 0.6) & (self.confidence_scores <= 0.8))
            low_conf = np.sum(self.confidence_scores <= 0.6)
            
            total = len(self.predictions)
            print(f"High confidence (>0.8): {high_conf} ({high_conf/total*100:.1f}%)")
            print(f"Medium confidence (0.6-0.8): {medium_conf} ({medium_conf/total*100:.1f}%)")
            print(f"Low confidence (≤0.6): {low_conf} ({low_conf/total*100:.1f}%)")
            
            # Confidence by class
            if self.prediction_probabilities is not None:
                print(f"\nConfidence by class (top 10):")
                for class_id in range(min(10, Config.N_CLASSES)):
                    class_mask = self.predictions == class_id
                    if np.sum(class_mask) > 0:
                        class_conf = np.mean(self.confidence_scores[class_mask])
                        class_count = np.sum(class_mask)
                        print(f"  Class {class_id:2d}: {class_conf:.3f} ({class_count:4d} samples)")
        
        # Calculate performance if true labels provided
        if y_true is not None:
            if len(y_true) != len(self.predictions):
                print("Warning: True label and prediction counts differ")
                return None
            
            macro_f1 = calculate_macro_f1(y_true, self.predictions)
            print(f"Macro F1 Score: {macro_f1:.4f}")
            
            # Class-wise performance analysis
            try:
                report = classification_report(y_true, self.predictions, output_dict=True, zero_division=0)
                
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
                        
                        if class_id < 10:
                            print(f"Class {class_id:2d} - F1: {report[class_key]['f1-score']:.4f}, "
                                  f"Support: {report[class_key]['support']:4d}, "
                                  f"Precision: {report[class_key]['precision']:.3f}, "
                                  f"Recall: {report[class_key]['recall']:.3f}")
                
                # Identify problematic classes
                low_performance_classes = [
                    m for m in class_metrics 
                    if m['f1_score'] < Config.CLASS_PERFORMANCE_THRESHOLD and m['support'] > 0
                ]
                
                if low_performance_classes:
                    print(f"\nLow-performance classes ({len(low_performance_classes)}):")
                    for m in low_performance_classes[:8]:
                        print(f"  Class {m['class']:2d}: F1={m['f1_score']:.3f}, Support={m['support']:4d}")
                
                return {
                    'macro_f1': macro_f1,
                    'class_metrics': class_metrics,
                    'low_performance_classes': low_performance_classes,
                    'prediction_distribution': self.analyze_prediction_distribution(),
                    'confidence_stats': {
                        'mean': np.mean(self.confidence_scores) if self.confidence_scores is not None else 0,
                        'median': np.median(self.confidence_scores) if self.confidence_scores is not None else 0,
                        'std': np.std(self.confidence_scores) if self.confidence_scores is not None else 0
                    }
                }
                
            except Exception as e:
                print(f"Performance analysis failed: {e}")
                return {'macro_f1': macro_f1}
        
        return {
            'prediction_distribution': self.analyze_prediction_distribution(),
            'confidence_stats': {
                'mean': np.mean(self.confidence_scores) if self.confidence_scores is not None else 0,
                'median': np.median(self.confidence_scores) if self.confidence_scores is not None else 0,
                'std': np.std(self.confidence_scores) if self.confidence_scores is not None else 0
            }
        }
    
    def get_prediction_confidence(self):
        """Analyze prediction confidence with detailed breakdown"""
        if self.prediction_probabilities is None:
            print("No prediction probabilities available for confidence analysis")
            return None
        
        max_probs = np.max(self.prediction_probabilities, axis=1)
        
        confidence_stats = {
            'mean_confidence': np.mean(max_probs),
            'median_confidence': np.median(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs),
            'q25_confidence': np.percentile(max_probs, 25),
            'q75_confidence': np.percentile(max_probs, 75)
        }
        
        # Confidence interval distribution
        confidence_bins = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        bin_counts = np.histogram(max_probs, bins=confidence_bins)[0]
        
        confidence_distribution = {}
        for i in range(len(confidence_bins)-1):
            bin_name = f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"
            confidence_distribution[bin_name] = bin_counts[i]
        
        print("Prediction confidence analysis:")
        print(f"  Average confidence: {confidence_stats['mean_confidence']:.4f}")
        print(f"  Median confidence: {confidence_stats['median_confidence']:.4f}")
        print(f"  Confidence std: {confidence_stats['std_confidence']:.4f}")
        print(f"  Q25-Q75 range: {confidence_stats['q25_confidence']:.3f} - {confidence_stats['q75_confidence']:.3f}")
        
        print("\nConfidence interval distribution:")
        for bin_name, count in confidence_distribution.items():
            percentage = (count / len(max_probs)) * 100
            print(f"  {bin_name}: {count:4d} ({percentage:5.1f}%)")
        
        # Entropy analysis
        if hasattr(self, 'prediction_probabilities'):
            entropies = self._calculate_prediction_entropy(self.prediction_probabilities)
            entropy_stats = {
                'mean_entropy': np.mean(entropies),
                'median_entropy': np.median(entropies),
                'std_entropy': np.std(entropies)
            }
            
            print(f"\nPrediction entropy analysis:")
            print(f"  Mean entropy: {entropy_stats['mean_entropy']:.4f}")
            print(f"  Median entropy: {entropy_stats['median_entropy']:.4f}")
            print(f"  Entropy std: {entropy_stats['std_entropy']:.4f}")
            
            confidence_stats['entropy_stats'] = entropy_stats
        
        return {
            'confidence_stats': confidence_stats,
            'confidence_distribution': confidence_distribution,
            'confidence_scores': max_probs
        }