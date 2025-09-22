# prediction.py

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
from scipy.special import softmax
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
        """Model probability calibration"""
        if self.model is None:
            print("Model not loaded")
            return
        
        try:
            print("Starting model probability calibration")
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method='isotonic',
                cv=3
            )
            self.calibrated_model.fit(X_val, y_val)
            print("Model probability calibration completed")
        except Exception as e:
            print(f"Model calibration failed: {e}")
            self.calibrated_model = None
    
    @timer
    def predict(self, X_test, use_calibrated=True):
        """Model prediction"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        print("Performing model prediction")
        
        # Select model to use
        active_model = self.calibrated_model if (use_calibrated and self.calibrated_model is not None) else self.model
        
        # Prediction probabilities
        if hasattr(active_model, 'predict_proba'):
            self.prediction_probabilities = active_model.predict_proba(X_test)
            print(f"Prediction probability shape: {self.prediction_probabilities.shape}")
        else:
            print("Probability prediction not available")
            self.prediction_probabilities = None
        
        # Prediction classes
        self.predictions = active_model.predict(X_test)
        print(f"Prediction result shape: {self.predictions.shape}")
        
        # Calculate confidence scores
        if self.prediction_probabilities is not None:
            self.confidence_scores = np.max(self.prediction_probabilities, axis=1)
        
        # Validate prediction results
        validate_predictions(self.predictions, Config.N_CLASSES)
        
        return self.predictions
    
    def _adjust_predictions_by_confidence(self, predictions, probabilities, threshold=0.6):
        """Adjust predictions based on confidence"""
        if probabilities is None:
            return predictions
        
        adjusted_predictions = predictions.copy()
        max_probs = np.max(probabilities, axis=1)
        low_confidence_mask = max_probs < threshold
        
        print(f"Low confidence predictions (< {threshold}): {np.sum(low_confidence_mask)}")
        
        if np.sum(low_confidence_mask) > 0:
            # Consider top 2 classes for low confidence predictions
            low_conf_indices = np.where(low_confidence_mask)[0]
            
            for idx in low_conf_indices:
                probs = probabilities[idx]
                sorted_indices = np.argsort(probs)[::-1]
                
                # If difference between top 2 classes is small, choose more balanced class
                top1_prob = probs[sorted_indices[0]]
                top2_prob = probs[sorted_indices[1]]
                
                if top1_prob - top2_prob < 0.1:
                    # Consider current prediction distribution
                    current_pred_counts = np.bincount(adjusted_predictions, minlength=Config.N_CLASSES)
                    expected_count = len(predictions) / Config.N_CLASSES
                    
                    top1_class = sorted_indices[0]
                    top2_class = sorted_indices[1]
                    
                    # Choose class with fewer predictions
                    if current_pred_counts[top2_class] < current_pred_counts[top1_class]:
                        adjusted_predictions[idx] = top2_class
        
        return adjusted_predictions
    
    def _balance_prediction_distribution(self, predictions, probabilities=None, method='entropy'):
        """Balance prediction distribution"""
        if probabilities is None:
            print("No probability information available for distribution adjustment")
            return predictions
        
        print(f"Starting prediction distribution balancing ({method})")
        
        current_counts = np.bincount(predictions, minlength=Config.N_CLASSES)
        total_samples = len(predictions)
        expected_count = total_samples / Config.N_CLASSES
        
        # Set target distribution
        target_counts = np.full(Config.N_CLASSES, int(expected_count))
        remainder = total_samples - target_counts.sum()
        
        # Distribute remainder randomly
        if remainder > 0:
            random_classes = np.random.RandomState(Config.RANDOM_STATE).choice(
                Config.N_CLASSES, remainder, replace=False
            )
            for cls in random_classes:
                target_counts[cls] += 1
        
        balanced_predictions = predictions.copy()
        
        if method == 'entropy':
            # Entropy-based adjustment
            entropies = []
            for i in range(len(probabilities)):
                probs = probabilities[i]
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
            
            entropies = np.array(entropies)
            
            # Adjust for each class
            for class_id in range(Config.N_CLASSES):
                current_count = current_counts[class_id]
                target_count = target_counts[class_id]
                
                if current_count > target_count:
                    # Remove from over-predicted class
                    remove_count = current_count - target_count
                    class_indices = np.where(balanced_predictions == class_id)[0]
                    
                    # Prioritize removal of high entropy (uncertain) predictions
                    class_entropies = entropies[class_indices]
                    remove_indices = class_indices[np.argsort(class_entropies)[-remove_count:]]
                    
                    # Reassign to other classes
                    for idx in remove_indices:
                        probs = probabilities[idx].copy()
                        probs[class_id] = 0  # Exclude current class
                        
                        # Prioritize under-represented classes
                        for j in range(Config.N_CLASSES):
                            if current_counts[j] < target_counts[j]:
                                probs[j] *= 2
                        
                        best_class = np.argmax(probs)
                        balanced_predictions[idx] = best_class
                        current_counts[class_id] -= 1
                        current_counts[best_class] += 1
        
        return balanced_predictions
    
    def _ensemble_predictions(self, predictions_list, probabilities_list=None, method='voting'):
        """Combine ensemble predictions"""
        if len(predictions_list) == 1:
            return predictions_list[0]
        
        print(f"Combining ensemble predictions ({method})")
        
        if method == 'voting':
            # Majority voting
            predictions_array = np.array(predictions_list)
            ensemble_pred = []
            
            for i in range(predictions_array.shape[1]):
                votes = predictions_array[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                ensemble_pred.append(unique[np.argmax(counts)])
            
            return np.array(ensemble_pred)
        
        elif method == 'probability' and probabilities_list is not None:
            # Average probabilities
            avg_probabilities = np.mean(probabilities_list, axis=0)
            return np.argmax(avg_probabilities, axis=1)
        
        elif method == 'weighted' and probabilities_list is not None:
            # Weighted average (confidence-based)
            weights = []
            for probs in probabilities_list:
                confidence = np.mean(np.max(probs, axis=1))
                weights.append(confidence)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            weighted_probabilities = np.zeros_like(probabilities_list[0])
            for i, probs in enumerate(probabilities_list):
                weighted_probabilities += weights[i] * probs
            
            return np.argmax(weighted_probabilities, axis=1)
        
        else:
            # Return first prediction by default
            return predictions_list[0]
    
    @timer
    def create_submission_file(self, test_ids, output_path=None, predictions=None, 
                             apply_balancing=True, confidence_threshold=0.6):
        """Create submission file"""
        if predictions is None:
            predictions = self.predictions
        
        if predictions is None:
            raise ValueError("No predictions available")
        
        if output_path is None:
            output_path = Config.RESULT_FILE
        
        print("Creating submission file")
        
        # Confidence-based adjustment
        if self.prediction_probabilities is not None:
            predictions = self._adjust_predictions_by_confidence(
                predictions, self.prediction_probabilities, confidence_threshold
            )
        
        # Distribution balancing
        if apply_balancing and self.prediction_probabilities is not None:
            predictions = self._balance_prediction_distribution(
                predictions, self.prediction_probabilities
            )
        
        # Basic validation
        if len(test_ids) != len(predictions):
            raise ValueError(f"ID count ({len(test_ids)}) and prediction count ({len(predictions)}) do not match")
        
        submission_df = create_submission_template(
            test_ids, predictions,
            Config.ID_COLUMN, Config.TARGET_COLUMN
        )
        
        # Data validation
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
            
            # Fix with most frequent class
            most_frequent_class = submission_df[Config.TARGET_COLUMN].mode()[0]
            submission_df.loc[invalid_predictions, Config.TARGET_COLUMN] = most_frequent_class
            print(f"Fixed invalid predictions to {most_frequent_class}")
        
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
        
        # Analyze prediction distribution
        self.analyze_prediction_distribution(predictions)
        
        return submission_df
    
    def analyze_prediction_distribution(self, predictions=None):
        """Analyze prediction distribution"""
        if predictions is None:
            predictions = self.predictions
        
        if predictions is None:
            print("No predictions available")
            return None
        
        print("Prediction distribution analysis")
        
        unique, counts = np.unique(predictions, return_counts=True)
        total_predictions = len(predictions)
        
        print("Class-wise prediction counts:")
        for class_id in range(min(10, Config.N_CLASSES)):
            count = counts[unique == class_id][0] if class_id in unique else 0
            percentage = (count / total_predictions) * 100
            print(f"Class {class_id:2d}: {count:4d} ({percentage:5.2f}%)")
        
        if Config.N_CLASSES > 10:
            print(f"... (total {Config.N_CLASSES} classes)")
        
        # Distribution statistics
        expected_per_class = total_predictions / Config.N_CLASSES
        actual_counts = [counts[unique == i][0] if i in unique else 0 for i in range(Config.N_CLASSES)]
        
        print(f"\nTotal predictions: {total_predictions}")
        print(f"Expected per class: {expected_per_class:.1f}")
        print(f"Actual distribution std: {np.std(actual_counts):.2f}")
        
        # Calculate imbalance degree
        max_count = max(actual_counts)
        min_count = min([c for c in actual_counts if c > 0]) if any(actual_counts) else 1
        imbalance_ratio = max_count / min_count
        
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Missing classes
        missing_classes = [i for i in range(Config.N_CLASSES) if i not in unique]
        if missing_classes:
            print(f"Missing classes: {missing_classes}")
        
        return {
            'distribution': dict(zip(unique, counts)),
            'total_predictions': total_predictions,
            'expected_per_class': expected_per_class,
            'imbalance_ratio': imbalance_ratio,
            'missing_classes': missing_classes,
            'std': np.std(actual_counts)
        }
    
    def validate_predictions(self, y_true=None):
        """Validate prediction results"""
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
            
            high_conf = np.sum(self.confidence_scores > 0.8)
            medium_conf = np.sum((self.confidence_scores > 0.5) & (self.confidence_scores <= 0.8))
            low_conf = np.sum(self.confidence_scores <= 0.5)
            
            print(f"High confidence (>0.8): {high_conf} ({high_conf/len(self.predictions)*100:.1f}%)")
            print(f"Medium confidence (0.5-0.8): {medium_conf} ({medium_conf/len(self.predictions)*100:.1f}%)")
            print(f"Low confidence (≤0.5): {low_conf} ({low_conf/len(self.predictions)*100:.1f}%)")
        
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
                            print(f"Class {class_id:2d} - F1: {report[class_key]['f1-score']:.4f}, Support: {report[class_key]['support']:4d}")
                
                # Identify low-performance classes
                low_performance_classes = [
                    m for m in class_metrics 
                    if m['f1_score'] < Config.CLASS_PERFORMANCE_THRESHOLD and m['support'] > 0
                ]
                
                if low_performance_classes:
                    print(f"\nLow-performance classes ({len(low_performance_classes)}):")
                    for m in low_performance_classes[:5]:
                        print(f"  Class {m['class']:2d}: F1={m['f1_score']:.3f}, Support={m['support']:4d}")
                
                return {
                    'macro_f1': macro_f1,
                    'class_metrics': class_metrics,
                    'low_performance_classes': low_performance_classes,
                    'prediction_distribution': self.analyze_prediction_distribution()
                }
                
            except Exception as e:
                print(f"Performance analysis failed: {e}")
                return {'macro_f1': macro_f1}
        
        return None
    
    def get_prediction_confidence(self):
        """Analyze prediction confidence"""
        if self.prediction_probabilities is None:
            print("No prediction probabilities available for confidence analysis")
            return None
        
        max_probs = np.max(self.prediction_probabilities, axis=1)
        
        confidence_stats = {
            'mean_confidence': np.mean(max_probs),
            'median_confidence': np.median(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs)
        }
        
        # Confidence interval distribution
        confidence_bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        bin_counts = np.histogram(max_probs, bins=confidence_bins)[0]
        
        confidence_distribution = {}
        for i in range(len(confidence_bins)-1):
            bin_name = f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"
            confidence_distribution[bin_name] = bin_counts[i]
        
        print("Prediction confidence analysis:")
        print(f"  Average confidence: {confidence_stats['mean_confidence']:.4f}")
        print(f"  Median confidence: {confidence_stats['median_confidence']:.4f}")
        print(f"  Confidence std: {confidence_stats['std_confidence']:.4f}")
        
        print("\nConfidence interval distribution:")
        for bin_name, count in confidence_distribution.items():
            percentage = (count / len(max_probs)) * 100
            print(f"  {bin_name}: {count:4d} ({percentage:5.1f}%)")
        
        return {
            'confidence_stats': confidence_stats,
            'confidence_distribution': confidence_distribution,
            'confidence_scores': max_probs
        }