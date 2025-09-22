# evaluation.py

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_all_metrics, save_results

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        self.class_reports = {}
        
    @timer
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Perform model evaluation"""
        print(f"=== {model_name} Evaluation Start ===")
        
        # Perform predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Calculate basic metrics
        metrics = calculate_all_metrics(y_test, y_pred)
        
        # Class-wise performance analysis
        class_metrics = self._calculate_class_metrics(y_test, y_pred)
        
        # Confusion matrix analysis
        confusion_metrics = self._analyze_confusion_matrix(y_test, y_pred)
        
        # Probability-based metrics
        prob_metrics = {}
        if y_proba is not None:
            prob_metrics = self._calculate_probability_metrics(y_test, y_proba)
        
        # Class imbalance analysis
        imbalance_metrics = self._analyze_class_imbalance(y_test, y_pred)
        
        # Save results
        evaluation_result = {
            'model_name': model_name,
            'basic_metrics': metrics,
            'class_metrics': class_metrics,
            'confusion_metrics': confusion_metrics,
            'probability_metrics': prob_metrics,
            'imbalance_metrics': imbalance_metrics,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        self.evaluation_results[model_name] = evaluation_result
        
        # Output results
        self._print_evaluation_summary(evaluation_result)
        
        return evaluation_result
    
    def _calculate_class_metrics(self, y_true, y_pred):
        """Calculate class-wise performance metrics"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=range(Config.N_CLASSES)
        )
        
        class_metrics = []
        for i in range(Config.N_CLASSES):
            if i < len(precision):
                class_metrics.append({
                    'class': i,
                    'precision': precision[i] if not np.isnan(precision[i]) else 0.0,
                    'recall': recall[i] if not np.isnan(recall[i]) else 0.0,
                    'f1_score': f1[i] if not np.isnan(f1[i]) else 0.0,
                    'support': support[i] if i < len(support) else 0
                })
            else:
                class_metrics.append({
                    'class': i,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'support': 0
                })
        
        return class_metrics
    
    def _analyze_confusion_matrix(self, y_true, y_pred):
        """Analyze confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=range(Config.N_CLASSES))
        
        # Normalized confusion matrix
        cm_sum = cm.sum(axis=1)
        cm_normalized = np.zeros_like(cm, dtype=float)
        
        for i in range(len(cm_sum)):
            if cm_sum[i] > 0:
                cm_normalized[i] = cm[i] / cm_sum[i]
        
        # Diagonal components
        diagonal_accuracy = np.diag(cm_normalized)
        
        # Most confused class pairs
        confusion_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_class': i,
                        'pred_class': j,
                        'count': int(cm[i, j]),
                        'rate': float(cm_normalized[i, j]) if cm_sum[i] > 0 else 0.0
                    })
        
        confusion_pairs = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)
        
        # Class-wise accuracy
        class_accuracies = []
        for i in range(Config.N_CLASSES):
            if cm_sum[i] > 0:
                accuracy = cm[i, i] / cm_sum[i]
            else:
                accuracy = 0.0
            class_accuracies.append({
                'class': i,
                'accuracy': float(accuracy),
                'total_samples': int(cm_sum[i])
            })
        
        return {
            'confusion_matrix': cm.tolist(),
            'normalized_cm': cm_normalized.tolist(),
            'diagonal_accuracy': diagonal_accuracy.tolist(),
            'top_confusions': confusion_pairs[:15],
            'class_accuracies': class_accuracies
        }
    
    def _calculate_probability_metrics(self, y_true, y_proba):
        """Calculate probability-based metrics"""
        # Maximum probability for each prediction
        max_proba = np.max(y_proba, axis=1)
        
        # Prediction confidence analysis
        confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        confidence_analysis = {}
        
        for threshold in confidence_thresholds:
            high_conf_mask = max_proba >= threshold
            high_conf_count = np.sum(high_conf_mask)
            
            if high_conf_count > 0:
                high_conf_acc = accuracy_score(
                    y_true[high_conf_mask], 
                    np.argmax(y_proba[high_conf_mask], axis=1)
                )
            else:
                high_conf_acc = 0.0
            
            confidence_analysis[threshold] = {
                'count': int(high_conf_count),
                'percentage': float(high_conf_count / len(y_true) * 100),
                'accuracy': float(high_conf_acc)
            }
        
        # Class-wise average probabilities
        class_proba_avg = []
        for class_id in range(Config.N_CLASSES):
            class_mask = y_true == class_id
            if np.sum(class_mask) > 0:
                avg_proba = np.mean(y_proba[class_mask, class_id])
            else:
                avg_proba = 0.0
            class_proba_avg.append(float(avg_proba))
        
        # Probability calibration analysis
        calibration_metrics = self._analyze_calibration(y_true, y_proba)
        
        # Entropy-based uncertainty analysis
        entropy_analysis = self._analyze_prediction_entropy(y_proba)
        
        return {
            'max_probabilities': max_proba.tolist(),
            'mean_max_probability': float(np.mean(max_proba)),
            'confidence_analysis': confidence_analysis,
            'class_avg_probabilities': class_proba_avg,
            'calibration_metrics': calibration_metrics,
            'entropy_analysis': entropy_analysis
        }
    
    def _analyze_calibration(self, y_true, y_proba):
        """Analyze probability calibration"""
        try:
            calibration_results = {}
            
            for class_id in range(min(5, Config.N_CLASSES)):  # Top 5 classes only
                if np.sum(y_true == class_id) > 10:  # Only if sufficient samples
                    y_binary = (y_true == class_id).astype(int)
                    y_prob_binary = y_proba[:, class_id]
                    
                    try:
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            y_binary, y_prob_binary, n_bins=10, strategy='uniform'
                        )
                        
                        # Calculate calibration error
                        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                        
                        calibration_results[f'class_{class_id}'] = {
                            'calibration_error': float(calibration_error),
                            'fraction_of_positives': fraction_of_positives.tolist(),
                            'mean_predicted_value': mean_predicted_value.tolist()
                        }
                    except:
                        calibration_results[f'class_{class_id}'] = {
                            'calibration_error': 0.0,
                            'fraction_of_positives': [],
                            'mean_predicted_value': []
                        }
            
            return calibration_results
            
        except Exception as e:
            print(f"Calibration analysis failed: {e}")
            return {}
    
    def _analyze_prediction_entropy(self, y_proba):
        """Analyze prediction entropy"""
        try:
            # Calculate entropy
            entropies = []
            for i in range(len(y_proba)):
                probs = y_proba[i]
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
            
            entropies = np.array(entropies)
            
            # Entropy statistics
            entropy_stats = {
                'mean_entropy': float(np.mean(entropies)),
                'std_entropy': float(np.std(entropies)),
                'min_entropy': float(np.min(entropies)),
                'max_entropy': float(np.max(entropies))
            }
            
            # Entropy interval distribution
            entropy_bins = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            bin_counts = np.histogram(entropies, bins=entropy_bins)[0]
            
            entropy_distribution = {}
            for i in range(len(entropy_bins)-1):
                bin_name = f"{entropy_bins[i]:.1f}-{entropy_bins[i+1]:.1f}"
                entropy_distribution[bin_name] = int(bin_counts[i])
            
            return {
                'entropy_stats': entropy_stats,
                'entropy_distribution': entropy_distribution,
                'entropies': entropies.tolist()
            }
            
        except Exception as e:
            print(f"Entropy analysis failed: {e}")
            return {}
    
    def _analyze_class_imbalance(self, y_true, y_pred):
        """Analyze class imbalance"""
        # Actual distribution
        true_counts = np.bincount(y_true, minlength=Config.N_CLASSES)
        pred_counts = np.bincount(y_pred, minlength=Config.N_CLASSES)
        
        total_samples = len(y_true)
        
        # Calculate imbalance metrics
        true_distribution = true_counts / total_samples
        pred_distribution = pred_counts / total_samples
        
        # Distribution difference
        distribution_diff = np.abs(true_distribution - pred_distribution)
        
        # Imbalance ratios
        true_max = np.max(true_counts)
        true_min = np.min(true_counts[true_counts > 0]) if np.any(true_counts > 0) else 1
        true_imbalance_ratio = true_max / true_min
        
        pred_max = np.max(pred_counts)
        pred_min = np.min(pred_counts[pred_counts > 0]) if np.any(pred_counts > 0) else 1
        pred_imbalance_ratio = pred_max / pred_min
        
        # Missing classes
        missing_true_classes = np.where(true_counts == 0)[0].tolist()
        missing_pred_classes = np.where(pred_counts == 0)[0].tolist()
        
        # Over/under-predicted classes
        over_predicted = []
        under_predicted = []
        
        for i in range(Config.N_CLASSES):
            if pred_counts[i] > true_counts[i] * 1.5:  # 50%+ over-prediction
                over_predicted.append(i)
            elif pred_counts[i] < true_counts[i] * 0.5:  # 50%+ under-prediction
                under_predicted.append(i)
        
        return {
            'true_distribution': true_distribution.tolist(),
            'pred_distribution': pred_distribution.tolist(),
            'distribution_difference': distribution_diff.tolist(),
            'true_imbalance_ratio': float(true_imbalance_ratio),
            'pred_imbalance_ratio': float(pred_imbalance_ratio),
            'missing_true_classes': missing_true_classes,
            'missing_pred_classes': missing_pred_classes,
            'over_predicted_classes': over_predicted,
            'under_predicted_classes': under_predicted,
            'total_samples': total_samples
        }
    
    def _print_evaluation_summary(self, evaluation_result):
        """Print evaluation result summary"""
        model_name = evaluation_result['model_name']
        metrics = evaluation_result['basic_metrics']
        
        print(f"\n=== {model_name} Evaluation Result Summary ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        
        # Best/worst performing classes
        class_metrics = evaluation_result['class_metrics']
        if class_metrics:
            class_f1_scores = [cm['f1_score'] for cm in class_metrics if cm['support'] > 0]
            
            if class_f1_scores:
                best_f1 = max(class_f1_scores)
                worst_f1 = min(class_f1_scores)
                
                best_class_idx = next(i for i, cm in enumerate(class_metrics) 
                                    if cm['f1_score'] == best_f1 and cm['support'] > 0)
                worst_class_idx = next(i for i, cm in enumerate(class_metrics) 
                                     if cm['f1_score'] == worst_f1 and cm['support'] > 0)
                
                print(f"\nBest performing class: {best_class_idx} (F1: {best_f1:.4f})")
                print(f"Worst performing class: {worst_class_idx} (F1: {worst_f1:.4f})")
        
        # Probability-based metrics
        if evaluation_result['probability_metrics']:
            prob_metrics = evaluation_result['probability_metrics']
            mean_confidence = prob_metrics['mean_max_probability']
            print(f"Average prediction confidence: {mean_confidence:.4f}")
            
            if 'entropy_analysis' in prob_metrics:
                entropy_stats = prob_metrics['entropy_analysis']['entropy_stats']
                print(f"Average prediction entropy: {entropy_stats['mean_entropy']:.4f}")
        
        # Class imbalance information
        imbalance_metrics = evaluation_result['imbalance_metrics']
        if imbalance_metrics:
            true_ratio = imbalance_metrics['true_imbalance_ratio']
            pred_ratio = imbalance_metrics['pred_imbalance_ratio']
            print(f"True imbalance ratio: {true_ratio:.2f}:1")
            print(f"Predicted imbalance ratio: {pred_ratio:.2f}:1")
            
            over_pred = imbalance_metrics['over_predicted_classes']
            under_pred = imbalance_metrics['under_predicted_classes']
            if over_pred:
                print(f"Over-predicted classes: {over_pred}")
            if under_pred:
                print(f"Under-predicted classes: {under_pred}")
    
    @timer
    def compare_models(self, model_results):
        """Compare multiple model performance"""
        print("=== Model Performance Comparison ===")
        
        comparison_data = []
        
        for model_name, result in model_results.items():
            metrics = result['basic_metrics']
            row = {'model': model_name}
            row.update(metrics)
            
            # Additional metrics
            if result.get('probability_metrics'):
                prob_metrics = result['probability_metrics']
                row['mean_confidence'] = prob_metrics['mean_max_probability']
                
                if 'entropy_analysis' in prob_metrics:
                    entropy_stats = prob_metrics['entropy_analysis']['entropy_stats']
                    row['mean_entropy'] = entropy_stats['mean_entropy']
            
            if result.get('imbalance_metrics'):
                imbalance_metrics = result['imbalance_metrics']
                row['pred_imbalance_ratio'] = imbalance_metrics['pred_imbalance_ratio']
            
            comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('macro_f1', ascending=False)
            
            print(f"\nModel performance ranking:")
            print(comparison_df.to_string(index=False, float_format='%.4f'))
            
            # Best performing model
            if not comparison_df.empty:
                best_model = comparison_df.iloc[0]
                print(f"\nBest performing model: {best_model['model']}")
                print(f"Macro F1 Score: {best_model['macro_f1']:.4f}")
        
        return comparison_df if comparison_data else None
    
    @timer
    def analyze_class_performance(self, evaluation_result):
        """Analyze class-wise performance"""
        model_name = evaluation_result['model_name']
        class_metrics = evaluation_result['class_metrics']
        
        print(f"=== {model_name} Class-wise Performance Analysis ===")
        
        if not class_metrics:
            print("No class metrics available")
            return None
        
        class_df = pd.DataFrame(class_metrics)
        
        print(f"\nClass-wise performance (top 15):")
        display_df = class_df.head(15)
        print(display_df.to_string(index=False, float_format='%.4f'))
        
        if len(class_df) > 15:
            print(f"... (total {len(class_df)} classes)")
        
        # Performance statistics (classes with support samples only)
        valid_classes = class_df[class_df['support'] > 0]
        
        if not valid_classes.empty:
            print(f"\nF1 Score statistics ({len(valid_classes)} classes):")
            print(f"Mean: {valid_classes['f1_score'].mean():.4f}")
            print(f"Std: {valid_classes['f1_score'].std():.4f}")
            print(f"Max: {valid_classes['f1_score'].max():.4f}")
            print(f"Min: {valid_classes['f1_score'].min():.4f}")
            
            # Identify low-performance classes
            low_performance_threshold = Config.CLASS_PERFORMANCE_THRESHOLD
            low_performance_classes = valid_classes[valid_classes['f1_score'] < low_performance_threshold]
            
            if not low_performance_classes.empty:
                print(f"\nLow-performance classes (F1 < {low_performance_threshold}):")
                low_perf_display = low_performance_classes[['class', 'f1_score', 'support']].head(10)
                print(low_perf_display.to_string(index=False, float_format='%.4f'))
                
                if len(low_performance_classes) > 10:
                    print(f"... (total {len(low_performance_classes)})")
        
        return class_df
    
    @timer
    def analyze_confusion_patterns(self, evaluation_result):
        """Analyze confusion patterns"""
        model_name = evaluation_result['model_name']
        confusion_metrics = evaluation_result['confusion_metrics']
        
        print(f"=== {model_name} Confusion Pattern Analysis ===")
        
        if not confusion_metrics:
            print("No confusion matrix metrics available")
            return None
        
        # Most confused class pairs
        top_confusions = confusion_metrics['top_confusions']
        
        print(f"\nMost confused class pairs (top 10):")
        for i, confusion in enumerate(top_confusions[:10]):
            print(f"{i+1:2d}. Class {confusion['true_class']:2d} → {confusion['pred_class']:2d}: "
                  f"{confusion['count']:4d} times ({confusion['rate']*100:5.1f}%)")
        
        # Class-wise accuracy
        class_accuracies = confusion_metrics['class_accuracies']
        
        print(f"\nClass-wise accuracy (top 15):")
        for i, acc_info in enumerate(class_accuracies[:15]):
            class_id = acc_info['class']
            accuracy = acc_info['accuracy']
            total = acc_info['total_samples']
            print(f"Class {class_id:2d}: {accuracy:.4f} (samples: {total:4d})")
        
        if len(class_accuracies) > 15:
            print(f"... (total {len(class_accuracies)} classes)")
        
        return confusion_metrics
    
    @timer
    def analyze_prediction_confidence(self, evaluation_result):
        """Analyze prediction confidence"""
        model_name = evaluation_result['model_name']
        prob_metrics = evaluation_result.get('probability_metrics')
        
        if not prob_metrics:
            print(f"{model_name}: No probability metrics available")
            return None
        
        print(f"=== {model_name} Prediction Confidence Analysis ===")
        
        confidence_analysis = prob_metrics['confidence_analysis']
        
        print(f"\nConfidence-based prediction analysis:")
        print(f"{'Threshold':>9} {'Samples':>8} {'Ratio(%)':>8} {'Accuracy':>8}")
        print("-" * 34)
        
        for threshold, analysis in confidence_analysis.items():
            count = analysis['count']
            percentage = analysis['percentage']
            accuracy = analysis['accuracy']
            print(f"{threshold:9.2f} {count:8d} {percentage:7.1f} {accuracy:8.4f}")
        
        # Entropy analysis
        if 'entropy_analysis' in prob_metrics:
            entropy_analysis = prob_metrics['entropy_analysis']
            entropy_stats = entropy_analysis['entropy_stats']
            
            print(f"\nPrediction uncertainty analysis:")
            print(f"Mean entropy: {entropy_stats['mean_entropy']:.4f}")
            print(f"Entropy std: {entropy_stats['std_entropy']:.4f}")
            
            entropy_distribution = entropy_analysis['entropy_distribution']
            print(f"\nEntropy interval distribution:")
            for bin_name, count in entropy_distribution.items():
                percentage = count / len(prob_metrics['max_probabilities']) * 100
                print(f"  {bin_name}: {count:4d} ({percentage:5.1f}%)")
        
        # Calibration analysis
        calibration_metrics = prob_metrics.get('calibration_metrics', {})
        if calibration_metrics:
            print(f"\nProbability calibration analysis:")
            for class_info, metrics in calibration_metrics.items():
                error = metrics['calibration_error']
                print(f"{class_info}: calibration error {error:.4f}")
        
        return prob_metrics
    
    @timer
    def generate_report(self, evaluation_result, save_path=None):
        """Generate evaluation report"""
        model_name = evaluation_result['model_name']
        
        report_lines = []
        report_lines.append(f"Model Evaluation Report: {model_name}")
        report_lines.append("=" * 60)
        
        # 1. Basic performance metrics
        metrics = evaluation_result['basic_metrics']
        report_lines.append(f"\n1. Basic Performance Metrics")
        report_lines.append("-" * 30)
        for metric_name, value in metrics.items():
            report_lines.append(f"{metric_name:20s}: {value:.4f}")
        
        # 2. Class-wise performance summary
        class_metrics = evaluation_result['class_metrics']
        if class_metrics:
            report_lines.append(f"\n2. Class-wise Performance Summary")
            report_lines.append("-" * 30)
            
            valid_classes = [cm for cm in class_metrics if cm['support'] > 0]
            if valid_classes:
                f1_scores = [cm['f1_score'] for cm in valid_classes]
                report_lines.append(f"Evaluable class count: {len(valid_classes)}")
                report_lines.append(f"Mean F1 Score: {np.mean(f1_scores):.4f}")
                report_lines.append(f"F1 Score std: {np.std(f1_scores):.4f}")
                report_lines.append(f"Best F1 Score: {np.max(f1_scores):.4f}")
                report_lines.append(f"Worst F1 Score: {np.min(f1_scores):.4f}")
        
        # 3. Major confusion patterns
        confusion_metrics = evaluation_result['confusion_metrics']
        if confusion_metrics and confusion_metrics['top_confusions']:
            report_lines.append(f"\n3. Major Confusion Patterns")
            report_lines.append("-" * 30)
            
            top_confusions = confusion_metrics['top_confusions']
            for i, confusion in enumerate(top_confusions[:8]):
                report_lines.append(
                    f"{i+1}. Class {confusion['true_class']} → {confusion['pred_class']}: "
                    f"{confusion['count']} times ({confusion['rate']*100:.1f}%)"
                )
        
        # 4. Prediction confidence analysis
        prob_metrics = evaluation_result.get('probability_metrics')
        if prob_metrics:
            report_lines.append(f"\n4. Prediction Confidence Analysis")
            report_lines.append("-" * 30)
            
            mean_conf = prob_metrics['mean_max_probability']
            report_lines.append(f"Mean max probability: {mean_conf:.4f}")
            
            confidence_analysis = prob_metrics['confidence_analysis']
            report_lines.append(f"\nConfidence-based analysis:")
            for threshold, analysis in confidence_analysis.items():
                if threshold in [0.7, 0.8, 0.9]:  # Major thresholds only
                    count = analysis['count']
                    percentage = analysis['percentage']
                    accuracy = analysis['accuracy']
                    report_lines.append(
                        f"Threshold {threshold}: {count} ({percentage:.1f}%), "
                        f"accuracy {accuracy:.4f}"
                    )
            
            # Add entropy analysis
            if 'entropy_analysis' in prob_metrics:
                entropy_stats = prob_metrics['entropy_analysis']['entropy_stats']
                report_lines.append(f"Mean prediction entropy: {entropy_stats['mean_entropy']:.4f}")
        
        # 5. Class imbalance analysis
        imbalance_metrics = evaluation_result.get('imbalance_metrics')
        if imbalance_metrics:
            report_lines.append(f"\n5. Class Imbalance Analysis")
            report_lines.append("-" * 30)
            
            true_ratio = imbalance_metrics['true_imbalance_ratio']
            pred_ratio = imbalance_metrics['pred_imbalance_ratio']
            report_lines.append(f"True imbalance ratio: {true_ratio:.2f}:1")
            report_lines.append(f"Predicted imbalance ratio: {pred_ratio:.2f}:1")
            
            missing_pred = imbalance_metrics['missing_pred_classes']
            if missing_pred:
                report_lines.append(f"Unpredicted classes: {missing_pred}")
            
            over_pred = imbalance_metrics['over_predicted_classes']
            under_pred = imbalance_metrics['under_predicted_classes']
            if over_pred:
                report_lines.append(f"Over-predicted classes: {over_pred}")
            if under_pred:
                report_lines.append(f"Under-predicted classes: {under_pred}")
        
        # Generate report text
        report_text = "\n".join(report_lines)
        
        # Save file
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                print(f"Report saved: {save_path}")
            except Exception as e:
                print(f"Report save failed: {e}")
        
        return report_text
    
    @timer
    def save_evaluation_results(self, output_dir=None):
        """Save evaluation results"""
        if output_dir is None:
            output_dir = Config.MODEL_DIR
        
        if not self.evaluation_results:
            print("No evaluation results to save")
            return
        
        # Save all evaluation results to CSV
        all_results = []
        
        for model_name, result in self.evaluation_results.items():
            metrics = result['basic_metrics']
            row = {'model': model_name}
            row.update(metrics)
            
            # Additional metrics
            if result.get('probability_metrics'):
                prob_metrics = result['probability_metrics']
                row['mean_confidence'] = prob_metrics['mean_max_probability']
                
                if 'entropy_analysis' in prob_metrics:
                    entropy_stats = prob_metrics['entropy_analysis']['entropy_stats']
                    row['mean_entropy'] = entropy_stats['mean_entropy']
                    row['entropy_std'] = entropy_stats['std_entropy']
            
            if result.get('imbalance_metrics'):
                imbalance_metrics = result['imbalance_metrics']
                row['pred_imbalance_ratio'] = imbalance_metrics['pred_imbalance_ratio']
            
            all_results.append(row)
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            save_path = output_dir / 'evaluation_results.csv'
            save_results(results_df, save_path)
        
        # Save class-wise performance
        for model_name, result in self.evaluation_results.items():
            class_metrics = result['class_metrics']
            if class_metrics:
                class_df = pd.DataFrame(class_metrics)
                save_path = output_dir / f'{model_name}_class_performance.csv'
                save_results(class_df, save_path)
        
        print("Evaluation results saved successfully")
    
    def create_performance_summary(self):
        """Create performance summary"""
        if not self.evaluation_results:
            print("No evaluation results available")
            return None
        
        summary = {
            'total_models': len(self.evaluation_results),
            'models_evaluated': list(self.evaluation_results.keys()),
            'best_models': {},
            'performance_statistics': {}
        }
        
        # Best model for each metric
        metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
        
        for metric in metrics:
            best_score = -1
            best_model = None
            
            for model_name, result in self.evaluation_results.items():
                if 'basic_metrics' in result:
                    score = result['basic_metrics'].get(metric, 0)
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                summary['best_models'][metric] = {
                    'model': best_model,
                    'score': best_score
                }
        
        # Performance statistics
        all_scores = {metric: [] for metric in metrics}
        
        for result in self.evaluation_results.values():
            if 'basic_metrics' in result:
                for metric in metrics:
                    score = result['basic_metrics'].get(metric, 0)
                    all_scores[metric].append(score)
        
        for metric in metrics:
            scores = all_scores[metric]
            if scores:
                summary['performance_statistics'][metric] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
        
        return summary
    
    def get_best_model_recommendation(self):
        """Get best model recommendation"""
        if not self.evaluation_results:
            return None
        
        # Select best model based on Macro F1
        best_score = 0
        best_model = None
        best_result = None
        
        for model_name, result in self.evaluation_results.items():
            if 'basic_metrics' in result:
                macro_f1 = result['basic_metrics'].get('macro_f1', 0)
                if macro_f1 > best_score:
                    best_score = macro_f1
                    best_model = model_name
                    best_result = result
        
        if best_model and best_result:
            recommendation = {
                'recommended_model': best_model,
                'macro_f1_score': best_score,
                'key_strengths': [],
                'potential_concerns': []
            }
            
            # Strength analysis
            if best_score >= 0.8:
                recommendation['key_strengths'].append("High Macro F1 score")
            
            if best_result.get('probability_metrics'):
                mean_conf = best_result['probability_metrics']['mean_max_probability']
                if mean_conf >= 0.8:
                    recommendation['key_strengths'].append("High prediction confidence")
                
                if 'entropy_analysis' in best_result['probability_metrics']:
                    entropy_stats = best_result['probability_metrics']['entropy_analysis']['entropy_stats']
                    if entropy_stats['mean_entropy'] < 1.0:
                        recommendation['key_strengths'].append("Low prediction uncertainty")
            
            # Concern analysis
            if best_result.get('imbalance_metrics'):
                missing_classes = best_result['imbalance_metrics']['missing_pred_classes']
                if missing_classes:
                    recommendation['potential_concerns'].append(f"Unpredicted classes: {len(missing_classes)}")
                
                over_pred = best_result['imbalance_metrics']['over_predicted_classes']
                under_pred = best_result['imbalance_metrics']['under_predicted_classes']
                if len(over_pred) + len(under_pred) > 5:
                    recommendation['potential_concerns'].append("Imbalanced prediction for many classes")
            
            class_metrics = best_result.get('class_metrics', [])
            valid_classes = [cm for cm in class_metrics if cm['support'] > 0]
            if valid_classes:
                f1_scores = [cm['f1_score'] for cm in valid_classes]
                low_performance_count = sum(1 for score in f1_scores if score < Config.CLASS_PERFORMANCE_THRESHOLD)
                if low_performance_count > len(f1_scores) * 0.3:
                    recommendation['potential_concerns'].append("Low performance for many classes")
            
            return recommendation
        
        return None