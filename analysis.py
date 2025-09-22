# analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_macro_f1
import gc

# Set matplotlib backend and style
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

class ModelAnalyzer:
    def __init__(self):
        self.analysis_results = {}
        self.figures = {}
        
    def draw_ascii_chart(self, data_dict, title="Performance Chart", max_width=40):
        """Draw ASCII chart in console"""
        if not data_dict:
            return
        
        # Sort by value
        sorted_data = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
        max_value = max(data_dict.values()) if data_dict.values() else 1
        
        # Calculate box width
        box_width = max_width + 20
        
        print("┌" + "─" * (box_width - 2) + "┐")
        print(f"│ {title:^{box_width - 4}} │")
        print("├" + "─" * (box_width - 2) + "┤")
        
        for name, value in sorted_data:
            bar_length = int((value / max_value) * max_width) if max_value > 0 else 0
            bar = "█" * bar_length
            name_truncated = name[:12] if len(name) > 12 else name
            print(f"│ {name_truncated:<12} {bar:<{max_width}} {value:.4f}│")
        
        print("└" + "─" * (box_width - 2) + "┘")
        print()
    
    def draw_class_performance_chart(self, class_metrics, title="Class Performance", max_width=35):
        """Draw class performance ASCII chart"""
        if not class_metrics:
            return
        
        # Filter valid classes and sort by F1 score
        valid_classes = [cm for cm in class_metrics if cm['support'] > 0]
        if not valid_classes:
            return
        
        sorted_classes = sorted(valid_classes, key=lambda x: x['f1_score'], reverse=True)[:15]
        max_f1 = max(cm['f1_score'] for cm in sorted_classes) if sorted_classes else 1
        
        box_width = max_width + 25
        
        print("┌" + "─" * (box_width - 2) + "┐")
        print(f"│ {title:^{box_width - 4}} │")
        print("├" + "─" * (box_width - 2) + "┤")
        
        for cm in sorted_classes[:10]:  # Top 10 classes
            bar_length = int((cm['f1_score'] / max_f1) * max_width) if max_f1 > 0 else 0
            bar = "█" * bar_length
            class_name = f"Class {cm['class']:2d}"
            print(f"│ {class_name:<8} {bar:<{max_width}} {cm['f1_score']:.4f} ({cm['support']:3d})│")
        
        print("└" + "─" * (box_width - 2) + "┘")
        print()
    
    @timer
    def analyze_learning_curve(self, model, X_train, y_train, model_name="Model"):
        """Generate learning curve analysis"""
        print(f"Analyzing learning curve for {model_name}")
        
        try:
            train_sizes = np.linspace(0.1, 1.0, 10)
            
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
            
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                cv=cv_strategy,
                train_sizes=train_sizes,
                scoring='f1_macro',
                n_jobs=1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            ax.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
            ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            ax.set_title(f'{model_name} Learning Curve', fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Set Size', fontsize=12)
            ax.set_ylabel('Macro F1 Score', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.figures[f'{model_name}_learning_curve'] = fig
            
            # Analysis results
            final_gap = train_mean[-1] - val_mean[-1]
            performance_trend = 'increasing' if val_mean[-1] > val_mean[0] else 'decreasing'
            
            analysis = {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores': train_mean.tolist(),
                'val_scores': val_mean.tolist(),
                'final_train_score': float(train_mean[-1]),
                'final_val_score': float(val_mean[-1]),
                'overfitting_gap': float(final_gap),
                'performance_trend': performance_trend,
                'recommendations': self._generate_learning_recommendations(final_gap, performance_trend)
            }
            
            self.analysis_results[f'{model_name}_learning_curve'] = analysis
            
            return analysis
            
        except Exception as e:
            print(f"Learning curve analysis failed: {e}")
            return None
    
    def _generate_learning_recommendations(self, gap, trend):
        """Generate recommendations based on learning curve"""
        recommendations = []
        
        if gap > 0.05:
            recommendations.append("High overfitting detected - consider regularization")
        elif gap < 0.01:
            recommendations.append("Good training-validation balance")
        
        if trend == 'increasing':
            recommendations.append("More training data may help")
        else:
            recommendations.append("Model may benefit from feature engineering")
        
        return recommendations
    
    @timer
    def analyze_feature_importance(self, model, X_train, y_train, feature_names, model_name="Model"):
        """Analyze feature importance"""
        print(f"Analyzing feature importance for {model_name}")
        
        try:
            importance_scores = None
            method = "built_in"
            
            # Try built-in feature importance first
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                importance_scores = model.get_feature_importance()
            else:
                # Use permutation importance
                method = "permutation"
                perm_importance = permutation_importance(
                    model, X_train, y_train,
                    n_repeats=5,
                    random_state=Config.RANDOM_STATE,
                    scoring='f1_macro',
                    n_jobs=1
                )
                importance_scores = perm_importance.importances_mean
            
            if importance_scores is not None:
                # Create DataFrame for analysis
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_scores
                }).sort_values('importance', ascending=False)
                
                # Create matplotlib horizontal bar chart
                fig, ax = plt.subplots(figsize=(12, 8))
                
                top_features = importance_df.head(20)
                colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
                
                bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.set_xlabel('Importance Score', fontsize=12)
                ax.set_title(f'{model_name} Feature Importance ({method})', fontsize=14, fontweight='bold')
                ax.grid(True, axis='x', alpha=0.3)
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', ha='left', va='center', fontsize=9)
                
                plt.tight_layout()
                self.figures[f'{model_name}_feature_importance'] = fig
                
                # Console ASCII chart
                top_10_dict = dict(zip(top_features['feature'].head(10), top_features['importance'].head(10)))
                self.draw_ascii_chart(top_10_dict, f"Top 10 Features - {model_name}")
                
                # Analysis results
                top_features_list = importance_df.head(10)['feature'].tolist()
                low_importance_features = importance_df.tail(10)['feature'].tolist()
                
                analysis = {
                    'method': method,
                    'feature_names': feature_names,
                    'importance_scores': importance_scores.tolist(),
                    'top_features': top_features_list,
                    'low_importance_features': low_importance_features,
                    'importance_ratio': float(importance_scores.max() / (importance_scores.min() + 1e-8)),
                    'recommendations': self._generate_feature_recommendations(importance_df)
                }
                
                self.analysis_results[f'{model_name}_feature_importance'] = analysis
                return analysis
                
        except Exception as e:
            print(f"Feature importance analysis failed: {e}")
            return None
    
    def _generate_feature_recommendations(self, importance_df):
        """Generate feature engineering recommendations"""
        recommendations = []
        
        zero_importance = len(importance_df[importance_df['importance'] == 0])
        if zero_importance > 0:
            recommendations.append(f"Consider removing {zero_importance} zero-importance features")
        
        low_importance_threshold = importance_df['importance'].quantile(0.2)
        low_importance_count = len(importance_df[importance_df['importance'] < low_importance_threshold])
        if low_importance_count > len(importance_df) * 0.3:
            recommendations.append("Many low-importance features - feature selection needed")
        
        return recommendations
    
    @timer
    def analyze_confusion_matrix(self, model, X_test, y_test, model_name="Model"):
        """Analyze confusion matrix"""
        print(f"Analyzing confusion matrix for {model_name}")
        
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=range(Config.N_CLASSES))
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            
            mask = cm == 0
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', mask=mask, 
                       xticklabels=[f'C{i}' for i in range(Config.N_CLASSES)],
                       yticklabels=[f'C{i}' for i in range(Config.N_CLASSES)],
                       ax=ax, cbar_kws={'shrink': 0.8})
            
            ax.set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Class', fontsize=12)
            ax.set_ylabel('Actual Class', fontsize=12)
            
            plt.tight_layout()
            self.figures[f'{model_name}_confusion_matrix'] = fig
            
            # Calculate per-class metrics
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Find most confused classes
            confusion_pairs = []
            for i in range(Config.N_CLASSES):
                for j in range(Config.N_CLASSES):
                    if i != j and cm[i, j] > 0:
                        confusion_pairs.append({
                            'true_class': i,
                            'pred_class': j,
                            'count': int(cm[i, j])
                        })
            
            confusion_pairs = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)
            
            analysis = {
                'confusion_matrix': cm.tolist(),
                'macro_f1': float(report['macro avg']['f1-score']),
                'weighted_f1': float(report['weighted avg']['f1-score']),
                'per_class_f1': {str(i): float(report[str(i)]['f1-score']) 
                                for i in range(Config.N_CLASSES) if str(i) in report},
                'top_confusions': confusion_pairs[:10],
                'recommendations': self._generate_confusion_recommendations(confusion_pairs, report)
            }
            
            self.analysis_results[f'{model_name}_confusion_matrix'] = analysis
            return analysis
            
        except Exception as e:
            print(f"Confusion matrix analysis failed: {e}")
            return None
    
    def _generate_confusion_recommendations(self, confusion_pairs, report):
        """Generate recommendations based on confusion matrix"""
        recommendations = []
        
        if confusion_pairs:
            most_confused = confusion_pairs[0]
            recommendations.append(
                f"Most confusion: Class {most_confused['true_class']} -> Class {most_confused['pred_class']}"
            )
        
        # Find classes with low F1
        low_f1_classes = []
        for i in range(Config.N_CLASSES):
            if str(i) in report and report[str(i)]['f1-score'] < 0.5:
                low_f1_classes.append(i)
        
        if low_f1_classes:
            recommendations.append(f"Low F1 classes: {low_f1_classes}")
            recommendations.append("Consider class-specific data augmentation")
        
        return recommendations
    
    @timer
    def analyze_cross_validation_stability(self, cv_scores_dict):
        """Analyze cross-validation stability"""
        print("Analyzing cross-validation stability")
        
        try:
            model_names = list(cv_scores_dict.keys())
            stability_data = []
            
            for model_name, scores in cv_scores_dict.items():
                if 'scores' in scores:
                    cv_scores = scores['scores']
                    stability_data.append({
                        'model': model_name,
                        'mean': float(scores['mean']),
                        'std': float(scores['std']),
                        'min': float(np.min(cv_scores)),
                        'max': float(np.max(cv_scores)),
                        'stability': float(scores.get('stability', scores['mean'] - scores['std']))
                    })
            
            if stability_data:
                stability_df = pd.DataFrame(stability_data)
                
                # Create comparison chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Mean performance chart
                models = stability_df['model']
                means = stability_df['mean']
                stds = stability_df['std']
                
                bars1 = ax1.bar(models, means, yerr=stds, capsize=5, color='skyblue', alpha=0.7)
                ax1.set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Macro F1 Score', fontsize=12)
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, bar in enumerate(bars1):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + stds.iloc[i],
                            f'{height:.4f}', ha='center', va='bottom', fontsize=9)
                
                # Stability score chart
                stability_scores = stability_df['stability']
                bars2 = ax2.bar(models, stability_scores, color='lightcoral', alpha=0.7)
                ax2.set_title('Model Stability Scores', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Stability Score', fontsize=12)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, bar in enumerate(bars2):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                self.figures['cv_stability'] = fig
                
                # Console ASCII chart
                stability_dict = dict(zip(stability_df['model'], stability_df['stability']))
                self.draw_ascii_chart(stability_dict, "Model Stability Scores")
                
                # Analysis results
                best_model = stability_df.loc[stability_df['stability'].idxmax()]
                most_stable = stability_df.loc[stability_df['std'].idxmin()]
                
                analysis = {
                    'stability_data': stability_data,
                    'best_performance': {
                        'model': best_model['model'],
                        'score': best_model['stability']
                    },
                    'most_stable': {
                        'model': most_stable['model'],
                        'std': most_stable['std']
                    },
                    'recommendations': self._generate_stability_recommendations(stability_df)
                }
                
                self.analysis_results['cv_stability'] = analysis
                return analysis
                
        except Exception as e:
            print(f"CV stability analysis failed: {e}")
            return None
    
    def _generate_stability_recommendations(self, stability_df):
        """Generate stability recommendations"""
        recommendations = []
        
        high_variance_models = stability_df[stability_df['std'] > 0.05]
        if not high_variance_models.empty:
            recommendations.append(
                f"High variance models: {list(high_variance_models['model'])}"
            )
            recommendations.append("Consider ensemble methods for stability")
        
        performance_gap = stability_df['mean'].max() - stability_df['mean'].min()
        if performance_gap > 0.1:
            recommendations.append("Large performance gap between models")
            recommendations.append("Focus on best performing model type")
        
        return recommendations
    
    @timer
    def quick_performance_analysis(self, model, X_train, X_test, y_train, y_test, 
                                 feature_names, model_name="QuickModel"):
        """Quick performance analysis for quick mode"""
        print(f"Running quick performance analysis for {model_name}")
        
        analysis_summary = {
            'model_name': model_name,
            'recommendations': [],
            'performance_metrics': {},
            'feature_analysis': {}
        }
        
        try:
            # Quick performance metrics
            y_pred = model.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            analysis_summary['performance_metrics'] = {
                'macro_f1': float(macro_f1),
                'test_samples': len(X_test),
                'train_samples': len(X_train),
                'features_used': len(feature_names)
            }
            
            # Quick feature importance
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                top_indices = np.argsort(importance_scores)[-5:][::-1]
                
                analysis_summary['feature_analysis'] = {
                    'top_5_features': [feature_names[i] for i in top_indices],
                    'top_5_scores': [float(importance_scores[i]) for i in top_indices],
                    'zero_importance_count': int(np.sum(importance_scores == 0))
                }
                
                # Console ASCII chart for top features
                top_features_dict = {
                    feature_names[i]: float(importance_scores[i]) 
                    for i in top_indices
                }
                self.draw_ascii_chart(top_features_dict, f"Top 5 Features - {model_name}")
            
            # Generate quick recommendations
            recommendations = []
            
            if macro_f1 < 0.6:
                recommendations.append("Low F1 score - consider feature engineering")
            elif macro_f1 > 0.8:
                recommendations.append("Good performance - consider ensemble methods")
            
            if len(X_train) < 5000:
                recommendations.append("Small dataset - consider data augmentation")
            
            if 'zero_importance_count' in analysis_summary['feature_analysis']:
                zero_count = analysis_summary['feature_analysis']['zero_importance_count']
                if zero_count > len(feature_names) * 0.2:
                    recommendations.append(f"Many unused features ({zero_count}) - feature selection needed")
            
            analysis_summary['recommendations'] = recommendations
            
            # Create quick visualization
            self._create_quick_visualization(analysis_summary)
            
            return analysis_summary
            
        except Exception as e:
            print(f"Quick analysis failed: {e}")
            analysis_summary['error'] = str(e)
            return analysis_summary
    
    def _create_quick_visualization(self, analysis_summary):
        """Create simple visualization for quick mode"""
        try:
            if 'feature_analysis' in analysis_summary and 'top_5_features' in analysis_summary['feature_analysis']:
                features = analysis_summary['feature_analysis']['top_5_features']
                scores = analysis_summary['feature_analysis']['top_5_scores']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
                bars = ax.barh(features, scores, color=colors)
                
                ax.set_xlabel('Importance Score', fontsize=12)
                ax.set_title(f"Top 5 Important Features - {analysis_summary['model_name']}", 
                           fontsize=14, fontweight='bold')
                ax.grid(True, axis='x', alpha=0.3)
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', ha='left', va='center', fontsize=10)
                
                plt.tight_layout()
                self.figures['quick_feature_importance'] = fig
                
        except Exception as e:
            print(f"Quick visualization failed: {e}")
    
    @timer
    def generate_performance_report(self, output_path=None):
        """Generate comprehensive performance report"""
        print("Generating performance analysis report")
        
        if not self.analysis_results:
            print("No analysis results available")
            return None
        
        report_lines = []
        report_lines.append("Model Performance Analysis Report")
        report_lines.append("=" * 50)
        
        for analysis_name, results in self.analysis_results.items():
            report_lines.append(f"\n## {analysis_name.replace('_', ' ').title()}")
            report_lines.append("-" * 30)
            
            if 'recommendations' in results:
                report_lines.append("Recommendations:")
                for rec in results['recommendations']:
                    report_lines.append(f"  - {rec}")
            
            # Add key metrics
            if 'macro_f1' in results:
                report_lines.append(f"Macro F1: {results['macro_f1']:.4f}")
            
            if 'final_val_score' in results:
                report_lines.append(f"Final Validation Score: {results['final_val_score']:.4f}")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                print(f"Performance report saved: {output_path}")
            except Exception as e:
                print(f"Report save failed: {e}")
        
        return report_text
    
    def save_visualizations(self, output_dir=None, save_format='png'):
        """Save all generated visualizations as PNG or PDF"""
        if output_dir is None:
            output_dir = Config.MODEL_DIR
        
        if not self.figures:
            print("No visualizations to save")
            return
        
        saved_files = []
        
        for fig_name, fig in self.figures.items():
            try:
                if save_format.lower() == 'pdf':
                    output_path = output_dir / f"{fig_name}.pdf"
                else:
                    output_path = output_dir / f"{fig_name}.png"
                
                fig.savefig(str(output_path), format=save_format.lower(), dpi=300, bbox_inches='tight')
                saved_files.append(output_path)
                plt.close(fig)  # Close figure to free memory
                
            except Exception as e:
                print(f"Failed to save {fig_name}: {e}")
        
        if saved_files:
            print(f"Saved {len(saved_files)} visualization files as {save_format.upper()}")
            
        gc.collect()
    
    def show_visualization(self, fig_name):
        """Display specific visualization"""
        if fig_name in self.figures:
            self.figures[fig_name].show()
        else:
            print(f"Visualization '{fig_name}' not found")
            print(f"Available visualizations: {list(self.figures.keys())}")
    
    def get_improvement_suggestions(self):
        """Get consolidated improvement suggestions"""
        all_recommendations = []
        
        for analysis_name, results in self.analysis_results.items():
            if 'recommendations' in results:
                for rec in results['recommendations']:
                    all_recommendations.append(f"{analysis_name}: {rec}")
        
        # Prioritize recommendations
        priority_keywords = ['overfitting', 'low f1', 'feature selection', 'data augmentation']
        priority_recs = []
        other_recs = []
        
        for rec in all_recommendations:
            if any(keyword in rec.lower() for keyword in priority_keywords):
                priority_recs.append(rec)
            else:
                other_recs.append(rec)
        
        return {
            'high_priority': priority_recs,
            'general': other_recs,
            'total_count': len(all_recommendations)
        }
    
    def create_performance_dashboard(self, cv_scores_dict, class_metrics=None, model_name="Dashboard"):
        """Create comprehensive performance dashboard"""
        try:
            fig = plt.figure(figsize=(20, 12))
            
            # Model performance comparison
            ax1 = plt.subplot(2, 3, 1)
            if cv_scores_dict:
                models = list(cv_scores_dict.keys())
                scores = [cv_scores_dict[m]['mean'] for m in models]
                stds = [cv_scores_dict[m]['std'] for m in models]
                
                bars = ax1.bar(models, scores, yerr=stds, capsize=5, color='skyblue', alpha=0.7)
                ax1.set_title('Model Performance Comparison', fontweight='bold')
                ax1.set_ylabel('Macro F1 Score')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + stds[i],
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Class performance distribution
            ax2 = plt.subplot(2, 3, 2)
            if class_metrics:
                valid_classes = [cm for cm in class_metrics if cm['support'] > 0]
                if valid_classes:
                    f1_scores = [cm['f1_score'] for cm in valid_classes]
                    ax2.hist(f1_scores, bins=10, color='lightcoral', alpha=0.7, edgecolor='black')
                    ax2.set_title('F1 Score Distribution', fontweight='bold')
                    ax2.set_xlabel('F1 Score')
                    ax2.set_ylabel('Number of Classes')
                    ax2.grid(True, alpha=0.3)
            
            # Performance trends
            ax3 = plt.subplot(2, 3, 3)
            if cv_scores_dict:
                stability_scores = [cv_scores_dict[m].get('stability', cv_scores_dict[m]['mean'] - cv_scores_dict[m]['std']) 
                                  for m in models]
                ax3.scatter(scores, stability_scores, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
                ax3.set_xlabel('Mean F1 Score')
                ax3.set_ylabel('Stability Score')
                ax3.set_title('Performance vs Stability', fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                for i, model in enumerate(models):
                    ax3.annotate(model, (scores[i], stability_scores[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Model variance comparison
            ax4 = plt.subplot(2, 3, 4)
            if cv_scores_dict:
                ax4.bar(models, stds, color='orange', alpha=0.7)
                ax4.set_title('Model Variance Comparison', fontweight='bold')
                ax4.set_ylabel('Standard Deviation')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
            
            # Performance target analysis
            ax5 = plt.subplot(2, 3, 5)
            if cv_scores_dict:
                target_score = 0.83
                gaps = [target_score - score for score in scores]
                colors = ['green' if gap <= 0 else 'red' for gap in gaps]
                
                bars = ax5.bar(models, gaps, color=colors, alpha=0.7)
                ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax5.set_title('Gap to Target (0.83)', fontweight='bold')
                ax5.set_ylabel('Gap to Target')
                ax5.tick_params(axis='x', rotation=45)
                ax5.grid(True, alpha=0.3)
            
            # Summary statistics
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            if cv_scores_dict:
                best_model = max(cv_scores_dict.keys(), key=lambda x: cv_scores_dict[x].get('stability', 0))
                best_score = cv_scores_dict[best_model]['mean']
                
                summary_text = f"""Performance Summary:
                
Best Model: {best_model}
Best Score: {best_score:.4f}
Target Gap: {0.83 - best_score:.4f}

Model Count: {len(models)}
Score Range: {max(scores) - min(scores):.4f}
Avg Variance: {np.mean(stds):.4f}
                """
                
                ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            plt.tight_layout()
            self.figures[f'{model_name}_dashboard'] = fig
            
            # Console summary
            if cv_scores_dict:
                print("\n" + "="*60)
                print("PERFORMANCE DASHBOARD SUMMARY")
                print("="*60)
                
                # Best model console display
                stability_dict = {
                    model: cv_scores_dict[model].get('stability', cv_scores_dict[model]['mean'] - cv_scores_dict[model]['std'])
                    for model in models
                }
                self.draw_ascii_chart(stability_dict, "Model Stability Overview")
            
            return fig
            
        except Exception as e:
            print(f"Dashboard creation failed: {e}")
            return None
    
    def clear_analysis(self):
        """Clear all analysis results and figures"""
        # Close all matplotlib figures
        for fig in self.figures.values():
            try:
                plt.close(fig)
            except:
                pass
        
        self.analysis_results.clear()
        self.figures.clear()
        gc.collect()
        print("Analysis data cleared")