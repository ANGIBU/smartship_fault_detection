# analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_macro_f1
import gc

plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class ModelAnalyzer:
    def __init__(self):
        self.analysis_results = {}
        self.figures = {}
        
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
            
            # Create plotly figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=train_sizes_abs,
                y=train_mean,
                mode='lines+markers',
                name='Training Score',
                line=dict(color='blue'),
                error_y=dict(type='data', array=train_std, visible=True)
            ))
            
            fig.add_trace(go.Scatter(
                x=train_sizes_abs,
                y=val_mean,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='red'),
                error_y=dict(type='data', array=val_std, visible=True)
            ))
            
            fig.update_layout(
                title=f'{model_name} Learning Curve',
                xaxis_title='Training Set Size',
                yaxis_title='Macro F1 Score',
                template='plotly_white',
                width=800,
                height=500
            )
            
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
                }).sort_values('importance', ascending=True)
                
                # Create horizontal bar chart
                fig = px.bar(
                    importance_df.tail(20),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f'{model_name} Feature Importance ({method})',
                    template='plotly_white',
                    width=800,
                    height=600
                )
                
                self.figures[f'{model_name}_feature_importance'] = fig
                
                # Analysis results
                top_features = importance_df.tail(10)['feature'].tolist()
                low_importance_features = importance_df.head(10)['feature'].tolist()
                
                analysis = {
                    'method': method,
                    'feature_names': feature_names,
                    'importance_scores': importance_scores.tolist(),
                    'top_features': top_features,
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
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=[f"Class {i}" for i in range(Config.N_CLASSES)],
                y=[f"Class {i}" for i in range(Config.N_CLASSES)],
                title=f'{model_name} Confusion Matrix',
                template='plotly_white',
                width=800,
                height=600,
                aspect="auto"
            )
            
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
                fig = go.Figure()
                
                for _, row in stability_df.iterrows():
                    fig.add_trace(go.Bar(
                        name=row['model'],
                        x=[row['model']],
                        y=[row['mean']],
                        error_y=dict(type='data', array=[row['std']], visible=True)
                    ))
                
                fig.update_layout(
                    title='Cross-Validation Performance Comparison',
                    xaxis_title='Models',
                    yaxis_title='Macro F1 Score',
                    template='plotly_white',
                    width=800,
                    height=500,
                    showlegend=False
                )
                
                self.figures['cv_stability'] = fig
                
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
        """Quick performance analysis for --quick mode"""
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
                top_features = np.argsort(importance_scores)[-5:][::-1]
                
                analysis_summary['feature_analysis'] = {
                    'top_5_features': [feature_names[i] for i in top_features],
                    'top_5_scores': [float(importance_scores[i]) for i in top_features],
                    'zero_importance_count': int(np.sum(importance_scores == 0))
                }
            
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
            
            # Simple performance visualization
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
                
                fig = px.bar(
                    x=scores,
                    y=features,
                    orientation='h',
                    title=f"Top 5 Important Features - {analysis_summary['model_name']}",
                    template='plotly_white',
                    width=600,
                    height=400
                )
                
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
    
    def save_visualizations(self, output_dir=None):
        """Save all generated visualizations"""
        if output_dir is None:
            output_dir = Config.MODEL_DIR
        
        if not self.figures:
            print("No visualizations to save")
            return
        
        saved_files = []
        
        for fig_name, fig in self.figures.items():
            try:
                output_path = output_dir / f"{fig_name}.html"
                fig.write_html(str(output_path))
                saved_files.append(output_path)
            except Exception as e:
                print(f"Failed to save {fig_name}: {e}")
        
        if saved_files:
            print(f"Saved {len(saved_files)} visualization files")
            
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
    
    def clear_analysis(self):
        """Clear all analysis results and figures"""
        self.analysis_results.clear()
        self.figures.clear()
        gc.collect()
        print("Analysis data cleared")