# evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_all_metrics, print_confusion_matrix, save_results

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        self.plots_saved = []
        
    @timer
    def comprehensive_evaluation(self, model, X_test, y_test, model_name="Model"):
        """종합적 모델 평가"""
        print(f"=== {model_name} 종합 평가 시작 ===")
        
        # 예측 수행
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        # 기본 메트릭 계산
        metrics = calculate_all_metrics(y_test, y_pred)
        
        # 클래스별 성능 분석
        class_metrics = self._calculate_class_metrics(y_test, y_pred)
        
        # 혼동 행렬 분석
        confusion_metrics = self._analyze_confusion_matrix(y_test, y_pred)
        
        # 확률 기반 메트릭 (가능한 경우)
        prob_metrics = {}
        if y_proba is not None:
            prob_metrics = self._calculate_probability_metrics(y_test, y_proba)
        
        # 결과 저장
        evaluation_result = {
            'model_name': model_name,
            'basic_metrics': metrics,
            'class_metrics': class_metrics,
            'confusion_metrics': confusion_metrics,
            'probability_metrics': prob_metrics,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        self.evaluation_results[model_name] = evaluation_result
        
        # 결과 출력
        self._print_evaluation_summary(evaluation_result)
        
        return evaluation_result
    
    def _calculate_class_metrics(self, y_true, y_pred):
        """클래스별 성능 메트릭 계산"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        class_metrics = []
        for i in range(len(precision)):
            class_metrics.append({
                'class': i,
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            })
        
        return class_metrics
    
    def _analyze_confusion_matrix(self, y_true, y_pred):
        """혼동 행렬 분석"""
        cm = confusion_matrix(y_true, y_pred)
        
        # 정규화된 혼동 행렬
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 대각선 성분 (정확히 분류된 비율)
        diagonal_accuracy = np.diag(cm_normalized)
        
        # 가장 많이 혼동되는 클래스 쌍
        confusion_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_class': i,
                        'pred_class': j,
                        'count': cm[i, j],
                        'rate': cm_normalized[i, j]
                    })
        
        confusion_pairs = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)
        
        return {
            'confusion_matrix': cm,
            'normalized_cm': cm_normalized,
            'diagonal_accuracy': diagonal_accuracy,
            'top_confusions': confusion_pairs[:10]
        }
    
    def _calculate_probability_metrics(self, y_true, y_proba):
        """확률 기반 메트릭 계산"""
        # 각 예측의 최대 확률
        max_proba = np.max(y_proba, axis=1)
        
        # 예측 신뢰도 분석
        confidence_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
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
                high_conf_acc = 0
            
            confidence_analysis[threshold] = {
                'count': high_conf_count,
                'percentage': high_conf_count / len(y_true) * 100,
                'accuracy': high_conf_acc
            }
        
        # 클래스별 평균 확률
        class_proba_avg = []
        for class_id in range(y_proba.shape[1]):
            class_mask = y_true == class_id
            if np.sum(class_mask) > 0:
                avg_proba = np.mean(y_proba[class_mask, class_id])
                class_proba_avg.append(avg_proba)
            else:
                class_proba_avg.append(0)
        
        return {
            'max_probabilities': max_proba,
            'mean_max_probability': np.mean(max_proba),
            'confidence_analysis': confidence_analysis,
            'class_avg_probabilities': class_proba_avg
        }
    
    def _print_evaluation_summary(self, evaluation_result):
        """평가 결과 요약 출력"""
        model_name = evaluation_result['model_name']
        metrics = evaluation_result['basic_metrics']
        
        print(f"\n=== {model_name} 평가 결과 요약 ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        
        # 최고/최저 성능 클래스
        class_metrics = evaluation_result['class_metrics']
        class_f1_scores = [cm['f1_score'] for cm in class_metrics]
        
        best_class_idx = np.argmax(class_f1_scores)
        worst_class_idx = np.argmin(class_f1_scores)
        
        print(f"\n최고 성능 클래스: {best_class_idx} (F1: {class_f1_scores[best_class_idx]:.4f})")
        print(f"최저 성능 클래스: {worst_class_idx} (F1: {class_f1_scores[worst_class_idx]:.4f})")
        
        # 확률 기반 메트릭 (있는 경우)
        if evaluation_result['probability_metrics']:
            prob_metrics = evaluation_result['probability_metrics']
            mean_confidence = prob_metrics['mean_max_probability']
            print(f"평균 예측 신뢰도: {mean_confidence:.4f}")
    
    @timer
    def compare_models(self, model_results):
        """여러 모델 성능 비교"""
        print("=== 모델 성능 비교 ===")
        
        comparison_data = []
        
        for model_name, result in model_results.items():
            metrics = result['basic_metrics']
            comparison_data.append({
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'weighted_f1': metrics['weighted_f1'],
                'macro_precision': metrics['macro_precision'],
                'macro_recall': metrics['macro_recall']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('macro_f1', ascending=False)
        
        print("\n모델별 성능 순위:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # 최고 성능 모델
        best_model = comparison_df.iloc[0]
        print(f"\n최고 성능 모델: {best_model['model']}")
        print(f"Macro F1 Score: {best_model['macro_f1']:.4f}")
        
        return comparison_df
    
    @timer
    def analyze_class_performance(self, evaluation_result):
        """클래스별 성능 상세 분석"""
        model_name = evaluation_result['model_name']
        class_metrics = evaluation_result['class_metrics']
        
        print(f"=== {model_name} 클래스별 성능 분석 ===")
        
        class_df = pd.DataFrame(class_metrics)
        
        print("\n클래스별 성능:")
        print(class_df.to_string(index=False, float_format='%.4f'))
        
        # 성능 통계
        print(f"\nF1 Score 통계:")
        print(f"평균: {class_df['f1_score'].mean():.4f}")
        print(f"표준편차: {class_df['f1_score'].std():.4f}")
        print(f"최대: {class_df['f1_score'].max():.4f}")
        print(f"최소: {class_df['f1_score'].min():.4f}")
        
        # 성능이 낮은 클래스 식별
        low_performance_threshold = 0.7
        low_performance_classes = class_df[class_df['f1_score'] < low_performance_threshold]
        
        if not low_performance_classes.empty:
            print(f"\n성능이 낮은 클래스 (F1 < {low_performance_threshold}):")
            print(low_performance_classes[['class', 'f1_score', 'support']].to_string(index=False))
        
        return class_df
    
    @timer
    def analyze_confusion_patterns(self, evaluation_result):
        """혼동 패턴 분석"""
        model_name = evaluation_result['model_name']
        confusion_metrics = evaluation_result['confusion_metrics']
        
        print(f"=== {model_name} 혼동 패턴 분석 ===")
        
        # 혼동 행렬 출력
        cm = confusion_metrics['confusion_matrix']
        print("\n혼동 행렬:")
        print(cm)
        
        # 가장 많이 혼동되는 클래스 쌍
        top_confusions = confusion_metrics['top_confusions']
        
        print("\n가장 많이 혼동되는 클래스 쌍:")
        for i, confusion in enumerate(top_confusions[:5]):
            print(f"{i+1}. 클래스 {confusion['true_class']} → {confusion['pred_class']}: "
                  f"{confusion['count']}회 ({confusion['rate']*100:.1f}%)")
        
        # 클래스별 정확도
        diagonal_acc = confusion_metrics['diagonal_accuracy']
        print("\n클래스별 정확도:")
        for i, acc in enumerate(diagonal_acc):
            print(f"클래스 {i}: {acc:.4f}")
        
        return confusion_metrics
    
    @timer
    def generate_detailed_report(self, evaluation_result, save_path=None):
        """상세 평가 보고서 생성"""
        model_name = evaluation_result['model_name']
        
        report_lines = []
        report_lines.append(f"모델 평가 보고서: {model_name}")
        report_lines.append("=" * 50)
        
        # 기본 메트릭
        metrics = evaluation_result['basic_metrics']
        report_lines.append("\n1. 기본 성능 메트릭")
        report_lines.append("-" * 20)
        for metric_name, value in metrics.items():
            report_lines.append(f"{metric_name}: {value:.4f}")
        
        # 클래스별 성능
        class_metrics = evaluation_result['class_metrics']
        report_lines.append("\n2. 클래스별 성능")
        report_lines.append("-" * 20)
        
        class_df = pd.DataFrame(class_metrics)
        report_lines.append(class_df.to_string(index=False, float_format='%.4f'))
        
        # 혼동 행렬
        confusion_metrics = evaluation_result['confusion_metrics']
        report_lines.append("\n3. 혼동 행렬")
        report_lines.append("-" * 20)
        
        cm = confusion_metrics['confusion_matrix']
        report_lines.append(str(cm))
        
        # 주요 혼동 패턴
        top_confusions = confusion_metrics['top_confusions']
        report_lines.append("\n4. 주요 혼동 패턴")
        report_lines.append("-" * 20)
        
        for i, confusion in enumerate(top_confusions[:5]):
            report_lines.append(
                f"{i+1}. 클래스 {confusion['true_class']} → {confusion['pred_class']}: "
                f"{confusion['count']}회 ({confusion['rate']*100:.1f}%)"
            )
        
        # 확률 분석 (있는 경우)
        if evaluation_result['probability_metrics']:
            prob_metrics = evaluation_result['probability_metrics']
            report_lines.append("\n5. 확률 분석")
            report_lines.append("-" * 20)
            
            report_lines.append(f"평균 최대 확률: {prob_metrics['mean_max_probability']:.4f}")
            
            confidence_analysis = prob_metrics['confidence_analysis']
            report_lines.append("\n신뢰도별 분석:")
            for threshold, analysis in confidence_analysis.items():
                report_lines.append(
                    f"임계값 {threshold}: {analysis['count']}개 ({analysis['percentage']:.1f}%), "
                    f"정확도 {analysis['accuracy']:.4f}"
                )
        
        # 보고서 텍스트 생성
        report_text = "\n".join(report_lines)
        
        # 파일 저장
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                print(f"보고서 저장: {save_path}")
            except Exception as e:
                print(f"보고서 저장 실패: {e}")
        
        return report_text
    
    @timer
    def save_evaluation_results(self, output_dir=None):
        """평가 결과 저장"""
        if output_dir is None:
            output_dir = Config.MODEL_DIR
        
        # 모든 평가 결과를 CSV로 저장
        all_results = []
        
        for model_name, result in self.evaluation_results.items():
            metrics = result['basic_metrics']
            row = {'model': model_name}
            row.update(metrics)
            all_results.append(row)
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            save_path = output_dir / 'evaluation_results.csv'
            save_results(results_df, save_path)
        
        # 클래스별 성능 저장
        for model_name, result in self.evaluation_results.items():
            class_metrics = result['class_metrics']
            class_df = pd.DataFrame(class_metrics)
            save_path = output_dir / f'{model_name}_class_performance.csv'
            save_results(class_df, save_path)
        
        print("평가 결과 저장 완료")
    
    def create_performance_summary(self):
        """성능 요약 생성"""
        if not self.evaluation_results:
            print("평가 결과가 없습니다.")
            return None
        
        summary = {
            'total_models': len(self.evaluation_results),
            'models_evaluated': list(self.evaluation_results.keys()),
            'best_models': {},
            'performance_statistics': {}
        }
        
        # 각 메트릭별 최고 모델
        metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
        
        for metric in metrics:
            best_score = -1
            best_model = None
            
            for model_name, result in self.evaluation_results.items():
                score = result['basic_metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            summary['best_models'][metric] = {
                'model': best_model,
                'score': best_score
            }
        
        # 성능 통계
        all_scores = {metric: [] for metric in metrics}
        
        for result in self.evaluation_results.values():
            for metric in metrics:
                all_scores[metric].append(result['basic_metrics'][metric])
        
        for metric in metrics:
            scores = all_scores[metric]
            summary['performance_statistics'][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return summary