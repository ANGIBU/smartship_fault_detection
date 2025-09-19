# evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, calculate_all_metrics, print_confusion_matrix, save_results

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        self.plots_saved = []
        self.class_reports = {}
        
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
        
        # 확률 기반 메트릭
        prob_metrics = {}
        if y_proba is not None:
            prob_metrics = self._calculate_probability_metrics(y_test, y_proba)
        
        # 클래스 불균형 분석
        imbalance_metrics = self._analyze_class_imbalance(y_test, y_pred)
        
        # 결과 저장
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
        
        # 결과 출력
        self._print_evaluation_summary(evaluation_result)
        
        return evaluation_result
    
    def _calculate_class_metrics(self, y_true, y_pred):
        """클래스별 성능 메트릭 계산"""
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
        """혼동 행렬 분석"""
        cm = confusion_matrix(y_true, y_pred, labels=range(Config.N_CLASSES))
        
        # 정규화된 혼동 행렬
        cm_sum = cm.sum(axis=1)
        cm_normalized = np.zeros_like(cm, dtype=float)
        
        for i in range(len(cm_sum)):
            if cm_sum[i] > 0:
                cm_normalized[i] = cm[i] / cm_sum[i]
        
        # 대각선 성분
        diagonal_accuracy = np.diag(cm_normalized)
        
        # 가장 많이 혼동되는 클래스 쌍
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
        
        # 클래스별 정확도
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
        """확률 기반 메트릭 계산"""
        # 각 예측의 최대 확률
        max_proba = np.max(y_proba, axis=1)
        
        # 예측 신뢰도 분석
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
        
        # 클래스별 평균 확률
        class_proba_avg = []
        for class_id in range(Config.N_CLASSES):
            class_mask = y_true == class_id
            if np.sum(class_mask) > 0:
                avg_proba = np.mean(y_proba[class_mask, class_id])
            else:
                avg_proba = 0.0
            class_proba_avg.append(float(avg_proba))
        
        # 확률 보정 분석
        calibration_metrics = self._analyze_calibration(y_true, y_proba)
        
        return {
            'max_probabilities': max_proba.tolist(),
            'mean_max_probability': float(np.mean(max_proba)),
            'confidence_analysis': confidence_analysis,
            'class_avg_probabilities': class_proba_avg,
            'calibration_metrics': calibration_metrics
        }
    
    def _analyze_calibration(self, y_true, y_proba):
        """확률 보정 분석"""
        try:
            # 전체적인 보정 분석
            y_pred_proba = np.max(y_proba, axis=1)
            y_pred = np.argmax(y_proba, axis=1)
            
            # 이진 분류로 변환하여 보정 분석
            calibration_results = {}
            
            for class_id in range(min(5, Config.N_CLASSES)):  # 상위 5개 클래스만
                if np.sum(y_true == class_id) > 10:  # 충분한 샘플이 있는 경우만
                    y_binary = (y_true == class_id).astype(int)
                    y_prob_binary = y_proba[:, class_id]
                    
                    try:
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            y_binary, y_prob_binary, n_bins=10, strategy='uniform'
                        )
                        
                        # 보정 오차 계산
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
            print(f"보정 분석 실패: {e}")
            return {}
    
    def _analyze_class_imbalance(self, y_true, y_pred):
        """클래스 불균형 분석"""
        # 실제 분포
        true_counts = np.bincount(y_true, minlength=Config.N_CLASSES)
        pred_counts = np.bincount(y_pred, minlength=Config.N_CLASSES)
        
        total_samples = len(y_true)
        
        # 불균형 메트릭 계산
        true_distribution = true_counts / total_samples
        pred_distribution = pred_counts / total_samples
        
        # 분포 차이
        distribution_diff = np.abs(true_distribution - pred_distribution)
        
        # 불균형 비율
        true_max = np.max(true_counts)
        true_min = np.min(true_counts[true_counts > 0]) if np.any(true_counts > 0) else 1
        true_imbalance_ratio = true_max / true_min
        
        pred_max = np.max(pred_counts)
        pred_min = np.min(pred_counts[pred_counts > 0]) if np.any(pred_counts > 0) else 1
        pred_imbalance_ratio = pred_max / pred_min
        
        # 누락된 클래스
        missing_true_classes = np.where(true_counts == 0)[0].tolist()
        missing_pred_classes = np.where(pred_counts == 0)[0].tolist()
        
        return {
            'true_distribution': true_distribution.tolist(),
            'pred_distribution': pred_distribution.tolist(),
            'distribution_difference': distribution_diff.tolist(),
            'true_imbalance_ratio': float(true_imbalance_ratio),
            'pred_imbalance_ratio': float(pred_imbalance_ratio),
            'missing_true_classes': missing_true_classes,
            'missing_pred_classes': missing_pred_classes,
            'total_samples': total_samples
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
        if class_metrics:
            class_f1_scores = [cm['f1_score'] for cm in class_metrics if cm['support'] > 0]
            
            if class_f1_scores:
                best_f1 = max(class_f1_scores)
                worst_f1 = min(class_f1_scores)
                
                best_class_idx = next(i for i, cm in enumerate(class_metrics) 
                                    if cm['f1_score'] == best_f1 and cm['support'] > 0)
                worst_class_idx = next(i for i, cm in enumerate(class_metrics) 
                                     if cm['f1_score'] == worst_f1 and cm['support'] > 0)
                
                print(f"\n최고 성능 클래스: {best_class_idx} (F1: {best_f1:.4f})")
                print(f"최저 성능 클래스: {worst_class_idx} (F1: {worst_f1:.4f})")
        
        # 확률 기반 메트릭
        if evaluation_result['probability_metrics']:
            prob_metrics = evaluation_result['probability_metrics']
            mean_confidence = prob_metrics['mean_max_probability']
            print(f"평균 예측 신뢰도: {mean_confidence:.4f}")
        
        # 클래스 불균형 정보
        imbalance_metrics = evaluation_result['imbalance_metrics']
        if imbalance_metrics:
            true_ratio = imbalance_metrics['true_imbalance_ratio']
            pred_ratio = imbalance_metrics['pred_imbalance_ratio']
            print(f"실제 불균형 비율: {true_ratio:.2f}:1")
            print(f"예측 불균형 비율: {pred_ratio:.2f}:1")
    
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
        
        print(f"\n모델별 성능 순위:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # 최고 성능 모델
        if not comparison_df.empty:
            best_model = comparison_df.iloc[0]
            print(f"\n최고 성능 모델: {best_model['model']}")
            print(f"Macro F1 Score: {best_model['macro_f1']:.4f}")
        
        return comparison_df
    
    @timer
    def analyze_class_performance(self, evaluation_result):
        """클래스별 성능 분석"""
        model_name = evaluation_result['model_name']
        class_metrics = evaluation_result['class_metrics']
        
        print(f"=== {model_name} 클래스별 성능 분석 ===")
        
        if not class_metrics:
            print("클래스 메트릭이 없습니다.")
            return None
        
        class_df = pd.DataFrame(class_metrics)
        
        print(f"\n클래스별 성능 (상위 15개):")
        display_df = class_df.head(15)
        print(display_df.to_string(index=False, float_format='%.4f'))
        
        if len(class_df) > 15:
            print(f"... (총 {len(class_df)}개 클래스)")
        
        # 성능 통계 (지원 샘플이 있는 클래스만)
        valid_classes = class_df[class_df['support'] > 0]
        
        if not valid_classes.empty:
            print(f"\nF1 Score 통계 ({len(valid_classes)}개 클래스):")
            print(f"평균: {valid_classes['f1_score'].mean():.4f}")
            print(f"표준편차: {valid_classes['f1_score'].std():.4f}")
            print(f"최대: {valid_classes['f1_score'].max():.4f}")
            print(f"최소: {valid_classes['f1_score'].min():.4f}")
            
            # 성능이 낮은 클래스 식별
            low_performance_threshold = 0.6
            low_performance_classes = valid_classes[valid_classes['f1_score'] < low_performance_threshold]
            
            if not low_performance_classes.empty:
                print(f"\n성능이 낮은 클래스 (F1 < {low_performance_threshold}):")
                low_perf_display = low_performance_classes[['class', 'f1_score', 'support']].head(10)
                print(low_perf_display.to_string(index=False, float_format='%.4f'))
                
                if len(low_performance_classes) > 10:
                    print(f"... (총 {len(low_performance_classes)}개)")
        
        return class_df
    
    @timer
    def analyze_confusion_patterns(self, evaluation_result):
        """혼동 패턴 분석"""
        model_name = evaluation_result['model_name']
        confusion_metrics = evaluation_result['confusion_metrics']
        
        print(f"=== {model_name} 혼동 패턴 분석 ===")
        
        if not confusion_metrics:
            print("혼동 행렬 메트릭이 없습니다.")
            return None
        
        # 가장 많이 혼동되는 클래스 쌍
        top_confusions = confusion_metrics['top_confusions']
        
        print(f"\n가장 많이 혼동되는 클래스 쌍 (상위 10개):")
        for i, confusion in enumerate(top_confusions[:10]):
            print(f"{i+1:2d}. 클래스 {confusion['true_class']:2d} → {confusion['pred_class']:2d}: "
                  f"{confusion['count']:4d}회 ({confusion['rate']*100:5.1f}%)")
        
        # 클래스별 정확도
        class_accuracies = confusion_metrics['class_accuracies']
        
        print(f"\n클래스별 정확도 (상위 15개):")
        for i, acc_info in enumerate(class_accuracies[:15]):
            class_id = acc_info['class']
            accuracy = acc_info['accuracy']
            total = acc_info['total_samples']
            print(f"클래스 {class_id:2d}: {accuracy:.4f} (샘플: {total:4d}개)")
        
        if len(class_accuracies) > 15:
            print(f"... (총 {len(class_accuracies)}개 클래스)")
        
        return confusion_metrics
    
    @timer
    def analyze_prediction_confidence(self, evaluation_result):
        """예측 신뢰도 분석"""
        model_name = evaluation_result['model_name']
        prob_metrics = evaluation_result.get('probability_metrics')
        
        if not prob_metrics:
            print(f"{model_name}: 확률 메트릭이 없습니다.")
            return None
        
        print(f"=== {model_name} 예측 신뢰도 분석 ===")
        
        confidence_analysis = prob_metrics['confidence_analysis']
        
        print(f"\n신뢰도별 예측 분석:")
        print(f"{'임계값':>6} {'샘플수':>8} {'비율(%)':>8} {'정확도':>8}")
        print("-" * 32)
        
        for threshold, analysis in confidence_analysis.items():
            count = analysis['count']
            percentage = analysis['percentage']
            accuracy = analysis['accuracy']
            print(f"{threshold:6.2f} {count:8d} {percentage:7.1f} {accuracy:8.4f}")
        
        # 보정 분석
        calibration_metrics = prob_metrics.get('calibration_metrics', {})
        if calibration_metrics:
            print(f"\n확률 보정 분석:")
            for class_info, metrics in calibration_metrics.items():
                error = metrics['calibration_error']
                print(f"{class_info}: 보정 오차 {error:.4f}")
        
        return prob_metrics
    
    @timer
    def generate_detailed_report(self, evaluation_result, save_path=None):
        """상세 평가 보고서 생성"""
        model_name = evaluation_result['model_name']
        
        report_lines = []
        report_lines.append(f"모델 평가 보고서: {model_name}")
        report_lines.append("=" * 60)
        
        # 1. 기본 성능 메트릭
        metrics = evaluation_result['basic_metrics']
        report_lines.append(f"\n1. 기본 성능 메트릭")
        report_lines.append("-" * 30)
        for metric_name, value in metrics.items():
            report_lines.append(f"{metric_name:20s}: {value:.4f}")
        
        # 2. 클래스별 성능 요약
        class_metrics = evaluation_result['class_metrics']
        if class_metrics:
            report_lines.append(f"\n2. 클래스별 성능 요약")
            report_lines.append("-" * 30)
            
            valid_classes = [cm for cm in class_metrics if cm['support'] > 0]
            if valid_classes:
                f1_scores = [cm['f1_score'] for cm in valid_classes]
                report_lines.append(f"평가 가능 클래스 수: {len(valid_classes)}")
                report_lines.append(f"평균 F1 Score: {np.mean(f1_scores):.4f}")
                report_lines.append(f"F1 Score 표준편차: {np.std(f1_scores):.4f}")
                report_lines.append(f"최고 F1 Score: {np.max(f1_scores):.4f}")
                report_lines.append(f"최저 F1 Score: {np.min(f1_scores):.4f}")
        
        # 3. 주요 혼동 패턴
        confusion_metrics = evaluation_result['confusion_metrics']
        if confusion_metrics and confusion_metrics['top_confusions']:
            report_lines.append(f"\n3. 주요 혼동 패턴")
            report_lines.append("-" * 30)
            
            top_confusions = confusion_metrics['top_confusions']
            for i, confusion in enumerate(top_confusions[:8]):
                report_lines.append(
                    f"{i+1}. 클래스 {confusion['true_class']} → {confusion['pred_class']}: "
                    f"{confusion['count']}회 ({confusion['rate']*100:.1f}%)"
                )
        
        # 4. 예측 신뢰도 분석
        prob_metrics = evaluation_result.get('probability_metrics')
        if prob_metrics:
            report_lines.append(f"\n4. 예측 신뢰도 분석")
            report_lines.append("-" * 30)
            
            mean_conf = prob_metrics['mean_max_probability']
            report_lines.append(f"평균 최대 확률: {mean_conf:.4f}")
            
            confidence_analysis = prob_metrics['confidence_analysis']
            report_lines.append(f"\n신뢰도별 분석:")
            for threshold, analysis in confidence_analysis.items():
                if threshold in [0.7, 0.8, 0.9]:  # 주요 임계값만
                    count = analysis['count']
                    percentage = analysis['percentage']
                    accuracy = analysis['accuracy']
                    report_lines.append(
                        f"임계값 {threshold}: {count}개 ({percentage:.1f}%), "
                        f"정확도 {accuracy:.4f}"
                    )
        
        # 5. 클래스 불균형 분석
        imbalance_metrics = evaluation_result.get('imbalance_metrics')
        if imbalance_metrics:
            report_lines.append(f"\n5. 클래스 불균형 분석")
            report_lines.append("-" * 30)
            
            true_ratio = imbalance_metrics['true_imbalance_ratio']
            pred_ratio = imbalance_metrics['pred_imbalance_ratio']
            report_lines.append(f"실제 불균형 비율: {true_ratio:.2f}:1")
            report_lines.append(f"예측 불균형 비율: {pred_ratio:.2f}:1")
            
            missing_pred = imbalance_metrics['missing_pred_classes']
            if missing_pred:
                report_lines.append(f"예측되지 않은 클래스: {missing_pred}")
        
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
        
        if not self.evaluation_results:
            print("저장할 평가 결과가 없습니다.")
            return
        
        # 모든 평가 결과를 CSV로 저장
        all_results = []
        
        for model_name, result in self.evaluation_results.items():
            metrics = result['basic_metrics']
            row = {'model': model_name}
            row.update(metrics)
            
            # 추가 메트릭
            if result.get('probability_metrics'):
                prob_metrics = result['probability_metrics']
                row['mean_confidence'] = prob_metrics['mean_max_probability']
            
            if result.get('imbalance_metrics'):
                imbalance_metrics = result['imbalance_metrics']
                row['pred_imbalance_ratio'] = imbalance_metrics['pred_imbalance_ratio']
            
            all_results.append(row)
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            save_path = output_dir / 'evaluation_results.csv'
            save_results(results_df, save_path)
        
        # 클래스별 성능 저장
        for model_name, result in self.evaluation_results.items():
            class_metrics = result['class_metrics']
            if class_metrics:
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
        
        # 성능 통계
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
        """최적 모델 추천"""
        if not self.evaluation_results:
            return None
        
        # Macro F1을 기준으로 최고 모델 선정
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
            
            # 강점 분석
            if best_score >= 0.8:
                recommendation['key_strengths'].append("높은 Macro F1 점수")
            
            if best_result.get('probability_metrics'):
                mean_conf = best_result['probability_metrics']['mean_max_probability']
                if mean_conf >= 0.8:
                    recommendation['key_strengths'].append("높은 예측 신뢰도")
            
            # 우려사항 분석
            if best_result.get('imbalance_metrics'):
                missing_classes = best_result['imbalance_metrics']['missing_pred_classes']
                if missing_classes:
                    recommendation['potential_concerns'].append(f"예측되지 않은 클래스: {len(missing_classes)}개")
            
            class_metrics = best_result.get('class_metrics', [])
            valid_classes = [cm for cm in class_metrics if cm['support'] > 0]
            if valid_classes:
                f1_scores = [cm['f1_score'] for cm in valid_classes]
                low_performance_count = sum(1 for score in f1_scores if score < 0.6)
                if low_performance_count > len(f1_scores) * 0.3:
                    recommendation['potential_concerns'].append("다수 클래스의 낮은 성능")
            
            return recommendation
        
        return None