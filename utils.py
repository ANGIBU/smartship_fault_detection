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
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

def setup_logging(log_file=None, level='INFO'):
    """로깅 설정"""
    if log_file is None:
        log_file = Path('logs') / 'training.log'
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 기존 핸들러 제거
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
    """메모리 효율적 데이터 로드"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 파일 크기 확인
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if chunk_size and file_size_mb > 100:
            # 큰 파일은 청크 단위로 로드
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunks.append(chunk)
            data = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
        else:
            data = pd.read_csv(file_path)
        
        return data
    except Exception as e:
        raise Exception(f"데이터 로드 실패: {e}")

def save_model(model, file_path):
    """모델 저장"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 먼저 joblib로 시도
        try:
            joblib.dump(model, file_path, compress=3)
            print(f"모델 저장 완료 (joblib): {file_path}")
            return
        except Exception as joblib_error:
            print(f"joblib 저장 실패: {joblib_error}")
        
        # joblib 실패시 pickle로 시도
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 파일 크기 확인
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"모델 저장 완료 (pickle): {file_path} ({file_size_mb:.1f}MB)")
            
        except Exception as pickle_error:
            print(f"pickle 저장 실패: {pickle_error}")
            
            # 특수한 경우: 앙상블 모델일 때
            if hasattr(model, 'estimators_'):
                print("앙상블 모델 감지, 개별 저장 시도")
                try:
                    # 개별 모델들 저장
                    estimators_data = []
                    for name, estimator in model.estimators_:
                        estimator_path = file_path.parent / f"{name}_estimator.pkl"
                        try:
                            joblib.dump(estimator, estimator_path, compress=3)
                            estimators_data.append((name, str(estimator_path)))
                            print(f"개별 모델 저장: {name}")
                        except Exception as est_error:
                            print(f"개별 모델 저장 실패 {name}: {est_error}")
                    
                    # 메타 정보 저장
                    meta_info = {
                        'model_type': type(model).__name__,
                        'estimators': estimators_data,
                        'voting': getattr(model, 'voting', 'soft'),
                        'weights': getattr(model, 'weights', None)
                    }
                    
                    meta_path = file_path.parent / 'ensemble_meta.pkl'
                    with open(meta_path, 'wb') as f:
                        pickle.dump(meta_info, f)
                    
                    print(f"앙상블 메타 정보 저장: {meta_path}")
                    return
                    
                except Exception as ensemble_error:
                    print(f"앙상블 개별 저장 실패: {ensemble_error}")
            
            # 마지막 시도: 모델 상태만 저장
            try:
                model_state = {
                    'model_type': type(model).__name__,
                    'model_params': getattr(model, 'get_params', lambda: {})(),
                    'classes_': getattr(model, 'classes_', None),
                    'n_classes_': getattr(model, 'n_classes_', None)
                }
                
                state_path = file_path.parent / 'model_state.pkl'
                with open(state_path, 'wb') as f:
                    pickle.dump(model_state, f)
                
                print(f"모델 상태 저장: {state_path}")
                
            except Exception as state_error:
                print(f"모델 상태 저장도 실패: {state_error}")
                raise Exception(f"모든 저장 방법 실패: pickle={pickle_error}, joblib={joblib_error}")
        
    except Exception as e:
        print(f"모델 저장 실패: {e}")
        print("경고: 모델 저장에 실패했지만 프로그램을 계속 진행합니다.")

def load_model(file_path):
    """모델 로드"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {file_path}")
        
        # 먼저 joblib로 시도
        try:
            model = joblib.load(file_path)
            print(f"모델 로드 완료 (joblib): {file_path}")
            return model
        except:
            pass
        
        # joblib 실패시 pickle로 시도
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            print(f"모델 로드 완료 (pickle): {file_path}")
            return model
        except:
            pass
        
        # 앙상블 메타 정보 확인
        meta_path = file_path.parent / 'ensemble_meta.pkl'
        if meta_path.exists():
            print("앙상블 메타 정보 발견, 재구성 시도")
            try:
                with open(meta_path, 'rb') as f:
                    meta_info = pickle.load(f)
                
                estimators = []
                for name, estimator_path in meta_info['estimators']:
                    try:
                        estimator = joblib.load(estimator_path)
                        estimators.append((name, estimator))
                    except:
                        print(f"개별 모델 로드 실패: {name}")
                
                if estimators:
                    from sklearn.ensemble import VotingClassifier
                    model = VotingClassifier(
                        estimators=estimators,
                        voting=meta_info.get('voting', 'soft'),
                        weights=meta_info.get('weights')
                    )
                    print("앙상블 모델 재구성 완료")
                    return model
            except Exception as e:
                print(f"앙상블 재구성 실패: {e}")
        
        raise Exception("모든 로드 방법 실패")
        
    except Exception as e:
        raise Exception(f"모델 로드 실패: {e}")

def save_joblib(obj, file_path, compress=3):
    """joblib을 사용한 객체 저장"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path, compress=compress)
        
        # 파일 크기 확인
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"객체 저장 완료: {file_path} ({file_size_mb:.1f}MB)")
        else:
            print(f"객체 저장 완료: {file_path}")
            
    except Exception as e:
        raise Exception(f"객체 저장 실패: {e}")

def load_joblib(file_path):
    """joblib을 사용한 객체 로드"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        obj = joblib.load(file_path)
        print(f"객체 로드 완료: {file_path}")
        return obj
    except Exception as e:
        raise Exception(f"객체 로드 실패: {e}")

def calculate_macro_f1(y_true, y_pred):
    """Macro F1 스코어 계산"""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def calculate_weighted_f1(y_true, y_pred):
    """Weighted F1 스코어 계산"""
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

def calculate_all_metrics(y_true, y_pred):
    """모든 평가 메트릭 계산"""
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
    except Exception as e:
        print(f"메트릭 계산 중 오류: {e}")
        # 기본값 반환
        metrics = {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'micro_f1': 0.0,
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'weighted_precision': 0.0,
            'weighted_recall': 0.0
        }
    
    return metrics

def calculate_class_metrics(y_true, y_pred, labels=None):
    """클래스별 메트릭 계산"""
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
        print(f"클래스 메트릭 계산 중 오류: {e}")
        return []

def analyze_sensor_data_quality(df, sensor_columns):
    """센서 데이터 품질 분석"""
    print("센서 데이터 품질 분석")
    
    quality_report = {
        'total_sensors': len(sensor_columns),
        'sensor_stats': {},
        'quality_issues': [],
        'recommendations': []
    }
    
    for sensor in sensor_columns:
        if sensor not in df.columns:
            quality_report['quality_issues'].append(f"센서 누락: {sensor}")
            continue
        
        sensor_data = df[sensor]
        
        # 기본 통계
        stats = {
            'count': len(sensor_data),
            'missing': sensor_data.isnull().sum(),
            'zeros': (sensor_data == 0).sum(),
            'mean': sensor_data.mean(),
            'std': sensor_data.std(),
            'min': sensor_data.min(),
            'max': sensor_data.max(),
            'range': sensor_data.max() - sensor_data.min()
        }
        
        # 이상치 검출
        if len(sensor_data.dropna()) > 0:
            z_scores = np.abs(zscore(sensor_data.dropna()))
            outliers = np.sum(z_scores > 3)
            stats['outliers'] = outliers
            stats['outlier_rate'] = outliers / len(sensor_data) * 100
        else:
            stats['outliers'] = 0
            stats['outlier_rate'] = 0
        
        # 변동성 분석
        if stats['std'] > 0:
            stats['cv'] = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else np.inf
        else:
            stats['cv'] = 0
        
        # 신호 안정성
        if len(sensor_data) > 1:
            diff = np.diff(sensor_data.fillna(sensor_data.mean()))
            stats['signal_stability'] = np.std(diff)
        else:
            stats['signal_stability'] = 0
        
        quality_report['sensor_stats'][sensor] = stats
        
        # 품질 문제 식별
        if stats['missing'] > len(sensor_data) * 0.1:
            quality_report['quality_issues'].append(f"{sensor}: 결측치 비율 높음 ({stats['missing']/len(sensor_data)*100:.1f}%)")
        
        if stats['outlier_rate'] > 5:
            quality_report['quality_issues'].append(f"{sensor}: 이상치 비율 높음 ({stats['outlier_rate']:.1f}%)")
        
        if stats['cv'] > 2:
            quality_report['quality_issues'].append(f"{sensor}: 변동성 매우 높음 (CV: {stats['cv']:.2f})")
        
        if stats['zeros'] > len(sensor_data) * 0.2:
            quality_report['quality_issues'].append(f"{sensor}: 0값 비율 높음 ({stats['zeros']/len(sensor_data)*100:.1f}%)")
    
    # 권장사항 생성
    if quality_report['quality_issues']:
        quality_report['recommendations'].append("센서 데이터 정제 필요")
        quality_report['recommendations'].append("이상치 처리 방법 검토")
        quality_report['recommendations'].append("센서 캘리브레이션 확인")
    
    return quality_report

def detect_sensor_anomalies(df, sensor_columns, method='zscore'):
    """센서 이상 감지"""
    print(f"센서 이상 감지 ({method})")
    
    anomaly_report = {
        'method': method,
        'sensor_anomalies': {},
        'total_anomalies': 0
    }
    
    for sensor in sensor_columns:
        if sensor not in df.columns:
            continue
        
        sensor_data = df[sensor].dropna()
        if len(sensor_data) == 0:
            continue
        
        if method == 'zscore':
            # Z-score 기반 이상 감지
            z_scores = np.abs(zscore(sensor_data))
            anomaly_mask = z_scores > 3
            anomaly_indices = sensor_data.index[anomaly_mask]
            
        elif method == 'iqr':
            # IQR 기반 이상 감지
            Q1 = sensor_data.quantile(0.25)
            Q3 = sensor_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomaly_mask = (sensor_data < lower_bound) | (sensor_data > upper_bound)
            anomaly_indices = sensor_data.index[anomaly_mask]
            
        elif method == 'peaks':
            # 피크 기반 이상 감지
            peaks, _ = find_peaks(sensor_data, height=sensor_data.mean() + 2*sensor_data.std())
            anomaly_indices = sensor_data.index[peaks]
            
        else:
            anomaly_indices = []
        
        anomaly_count = len(anomaly_indices)
        anomaly_rate = anomaly_count / len(sensor_data) * 100
        
        anomaly_report['sensor_anomalies'][sensor] = {
            'count': anomaly_count,
            'rate': anomaly_rate,
            'indices': anomaly_indices.tolist()
        }
        
        anomaly_report['total_anomalies'] += anomaly_count
        
        if anomaly_rate > 5:
            print(f"{sensor}: 이상 비율 높음 ({anomaly_rate:.1f}%)")
    
    return anomaly_report

def analyze_sensor_correlations(df, sensor_groups):
    """센서 그룹별 상관관계 분석"""
    print("센서 그룹별 상관관계 분석")
    
    correlation_report = {
        'group_correlations': {},
        'cross_group_correlations': {},
        'high_correlations': [],
        'low_correlations': []
    }
    
    # 그룹 내 상관관계
    for group_name, sensors in sensor_groups.items():
        valid_sensors = [s for s in sensors if s in df.columns]
        if len(valid_sensors) < 2:
            continue
        
        group_data = df[valid_sensors]
        corr_matrix = group_data.corr()
        
        # 상삼각행렬만 추출
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 통계 계산
        correlations = upper_triangle.stack().dropna()
        
        group_stats = {
            'sensors': valid_sensors,
            'mean_correlation': correlations.mean(),
            'max_correlation': correlations.max(),
            'min_correlation': correlations.min(),
            'std_correlation': correlations.std()
        }
        
        correlation_report['group_correlations'][group_name] = group_stats
        
        # 높은 상관관계 (>0.8) 식별
        high_corr = correlations[correlations > 0.8]
        for (sensor1, sensor2), corr_val in high_corr.items():
            correlation_report['high_correlations'].append({
                'group': group_name,
                'sensor1': sensor1,
                'sensor2': sensor2,
                'correlation': corr_val
            })
        
        # 낮은 상관관계 (<0.1) 식별
        low_corr = correlations[correlations < 0.1]
        for (sensor1, sensor2), corr_val in low_corr.items():
            correlation_report['low_correlations'].append({
                'group': group_name,
                'sensor1': sensor1,
                'sensor2': sensor2,
                'correlation': corr_val
            })
    
    # 그룹 간 상관관계
    group_names = list(sensor_groups.keys())
    for i, group1 in enumerate(group_names):
        for j, group2 in enumerate(group_names[i+1:], i+1):
            sensors1 = [s for s in sensor_groups[group1] if s in df.columns]
            sensors2 = [s for s in sensor_groups[group2] if s in df.columns]
            
            if len(sensors1) > 0 and len(sensors2) > 0:
                # 각 그룹의 대표값 (평균) 계산
                group1_mean = df[sensors1].mean(axis=1)
                group2_mean = df[sensors2].mean(axis=1)
                
                cross_corr = group1_mean.corr(group2_mean)
                
                correlation_report['cross_group_correlations'][f'{group1}_vs_{group2}'] = cross_corr
    
    return correlation_report

def print_classification_metrics(y_true, y_pred, class_names=None, target_names=None):
    """분류 성능 메트릭 출력"""
    try:
        metrics = calculate_all_metrics(y_true, y_pred)
        
        print("분류 성능 메트릭")
        for metric_name, value in metrics.items():
            print(f"{metric_name:20s}: {value:.4f}")
        
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
            print("\n분류 리포트:")
            print(report)
        except Exception as e:
            print(f"분류 리포트 생성 실패: {e}")
        
        return metrics['macro_f1']
    except Exception as e:
        print(f"성능 메트릭 출력 중 오류: {e}")
        return 0.0

def create_cv_folds(X, y, n_splits=5, random_state=42):
    """교차 검증 폴드 생성"""
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(skf.split(X, y))
    except Exception as e:
        print(f"CV 폴드 생성 실패: {e}")
        # 기본 분할 반환
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(X))
        folds = []
        for i in range(n_splits):
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, random_state=random_state + i, stratify=y
            )
            folds.append((train_idx, val_idx))
        return folds

def calculate_class_weights(y, method='balanced'):
    """클래스 가중치 계산"""
    try:
        if method == 'balanced':
            unique_classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
            return dict(zip(unique_classes, class_weights))
        elif method == 'inverse_freq':
            class_counts = Counter(y)
            total_samples = len(y)
            weights = {}
            for class_label, count in class_counts.items():
                weights[class_label] = total_samples / (len(class_counts) * count)
            return weights
        else:
            return None
    except Exception as e:
        print(f"클래스 가중치 계산 실패: {e}")
        return None

def timer(func):
    """함수 실행 시간 측정 데코레이터"""
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
                print(f"{func.__name__} 실행 시간: {elapsed:.2f}초")
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                print(f"{func.__name__} 실행 시간: {minutes}분 {seconds:.2f}초")
            
            if memory_increase > 10:  # 10MB 이상 증가시 출력
                print(f"메모리 증가: {memory_increase:.1f}MB")
            
            return result
            
        except Exception as e:
            print(f"{func.__name__} 실행 중 오류 발생: {e}")
            raise
            
    return wrapper

def check_data_quality(df, feature_columns):
    """데이터 품질 검사"""
    try:
        print("데이터 품질 검사")
        print(f"데이터 형태: {df.shape}")
        
        # 결측치 확인
        missing_info = df[feature_columns].isnull().sum()
        total_missing = missing_info.sum()
        print(f"총 결측치 개수: {total_missing}")
        
        if total_missing > 0:
            print("결측치가 있는 컬럼 (상위 5개):")
            missing_cols = missing_info[missing_info > 0].sort_values(ascending=False)
            for col in missing_cols.head(5).index:
                print(f"  {col}: {missing_info[col]}개 ({missing_info[col]/len(df)*100:.2f}%)")
        
        # 무한값 확인
        numeric_cols = df[feature_columns].select_dtypes(include=[np.number]).columns
        inf_counts = {}
        total_inf = 0
        
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
                total_inf += inf_count
        
        print(f"무한값 개수: {total_inf}")
        
        if total_inf > 0:
            print("무한값이 있는 컬럼 (상위 5개):")
            sorted_inf = sorted(inf_counts.items(), key=lambda x: x[1], reverse=True)
            for col, count in sorted_inf[:5]:
                print(f"  {col}: {count}개 ({count/len(df)*100:.2f}%)")
        
        # 데이터 타입 확인
        print(f"\n데이터 타입:")
        dtype_counts = df[feature_columns].dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count}개 컬럼")
        
        # 기본 통계 (수치형 컬럼만)
        if len(numeric_cols) > 0:
            print(f"\n기본 통계 (수치형 컬럼 {len(numeric_cols)}개):")
            stats = df[numeric_cols].describe()
            print(f"  평균의 범위: {stats.loc['mean'].min():.4f} ~ {stats.loc['mean'].max():.4f}")
            print(f"  표준편차의 범위: {stats.loc['std'].min():.4f} ~ {stats.loc['std'].max():.4f}")
            print(f"  최솟값: {stats.loc['min'].min():.4f}")
            print(f"  최댓값: {stats.loc['max'].max():.4f}")
        
        # 메모리 사용량
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"메모리 사용량: {memory_mb:.2f} MB")
        
        return total_missing == 0 and total_inf == 0
        
    except Exception as e:
        print(f"데이터 품질 검사 중 오류: {e}")
        return False

def memory_usage_check():
    """메모리 사용량 확인"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except Exception:
        return 0

def save_results(results, file_path):
    """결과를 CSV 파일로 저장"""
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
        
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"결과 저장 완료: {file_path}")
    except Exception as e:
        print(f"결과 저장 실패: {e}")

def validate_predictions(y_pred, n_classes, sample_ids=None):
    """예측 결과 검증"""
    try:
        print("예측 결과 검증")
        
        if sample_ids is not None:
            print(f"샘플 개수: {len(sample_ids)}")
            print(f"예측 개수: {len(y_pred)}")
            
            if len(sample_ids) != len(y_pred):
                print("경고: 샘플 개수와 예측 개수가 일치하지 않습니다.")
        
        # 예측값 범위 확인
        min_pred = np.min(y_pred)
        max_pred = np.max(y_pred)
        unique_pred = len(np.unique(y_pred))
        
        print(f"예측값 범위: {min_pred} ~ {max_pred}")
        print(f"고유 예측값 개수: {unique_pred}")
        print(f"총 클래스 개수: {n_classes}")
        
        # 유효성 검사
        is_valid = True
        if min_pred < 0 or max_pred >= n_classes:
            print(f"경고: 예측값이 유효한 범위(0 ~ {n_classes-1})를 벗어났습니다.")
            is_valid = False
        
        # 분포 확인
        unique, counts = np.unique(y_pred, return_counts=True)
        print(f"\n예측 분포 (상위 15개):")
        
        # 빈도순으로 정렬
        sorted_indices = np.argsort(counts)[::-1]
        display_count = min(15, len(unique))
        
        for i in range(display_count):
            idx = sorted_indices[i]
            class_id = unique[idx]
            count = counts[idx]
            percentage = count / len(y_pred) * 100
            print(f"  클래스 {class_id:2d}: {count:4d}개 ({percentage:5.2f}%)")
        
        if len(unique) > 15:
            print(f"  ... (총 {len(unique)}개 클래스)")
        
        # 누락된 클래스 확인
        all_classes = set(range(n_classes))
        predicted_classes = set(unique)
        missing_classes = all_classes - predicted_classes
        
        if missing_classes:
            missing_list = sorted(list(missing_classes))
            if len(missing_list) <= 10:
                print(f"누락된 클래스: {missing_list}")
            else:
                print(f"누락된 클래스: {missing_list[:10]} ... (총 {len(missing_list)}개)")
            is_valid = False
        
        return is_valid
        
    except Exception as e:
        print(f"예측 결과 검증 중 오류: {e}")
        return False

def create_submission_template(test_ids, predictions, id_col='ID', target_col='target'):
    """제출 파일 템플릿 생성"""
    try:
        submission = pd.DataFrame({
            id_col: test_ids,
            target_col: predictions
        })
        
        # 데이터 타입 최적화
        submission[target_col] = submission[target_col].astype('int16')
        
        return submission
    except Exception as e:
        print(f"제출 파일 템플릿 생성 실패: {e}")
        return None

def analyze_class_distribution(y, class_names=None):
    """클래스 분포 분석"""
    try:
        print("클래스 분포 분석")
        
        unique, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        print(f"총 샘플 수: {total_samples}")
        print(f"클래스 개수: {len(unique)}")
        
        # 분포 통계
        print(f"\n클래스별 분포:")
        
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
            
            if class_id < 10:  # 상위 10개만 출력
                print(f"  {class_name:>12}: {count:5d}개 ({percentage:5.2f}%)")
        
        # 불균형 정도 계산
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\n분포 통계:")
        print(f"  최대 클래스 크기: {max_count}")
        print(f"  최소 클래스 크기: {min_count}")
        print(f"  불균형 비율: {imbalance_ratio:.2f}:1")
        print(f"  표준편차: {np.std(counts):.2f}")
        
        return distribution_data
        
    except Exception as e:
        print(f"클래스 분포 분석 중 오류: {e}")
        return []

def garbage_collect():
    """가비지 컬렉션 수행"""
    try:
        collected = gc.collect()
        if collected > 0:
            print(f"메모리 정리: {collected}개 객체 해제")
        return collected
    except Exception as e:
        print(f"가비지 컬렉션 중 오류: {e}")
        return 0

def format_time(seconds):
    """초를 읽기 쉬운 형식으로 변환"""
    try:
        if seconds < 60:
            return f"{seconds:.2f}초"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}분 {remaining_seconds:.2f}초"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}시간 {remaining_minutes}분 {remaining_seconds:.2f}초"
    except Exception:
        return f"{seconds}초"

def safe_divide(numerator, denominator, default=0.0):
    """안전한 나눗셈"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError, ValueError):
        return default

def check_system_resources():
    """시스템 리소스 확인"""
    try:
        # CPU 정보
        cpu_count = os.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        memory_usage_percent = memory.percent
        
        # 디스크 정보
        disk = psutil.disk_usage('.')
        disk_total_gb = disk.total / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        
        print(f"시스템 리소스 상태:")
        print(f"  CPU: {cpu_count}코어, 사용률 {cpu_usage}%")
        print(f"  메모리: {memory_available_gb:.1f}GB 사용가능 / {memory_total_gb:.1f}GB 전체 ({memory_usage_percent:.1f}% 사용중)")
        print(f"  디스크: {disk_free_gb:.1f}GB 여유공간 / {disk_total_gb:.1f}GB 전체")
        
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
        print(f"시스템 리소스 확인 실패: {e}")
        return None

def optimize_dataframe_memory(df):
    """DataFrame 메모리 사용량 최적화"""
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
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        memory_reduction = (initial_memory - final_memory) / initial_memory * 100
        
        print(f"메모리 사용량 최적화: {initial_memory:.1f}MB -> {final_memory:.1f}MB ({memory_reduction:.1f}% 감소)")
        
        return df
        
    except Exception as e:
        print(f"메모리 최적화 실패: {e}")
        return df

def generate_system_report(validation_scores, cv_scores, memory_info, execution_time):
    """시스템 성능 보고서 생성"""
    print("\n" + "=" * 60)
    print("시스템 성능 보고서")
    print("=" * 60)
    
    # 성능 점수 분석
    if validation_scores:
        print("\n검증 점수 분석:")
        for val_type, score in validation_scores.items():
            print(f"  {val_type:15s}: {score:.4f}")
        
        scores = list(validation_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  평균 점수       : {mean_score:.4f}")
        print(f"  표준편차        : {std_score:.4f}")
        print(f"  안정성 지수     : {1-std_score:.4f}")
    
    # 교차 검증 결과
    if cv_scores:
        print("\n교차 검증 결과:")
        for model_name, scores in cv_scores.items():
            stability_score = scores.get('stability_score', scores['mean'])
            print(f"  {model_name:15s}: {stability_score:.4f}")
    
    # 메모리 사용량
    if memory_info:
        print("\n메모리 사용량:")
        print(f"  초기 메모리     : {memory_info['initial']:.2f} MB")
        print(f"  최종 메모리     : {memory_info['final']:.2f} MB")
        print(f"  증가량          : {memory_info['increase']:.2f} MB")
    
    # 실행 시간
    if execution_time:
        print(f"\n총 실행 시간     : {format_time(execution_time)}")
    
    # 시스템 리소스
    resource_info = check_system_resources()
    
    print("=" * 60)