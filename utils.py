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

def load_data(file_path):
    """데이터 로드"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
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
        
        with open(file_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"모델 저장 완료: {file_path}")
    except Exception as e:
        raise Exception(f"모델 저장 실패: {e}")

def load_model(file_path):
    """모델 로드"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {file_path}")
        
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"모델 로드 완료: {file_path}")
        return model
    except Exception as e:
        raise Exception(f"모델 로드 실패: {e}")

def save_joblib(obj, file_path, compress=3):
    """joblib을 사용한 객체 저장"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path, compress=compress)
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
    
    return metrics

def calculate_class_metrics(y_true, y_pred, labels=None):
    """클래스별 메트릭 계산"""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    class_metrics = []
    for i, label in enumerate(labels):
        class_metrics.append({
            'class': label,
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        })
    
    return class_metrics

def print_classification_metrics(y_true, y_pred, class_names=None, target_names=None):
    """분류 성능 메트릭 출력"""
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

def create_cv_folds(X, y, n_splits=5, random_state=42):
    """교차 검증 폴드 생성"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))

def calculate_class_weights(y, method='balanced'):
    """클래스 가중치 계산"""
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

def timer(func):
    """함수 실행 시간 측정 데코레이터"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        
        if elapsed < 60:
            print(f"{func.__name__} 실행 시간: {elapsed:.2f}초")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"{func.__name__} 실행 시간: {minutes}분 {seconds:.2f}초")
        
        return result
    return wrapper

def check_data_quality(df, feature_columns):
    """데이터 품질 검사"""
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

def create_submission_template(test_ids, predictions, id_col='ID', target_col='target'):
    """제출 파일 템플릿 생성"""
    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: predictions
    })
    
    # 데이터 타입 최적화
    submission[target_col] = submission[target_col].astype('int32')
    
    return submission

def analyze_class_distribution(y, class_names=None):
    """클래스 분포 분석"""
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

def garbage_collect():
    """가비지 컬렉션 수행"""
    collected = gc.collect()
    if collected > 0:
        print(f"메모리 정리: {collected}개 객체 해제")
    return collected

def format_time(seconds):
    """초를 읽기 쉬운 형식으로 변환"""
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

def safe_divide(numerator, denominator, default=0.0):
    """안전한 나눗셈"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError, ValueError):
        return default