# utils.py

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time
import logging
import warnings
import psutil
from pathlib import Path

warnings.filterwarnings('ignore')

def setup_logging(log_file=None, level='INFO'):
    """로깅 설정"""
    if log_file is None:
        log_file = Path('logs') / 'training.log'
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
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
            pickle.dump(model, f)
        
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

def save_joblib(obj, file_path):
    """joblib을 사용한 객체 저장"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)
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
    return f1_score(y_true, y_pred, average='macro')

def calculate_all_metrics(y_true, y_pred):
    """모든 평가 메트릭 계산"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    return metrics

def print_classification_metrics(y_true, y_pred, class_names=None):
    """분류 성능 메트릭 출력"""
    metrics = calculate_all_metrics(y_true, y_pred)
    
    print("=== 분류 성능 메트릭 ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    if class_names is None:
        class_names = sorted(list(set(y_true)))
    
    report = classification_report(
        y_true, y_pred, 
        target_names=[str(c) for c in class_names],
        zero_division=0
    )
    print("\n분류 리포트:")
    print(report)
    
    return metrics['macro_f1']

def print_confusion_matrix(y_true, y_pred, class_names=None):
    """혼동 행렬 출력"""
    if class_names is None:
        class_names = sorted(list(set(y_true)))
    
    cm = confusion_matrix(y_true, y_pred)
    print("혼동 행렬:")
    
    # 헤더 출력
    header = "실제\\예측 " + " ".join([f"{str(c):>6}" for c in class_names])
    print(header)
    
    # 각 행 출력
    for i, true_class in enumerate(class_names):
        row = f"{str(true_class):>8} " + " ".join([f"{cm[i][j]:>6}" for j in range(len(class_names))])
        print(row)

def create_cv_folds(X, y, n_splits=5, random_state=42):
    """교차 검증 폴드 생성"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))

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
    print("=== 데이터 품질 검사 ===")
    print(f"데이터 형태: {df.shape}")
    
    # 결측치 확인
    missing_info = df[feature_columns].isnull().sum()
    total_missing = missing_info.sum()
    print(f"총 결측치 개수: {total_missing}")
    
    if total_missing > 0:
        print("결측치가 있는 컬럼:")
        for col in feature_columns:
            if missing_info[col] > 0:
                print(f"  {col}: {missing_info[col]}개")
    
    # 무한값 확인
    inf_info = np.isinf(df[feature_columns].select_dtypes(include=[np.number])).sum()
    total_inf = inf_info.sum()
    print(f"무한값 개수: {total_inf}")
    
    if total_inf > 0:
        print("무한값이 있는 컬럼:")
        for col in feature_columns:
            if col in inf_info.index and inf_info[col] > 0:
                print(f"  {col}: {inf_info[col]}개")
    
    # 데이터 타입 확인
    print("\n데이터 타입:")
    print(df[feature_columns].dtypes.value_counts())
    
    # 기본 통계
    print("\n기본 통계:")
    print(df[feature_columns].describe())
    
    return total_missing == 0 and total_inf == 0

def memory_usage_check():
    """메모리 사용량 확인"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"현재 메모리 사용량: {memory_mb:.2f} MB")
        return memory_mb
    except Exception:
        print("메모리 사용량을 확인할 수 없습니다.")
        return 0

def optimize_memory_usage(df):
    """메모리 사용량 최적화"""
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # 수치형 컬럼 최적화
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # 문자열 컬럼 최적화
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    savings = (initial_memory - final_memory) / initial_memory * 100
    
    print(f"메모리 최적화: {initial_memory:.2f}MB -> {final_memory:.2f}MB "
          f"({savings:.1f}% 절약)")
    
    return df

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
        
        df.to_csv(file_path, index=False)
        print(f"결과 저장 완료: {file_path}")
    except Exception as e:
        print(f"결과 저장 실패: {e}")

def validate_predictions(y_pred, n_classes, sample_ids=None):
    """예측 결과 검증"""
    print("=== 예측 결과 검증 ===")
    
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
    
    # 유효성 검사
    if min_pred < 0 or max_pred >= n_classes:
        print(f"경고: 예측값이 유효한 범위(0 ~ {n_classes-1})를 벗어났습니다.")
        return False
    
    # 분포 확인
    unique, counts = np.unique(y_pred, return_counts=True)
    print("\n예측 분포:")
    for class_id, count in zip(unique, counts):
        percentage = count / len(y_pred) * 100
        print(f"클래스 {class_id}: {count}개 ({percentage:.2f}%)")
    
    return True

def create_submission_template(test_ids, predictions, id_col='ID', target_col='target'):
    """제출 파일 템플릿 생성"""
    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: predictions
    })
    
    return submission