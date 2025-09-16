# utils.py

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
import time
import logging

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_data(file_path):
    """데이터 로드"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"데이터 로드 실패: {e}")

def save_model(model, file_path):
    """모델 저장"""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        raise Exception(f"모델 저장 실패: {e}")

def load_model(file_path):
    """모델 로드"""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise Exception(f"모델 로드 실패: {e}")

def calculate_macro_f1(y_true, y_pred):
    """Macro F1 스코어 계산"""
    return f1_score(y_true, y_pred, average='macro')

def print_classification_metrics(y_true, y_pred, class_names=None):
    """분류 성능 메트릭 출력"""
    macro_f1 = calculate_macro_f1(y_true, y_pred)
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    if class_names is None:
        class_names = sorted(list(set(y_true)))
    
    report = classification_report(y_true, y_pred, target_names=[str(c) for c in class_names])
    print("\n분류 리포트:")
    print(report)
    
    return macro_f1

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
        print(f"{func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result
    return wrapper

def check_data_quality(df, feature_columns):
    """데이터 품질 검사"""
    print("=== 데이터 품질 검사 ===")
    print(f"데이터 형태: {df.shape}")
    
    # 결측치 확인
    missing_count = df[feature_columns].isnull().sum().sum()
    print(f"총 결측치 개수: {missing_count}")
    
    # 무한값 확인
    inf_count = np.isinf(df[feature_columns]).sum().sum()
    print(f"무한값 개수: {inf_count}")
    
    # 데이터 타입 확인
    print("\n데이터 타입:")
    print(df[feature_columns].dtypes.value_counts())
    
    # 기본 통계
    print("\n기본 통계:")
    print(df[feature_columns].describe())
    
    return missing_count == 0 and inf_count == 0

def memory_usage_check():
    """메모리 사용량 확인"""
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"현재 메모리 사용량: {memory_mb:.2f} MB")
    return memory_mb

def optimize_memory_usage(df):
    """메모리 사용량 최적화"""
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # 수치형 컬럼 최적화
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"메모리 최적화: {initial_memory:.2f}MB -> {final_memory:.2f}MB "
          f"({(initial_memory-final_memory)/initial_memory*100:.1f}% 절약)")
    
    return df