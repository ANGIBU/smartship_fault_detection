# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, welch
from scipy import fft
import gc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, check_data_quality, save_joblib

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.imputer = None
        self.sampler = None
        self.feature_columns = Config.FEATURE_COLUMNS
        self.selected_features = None
        self.train_stats = {}
        self.sensor_groups = Config.SENSOR_GROUPS
        self.generated_feature_names = []
        self.time_series_features = []
        self.domain_features = []
        
    @timer
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print("데이터 로드 시작")
        
        dtypes = {col: 'float32' for col in self.feature_columns}
        
        train_chunks = []
        test_chunks = []
        
        try:
            for chunk in pd.read_csv(Config.TRAIN_FILE, dtype=dtypes, chunksize=Config.CHUNK_SIZE):
                if Config.TARGET_COLUMN in chunk.columns:
                    chunk[Config.TARGET_COLUMN] = chunk[Config.TARGET_COLUMN].astype('int16')
                train_chunks.append(chunk)
            
            train_df = pd.concat(train_chunks, ignore_index=True)
            del train_chunks
            gc.collect()
            
            for chunk in pd.read_csv(Config.TEST_FILE, dtype=dtypes, chunksize=Config.CHUNK_SIZE):
                test_chunks.append(chunk)
            
            test_df = pd.concat(test_chunks, ignore_index=True)
            del test_chunks
            gc.collect()
            
        except Exception as e:
            print(f"청크 로드 실패, 일반 로드 시도: {e}")
            train_df = pd.read_csv(Config.TRAIN_FILE)
            test_df = pd.read_csv(Config.TEST_FILE)
            
            for col in self.feature_columns:
                if col in train_df.columns:
                    train_df[col] = train_df[col].astype('float32')
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype('float32')
            
            if Config.TARGET_COLUMN in train_df.columns:
                train_df[Config.TARGET_COLUMN] = train_df[Config.TARGET_COLUMN].astype('int16')
        
        print(f"Train 데이터 형태: {train_df.shape}")
        print(f"Test 데이터 형태: {test_df.shape}")
        
        train_quality = check_data_quality(train_df, self.feature_columns)
        test_quality = check_data_quality(test_df, self.feature_columns)
        
        if not (train_quality and test_quality):
            print("데이터 품질 문제 처리 중")
            train_df, test_df = self._handle_data_issues(train_df, test_df)
        
        return train_df, test_df
    
    def _handle_data_issues(self, train_df, test_df):
        """데이터 이슈 처리"""
        for col in self.feature_columns:
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
            
            if train_df[col].isnull().sum() > 0:
                fill_val = train_df[col].median()
                if pd.isna(fill_val):
                    fill_val = 0.0
                    
                train_df[col].fillna(fill_val, inplace=True)
                test_df[col].fillna(fill_val, inplace=True)
        
        return train_df, test_df
    
    def _create_time_series_features(self, X_train, X_test):
        """시계열 피처 생성"""
        print("시계열 피처 생성 중")
        
        # 가정: 데이터가 시간 순서로 정렬되어 있음
        for df_name, df in [('train', X_train), ('test', X_test)]:
            # 차분 피처
            for col in self.feature_columns[:20]:  # 주요 센서만
                if col in df.columns:
                    for lag in Config.LAG_FEATURES:
                        if len(df) > lag:
                            diff_col = f'{col}_diff_{lag}'
                            df[diff_col] = df[col].diff(lag).fillna(0).astype('float32')
                            self.time_series_features.append(diff_col)
            
            # 롤링 통계
            for window in Config.ROLLING_STATS_WINDOWS:
                if len(df) > window:
                    # 전체 센서 평균의 롤링 통계
                    sensor_mean = df[self.feature_columns].mean(axis=1)
                    
                    df[f'rolling_mean_{window}'] = sensor_mean.rolling(window=window, min_periods=1).mean().astype('float32')
                    df[f'rolling_std_{window}'] = sensor_mean.rolling(window=window, min_periods=1).std().fillna(0).astype('float32')
                    df[f'rolling_skew_{window}'] = sensor_mean.rolling(window=window, min_periods=3).skew().fillna(0).astype('float32')
                    
                    self.time_series_features.extend([
                        f'rolling_mean_{window}', 
                        f'rolling_std_{window}', 
                        f'rolling_skew_{window}'
                    ])
            
            # 변화율 피처
            sensor_data = df[self.feature_columns].values
            if len(sensor_data) > 1:
                change_rate = np.abs(np.diff(sensor_data, axis=0, prepend=sensor_data[[0]]))
                df['sensor_change_rate_mean'] = np.mean(change_rate, axis=1).astype('float32')
                df['sensor_change_rate_max'] = np.max(change_rate, axis=1).astype('float32')
                df['sensor_change_acceleration'] = np.abs(np.diff(change_rate.mean(axis=1), prepend=0)).astype('float32')
                
                self.time_series_features.extend([
                    'sensor_change_rate_mean', 
                    'sensor_change_rate_max', 
                    'sensor_change_acceleration'
                ])
        
        return X_train, X_test
    
    def _create_domain_features(self, X_train, X_test):
        """도메인 지식 기반 피처 생성"""
        print("도메인 지식 피처 생성 중")
        
        for df in [X_train, X_test]:
            # 물리법칙 기반 피처
            vibration_sensors = [s for s in Config.SENSOR_GROUPS['vibration'] if s in df.columns]
            power_sensors = [s for s in Config.SENSOR_GROUPS['power'] if s in df.columns]
            temperature_sensors = [s for s in Config.SENSOR_GROUPS['temperature'] if s in df.columns]
            
            if len(vibration_sensors) >= 2:
                # 진동 에너지
                vibration_data = df[vibration_sensors].values
                df['vibration_energy'] = np.sum(vibration_data**2, axis=1).astype('float32')
                
                # 진동 불균형
                df['vibration_imbalance'] = np.std(vibration_data, axis=1).astype('float32')
                self.domain_features.extend(['vibration_energy', 'vibration_imbalance'])
            
            if len(power_sensors) >= 2:
                # 전력 효율
                power_data = df[power_sensors].values
                power_total = np.sum(power_data, axis=1)
                power_efficiency = power_total / (np.max(power_data, axis=1) + 1e-10)
                df['power_efficiency'] = power_efficiency.astype('float32')
                self.domain_features.append('power_efficiency')
            
            if len(temperature_sensors) >= 2:
                # 온도 구배
                temp_data = df[temperature_sensors].values
                df['temperature_gradient'] = (np.max(temp_data, axis=1) - np.min(temp_data, axis=1)).astype('float32')
                self.domain_features.append('temperature_gradient')
            
            # 임계값 기반 피처
            for group_name, sensors in Config.SENSOR_GROUPS.items():
                valid_sensors = [s for s in sensors if s in df.columns]
                thresholds = Config.get_domain_thresholds(group_name)
                
                if valid_sensors and thresholds:
                    group_data = df[valid_sensors].values
                    group_mean = np.mean(group_data, axis=1)
                    
                    # 임계값 초과 여부
                    if 'high' in thresholds:
                        high_threshold = thresholds['high']
                        df[f'{group_name}_above_high'] = (group_mean > high_threshold).astype('float32')
                        self.domain_features.append(f'{group_name}_above_high')
                    
                    if 'medium' in thresholds:
                        medium_threshold = thresholds['medium']
                        df[f'{group_name}_above_medium'] = (group_mean > medium_threshold).astype('float32')
                        self.domain_features.append(f'{group_name}_above_medium')
            
            # 센서 간 물리적 관계
            # 진동-전력 관계
            if len(vibration_sensors) > 0 and len(power_sensors) > 0:
                vibration_avg = df[vibration_sensors].mean(axis=1)
                power_avg = df[power_sensors].mean(axis=1)
                
                # 진동 대비 전력 비율
                vibration_power_ratio = vibration_avg / (power_avg + 1e-10)
                df['vibration_power_ratio'] = vibration_power_ratio.astype('float32')
                self.domain_features.append('vibration_power_ratio')
            
            # 온도-전력 관계
            if len(temperature_sensors) > 0 and len(power_sensors) > 0:
                temp_avg = df[temperature_sensors].mean(axis=1)
                power_avg = df[power_sensors].mean(axis=1)
                
                # 온도 효율성
                temp_efficiency = power_avg / (temp_avg + 1e-10)
                df['temperature_efficiency'] = temp_efficiency.astype('float32')
                self.domain_features.append('temperature_efficiency')
        
        return X_train, X_test
    
    def _create_signal_features(self, X_train, X_test):
        """신호 처리 피처 생성"""
        print("신호 처리 피처 생성 중")
        
        for group_name, sensors in self.sensor_groups.items():
            valid_sensors = [s for s in sensors if s in self.feature_columns]
            if len(valid_sensors) >= 2:
                for df in [X_train, X_test]:
                    group_data = df[valid_sensors].values.astype('float32')
                    
                    # FFT 기반 주파수 도메인 특성
                    fft_features = []
                    for i in range(group_data.shape[0]):
                        signal = group_data[i]
                        
                        fft_vals = np.abs(fft.fft(signal))
                        
                        dominant_freq_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
                        dominant_freq_power = fft_vals[dominant_freq_idx]
                        
                        low_freq_energy = np.sum(fft_vals[1:len(fft_vals)//4])
                        mid_freq_energy = np.sum(fft_vals[len(fft_vals)//4:len(fft_vals)//2])
                        high_freq_energy = np.sum(fft_vals[len(fft_vals)//2:])
                        
                        # 주파수 중심
                        freq_centroid = np.sum(np.arange(len(fft_vals)) * fft_vals) / (np.sum(fft_vals) + 1e-10)
                        
                        fft_features.append([
                            dominant_freq_power, 
                            low_freq_energy, 
                            mid_freq_energy, 
                            high_freq_energy, 
                            freq_centroid
                        ])
                    
                    fft_features = np.array(fft_features, dtype='float32')
                    
                    df[f'{group_name}_dominant_freq'] = fft_features[:, 0]
                    df[f'{group_name}_low_freq_energy'] = fft_features[:, 1]
                    df[f'{group_name}_mid_freq_energy'] = fft_features[:, 2]
                    df[f'{group_name}_high_freq_energy'] = fft_features[:, 3]
                    df[f'{group_name}_freq_centroid'] = fft_features[:, 4]
                    
                    # 시간 도메인 특성
                    df[f'{group_name}_rms'] = np.sqrt(np.mean(group_data**2, axis=1)).astype('float32')
                    df[f'{group_name}_peak_to_peak'] = (np.max(group_data, axis=1) - np.min(group_data, axis=1)).astype('float32')
                    df[f'{group_name}_crest_factor'] = (np.max(np.abs(group_data), axis=1) / np.sqrt(np.mean(group_data**2, axis=1))).astype('float32')
                    df[f'{group_name}_form_factor'] = (np.sqrt(np.mean(group_data**2, axis=1)) / np.mean(np.abs(group_data), axis=1)).astype('float32')
                    
                    # 스펙트럼 특성
                    df[f'{group_name}_spectral_entropy'] = np.array([
                        -np.sum((fft_vals[1:] + 1e-10) * np.log(fft_vals[1:] + 1e-10)) 
                        for fft_vals in [np.abs(fft.fft(row)) for row in group_data]
                    ], dtype='float32')
                    
                    # 상관관계 기반 특성
                    correlation_matrix = np.corrcoef(group_data.T)
                    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
                    
                    upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
                    avg_correlation = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
                    df[f'{group_name}_avg_correlation'] = avg_correlation
        
        return X_train, X_test
    
    def _create_statistical_features(self, X_train, X_test):
        """통계적 피처 생성"""
        print("통계적 피처 생성 중")
        
        batch_size = 2000
        
        for df_name, df in [('train', X_train), ('test', X_test)]:
            n_samples = len(df)
            
            global_features = {
                'sensor_mean': np.zeros(n_samples, dtype='float32'),
                'sensor_std': np.zeros(n_samples, dtype='float32'),
                'sensor_median': np.zeros(n_samples, dtype='float32'),
                'sensor_mad': np.zeros(n_samples, dtype='float32'),
                'sensor_iqr': np.zeros(n_samples, dtype='float32'),
                'sensor_skew': np.zeros(n_samples, dtype='float32'),
                'sensor_kurtosis': np.zeros(n_samples, dtype='float32'),
                'sensor_entropy': np.zeros(n_samples, dtype='float32'),
                'sensor_energy': np.zeros(n_samples, dtype='float32'),
                'sensor_stability': np.zeros(n_samples, dtype='float32'),
                'sensor_complexity': np.zeros(n_samples, dtype='float32')
            }
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_data = df[self.feature_columns].iloc[i:end_idx].values
                
                global_features['sensor_mean'][i:end_idx] = np.mean(batch_data, axis=1)
                global_features['sensor_std'][i:end_idx] = np.std(batch_data, axis=1)
                global_features['sensor_median'][i:end_idx] = np.median(batch_data, axis=1)
                
                medians = np.median(batch_data, axis=1, keepdims=True)
                global_features['sensor_mad'][i:end_idx] = np.median(np.abs(batch_data - medians), axis=1)
                
                q75 = np.percentile(batch_data, 75, axis=1)
                q25 = np.percentile(batch_data, 25, axis=1)
                global_features['sensor_iqr'][i:end_idx] = q75 - q25
                
                global_features['sensor_skew'][i:end_idx] = skew(batch_data, axis=1, nan_policy='omit')
                global_features['sensor_kurtosis'][i:end_idx] = kurtosis(batch_data, axis=1, nan_policy='omit')
                
                # 엔트로피
                entropies = []
                for row in batch_data:
                    hist, _ = np.histogram(row, bins=10, density=True)
                    hist = hist + 1e-10
                    entropy = -np.sum(hist * np.log2(hist))
                    entropies.append(entropy)
                global_features['sensor_entropy'][i:end_idx] = entropies
                
                # 에너지
                global_features['sensor_energy'][i:end_idx] = np.sum(batch_data**2, axis=1)
                
                # 안정성
                stability = np.sum((batch_data - np.mean(batch_data, axis=1, keepdims=True))**2, axis=1)
                global_features['sensor_stability'][i:end_idx] = stability
                
                # 복잡도
                complexity = np.sum(np.abs(np.diff(batch_data, axis=1)), axis=1)
                global_features['sensor_complexity'][i:end_idx] = complexity
            
            for feature_name, values in global_features.items():
                df[feature_name] = values
        
        return X_train, X_test
    
    def _create_interaction_features(self, X_train, X_test):
        """상호작용 피처 생성"""
        print("상호작용 피처 생성 중")
        
        important_groups = ['vibration', 'temperature', 'pressure', 'power']
        
        for i, group1 in enumerate(important_groups):
            for j, group2 in enumerate(important_groups[i+1:], i+1):
                sensors1 = [s for s in self.sensor_groups[group1] if s in self.feature_columns]
                sensors2 = [s for s in self.sensor_groups[group2] if s in self.feature_columns]
                
                if len(sensors1) >= 1 and len(sensors2) >= 1:
                    for df in [X_train, X_test]:
                        group1_repr = df[sensors1].mean(axis=1)
                        group2_repr = df[sensors2].mean(axis=1)
                        
                        # 비율
                        with np.errstate(divide='ignore', invalid='ignore'):
                            ratio = group1_repr / (group2_repr + 1e-10)
                            ratio = np.nan_to_num(ratio, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        df[f'{group1}_{group2}_ratio'] = ratio.astype('float32')
                        
                        # 곱
                        df[f'{group1}_{group2}_product'] = (group1_repr * group2_repr).astype('float32')
                        
                        # 차이
                        df[f'{group1}_{group2}_diff'] = (group1_repr - group2_repr).astype('float32')
                        
                        # 정규화된 차이
                        normalized_diff = np.abs(group1_repr - group2_repr) / (group1_repr + group2_repr + 1e-10)
                        df[f'{group1}_{group2}_normalized_diff'] = normalized_diff.astype('float32')
        
        return X_train, X_test
    
    def _create_sensor_health_features(self, X_train, X_test):
        """센서 상태 피처 생성"""
        print("센서 상태 피처 생성 중")
        
        for df in [X_train, X_test]:
            sensor_data = df[self.feature_columns].values
            
            # 변화율
            diff_features = np.diff(sensor_data, axis=1, prepend=sensor_data[:, [0]])
            df['sensor_change_mean'] = np.mean(np.abs(diff_features), axis=1).astype('float32')
            df['sensor_change_std'] = np.std(diff_features, axis=1).astype('float32')
            df['sensor_change_max'] = np.max(np.abs(diff_features), axis=1).astype('float32')
            
            # 이상치 개수
            q75 = np.percentile(sensor_data, 75, axis=1, keepdims=True)
            q25 = np.percentile(sensor_data, 25, axis=1, keepdims=True)
            iqr = q75 - q25
            
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outliers = ((sensor_data < lower_bound) | (sensor_data > upper_bound)).sum(axis=1)
            df['sensor_outlier_count'] = outliers.astype('float32')
            df['sensor_outlier_ratio'] = (outliers / len(self.feature_columns)).astype('float32')
            
            # 센서 범위 활용도
            sensor_ranges = np.max(sensor_data, axis=1) - np.min(sensor_data, axis=1)
            df['sensor_range_utilization'] = sensor_ranges.astype('float32')
            
            # 0에 가까운 센서 개수
            near_zero = (np.abs(sensor_data) < 0.01).sum(axis=1)
            df['sensor_near_zero_count'] = near_zero.astype('float32')
            
            # 센서 동기화 정도
            correlation_scores = []
            for i in range(sensor_data.shape[0]):
                row = sensor_data[i]
                if np.std(row) > 0:
                    autocorr = np.corrcoef(row[:-1], row[1:])[0, 1]
                    correlation_scores.append(autocorr if not np.isnan(autocorr) else 0.0)
                else:
                    correlation_scores.append(0.0)
            
            df['sensor_synchronization'] = np.array(correlation_scores, dtype='float32')
        
        return X_train, X_test
    
    @timer
    def feature_engineering(self, train_df, test_df):
        """피처 엔지니어링"""
        print("피처 엔지니어링 시작")
        
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        y_train = train_df[Config.TARGET_COLUMN].copy()
        
        del train_df, test_df
        gc.collect()
        
        # 시계열 피처 생성
        if Config.TIME_SERIES_FEATURES:
            X_train, X_test = self._create_time_series_features(X_train, X_test)
        
        # 도메인 지식 기반 피처 생성
        if Config.DOMAIN_FEATURES:
            X_train, X_test = self._create_domain_features(X_train, X_test)
        
        # 신호 처리 피처 생성
        if Config.SIGNAL_FEATURES:
            X_train, X_test = self._create_signal_features(X_train, X_test)
        
        # 통계적 피처 생성
        if Config.STATISTICAL_FEATURES:
            X_train, X_test = self._create_statistical_features(X_train, X_test)
        
        # 상호작용 피처 생성
        if Config.INTERACTION_FEATURES:
            X_train, X_test = self._create_interaction_features(X_train, X_test)
        
        # 센서 상태 피처 생성
        X_train, X_test = self._create_sensor_health_features(X_train, X_test)
        
        # 최종 데이터 정리
        X_train, X_test = self._final_cleanup(X_train, X_test)
        
        return X_train, X_test, y_train
    
    def _final_cleanup(self, X_train, X_test):
        """최종 데이터 정리"""
        print("데이터 정리 중")
        
        for df in [X_train, X_test]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            for col in numeric_cols:
                if df[col].isna().any():
                    fill_value = df[col].median()
                    if pd.isna(fill_value):
                        fill_value = 0.0
                    df[col].fillna(fill_value, inplace=True)
            
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype('float32')
        
        print(f"생성된 피처 수: {X_train.shape[1]}")
        
        self.generated_feature_names = X_train.columns.tolist()
        
        gc.collect()
        
        return X_train, X_test
    
    @timer
    def balance_classes(self, X_train, y_train):
        """클래스 불균형 해결"""
        print("클래스 불균형 처리 시작")
        
        # 클래스 분포 확인
        class_counts = y_train.value_counts().sort_index()
        print("클래스별 샘플 수:")
        for class_id, count in class_counts.items():
            print(f"  클래스 {class_id}: {count}개")
        
        # 불균형 정도 계산
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"불균형 비율: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > (1 / Config.IMBALANCE_THRESHOLD):
            print("SMOTE를 사용한 오버샘플링 적용")
            
            try:
                # SMOTE 적용
                smote = SMOTE(
                    sampling_strategy='auto',
                    random_state=Config.RANDOM_STATE,
                    k_neighbors=3
                )
                
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                
                print(f"리샘플링 전: {X_train.shape}")
                print(f"리샘플링 후: {X_resampled.shape}")
                
                # 리샘플링 후 분포 확인
                new_class_counts = pd.Series(y_resampled).value_counts().sort_index()
                print("리샘플링 후 클래스별 샘플 수:")
                for class_id, count in new_class_counts.items():
                    print(f"  클래스 {class_id}: {count}개")
                
                self.sampler = smote
                return X_resampled, y_resampled
                
            except Exception as e:
                print(f"SMOTE 적용 실패: {e}")
                print("원본 데이터 사용")
                return X_train, y_train
        else:
            print("클래스 균형이 적절하여 리샘플링 생략")
            return X_train, y_train
    
    @timer
    def scale_features(self, X_train, X_test, method='robust'):
        """피처 스케일링"""
        print(f"피처 스케일링 시작 ({method})")
        
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'quantile':
            self.scaler = QuantileTransformer(
                output_distribution='normal', 
                random_state=Config.RANDOM_STATE,
                n_quantiles=min(2000, X_train.shape[0])
            )
        else:
            self.scaler = RobustScaler()
        
        if len(X_train) > 15000:
            sample_size = min(15000, len(X_train))
            sample_idx = np.random.RandomState(Config.RANDOM_STATE).choice(
                len(X_train), sample_size, replace=False
            )
            X_sample = X_train.iloc[sample_idx]
            self.scaler.fit(X_sample)
            del X_sample
        else:
            self.scaler.fit(X_train)
        
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns, 
            index=X_train.index,
            dtype='float32'
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=X_test.columns, 
            index=X_test.index,
            dtype='float32'
        )
        
        save_joblib(self.scaler, Config.SCALER_FILE)
        
        gc.collect()
        
        return X_train_scaled, X_test_scaled
    
    @timer
    def select_features(self, X_train, X_test, y_train, method='domain_based', k=None):
        """피처 선택"""
        if k is None:
            k = min(Config.FEATURE_SELECTION_K, X_train.shape[1])
        
        k = min(k, X_train.shape[1])
        
        print(f"피처 선택 시작 ({method}, k={k})")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        if method == 'domain_based':
            print("도메인 기반 피처 선택 적용")
            
            # 1. 원본 센서 피처 우선
            original_features = [col for col in self.feature_columns if col in X_train.columns]
            
            # 2. 시계열 피처 추가
            time_features = [col for col in self.time_series_features if col in X_train.columns]
            
            # 3. 도메인 피처 추가
            domain_features = [col for col in self.domain_features if col in X_train.columns]
            
            # 4. 신호 처리 피처 추가
            signal_features = [col for col in X_train.columns if any(x in col for x in 
                              ['rms', 'peak_to_peak', 'crest_factor', 'dominant_freq', 'energy', 'spectral'])]
            
            # 5. 통계적 피처 추가
            stat_features = [col for col in X_train.columns if any(x in col for x in 
                           ['sensor_mean', 'sensor_std', 'sensor_mad', 'sensor_entropy', 'sensor_stability'])]
            
            # 우선순위별 피처 조합
            priority_features = (original_features + domain_features + time_features + 
                               signal_features + stat_features)
            
            # 중복 제거
            priority_features = list(dict.fromkeys(priority_features))
            
            if len(priority_features) > k:
                from sklearn.feature_selection import SelectKBest, f_classif
                
                selector = SelectKBest(f_classif, k=k)
                
                if len(X_train) > 8000:
                    sample_idx = np.random.RandomState(Config.RANDOM_STATE).choice(
                        len(X_train), 8000, replace=False
                    )
                    X_sample = X_train[priority_features].iloc[sample_idx]
                    y_sample = y_train.iloc[sample_idx]
                    selector.fit(X_sample, y_sample)
                    del X_sample, y_sample
                else:
                    selector.fit(X_train[priority_features], y_train)
                
                selected_mask = selector.get_support()
                self.selected_features = [priority_features[i] for i, selected in enumerate(selected_mask) if selected]
            else:
                self.selected_features = priority_features
            
        else:
            # 기존 보수적 방법
            from sklearn.feature_selection import VarianceThreshold
            var_threshold = VarianceThreshold(threshold=0.01)
            X_train_var = var_threshold.fit_transform(X_train)
            X_test_var = var_threshold.transform(X_test)
            
            var_features = X_train.columns[var_threshold.get_support()].tolist()
            print(f"분산 필터링 후 피처 수: {len(var_features)}")
            
            X_train_filtered = pd.DataFrame(X_train_var, columns=var_features, index=X_train.index)
            X_test_filtered = pd.DataFrame(X_test_var, columns=var_features, index=X_test.index)
            
            if len(var_features) > k:
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(f_classif, k=k)
                
                if len(X_train) > 8000:
                    sample_idx = np.random.RandomState(Config.RANDOM_STATE).choice(
                        len(X_train), 8000, replace=False
                    )
                    X_sample = X_train_filtered.iloc[sample_idx]
                    y_sample = y_train.iloc[sample_idx]
                    selector.fit(X_sample, y_sample)
                    del X_sample, y_sample
                else:
                    selector.fit(X_train_filtered, y_train)
                
                selected_mask = selector.get_support()
                self.selected_features = [var_features[i] for i, selected in enumerate(selected_mask) if selected]
            else:
                self.selected_features = var_features
        
        X_train_selected = X_train[self.selected_features].copy()
        X_test_selected = X_test[self.selected_features].copy()
        
        print(f"선택된 피처 개수: {len(self.selected_features)}")
        
        # 선택된 피처 유형별 분석
        feature_types = {
            'original': [f for f in self.selected_features if f in self.feature_columns],
            'time_series': [f for f in self.selected_features if f in self.time_series_features],
            'domain': [f for f in self.selected_features if f in self.domain_features],
            'signal': [f for f in self.selected_features if any(x in f for x in ['rms', 'freq', 'spectral'])],
            'statistical': [f for f in self.selected_features if 'sensor_' in f]
        }
        
        print("선택된 피처 유형별 개수:")
        for ftype, features in feature_types.items():
            print(f"  {ftype}: {len(features)}개")
        
        gc.collect()
        
        return X_train_selected, X_test_selected
    
    def get_processed_data(self, use_feature_selection=True, scaling_method='robust', balance_classes=True):
        """전체 전처리 파이프라인 실행"""
        print("전체 데이터 전처리 파이프라인 시작")
        
        try:
            # 1. 데이터 로드
            train_df, test_df = self.load_and_preprocess_data()
            
            # ID 컬럼 저장
            train_ids = train_df[Config.ID_COLUMN].copy()
            test_ids = test_df[Config.ID_COLUMN].copy()
            
            # 2. 피처 엔지니어링
            X_train, X_test, y_train = self.feature_engineering(train_df, test_df)
            
            del train_df, test_df
            gc.collect()
            
            # 3. 클래스 불균형 처리
            if balance_classes:
                X_train, y_train = self.balance_classes(X_train, y_train)
                # 인덱스 재정렬
                train_ids = train_ids.iloc[:len(X_train)]
            
            # 4. 스케일링
            X_train, X_test = self.scale_features(X_train, X_test, scaling_method)
            
            # 5. 피처 선택
            if use_feature_selection:
                available_features = min(Config.FEATURE_SELECTION_K, X_train.shape[1])
                X_train, X_test = self.select_features(
                    X_train, X_test, y_train, 
                    method='domain_based',
                    k=available_features
                )
            
            # 6. 최종 검증 및 정리
            print("최종 데이터 검증")
            
            train_nan_count = X_train.isna().sum().sum()
            test_nan_count = X_test.isna().sum().sum()
            
            print(f"최종 훈련 데이터 NaN: {train_nan_count}")
            print(f"최종 테스트 데이터 NaN: {test_nan_count}")
            
            if train_nan_count > 0 or test_nan_count > 0:
                X_train.fillna(0, inplace=True)
                X_test.fillna(0, inplace=True)
                print("잔여 NaN을 0으로 대체")
            
            print(f"최종 피처 개수: {X_train.shape[1]}")
            print(f"최종 데이터 형태 - 훈련: {X_train.shape}, 테스트: {X_test.shape}")
            
            gc.collect()
            
            return X_train, X_test, y_train, train_ids, test_ids
            
        except Exception as e:
            print(f"데이터 전처리 중 오류 발생: {e}")
            gc.collect()
            raise