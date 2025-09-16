# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from config import Config
from utils import timer, check_data_quality, optimize_memory_usage, save_model

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.feature_columns = Config.FEATURE_COLUMNS
        self.selected_features = None
        
    @timer
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print("=== 데이터 로드 및 전처리 시작 ===")
        
        # 데이터 로드
        train_df = pd.read_csv(Config.TRAIN_FILE)
        test_df = pd.read_csv(Config.TEST_FILE)
        
        print(f"Train 데이터 형태: {train_df.shape}")
        print(f"Test 데이터 형태: {test_df.shape}")
        
        # 메모리 최적화
        train_df = optimize_memory_usage(train_df)
        test_df = optimize_memory_usage(test_df)
        
        # 데이터 품질 검사
        train_quality = check_data_quality(train_df, self.feature_columns)
        test_quality = check_data_quality(test_df, self.feature_columns)
        
        if not (train_quality and test_quality):
            print("데이터 품질 문제 발견, 추가 전처리 진행")
            train_df, test_df = self._handle_data_issues(train_df, test_df)
        
        return train_df, test_df
    
    def _handle_data_issues(self, train_df, test_df):
        """데이터 이슈 처리"""
        print("데이터 이슈 처리 중...")
        
        # 결측치 처리
        for col in self.feature_columns:
            if train_df[col].isnull().sum() > 0:
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                test_df[col].fillna(median_val, inplace=True)
        
        # 무한값 처리
        train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        for col in self.feature_columns:
            if train_df[col].isnull().sum() > 0:
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                test_df[col].fillna(median_val, inplace=True)
        
        return train_df, test_df
    
    @timer
    def feature_engineering(self, train_df, test_df):
        """피처 엔지니어링"""
        print("=== 피처 엔지니어링 시작 ===")
        
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        y_train = train_df[Config.TARGET_COLUMN]
        
        # 기본 통계 피처 생성
        X_train, X_test = self._create_statistical_features(X_train, X_test)
        
        # 비율 피처 생성
        X_train, X_test = self._create_ratio_features(X_train, X_test)
        
        # 상호작용 피처 생성 (선택적)
        # X_train, X_test = self._create_interaction_features(X_train, X_test)
        
        return X_train, X_test, y_train
    
    def _create_statistical_features(self, X_train, X_test):
        """통계적 피처 생성"""
        print("통계적 피처 생성 중...")
        
        # 평균, 표준편차, 최대, 최소
        X_train['feature_mean'] = X_train[self.feature_columns].mean(axis=1)
        X_train['feature_std'] = X_train[self.feature_columns].std(axis=1)
        X_train['feature_max'] = X_train[self.feature_columns].max(axis=1)
        X_train['feature_min'] = X_train[self.feature_columns].min(axis=1)
        X_train['feature_range'] = X_train['feature_max'] - X_train['feature_min']
        
        X_test['feature_mean'] = X_test[self.feature_columns].mean(axis=1)
        X_test['feature_std'] = X_test[self.feature_columns].std(axis=1)
        X_test['feature_max'] = X_test[self.feature_columns].max(axis=1)
        X_test['feature_min'] = X_test[self.feature_columns].min(axis=1)
        X_test['feature_range'] = X_test['feature_max'] - X_test['feature_min']
        
        # 분위수 피처
        for q in [0.25, 0.5, 0.75]:
            X_train[f'feature_q{int(q*100)}'] = X_train[self.feature_columns].quantile(q, axis=1)
            X_test[f'feature_q{int(q*100)}'] = X_test[self.feature_columns].quantile(q, axis=1)
        
        return X_train, X_test
    
    def _create_ratio_features(self, X_train, X_test):
        """비율 피처 생성"""
        print("비율 피처 생성 중...")
        
        # 큰 값을 가지는 피처들 식별 (스케일이 다른 피처들)
        high_scale_features = []
        for col in self.feature_columns:
            if X_train[col].max() > 10:
                high_scale_features.append(col)
        
        # 고스케일 피처들의 비율
        if len(high_scale_features) >= 2:
            for i in range(min(3, len(high_scale_features))):
                for j in range(i+1, min(3, len(high_scale_features))):
                    feat1, feat2 = high_scale_features[i], high_scale_features[j]
                    ratio_name = f'ratio_{feat1}_{feat2}'
                    
                    # 0으로 나누기 방지
                    X_train[ratio_name] = X_train[feat1] / (X_train[feat2] + 1e-8)
                    X_test[ratio_name] = X_test[feat1] / (X_test[feat2] + 1e-8)
        
        return X_train, X_test
    
    def _create_interaction_features(self, X_train, X_test):
        """상호작용 피처 생성 (메모리 사용량 주의)"""
        print("상호작용 피처 생성 중...")
        
        # 상관관계가 높은 피처 쌍 찾기
        corr_matrix = X_train[self.feature_columns].corr()
        high_corr_pairs = []
        
        for i in range(len(self.feature_columns)):
            for j in range(i+1, len(self.feature_columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((self.feature_columns[i], self.feature_columns[j]))
        
        # 상관관계 높은 피처들의 곱셈 피처 생성 (최대 5개만)
        for i, (feat1, feat2) in enumerate(high_corr_pairs[:5]):
            interaction_name = f'interact_{feat1}_{feat2}'
            X_train[interaction_name] = X_train[feat1] * X_train[feat2]
            X_test[interaction_name] = X_test[feat1] * X_test[feat2]
        
        return X_train, X_test
    
    @timer
    def scale_features(self, X_train, X_test, method='robust'):
        """피처 스케일링"""
        print(f"=== 피처 스케일링 시작 ({method}) ===")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("지원하지 않는 스케일링 방법입니다.")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 피처명 유지
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # 스케일러 저장
        save_model(self.scaler, Config.SCALER_FILE)
        
        return X_train_scaled, X_test_scaled
    
    @timer
    def select_features(self, X_train, X_test, y_train, method='mutual_info', k=100):
        """피처 선택"""
        print(f"=== 피처 선택 시작 ({method}, k={k}) ===")
        print(f"원본 피처 개수: {X_train.shape[1]}")
        
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            self.feature_selector = SelectKBest(f_classif, k=k)
        elif method == 'random_forest':
            return self._select_features_rf(X_train, X_test, y_train, k)
        else:
            raise ValueError("지원하지 않는 피처 선택 방법입니다.")
        
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # 선택된 피처명 저장
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
        print(f"선택된 피처 개수: {len(self.selected_features)}")
        
        # DataFrame으로 변환
        X_train_selected = pd.DataFrame(X_train_selected, columns=self.selected_features, index=X_train.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=self.selected_features, index=X_test.index)
        
        return X_train_selected, X_test_selected
    
    def _select_features_rf(self, X_train, X_test, y_train, k):
        """랜덤 포레스트 기반 피처 선택"""
        print("랜덤 포레스트 기반 피처 중요도 계산 중...")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # 피처 중요도 기반 선택
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        
        self.selected_features = X_train.columns[indices].tolist()
        
        X_train_selected = X_train.iloc[:, indices]
        X_test_selected = X_test.iloc[:, indices]
        
        return X_train_selected, X_test_selected
    
    @timer
    def apply_pca(self, X_train, X_test, n_components=0.95):
        """PCA 적용"""
        print(f"=== PCA 적용 시작 (n_components={n_components}) ===")
        
        self.pca = PCA(n_components=n_components, random_state=Config.RANDOM_STATE)
        
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        
        print(f"PCA 후 컴포넌트 개수: {X_train_pca.shape[1]}")
        print(f"설명 가능한 분산 비율: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # DataFrame으로 변환
        pca_columns = [f'PC_{i+1}' for i in range(X_train_pca.shape[1])]
        X_train_pca = pd.DataFrame(X_train_pca, columns=pca_columns, index=X_train.index)
        X_test_pca = pd.DataFrame(X_test_pca, columns=pca_columns, index=X_test.index)
        
        return X_train_pca, X_test_pca
    
    def get_processed_data(self, use_feature_selection=True, use_pca=False):
        """전체 전처리 파이프라인 실행"""
        print("=== 전체 데이터 전처리 파이프라인 시작 ===")
        
        # 1. 데이터 로드 및 기본 전처리
        train_df, test_df = self.load_and_preprocess_data()
        
        # 2. 피처 엔지니어링
        X_train, X_test, y_train = self.feature_engineering(train_df, test_df)
        
        # 3. 스케일링
        X_train, X_test = self.scale_features(X_train, X_test, method='robust')
        
        # 4. 피처 선택 (선택사항)
        if use_feature_selection:
            X_train, X_test = self.select_features(X_train, X_test, y_train, method='mutual_info', k=80)
        
        # 5. PCA (선택사항)
        if use_pca:
            X_train, X_test = self.apply_pca(X_train, X_test, n_components=0.95)
        
        print("=== 전처리 완료 ===")
        print(f"최종 피처 개수: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, train_df[Config.ID_COLUMN], test_df[Config.ID_COLUMN]