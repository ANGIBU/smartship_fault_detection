# data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from scipy.stats import skew, kurtosis
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
        self.feature_columns = Config.FEATURE_COLUMNS
        self.selected_features = None
        self.feature_importance = {}
        
    @timer
    def load_data(self):
        """Load data"""
        print("Starting data loading")
        
        try:
            train_df = pd.read_csv(Config.TRAIN_FILE)
            test_df = pd.read_csv(Config.TEST_FILE)
            
            print(f"Train data shape: {train_df.shape}")
            print(f"Test data shape: {test_df.shape}")
            
            # Data type optimization
            for col in self.feature_columns:
                if col in train_df.columns:
                    train_df[col] = train_df[col].astype('float32')
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype('float32')
            
            if Config.TARGET_COLUMN in train_df.columns:
                train_df[Config.TARGET_COLUMN] = train_df[Config.TARGET_COLUMN].astype('int16')
            
            # Data quality check
            train_quality = check_data_quality(train_df, self.feature_columns)
            test_quality = check_data_quality(test_df, self.feature_columns)
            
            if not (train_quality and test_quality):
                print("Handling data quality issues")
                train_df, test_df = self._handle_data_issues(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Data loading failed: {e}")
            raise
    
    @timer
    def load_quick_data(self):
        """Load data for quick mode"""
        print("Loading data for quick mode")
        
        try:
            train_df = pd.read_csv(Config.TRAIN_FILE)
            test_df = pd.read_csv(Config.TEST_FILE)
            
            # Sample data for quick testing
            if len(train_df) > Config.QUICK_SAMPLE_SIZE:
                # Stratified sampling to maintain class distribution
                sample_indices = []
                for class_id in train_df[Config.TARGET_COLUMN].unique():
                    class_data = train_df[train_df[Config.TARGET_COLUMN] == class_id]
                    sample_size = min(len(class_data), Config.QUICK_SAMPLE_SIZE // Config.N_CLASSES)
                    if sample_size > 0:
                        sampled = class_data.sample(n=sample_size, random_state=Config.RANDOM_STATE)
                        sample_indices.extend(sampled.index.tolist())
                
                train_df = train_df.loc[sample_indices].reset_index(drop=True)
                print(f"Sampled training data: {len(train_df)} samples")
            
            # Sample test data
            if len(test_df) > Config.QUICK_SAMPLE_SIZE:
                test_df = test_df.sample(n=Config.QUICK_SAMPLE_SIZE, random_state=Config.RANDOM_STATE).reset_index(drop=True)
                print(f"Sampled test data: {len(test_df)} samples")
            
            # Data type optimization
            for col in self.feature_columns:
                if col in train_df.columns:
                    train_df[col] = train_df[col].astype('float32')
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype('float32')
            
            if Config.TARGET_COLUMN in train_df.columns:
                train_df[Config.TARGET_COLUMN] = train_df[Config.TARGET_COLUMN].astype('int16')
            
            # Basic data cleaning
            train_df, test_df = self._handle_data_issues(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Quick data loading failed: {e}")
            raise
    
    def _handle_data_issues(self, train_df, test_df):
        """Handle data issues"""
        for col in self.feature_columns:
            if col in train_df.columns and col in test_df.columns:
                # Handle infinite values
                train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
                test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
                
                # Handle missing values
                if train_df[col].isnull().sum() > 0 or test_df[col].isnull().sum() > 0:
                    fill_val = train_df[col].median()
                    if pd.isna(fill_val):
                        fill_val = 0.0
                    
                    train_df[col].fillna(fill_val, inplace=True)
                    test_df[col].fillna(fill_val, inplace=True)
        
        return train_df, test_df
    
    @timer
    def create_statistical_features(self, X_train, X_test):
        """Create statistical features"""
        print("Creating statistical features")
        
        for df in [X_train, X_test]:
            sensor_data = df[self.feature_columns].values
            
            # Basic statistics
            df['sensor_mean'] = np.mean(sensor_data, axis=1).astype('float32')
            df['sensor_std'] = np.std(sensor_data, axis=1).astype('float32')
            df['sensor_median'] = np.median(sensor_data, axis=1).astype('float32')
            df['sensor_min'] = np.min(sensor_data, axis=1).astype('float32')
            df['sensor_max'] = np.max(sensor_data, axis=1).astype('float32')
            df['sensor_range'] = (df['sensor_max'] - df['sensor_min']).astype('float32')
            
            # Distribution characteristics
            df['sensor_skew'] = skew(sensor_data, axis=1, nan_policy='omit').astype('float32')
            df['sensor_kurtosis'] = kurtosis(sensor_data, axis=1, nan_policy='omit').astype('float32')
            
            # Quantiles
            df['sensor_q25'] = np.percentile(sensor_data, 25, axis=1).astype('float32')
            df['sensor_q75'] = np.percentile(sensor_data, 75, axis=1).astype('float32')
            df['sensor_iqr'] = (df['sensor_q75'] - df['sensor_q25']).astype('float32')
            
            # Variability indicators
            df['sensor_cv'] = (df['sensor_std'] / (np.abs(df['sensor_mean']) + 1e-8)).astype('float32')
            
            # Outlier count
            q75 = df['sensor_q75'].values.reshape(-1, 1)
            q25 = df['sensor_q25'].values.reshape(-1, 1)
            iqr = (q75 - q25)
            
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outliers = ((sensor_data < lower_bound) | (sensor_data > upper_bound)).sum(axis=1)
            df['sensor_outlier_count'] = outliers.astype('float32')
            df['sensor_outlier_ratio'] = (outliers / len(self.feature_columns)).astype('float32')
        
        return X_train, X_test
    
    @timer
    def create_quick_statistical_features(self, X_train, X_test):
        """Create minimal statistical features for quick mode"""
        print("Creating basic statistical features for quick mode")
        
        for df in [X_train, X_test]:
            sensor_data = df[self.feature_columns].values
            
            # Only essential statistics
            df['sensor_mean'] = np.mean(sensor_data, axis=1).astype('float32')
            df['sensor_std'] = np.std(sensor_data, axis=1).astype('float32')
            df['sensor_range'] = (np.max(sensor_data, axis=1) - np.min(sensor_data, axis=1)).astype('float32')
        
        return X_train, X_test
    
    @timer
    def create_correlation_features(self, X_train, X_test):
        """Create correlation-based features"""
        print("Creating correlation features")
        
        # Calculate correlation from training data
        train_corr = X_train[self.feature_columns].corr()
        
        # Identify highly correlated sensor pairs
        high_corr_pairs = []
        for i in range(len(self.feature_columns)):
            for j in range(i+1, len(self.feature_columns)):
                corr_val = train_corr.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((self.feature_columns[i], self.feature_columns[j], corr_val))
        
        print(f"High correlation sensor pairs: {len(high_corr_pairs)}")
        
        # Create correlation-based features
        for df in [X_train, X_test]:
            for sensor1, sensor2, corr_val in high_corr_pairs[:10]:  # Top 10 only
                if sensor1 in df.columns and sensor2 in df.columns:
                    # Ratio
                    df[f'{sensor1}_{sensor2}_ratio'] = (df[sensor1] / (df[sensor2] + 1e-8)).astype('float32')
                    # Difference
                    df[f'{sensor1}_{sensor2}_diff'] = (df[sensor1] - df[sensor2]).astype('float32')
        
        return X_train, X_test
    
    @timer
    def handle_class_imbalance(self, X_train, y_train):
        """Handle class imbalance"""
        print("Starting class imbalance handling")
        
        # Check class distribution
        class_counts = y_train.value_counts().sort_index()
        print("Class sample counts:")
        for class_id, count in class_counts.items():
            if class_id < 10:
                print(f"  Class {class_id}: {count} samples")
        
        # Calculate imbalance degree
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3.0:
            print("Applying SMOTE")
            
            try:
                # Calculate appropriate k_neighbors
                min_class_samples = class_counts.min()
                k_neighbors = min(5, min_class_samples - 1)
                
                if k_neighbors < 1:
                    print("SMOTE not applicable, using original data")
                    return X_train, y_train
                
                smote = SMOTE(
                    sampling_strategy='auto',
                    random_state=Config.RANDOM_STATE,
                    k_neighbors=k_neighbors
                )
                
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                
                print(f"Before resampling: {X_train.shape}")
                print(f"After resampling: {X_resampled.shape}")
                
                return X_resampled, y_resampled
                
            except Exception as e:
                print(f"SMOTE application failed: {e}")
                return X_train, y_train
        else:
            print("Class balance is appropriate, skipping resampling")
            return X_train, y_train
    
    @timer
    def scale_features(self, X_train, X_test, method='robust'):
        """Feature scaling"""
        print(f"Starting feature scaling ({method})")
        
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'quantile':
            self.scaler = QuantileTransformer(
                output_distribution='normal',
                random_state=Config.RANDOM_STATE,
                n_quantiles=min(1000, X_train.shape[0])
            )
        else:
            self.scaler = RobustScaler()
        
        # Scale only numeric columns
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Apply scaling
        X_train_scaled[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
        X_test_scaled[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
        
        # Maintain data types
        for col in numeric_columns:
            X_train_scaled[col] = X_train_scaled[col].astype('float32')
            X_test_scaled[col] = X_test_scaled[col].astype('float32')
        
        save_joblib(self.scaler, Config.SCALER_FILE)
        
        return X_train_scaled, X_test_scaled
    
    @timer
    def select_features(self, X_train, X_test, y_train, n_features=None):
        """Feature selection"""
        if n_features is None:
            n_features = Config.TARGET_FEATURES
        
        n_features = min(n_features, X_train.shape[1])
        
        print(f"Starting feature selection (target: {n_features} features)")
        print(f"Original feature count: {X_train.shape[1]}")
        
        # Remove zero variance features
        constant_features = []
        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"Removing constant features: {len(constant_features)}")
            X_train = X_train.drop(columns=constant_features)
            X_test = X_test.drop(columns=constant_features)
        
        # Apply multiple feature selection methods
        feature_scores = {}
        
        # Mutual Information
        try:
            mi_selector = SelectKBest(mutual_info_classif, k=n_features)
            mi_selector.fit(X_train, y_train)
            mi_scores = mi_selector.scores_
            
            for i, col in enumerate(X_train.columns):
                if col not in feature_scores:
                    feature_scores[col] = 0
                feature_scores[col] += mi_scores[i]
        except Exception as e:
            print(f"Mutual Information calculation failed: {e}")
        
        # F-test
        try:
            f_selector = SelectKBest(f_classif, k=n_features)
            f_selector.fit(X_train, y_train)
            f_scores = f_selector.scores_
            
            for i, col in enumerate(X_train.columns):
                if col not in feature_scores:
                    feature_scores[col] = 0
                feature_scores[col] += f_scores[i] / np.max(f_scores)  # Normalize
        except Exception as e:
            print(f"F-test calculation failed: {e}")
        
        # Score-based feature selection
        if feature_scores:
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            self.selected_features = [feat for feat, score in sorted_features[:n_features]]
            
            # Save feature importance
            self.feature_importance = dict(sorted_features[:n_features])
        else:
            # Use original sensor features by default
            available_sensors = [col for col in self.feature_columns if col in X_train.columns]
            self.selected_features = available_sensors[:n_features]
            print("Feature selection failed, using original sensor features")
        
        X_train_selected = X_train[self.selected_features].copy()
        X_test_selected = X_test[self.selected_features].copy()
        
        print(f"Selected feature count: {len(self.selected_features)}")
        
        # Analyze selected feature types
        original_count = sum(1 for f in self.selected_features if f in self.feature_columns)
        statistical_count = sum(1 for f in self.selected_features if 'sensor_' in f)
        correlation_count = len(self.selected_features) - original_count - statistical_count
        
        print(f"Feature type selection:")
        print(f"  Original sensors: {original_count}")
        print(f"  Statistical: {statistical_count}")
        print(f"  Correlation: {correlation_count}")
        
        return X_train_selected, X_test_selected
    
    @timer
    def quick_feature_selection(self, X_train, X_test, y_train):
        """Quick feature selection for quick mode"""
        print(f"Quick feature selection (target: {Config.QUICK_FEATURE_COUNT} features)")
        
        # Remove constant features
        constant_features = []
        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            X_train = X_train.drop(columns=constant_features)
            X_test = X_test.drop(columns=constant_features)
        
        # Simple variance-based selection
        variances = X_train.var()
        top_variance_features = variances.nlargest(Config.QUICK_FEATURE_COUNT).index.tolist()
        
        self.selected_features = top_variance_features
        
        X_train_selected = X_train[self.selected_features].copy()
        X_test_selected = X_test[self.selected_features].copy()
        
        print(f"Selected {len(self.selected_features)} features by variance")
        
        return X_train_selected, X_test_selected
    
    @timer
    def get_processed_data(self, use_resampling=True, scaling_method='robust'):
        """Execute complete data preprocessing pipeline"""
        print("Starting complete data preprocessing pipeline")
        
        try:
            # 1. Data loading
            train_df, test_df = self.load_data()
            
            # Save ID columns
            train_ids = train_df[Config.ID_COLUMN].copy()
            test_ids = test_df[Config.ID_COLUMN].copy()
            
            # Separate features and target
            X_train = train_df[self.feature_columns].copy()
            X_test = test_df[self.feature_columns].copy()
            y_train = train_df[Config.TARGET_COLUMN].copy()
            
            del train_df, test_df
            gc.collect()
            
            # 2. Create statistical features
            X_train, X_test = self.create_statistical_features(X_train, X_test)
            
            # 3. Create correlation features
            X_train, X_test = self.create_correlation_features(X_train, X_test)
            
            # 4. Scaling
            X_train, X_test = self.scale_features(X_train, X_test, scaling_method)
            
            # 5. Feature selection
            X_train, X_test = self.select_features(X_train, X_test, y_train)
            
            # 6. Handle class imbalance
            if use_resampling:
                X_train, y_train = self.handle_class_imbalance(X_train, y_train)
                # Adjust IDs after resampling
                if len(X_train) != len(train_ids):
                    train_ids = pd.Series(range(len(X_train)), name=Config.ID_COLUMN)
            
            # 7. Final validation
            print("Final data validation")
            
            train_nan_count = X_train.isna().sum().sum()
            test_nan_count = X_test.isna().sum().sum()
            
            print(f"Final training data NaN: {train_nan_count}")
            print(f"Final test data NaN: {test_nan_count}")
            
            if train_nan_count > 0 or test_nan_count > 0:
                X_train.fillna(0, inplace=True)
                X_test.fillna(0, inplace=True)
                print("Replaced remaining NaN with 0")
            
            print(f"Final feature count: {X_train.shape[1]}")
            print(f"Final data shapes - training: {X_train.shape}, test: {X_test.shape}")
            
            gc.collect()
            
            return X_train, X_test, y_train, train_ids, test_ids
            
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            gc.collect()
            raise
    
    @timer
    def get_quick_processed_data(self):
        """Execute quick data preprocessing pipeline"""
        print("Starting quick data preprocessing pipeline")
        
        try:
            # 1. Quick data loading
            train_df, test_df = self.load_quick_data()
            
            # Save ID columns
            train_ids = train_df[Config.ID_COLUMN].copy()
            test_ids = test_df[Config.ID_COLUMN].copy()
            
            # Separate features and target
            X_train = train_df[self.feature_columns].copy()
            X_test = test_df[self.feature_columns].copy()
            y_train = train_df[Config.TARGET_COLUMN].copy()
            
            del train_df, test_df
            gc.collect()
            
            # 2. Create basic statistical features
            X_train, X_test = self.create_quick_statistical_features(X_train, X_test)
            
            # 3. Quick scaling
            X_train, X_test = self.scale_features(X_train, X_test, 'robust')
            
            # 4. Quick feature selection
            X_train, X_test = self.quick_feature_selection(X_train, X_test, y_train)
            
            # 5. Final validation
            train_nan_count = X_train.isna().sum().sum()
            test_nan_count = X_test.isna().sum().sum()
            
            if train_nan_count > 0 or test_nan_count > 0:
                X_train.fillna(0, inplace=True)
                X_test.fillna(0, inplace=True)
                print("Replaced remaining NaN with 0")
            
            print(f"Quick processing complete")
            print(f"Final data shapes - training: {X_train.shape}, test: {X_test.shape}")
            
            gc.collect()
            
            return X_train, X_test, y_train, train_ids, test_ids
            
        except Exception as e:
            print(f"Error during quick data preprocessing: {e}")
            gc.collect()
            raise
    
    def get_feature_importance(self):
        """Return feature importance"""
        return self.feature_importance