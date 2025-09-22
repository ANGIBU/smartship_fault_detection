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

warnings.filterwarnings('ignore')

def setup_logging(log_file=None, level='INFO'):
    """Setup logging configuration"""
    if log_file is None:
        log_file = Path('logs') / 'training.log'
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing handlers
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
    """Memory-efficient data loading"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if chunk_size and file_size_mb > 100:
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
        raise Exception(f"Data loading failed: {e}")

def save_model(model, file_path):
    """Save model"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try joblib first
        try:
            joblib.dump(model, file_path, compress=3)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"Model saved successfully (joblib): {file_path} ({file_size_mb:.1f}MB)")
            return
        except Exception as joblib_error:
            print(f"joblib save failed: {joblib_error}")
        
        # Try pickle
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"Model saved successfully (pickle): {file_path} ({file_size_mb:.1f}MB)")
            
        except Exception as pickle_error:
            print(f"pickle save failed: {pickle_error}")
            raise Exception(f"Model save failed: joblib={joblib_error}, pickle={pickle_error}")
        
    except Exception as e:
        print(f"Model save failed: {e}")
        raise

def load_model(file_path):
    """Load model"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Try joblib first
        try:
            model = joblib.load(file_path)
            print(f"Model loaded successfully (joblib): {file_path}")
            return model
        except:
            pass
        
        # Try pickle
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully (pickle): {file_path}")
            return model
        except:
            pass
        
        raise Exception("All loading methods failed")
        
    except Exception as e:
        raise Exception(f"Model loading failed: {e}")

def save_joblib(obj, file_path, compress=3):
    """Save object using joblib"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path, compress=compress)
        
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"Object saved successfully: {file_path} ({file_size_mb:.1f}MB)")
        else:
            print(f"Object saved successfully: {file_path}")
            
    except Exception as e:
        raise Exception(f"Object save failed: {e}")

def load_joblib(file_path):
    """Load object using joblib"""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        obj = joblib.load(file_path)
        print(f"Object loaded successfully: {file_path}")
        return obj
    except Exception as e:
        raise Exception(f"Object loading failed: {e}")

def calculate_macro_f1(y_true, y_pred):
    """Calculate Macro F1 score"""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def calculate_weighted_f1(y_true, y_pred):
    """Calculate Weighted F1 score"""
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

def calculate_all_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""
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
        print(f"Error calculating metrics: {e}")
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
    """Calculate class-wise metrics"""
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
        print(f"Error calculating class metrics: {e}")
        return []

def print_classification_metrics(y_true, y_pred, class_names=None, target_names=None):
    """Print classification performance metrics"""
    try:
        metrics = calculate_all_metrics(y_true, y_pred)
        
        print("Classification Performance Metrics")
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
            print("\nClassification Report:")
            print(report)
        except Exception as e:
            print(f"Classification report generation failed: {e}")
        
        return metrics['macro_f1']
    except Exception as e:
        print(f"Error printing performance metrics: {e}")
        return 0.0

def create_cv_folds(X, y, n_splits=5, random_state=42):
    """Create cross-validation folds"""
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(skf.split(X, y))
    except Exception as e:
        print(f"CV fold creation failed: {e}")
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
    """Calculate class weights"""
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
        print(f"Class weight calculation failed: {e}")
        return None

def timer(func):
    """Function execution time measurement decorator"""
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
                print(f"{func.__name__} execution time: {elapsed:.2f} seconds")
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                print(f"{func.__name__} execution time: {minutes}m {seconds:.2f}s")
            
            if memory_increase > 50:  # Display if increase > 50MB
                print(f"Memory increase: {memory_increase:.1f}MB")
            
            return result
            
        except Exception as e:
            print(f"Error during {func.__name__} execution: {e}")
            raise
            
    return wrapper

def check_data_quality(df, feature_columns):
    """Data quality check"""
    try:
        print("Data quality check")
        print(f"Data shape: {df.shape}")
        
        # Check missing values
        missing_info = df[feature_columns].isnull().sum()
        total_missing = missing_info.sum()
        print(f"Total missing values: {total_missing}")
        
        if total_missing > 0:
            print("Columns with missing values (top 5):")
            missing_cols = missing_info[missing_info > 0].sort_values(ascending=False)
            for col in missing_cols.head(5).index:
                print(f"  {col}: {missing_info[col]} ({missing_info[col]/len(df)*100:.2f}%)")
        
        # Check infinite values
        numeric_cols = df[feature_columns].select_dtypes(include=[np.number]).columns
        inf_counts = {}
        total_inf = 0
        
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
                total_inf += inf_count
        
        print(f"Infinite values: {total_inf}")
        
        if total_inf > 0:
            print("Columns with infinite values (top 5):")
            sorted_inf = sorted(inf_counts.items(), key=lambda x: x[1], reverse=True)
            for col, count in sorted_inf[:5]:
                print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
        
        # Check data types
        print(f"\nData types:")
        dtype_counts = df[feature_columns].dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Basic statistics
        if len(numeric_cols) > 0:
            print(f"\nBasic statistics ({len(numeric_cols)} numeric columns):")
            stats = df[numeric_cols].describe()
            print(f"  Mean range: {stats.loc['mean'].min():.4f} ~ {stats.loc['mean'].max():.4f}")
            print(f"  Std range: {stats.loc['std'].min():.4f} ~ {stats.loc['std'].max():.4f}")
            print(f"  Min value: {stats.loc['min'].min():.4f}")
            print(f"  Max value: {stats.loc['max'].max():.4f}")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"Memory usage: {memory_mb:.2f} MB")
        
        return total_missing == 0 and total_inf == 0
        
    except Exception as e:
        print(f"Error during data quality check: {e}")
        return False

def memory_usage_check():
    """Check memory usage"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except Exception:
        return 0

def save_results(results, file_path):
    """Save results to CSV file"""
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
        print(f"Results saved successfully: {file_path}")
    except Exception as e:
        print(f"Results save failed: {e}")

def validate_predictions(y_pred, n_classes, sample_ids=None):
    """Validate prediction results"""
    try:
        print("Prediction validation")
        
        if sample_ids is not None:
            print(f"Sample count: {len(sample_ids)}")
            print(f"Prediction count: {len(y_pred)}")
            
            if len(sample_ids) != len(y_pred):
                print("Warning: Sample count and prediction count do not match")
        
        # Check prediction value range
        min_pred = np.min(y_pred)
        max_pred = np.max(y_pred)
        unique_pred = len(np.unique(y_pred))
        
        print(f"Prediction range: {min_pred} ~ {max_pred}")
        print(f"Unique prediction count: {unique_pred}")
        print(f"Total class count: {n_classes}")
        
        # Validity check
        is_valid = True
        if min_pred < 0 or max_pred >= n_classes:
            print(f"Warning: Predictions outside valid range (0 ~ {n_classes-1})")
            is_valid = False
        
        # Distribution check
        unique, counts = np.unique(y_pred, return_counts=True)
        print(f"\nPrediction distribution (top 15):")
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        display_count = min(15, len(unique))
        
        for i in range(display_count):
            idx = sorted_indices[i]
            class_id = unique[idx]
            count = counts[idx]
            percentage = count / len(y_pred) * 100
            print(f"  Class {class_id:2d}: {count:4d} ({percentage:5.2f}%)")
        
        if len(unique) > 15:
            print(f"  ... (total {len(unique)} classes)")
        
        # Check missing classes
        all_classes = set(range(n_classes))
        predicted_classes = set(unique)
        missing_classes = all_classes - predicted_classes
        
        if missing_classes:
            missing_list = sorted(list(missing_classes))
            if len(missing_list) <= 10:
                print(f"Missing classes: {missing_list}")
            else:
                print(f"Missing classes: {missing_list[:10]} ... (total {len(missing_list)})")
            is_valid = False
        
        return is_valid
        
    except Exception as e:
        print(f"Error during prediction validation: {e}")
        return False

def create_submission_template(test_ids, predictions, id_col='ID', target_col='target'):
    """Create submission file template"""
    try:
        submission = pd.DataFrame({
            id_col: test_ids,
            target_col: predictions
        })
        
        # Data type optimization
        submission[target_col] = submission[target_col].astype('int16')
        
        return submission
    except Exception as e:
        print(f"Submission file template creation failed: {e}")
        return None

def analyze_class_distribution(y, class_names=None):
    """Analyze class distribution"""
    try:
        print("Class distribution analysis")
        
        unique, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        print(f"Total samples: {total_samples}")
        print(f"Class count: {len(unique)}")
        
        # Distribution statistics
        print(f"\nClass-wise distribution:")
        
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
            
            if class_id < 10:  # Display top 10 only
                print(f"  {class_name:>12}: {count:5d} ({percentage:5.2f}%)")
        
        # Calculate imbalance degree
        max_count = max(counts)
        min_count = min(counts[counts > 0]) if np.any(counts > 0) else 1
        imbalance_ratio = max_count / min_count
        
        print(f"\nDistribution statistics:")
        print(f"  Max class size: {max_count}")
        print(f"  Min class size: {min_count}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        print(f"  Standard deviation: {np.std(counts):.2f}")
        
        return distribution_data
        
    except Exception as e:
        print(f"Error during class distribution analysis: {e}")
        return []

def garbage_collect():
    """Perform garbage collection"""
    try:
        collected = gc.collect()
        if collected > 0:
            print(f"Memory cleanup: {collected} objects released")
        return collected
    except Exception as e:
        print(f"Error during garbage collection: {e}")
        return 0

def format_time(seconds):
    """Convert seconds to readable format"""
    try:
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.2f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}h {remaining_minutes}m {remaining_seconds:.2f}s"
    except Exception:
        return f"{seconds} seconds"

def safe_divide(numerator, denominator, default=0.0):
    """Safe division"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError, ValueError):
        return default

def check_system_resources():
    """Check system resources"""
    try:
        # CPU information
        cpu_count = os.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        memory_usage_percent = memory.percent
        
        # Disk information
        disk = psutil.disk_usage('.')
        disk_total_gb = disk.total / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        
        print(f"System resource status:")
        print(f"  CPU: {cpu_count} cores, usage {cpu_usage}%")
        print(f"  Memory: {memory_available_gb:.1f}GB available / {memory_total_gb:.1f}GB total ({memory_usage_percent:.1f}% used)")
        print(f"  Disk: {disk_free_gb:.1f}GB free / {disk_total_gb:.1f}GB total")
        
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
        print(f"System resource check failed: {e}")
        return None

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage"""
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
        
        print(f"Memory usage optimization: {initial_memory:.1f}MB -> {final_memory:.1f}MB ({memory_reduction:.1f}% reduction)")
        
        return df
        
    except Exception as e:
        print(f"Memory optimization failed: {e}")
        return df

def validate_data_consistency(train_df, test_df, feature_columns):
    """Validate training and test data consistency"""
    try:
        print("Data consistency validation")
        
        # Check feature columns
        train_features = set(train_df.columns) & set(feature_columns)
        test_features = set(test_df.columns) & set(feature_columns)
        
        missing_in_test = train_features - test_features
        missing_in_train = test_features - train_features
        
        if missing_in_test:
            print(f"Features missing in test data: {missing_in_test}")
        
        if missing_in_train:
            print(f"Features missing in train data: {missing_in_train}")
        
        # Check data type consistency
        common_features = train_features & test_features
        type_mismatches = []
        
        for feature in common_features:
            if train_df[feature].dtype != test_df[feature].dtype:
                type_mismatches.append({
                    'feature': feature,
                    'train_type': train_df[feature].dtype,
                    'test_type': test_df[feature].dtype
                })
        
        if type_mismatches:
            print("Data type mismatches:")
            for mismatch in type_mismatches:
                print(f"  {mismatch['feature']}: train={mismatch['train_type']}, test={mismatch['test_type']}")
        
        # Statistical distribution comparison
        distribution_differences = []
        
        for feature in list(common_features)[:10]:  # Check top 10 only
            train_mean = train_df[feature].mean()
            test_mean = test_df[feature].mean()
            train_std = train_df[feature].std()
            test_std = test_df[feature].std()
            
            mean_diff = abs(train_mean - test_mean) / (abs(train_mean) + 1e-8)
            std_diff = abs(train_std - test_std) / (abs(train_std) + 1e-8)
            
            if mean_diff > 0.1 or std_diff > 0.1:  # >10% difference
                distribution_differences.append({
                    'feature': feature,
                    'mean_diff': mean_diff,
                    'std_diff': std_diff
                })
        
        if distribution_differences:
            print("Features with large distribution differences (top 5):")
            for diff in distribution_differences[:5]:
                print(f"  {diff['feature']}: mean diff {diff['mean_diff']:.3f}, std diff {diff['std_diff']:.3f}")
        
        is_consistent = (len(missing_in_test) == 0 and len(missing_in_train) == 0 and 
                        len(type_mismatches) == 0 and len(distribution_differences) < 5)
        
        return is_consistent
        
    except Exception as e:
        print(f"Data consistency validation failed: {e}")
        return False