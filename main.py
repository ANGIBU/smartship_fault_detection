# main.py

import pandas as pd
import numpy as np
import warnings
import sys
import time
import psutil
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from model_training import ModelTraining
from prediction import PredictionProcessor
from utils import setup_logging, memory_usage_check, timer
from sklearn.model_selection import train_test_split

def main():
    """Main execution function"""
    start_time = time.time()
    logger = setup_logging()
    logger.info("System startup")
    Config.create_directories()
    initial_memory = memory_usage_check()
    
    try:
        # Hardware-specific configuration adjustment
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        cpu_cores = psutil.cpu_count()
        Config.update_for_hardware(available_memory_gb, cpu_cores)
        
        print(f"System configuration:")
        print(f"  Available memory: {available_memory_gb:.1f}GB")
        print(f"  CPU cores: {cpu_cores}")
        print(f"  Worker processes: {Config.N_JOBS}")
        
        # Configuration validation
        config_errors = Config.validate_config()
        if config_errors:
            print("Configuration errors found:")
            for error in config_errors:
                print(f"  - {error}")
            return None
        
        # 1. Data preprocessing
        print("\n" + "=" * 50)
        print("Stage 1: Data Preprocessing")
        print("=" * 50)
        
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_resampling=True,
            scaling_method=Config.SCALING_METHOD
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Target distribution (top 10): {pd.Series(y_train).value_counts().sort_index().head(10).to_dict()}")
        
        # 2. Validation strategy setup
        print("\n" + "=" * 50)
        print("Stage 2: Validation Strategy Setup")
        print("=" * 50)
        
        # Create validation set with stratified split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=Config.VALIDATION_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"Stratified split:")
        print(f"  Training set: {X_train_split.shape}")
        print(f"  Validation set: {X_val.shape}")
        
        # Check validation set distribution
        val_distribution = pd.Series(y_val).value_counts().sort_index()
        print(f"\nValidation set class distribution (top 10):")
        for class_id, count in val_distribution.head(10).items():
            percentage = (count / len(y_val)) * 100
            print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
        
        # 3. Model training
        print("\n" + "=" * 50)
        print("Stage 3: Model Training")
        print("=" * 50)
        
        trainer = ModelTraining()
        
        # Separate small validation set for probability calibration
        X_train_final, X_calibration, y_train_final, y_calibration = train_test_split(
            X_train_split, y_train_split,
            test_size=0.1,
            random_state=Config.RANDOM_STATE,
            stratify=y_train_split
        )
        
        print(f"Final training set: {X_train_final.shape}")
        print(f"Calibration set: {X_calibration.shape}")
        
        # Train models
        models, best_model = trainer.train_all_models(
            X_train_final, y_train_final,
            X_calibration, y_calibration,
            use_optimization=True
        )
        
        print(f"Number of trained models: {len(models)}")
        if best_model is not None:
            print(f"Best performing model: {type(best_model).__name__}")
        
        # 4. Model validation
        print("\n" + "=" * 50)
        print("Stage 4: Model Validation")
        print("=" * 50)
        
        validation_results = {}
        
        if best_model is not None:
            predictor = PredictionProcessor(best_model)
            
            # Model probability calibration
            predictor.calibrate_model(X_calibration, y_calibration)
            
            # Validation set prediction
            val_predictions = predictor.predict(X_val, use_calibrated=True)
            val_metrics = predictor.validate_predictions(y_val)
            
            if val_metrics:
                print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")
                validation_results['validation'] = val_metrics
                
                # Class-wise performance analysis
                low_performance_classes = val_metrics.get('low_performance_classes', [])
                if low_performance_classes:
                    print(f"\nLow-performance classes: {len(low_performance_classes)}")
                    for cls_info in low_performance_classes[:5]:
                        print(f"  Class {cls_info['class']}: F1={cls_info['f1_score']:.3f}")
        
        # 5. Test prediction
        print("\n" + "=" * 50)
        print("Stage 5: Test Prediction")
        print("=" * 50)
        
        if best_model is not None:
            # Retrain final model with full training data
            final_trainer = ModelTraining()
            
            # Retrain with same model type as best performing
            best_model_type = type(best_model).__name__
            print(f"Retraining {best_model_type} with full data")
            
            if 'LGBMClassifier' in best_model_type:
                final_model = final_trainer.train_lightgbm(X_train, y_train)
            elif 'XGBClassifier' in best_model_type:
                final_model = final_trainer.train_xgboost(X_train, y_train)
            elif 'CatBoost' in best_model_type:
                final_model = final_trainer.train_catboost(X_train, y_train)
            elif 'RandomForest' in best_model_type:
                final_model = final_trainer.train_random_forest(X_train, y_train)
            elif 'ExtraTrees' in best_model_type:
                final_model = final_trainer.train_extra_trees(X_train, y_train)
            else:
                # Use existing model for ensemble
                final_model = best_model
            
            if final_model is not None:
                final_predictor = PredictionProcessor(final_model)
                test_predictions = final_predictor.predict(X_test, use_calibrated=False)
                distribution_info = final_predictor.analyze_prediction_distribution()
            else:
                final_predictor = predictor
                test_predictions = predictor.predict(X_test, use_calibrated=True)
                distribution_info = predictor.analyze_prediction_distribution()
        else:
            print("No available model")
            return None
        
        # 6. Submission file creation
        print("\n" + "=" * 50)
        print("Stage 6: Submission File Creation")
        print("=" * 50)
        
        submission_df = final_predictor.create_submission_file(
            test_ids,
            apply_balancing=True,
            confidence_threshold=0.6
        )
        
        if submission_df is not None:
            print(f"Submission file created: {Config.RESULT_FILE}")
            print(f"Submission file shape: {submission_df.shape}")
        
        # 7. Performance analysis
        print("\n" + "=" * 50)
        print("Stage 7: Performance Analysis")
        print("=" * 50)
        
        # Cross-validation results output
        if trainer.cv_scores:
            print("\nCross-validation results (stability ranking):")
            sorted_scores = sorted(trainer.cv_scores.items(),
                                 key=lambda x: x[1].get('stability', x[1]['mean']),
                                 reverse=True)
            for model_name, scores in sorted_scores:
                stability_score = scores.get('stability', scores['mean'])
                std_score = scores['std']
                print(f"  {model_name:15s}: {stability_score:.4f} (±{std_score:.4f})")
        
        # Feature importance analysis
        feature_importance = trainer.get_feature_importance()
        if feature_importance:
            print("\nFeature importance (top models):")
            for model_name, importance in list(feature_importance.items())[:2]:
                if len(importance) > 0:
                    print(f"  {model_name}: Average importance {np.mean(importance):.4f}")
        
        # Final performance prediction
        print(f"\nPerformance prediction:")
        if validation_results and 'validation' in validation_results:
            val_score = validation_results['validation']['macro_f1']
            conservative_estimate = val_score * 0.95
            print(f"  Validation Macro F1: {val_score:.4f}")
            print(f"  Expected performance: {conservative_estimate:.4f}")
            print(f"  Gap to target: {0.83 - conservative_estimate:.4f} points {'achieved' if conservative_estimate >= 0.83 else 'short'}")
        
        # System resource usage
        final_memory = memory_usage_check()
        memory_increase = final_memory - initial_memory
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("System Execution Complete")
        print("=" * 60)
        print(f"Final prediction file: {Config.RESULT_FILE}")
        print(f"Best model file: {Config.MODEL_FILE}")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        print(f"Total execution time: {total_time/60:.1f} minutes")
        
        # Validation score summary
        if validation_results:
            for val_type, result in validation_results.items():
                if 'macro_f1' in result:
                    print(f"{val_type} Macro F1: {result['macro_f1']:.4f}")
        
        if trainer.cv_scores:
            best_cv_scores = [score.get('stability', score['mean']) 
                             for score in trainer.cv_scores.values()]
            best_cv_score = max(best_cv_scores) if best_cv_scores else 0
            print(f"Best stability score: {best_cv_score:.4f}")
        
        logger.info("System completed successfully")
        
        # Construct return value
        result_dict = {
            'models': models,
            'best_model': best_model,
            'final_model': final_model if 'final_model' in locals() else best_model,
            'validation_results': validation_results,
            'cv_scores': trainer.cv_scores,
            'distribution_info': distribution_info,
            'submission_df': submission_df,
            'feature_importance': feature_importance,
            'memory_usage': {
                'initial': initial_memory,
                'final': final_memory,
                'increase': memory_increase
            },
            'execution_time': total_time
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error during system execution: {e}")
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_fast_mode():
    """Fast execution mode"""
    print("=" * 50)
    print("   Fast Execution Mode")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        # Basic data preprocessing
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_resampling=False,
            scaling_method='robust'
        )
        
        # Simple split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"Training set: {X_train_split.shape}")
        print(f"Validation set: {X_val.shape}")
        
        # Basic model training (optimization disabled)
        trainer = ModelTraining()
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split,
            X_val, y_val,
            use_optimization=False
        )
        
        print(f"Number of trained models: {len(models)}")
        if best_model is not None:
            print(f"Best performing model: {type(best_model).__name__}")
        
        # Test prediction
        if best_model is not None:
            predictor = PredictionProcessor(best_model)
            test_predictions = predictor.predict(X_test, use_calibrated=False)
            submission_df = predictor.create_submission_file(
                test_ids, apply_balancing=True
            )
        else:
            submission_df = None
        
        print(f"Fast execution complete: {Config.RESULT_FILE}")
        
        return models, best_model, submission_df
        
    except Exception as e:
        print(f"Error during fast execution: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_validation_mode():
    """Validation mode"""
    print("=" * 50)
    print("   Validation Mode")
    print("=" * 50)
    
    try:
        Config.create_directories()
        
        # Data preprocessing
        processor = DataProcessor()
        X_train, X_test, y_train, train_ids, test_ids = processor.get_processed_data(
            use_resampling=True,
            scaling_method=Config.SCALING_METHOD
        )
        
        # Validation split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=0.3,
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        print(f"Validation training set: {X_train_split.shape}")
        print(f"Validation set: {X_val.shape}")
        
        # Model training
        trainer = ModelTraining()
        models, best_model = trainer.train_all_models(
            X_train_split, y_train_split,
            use_optimization=True
        )
        
        # Validation execution
        if best_model is not None:
            predictor = PredictionProcessor(best_model)
            val_predictions = predictor.predict(X_val, use_calibrated=False)
            val_metrics = predictor.validate_predictions(y_val)
            
            if val_metrics:
                print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")
        
        print(f"Validation mode complete")
        
        return val_metrics if 'val_metrics' in locals() else None
        
    except Exception as e:
        print(f"Error during validation mode: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "fast":
            run_fast_mode()
        elif mode == "validation":
            run_validation_mode()
        else:
            print("Usage:")
            print("  python main.py           # Full execution")
            print("  python main.py fast      # Fast execution")
            print("  python main.py validation # Validation mode")
    else:
        main()