#!/usr/bin/env python3
"""
ML Engine for Linux Scheduler - Main Orchestrator
==================================================
Runs the full ML pipeline: data collection, preprocessing, visualization,
model training, evaluation, and real-time simulation.

Usage:
    python main.py
    python main.py --duration 60 --interval 2
"""

import os
import sys
import argparse
import time

# ============================================================
# Import project modules
# ============================================================
from src.collect_data import run_data_collection
from src.preprocess import preprocess_pipeline
from src.visualize import (
    generate_all_eda_plots,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
)
from src.train import prepare_data, train_all_models, save_model
from src.evaluate import (
    evaluate_all_models,
    get_feature_importance,
    generate_comparison_report,
)
from src.pipeline import simulate_realtime_scheduler, compare_ml_vs_rules
from src import get_all_feature_columns


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ML Engine for Linux Scheduler - Full Pipeline"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Duration in seconds for data collection (default: 120)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Sampling interval in seconds for data collection (default: 1)",
    )
    return parser.parse_args()


def main():
    """Run the complete ML pipeline."""
    args = parse_args()

    # --------------------------------------------------------
    # Define paths
    # --------------------------------------------------------
    RAW_DATA = os.path.join("data", "raw", "system_metrics.csv")
    PROCESSED_DATA = os.path.join("data", "processed", "clean_metrics.csv")
    MODEL_DIR = os.path.join("models")
    FIGURES_DIR = os.path.join("output", "figures")
    RESULTS_DIR = os.path.join("output", "results")

    # Create directories
    for dir_path in [
        os.path.dirname(RAW_DATA),
        os.path.dirname(PROCESSED_DATA),
        MODEL_DIR,
        FIGURES_DIR,
        RESULTS_DIR,
    ]:
        os.makedirs(dir_path, exist_ok=True)

    print("=" * 70)
    print("ML ENGINE FOR LINUX SCHEDULER - FULL PIPELINE")
    print("=" * 70)
    print(f"Duration : {args.duration}s")
    print(f"Interval : {args.interval}s")
    print("=" * 70)

    pipeline_start = time.time()

    # ========================================================
    # Step 1: Collect Data
    # ========================================================
    try:
        print("\n[Step 1/9] Collecting system process data...")
        print(f"  Duration: {args.duration}s | Interval: {args.interval}s")
        df_raw = run_data_collection(
            output_path=RAW_DATA,
            duration=args.duration,
            interval=args.interval,
        )
        print(f"  -> Collected {len(df_raw)} raw records -> {RAW_DATA}")
    except Exception as e:
        print(f"  [ERROR] Data collection failed: {e}")
        sys.exit(1)

    # ========================================================
    # Step 2: Preprocess Data
    # ========================================================
    try:
        print("\n[Step 2/9] Preprocessing data...")
        df_processed = preprocess_pipeline(RAW_DATA, PROCESSED_DATA)
        print(f"  -> Processed dataset: {df_processed.shape[0]} rows, "
              f"{df_processed.shape[1]} columns -> {PROCESSED_DATA}")
    except Exception as e:
        print(f"  [ERROR] Preprocessing failed: {e}")
        sys.exit(1)

    # ========================================================
    # Step 3: Generate EDA Visualizations
    # ========================================================
    try:
        print("\n[Step 3/9] Generating EDA visualizations...")
        generate_all_eda_plots(df_processed, output_dir=FIGURES_DIR)
        print(f"  -> Plots saved to {FIGURES_DIR}")
    except Exception as e:
        print(f"  [WARNING] Visualization generation failed: {e}")
        print("  Continuing with training...")

    # ========================================================
    # Step 4: Prepare Data and Train Models
    # ========================================================
    try:
        print("\n[Step 4/9] Preparing data (70/30 split) and training models...")
        X_train, X_test, y_train, y_test = prepare_data(
            df_processed, test_size=0.30, random_state=42
        )
        print(f"  Training set : {X_train.shape[0]} samples")
        print(f"  Test set     : {X_test.shape[0]} samples")

        trained_models = train_all_models(X_train, y_train, include_kmeans=False)
        print(f"  -> Successfully trained {len(trained_models)} models")
    except Exception as e:
        print(f"  [ERROR] Training failed: {e}")
        sys.exit(1)

    # ========================================================
    # Step 5: Evaluate All Models
    # ========================================================
    try:
        print("\n[Step 5/9] Evaluating all models...")
        results_df = evaluate_all_models(trained_models, X_test, y_test)
        print("\n" + "=" * 70)
        print("MODEL COMPARISON TABLE")
        print("=" * 70)
        print(results_df.to_string(index=False))
        print("=" * 70)

        # Generate comparison report
        report_path = generate_comparison_report(results_df, output_dir=RESULTS_DIR)
        print(f"  -> Comparison report saved to: {report_path}")
    except Exception as e:
        print(f"  [ERROR] Evaluation failed: {e}")
        sys.exit(1)

    # ========================================================
    # Step 6: Best Model - Confusion Matrix & Feature Importance
    # ========================================================
    try:
        print("\n[Step 6/9] Generating best model analysis plots...")
        best_model_name = results_df.iloc[0]["model_name"]
        best_model = trained_models[best_model_name]
        best_accuracy = results_df.iloc[0]["accuracy"]
        print(f"  Best model: {best_model_name} (accuracy={best_accuracy:.4f})")

        # Confusion matrix
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        y_pred = best_model.predict(X_test)
        plot_confusion_matrix(
            y_test,
            y_pred,
            title=f"Confusion Matrix - {best_model_name}",
            save_path=os.path.join(FIGURES_DIR, "confusion_matrix_best.png"),
        )
        print(f"  -> Confusion matrix saved")

        # Feature importance
        feature_names = get_all_feature_columns()
        importances = get_feature_importance(best_model, feature_names)
        if importances is not None:
            plot_feature_importance(
                importances.values,
                importances.index.tolist(),
                title=f"Feature Importance - {best_model_name}",
                save_path=os.path.join(FIGURES_DIR, "feature_importance_best.png"),
            )
            print(f"  -> Feature importance plot saved")
        else:
            print("  -> Feature importance not available for this model type")

        # Model comparison chart
        plot_model_comparison(
            results_df,
            metric="accuracy",
            save_path=os.path.join(FIGURES_DIR, "model_comparison_accuracy.png"),
        )
        print(f"  -> Model comparison chart saved")
    except Exception as e:
        print(f"  [WARNING] Best model analysis failed: {e}")

    # ========================================================
    # Step 7: Save Best Model
    # ========================================================
    try:
        print("\n[Step 7/9] Saving best model...")
        model_filename = best_model_name.lower().replace(" ", "_") + "_scheduler.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)
        save_model(best_model, model_path)
        print(f"  -> Model saved to: {model_path}")
    except Exception as e:
        print(f"  [ERROR] Model saving failed: {e}")

    # ========================================================
    # Step 8: Real-Time Simulation
    # ========================================================
    try:
        print("\n[Step 8/9] Running real-time scheduler simulation...")
        print("-" * 50)
        simulation_df = simulate_realtime_scheduler(best_model, n_processes=10)
        print("-" * 50)
    except Exception as e:
        print(f"  [WARNING] Real-time simulation failed: {e}")

    # ========================================================
    # Step 9: ML vs Rule-Based Comparison
    # ========================================================
    try:
        print("\n[Step 9/9] Comparing ML model vs rule-based scheduler...")
        print("-" * 50)
        comparison = compare_ml_vs_rules(best_model, n_processes=15)
        print("-" * 50)
    except Exception as e:
        print(f"  [WARNING] ML vs rule-based comparison failed: {e}")

    # ========================================================
    # Final Summary
    # ========================================================
    elapsed = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Best Model     : {best_model_name}")
    print(f"  Best Accuracy  : {best_accuracy:.4f}")
    print(f"  Total Time     : {elapsed:.1f}s")
    print(f"  Raw Data       : {RAW_DATA}")
    print(f"  Processed Data : {PROCESSED_DATA}")
    print(f"  Model File     : {model_path}")
    print(f"  Figures        : {FIGURES_DIR}")
    print(f"  Results        : {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()