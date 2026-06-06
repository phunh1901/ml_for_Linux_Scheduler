"""
pipeline.py - Enhanced Real-Time Inference Simulation for ML-Based Linux Scheduler

This module provides utilities to:
  - Generate realistic mock process data with varied resource profiles.
  - Simulate a real-time scheduler using a trained ML model for inference.
  - Compare ML-based predictions against a deterministic rule-based baseline.
"""

import numpy as np
import pandas as pd

from src import LABEL_MAP, LABEL_NAMES, get_all_feature_columns


# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------

def generate_mock_processes(n: int = 10) -> pd.DataFrame:
    """
    Generate *n* random mock processes with realistic feature distributions.

    The generated processes include a mix of random profiles plus two
    specific scenarios injected for realism:
      - 1 high-CPU process  (cpu > 60, priority = 4)
      - 1 idle process      (cpu < 1, priority = 0)

    Feature columns produced match the project's standard feature set
    (see ``src.get_all_feature_columns``).

    Args:
        n (int): Total number of mock processes to generate (must be >= 2).

    Returns:
        pd.DataFrame: DataFrame with one row per process and the standard
            feature columns.
    """
    if n < 2:
        n = 2  # Need at least 2 to inject both specific scenarios

    # --- Random base values for *all* n rows ---
    arrival_time = np.random.uniform(0, 10, n)
    priority_level = np.random.choice(
        [0, 1, 2, 3, 4, 5], size=n,
        p=[0.1, 0.1, 0.4, 0.15, 0.15, 0.1]
    )
    # CPU utilisation drawn from an exponential distribution, clipped to [0, 100]
    cpu_utilization = np.clip(np.random.exponential(5, n), 0, 100)
    memory_utilization = np.random.uniform(0.1, 50, n)
    num_threads = np.random.randint(1, 20, n)
    process_status = np.random.choice(
        [0, 1, 2, 3], size=n,
        p=[0.05, 0.1, 0.35, 0.5]
    )

    # --- Inject specific realistic scenarios ---
    # Scenario 1: High-CPU process (index 0)
    priority_level[0] = 4
    cpu_utilization[0] = np.random.uniform(60, 95)
    memory_utilization[0] = np.random.uniform(30, 60)
    num_threads[0] = np.random.randint(8, 20)
    process_status[0] = 3  # Running

    # Scenario 2: Idle process (index 1)
    priority_level[1] = 0
    cpu_utilization[1] = np.random.uniform(0, 1)
    memory_utilization[1] = np.random.uniform(0.1, 2)
    num_threads[1] = 1
    process_status[1] = 1  # Sleeping

    # --- Derived feature ---
    combined_load_score = (
        priority_level * 20.0
        + cpu_utilization * 1.2
        + memory_utilization * 0.5
    )

    df = pd.DataFrame({
        'arrival_time': arrival_time,
        'priority_level': priority_level,
        'cpu_utilization': cpu_utilization,
        'memory_utilization': memory_utilization,
        'num_threads': num_threads,
        'process_status': process_status,
        'combined_load_score': combined_load_score,
    })

    return df


# ---------------------------------------------------------------------------
# Real-time scheduler simulation
# ---------------------------------------------------------------------------

def simulate_realtime_scheduler(model, n_processes: int = 10) -> pd.DataFrame:
    """
    Simulate a real-time scheduler using a trained ML model.

    Generates mock processes, runs inference to predict the scheduling
    class for each process, and prints a formatted results table.

    Args:
        model: A fitted scikit-learn estimator with a ``predict`` method.
        n_processes (int): Number of mock processes to generate.

    Returns:
        pd.DataFrame: The mock-process DataFrame with an added
            ``predicted_schedule`` column.
    """
    print(f"\n{'=' * 72}")
    print(f"  REAL-TIME SCHEDULER SIMULATION  ({n_processes} processes)")
    print(f"{'=' * 72}")

    # Generate processes and select feature columns for prediction
    processes_df = generate_mock_processes(n=n_processes)
    feature_cols = get_all_feature_columns()
    X = processes_df[feature_cols]

    # Inference
    predictions = model.predict(X)
    processes_df['predicted_schedule'] = predictions

    # Pretty-print results
    print(f"\n  {'#':<4} {'CPU%':>7} {'MEM%':>7} {'PRI':>4} {'Threads':>8} "
          f"{'Status':>7} {'Load':>8}   {'Prediction'}")
    print(f"  {'-' * 70}")

    for idx, row in processes_df.iterrows():
        label_text = LABEL_MAP.get(int(row['predicted_schedule']), 'UNKNOWN')
        print(f"  {idx + 1:<4} "
              f"{row['cpu_utilization']:>7.2f} "
              f"{row['memory_utilization']:>7.2f} "
              f"{int(row['priority_level']):>4} "
              f"{int(row['num_threads']):>8} "
              f"{int(row['process_status']):>7} "
              f"{row['combined_load_score']:>8.2f}   "
              f"{label_text}")

    print(f"{'=' * 72}\n")
    return processes_df


# ---------------------------------------------------------------------------
# ML vs. rule-based comparison
# ---------------------------------------------------------------------------

def compare_ml_vs_rules(model, n_processes: int = 10) -> pd.DataFrame:
    """
    Compare ML model predictions against a simple rule-based scheduler.

    Rule-based labelling logic mirrors the heuristic used during data
    collection (see ``src.collect_data``):
      - cpu > 45 OR priority >= 4  →  0  (IMMEDIATELY)
      - cpu > 2  OR priority >= 2  →  1  (NEXT)
      - otherwise                  →  2  (LATELY)

    Args:
        model: A fitted scikit-learn estimator with a ``predict`` method.
        n_processes (int): Number of mock processes to generate.

    Returns:
        pd.DataFrame: Comparison DataFrame containing process features,
            ML predictions, rule-based labels, and a match flag.
    """
    print(f"\n{'=' * 72}")
    print(f"  ML vs RULE-BASED COMPARISON  ({n_processes} processes)")
    print(f"{'=' * 72}")

    # Generate processes
    processes_df = generate_mock_processes(n=n_processes)
    feature_cols = get_all_feature_columns()
    X = processes_df[feature_cols]

    # ML predictions
    ml_predictions = model.predict(X)
    processes_df['ml_prediction'] = ml_predictions

    # Rule-based predictions
    rule_predictions = []
    for _, row in processes_df.iterrows():
        cpu = row['cpu_utilization']
        pri = row['priority_level']
        if cpu > 45.0 or pri >= 4:
            rule_predictions.append(0)   # IMMEDIATELY
        elif cpu > 2.0 or pri >= 2:
            rule_predictions.append(1)   # NEXT
        else:
            rule_predictions.append(2)   # LATELY
    processes_df['rule_prediction'] = rule_predictions

    # Match flag
    processes_df['match'] = (
        processes_df['ml_prediction'] == processes_df['rule_prediction']
    )

    # Summary statistics
    total = len(processes_df)
    matches = processes_df['match'].sum()
    agreement_pct = (matches / total) * 100

    print(f"\n  Agreement: {matches}/{total} ({agreement_pct:.1f}%)\n")

    # Print detailed table
    print(f"  {'#':<4} {'CPU%':>7} {'PRI':>4}   {'ML Pred':<22} {'Rule Pred':<22} {'Match'}")
    print(f"  {'-' * 70}")

    for idx, row in processes_df.iterrows():
        ml_label = LABEL_MAP.get(int(row['ml_prediction']), 'UNKNOWN')
        rule_label = LABEL_MAP.get(int(row['rule_prediction']), 'UNKNOWN')
        match_str = 'YES' if row['match'] else '*** NO ***'
        print(f"  {idx + 1:<4} "
              f"{row['cpu_utilization']:>7.2f} "
              f"{int(row['priority_level']):>4}   "
              f"{ml_label:<22} "
              f"{rule_label:<22} "
              f"{match_str}")

    # Show disagreements if any
    disagreements = processes_df[~processes_df['match']]
    if len(disagreements) > 0:
        print(f"\n  [!] {len(disagreements)} disagreement(s) found between "
              f"ML and rule-based scheduler.")
    else:
        print(f"\n  [OK] Perfect agreement between ML and rule-based scheduler.")

    print(f"{'=' * 72}\n")
    return processes_df