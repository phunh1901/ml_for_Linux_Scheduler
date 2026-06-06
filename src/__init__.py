"""
ML Engine for Linux Scheduler - Support Engine
A machine learning-based support engine for OS process scheduling.
"""

LABEL_MAP = {0: "IMMEDIATELY SCHEDULE", 1: "NEXT SCHEDULE", 2: "LATELY SCHEDULE"}
LABEL_NAMES = ['IMMEDIATELY', 'NEXT', 'LATELY']
RAW_FEATURE_COLUMNS = ['arrival_time', 'priority_level', 'cpu_utilization', 'memory_utilization', 'num_threads', 'process_status']
TARGET_COLUMN = 'target_schedule'
ENGINEERED_FEATURE = 'combined_load_score'

def get_all_feature_columns():
    """Return all feature columns including engineered features."""
    return RAW_FEATURE_COLUMNS + [ENGINEERED_FEATURE]
