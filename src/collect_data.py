"""
collect_data.py - Real-Time Process Data Collector

Collects live process telemetry from the host system using psutil.
Works on both Windows and Linux, with platform-specific priority mapping.
The collected data is labeled for supervised learning based on scheduling urgency.
"""

import os
import time
import platform
import pandas as pd
import psutil

from src import TARGET_COLUMN


# ---------------------------------------------------------------------------
# Platform-aware priority mapping
# ---------------------------------------------------------------------------

# Windows priority class constants returned by proc.nice()
# Maps Windows priority class values to a normalized 0-5 score.
WINDOWS_PRIORITY_MAP = {
    psutil.IDLE_PRIORITY_CLASS: 0,            # 64
    psutil.BELOW_NORMAL_PRIORITY_CLASS: 1,    # 16384
    psutil.NORMAL_PRIORITY_CLASS: 2,          # 32
    psutil.ABOVE_NORMAL_PRIORITY_CLASS: 3,    # 32768
    psutil.HIGH_PRIORITY_CLASS: 4,            # 128
    psutil.REALTIME_PRIORITY_CLASS: 5,        # 256
}


def _map_priority(nice_value: int) -> int:
    """
    Convert a platform-specific nice/priority value to a unified 0-5 score.

    On Windows, proc.nice() returns a priority class constant; on Linux it
    returns a nice value in the range -20 (highest) to 19 (lowest).

    Parameters
    ----------
    nice_value : int
        Raw value from psutil.Process.nice().

    Returns
    -------
    int
        Normalized priority score between 0 (lowest) and 5 (highest).
    """
    if platform.system() == "Windows":
        return WINDOWS_PRIORITY_MAP.get(nice_value, 2)  # default to NORMAL
    else:
        # Linux: nice ranges from -20 (highest priority) to 19 (lowest)
        # Map linearly to 0-5 score (inverted: lower nice = higher priority)
        clamped = max(-20, min(19, nice_value))
        return int(round((19 - clamped) / 39.0 * 5))


# ---------------------------------------------------------------------------
# Process status encoding
# ---------------------------------------------------------------------------

STATUS_ENCODING = {
    "running": 3,
    "sleeping": 2,
    "disk-sleep": 2,
    "stopped": 1,
    "idle": 1,
    "zombie": 0,
    "dead": 0,
}


def _encode_status(status_str: str) -> int:
    """
    Encode a process status string into a numeric value.

    Parameters
    ----------
    status_str : str
        Process status string from psutil (e.g., 'running', 'sleeping').

    Returns
    -------
    int
        Encoded integer: running=3, sleeping=2, stopped/idle=1, zombie/dead=0.
    """
    return STATUS_ENCODING.get(status_str, 0)


# ---------------------------------------------------------------------------
# Labeling logic
# ---------------------------------------------------------------------------

def _assign_label(cpu_util: float, priority_score: int) -> int:
    """
    Assign a scheduling urgency label based on CPU utilization and priority.

    Labels
    ------
    0 - IMMEDIATELY SCHEDULE : high CPU usage (>45%) or high priority (>=4)
    1 - NEXT SCHEDULE        : moderate CPU usage (>2%) or moderate priority (>=2)
    2 - LATELY SCHEDULE      : low resource usage and low priority

    Parameters
    ----------
    cpu_util : float
        CPU utilization percentage of the process.
    priority_score : int
        Normalized priority score (0-5).

    Returns
    -------
    int
        Scheduling label (0, 1, or 2).
    """
    if cpu_util > 45 or priority_score >= 4:
        return 0  # IMMEDIATELY
    elif cpu_util > 2 or priority_score >= 2:
        return 1  # NEXT
    else:
        return 2  # LATELY


# ---------------------------------------------------------------------------
# Main collection routine
# ---------------------------------------------------------------------------

def run_data_collection(
    output_path: str,
    duration: int = 120,
    interval: int = 1,
) -> pd.DataFrame:
    """
    Collect real-time process data and save to CSV.

    Iterates over all running processes at regular intervals, extracting
    scheduling-relevant features. A CPU warm-up pass is performed first so
    that ``cpu_percent()`` returns meaningful values on the first real sample.

    Parameters
    ----------
    output_path : str
        File path where the resulting CSV will be saved.
    duration : int, optional
        Total collection time in seconds (default 120).
    interval : int, optional
        Seconds between collection sweeps (default 1).

    Returns
    -------
    pd.DataFrame
        DataFrame containing all collected samples.
    """
    current_os = platform.system()
    print(f"[INFO] Detected OS: {current_os}")
    print(f"[INFO] Collection settings: duration={duration}s, interval={interval}s")
    print(f"[INFO] Output path: {output_path}")

    # ------------------------------------------------------------------
    # CPU warm-up: call cpu_percent() once on every process so the next
    # call returns a real delta rather than 0.0.
    # ------------------------------------------------------------------
    print("[INFO] Performing CPU warm-up pass ...")
    for proc in psutil.process_iter():
        try:
            proc.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    time.sleep(1)
    print("[INFO] Warm-up complete. Starting data collection ...")

    records = []
    start_time = time.time()
    sample_count = 0

    while (time.time() - start_time) < duration:
        for proc in psutil.process_iter():
            try:
                with proc.oneshot():
                    # Arrival time: seconds elapsed since collection start
                    arrival_time = round(time.time() - start_time, 4)

                    # Priority
                    nice_value = proc.nice()
                    priority_score = _map_priority(nice_value)

                    # Resource utilisation
                    cpu_util = proc.cpu_percent()
                    mem_util = proc.memory_percent()

                    # Thread count
                    num_threads = proc.num_threads()

                    # Process status
                    status_str = proc.status()
                    status_code = _encode_status(status_str)

                    # Target label
                    label = _assign_label(cpu_util, priority_score)

                    records.append({
                        "arrival_time": arrival_time,
                        "priority_level": priority_score,
                        "cpu_utilization": round(cpu_util, 2),
                        "memory_utilization": round(mem_util, 2),
                        "num_threads": num_threads,
                        "process_status": status_code,
                        TARGET_COLUMN: label,
                    })

                    sample_count += 1
                    if sample_count % 10 == 0:
                        print(".", end="", flush=True)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process disappeared or is inaccessible — skip silently
                continue

        time.sleep(interval)

    # ------------------------------------------------------------------
    # Build DataFrame and persist
    # ------------------------------------------------------------------
    print()  # newline after progress dots
    df = pd.DataFrame(records)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    print(f"\n[INFO] Data collection complete.")
    print(f"[INFO] Total records collected: {len(df)}")
    if not df.empty:
        print(f"[INFO] Class distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}")
    print(f"[INFO] Data saved to: {output_path}")

    return df