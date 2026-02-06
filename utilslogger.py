import csv
import os
import datetime
from typing import Dict, Any, Iterable


LOG_FILE = "llm_prediction_log.csv"

FIELDNAMES = [
    "Iteration",
    "Component_Name",
    "LLM_Selected",
    "LLM_Args",
    "Success",
    "Old_Cost",
    "New_Cost",
    "Delta",
    "Duration_ms",
    "Tour_Before",
    "Tour_After",
    "Timestamp",
]

def _header_needed(path: str, expected_first: str = "Iteration") -> bool:
    """Header is needed if the file doesn't exist or is empty (size=0)."""
    try:    
        if not os.path.exists(path):
                return True
        if os.path.getsize(path) == 0:
            return True
        # Peek first non-empty line
        with open(path, mode="r", encoding="utf-8", newline="") as f:
            first = f.readline().strip()
        return (expected_first not in first)  # very light check
    except Exception:
        # Be safe: if anything goes wrong, (re)write header
        return True

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    
def init_csv_log(log_file: str = LOG_FILE) -> None:
    """Create CSV log file with header if it does not exist yet."""
    _ensure_parent_dir(log_file)
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def log_iteration(
    iteration: int,
    component_name: str,
    llm_selected: bool,
    llm_args: dict,
    success: bool,
    old_cost: float,
    new_cost: float,
    duration_ms: float,
    tour_before,
    tour_after,
    log_file: str = LOG_FILE,
) -> None:
    """Append one iteration’s data to the CSV log."""
    _ensure_parent_dir(log_file)

    # Write header on demand
    if _header_needed(log_file):
        with open(log_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
    delta = new_cost - old_cost
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(
            {
                "Iteration": iteration,
                "Component_Name": component_name,
                "LLM_Selected": llm_selected,
                "LLM_Args": str(llm_args),
                "Success": success,
                "Old_Cost": old_cost,
                "New_Cost": new_cost,
                "Delta": delta,
                "Duration_ms": round(duration_ms, 3),
                "Tour_Before": tour_before,
                "Tour_After": tour_after,
                "Timestamp": datetime.datetime.now().isoformat(),
            }
        )