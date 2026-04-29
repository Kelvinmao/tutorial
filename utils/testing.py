"""
Simple test harness for chapter exercises.

Usage from any chapter directory:
    python -m pytest .          # if tests are in test_*.py files
    python ../../utils/testing.py   # run all *_test.py / test_*.py in cwd
"""

import importlib.util
import os
import sys
import traceback
from pathlib import Path


def run_script(script_path: str) -> bool:
    """
    Execute a Python script and return True if it exits cleanly.
    """
    print(f"  Running {script_path} ... ", end="", flush=True)
    spec = importlib.util.spec_from_file_location("__main__", script_path)
    if spec is None or spec.loader is None:
        print("SKIP (cannot load)")
        return True
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        print("OK")
        return True
    except SystemExit as e:
        if e.code == 0 or e.code is None:
            print("OK")
            return True
        print(f"FAIL (exit code {e.code})")
        return False
    except Exception:
        print("FAIL")
        traceback.print_exc()
        return False


def discover_and_run(directory: str = ".") -> int:
    """
    Discover and run all Python scripts in *directory* whose names
    do NOT start with '_' or 'test_'.  Returns count of failures.
    """
    directory = os.path.abspath(directory)
    scripts = sorted(
        p for p in Path(directory).glob("*.py")
        if not p.name.startswith("_")
    )
    if not scripts:
        print(f"No scripts found in {directory}")
        return 0

    print(f"=== Running scripts in {directory} ===")
    failures = 0
    for script in scripts:
        if not run_script(str(script)):
            failures += 1
    print(f"=== {len(scripts) - failures}/{len(scripts)} passed ===")
    return failures


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    failures = discover_and_run(target)
    sys.exit(1 if failures else 0)
