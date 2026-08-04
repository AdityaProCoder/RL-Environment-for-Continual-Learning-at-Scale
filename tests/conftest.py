"""
Pytest configuration and environment isolation setup.
Ensures sys.path isolates virtual environment dependencies from conflicting global Python installations.
"""

import sys
import os

# Filter out incompatible host Python site-packages from sys.path if present
current_py_tag = f"Python{sys.version_info.major}{sys.version_info.minor}"
cleaned_path = []
for path in sys.path:
    if "AppData\\Local\\Programs\\Python\\Python" in path and current_py_tag not in path:
        continue
    if "AppData/Local/Programs/Python/Python" in path and current_py_tag not in path:
        continue
    cleaned_path.append(path)

sys.path[:] = cleaned_path
