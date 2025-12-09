#!/usr/bin/env python3
"""Fix ch10 CudaBinaryBenchmark files for compliance."""

import re
from pathlib import Path

CH10_DIR = Path(__file__).parent / "ch10"

def fix_file(filepath: Path) -> bool:
    """Fix a single file. Returns True if changes were made."""
    content = filepath.read_text()
    original = content
    
    # Skip if not a CudaBinaryBenchmark or already has workload_params
    if "CudaBinaryBenchmark" not in content:
        return False
    if "workload_params=" in content:
        return False
    
    # Get type name from filename
    type_name = filepath.stem.replace("baseline_", "").replace("optimized_", "")
    
    # Pattern 1: timeout_seconds=XXX, followed by closing )
    pattern1 = r'(timeout_seconds=\d+,)\s*\)'
    replacement1 = rf'\1\n            workload_params={{"type": "{type_name}"}},\n        )\n        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)'
    
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
    else:
        # Pattern 2: time_regex=XXX, followed by closing )
        pattern2 = r'(time_regex=r"[^"]+",)\s*\)'
        replacement2 = rf'\1\n            workload_params={{"type": "{type_name}"}},\n        )\n        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)'
        
        if re.search(pattern2, content):
            content = re.sub(pattern2, replacement2, content)
        else:
            # Pattern 3: require_tma_instructions=True/False, followed by closing )
            pattern3 = r'(require_tma_instructions=(True|False),)\s*\)'
            replacement3 = rf'\1\n            workload_params={{"type": "{type_name}"}},\n        )\n        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)'
            
            if re.search(pattern3, content):
                content = re.sub(pattern3, replacement3, content)
    
    if content != original:
        filepath.write_text(content)
        return True
    return False


def main():
    fixed = 0
    for py_file in sorted(CH10_DIR.glob("*.py")):
        if py_file.name.startswith("__") or py_file.name == "fix_ch10.py":
            continue
        try:
            if fix_file(py_file):
                print(f"Fixed: {py_file.name}")
                fixed += 1
        except Exception as e:
            print(f"Error fixing {py_file.name}: {e}")
    
    print(f"\nFixed {fixed} files")


if __name__ == "__main__":
    main()
