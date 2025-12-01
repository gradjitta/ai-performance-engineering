#!/usr/bin/env python3
"""Migrate all benchmark files to use the new safe benchmark_main() pattern.

This script finds all benchmark files using the old error-prone pattern and
replaces them with the new safe pattern.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_benchmark_files() -> list[Path]:
    """Find all Python files with the old __main__ pattern."""
    files = []
    
    # Search in ch* and labs directories
    for pattern in ['ch*/*.py', 'ch*/**/*.py', 'labs/**/*.py']:
        for f in REPO_ROOT.glob(pattern):
            if f.is_file():
                content = f.read_text()
                # Check for old pattern - has BenchmarkHarness and __main__
                if 'if __name__' in content and ('BenchmarkHarness' in content or 'harness.benchmark(' in content):
                    # Skip files that already use ONLY the new pattern
                    if 'benchmark_main(' in content:
                        # Make sure there's no OLD pattern still present
                        if 'harness.benchmark(' not in content:
                            continue
                    files.append(f)
    
    return sorted(set(files))


def migrate_file(filepath: Path, dry_run: bool = False) -> bool:
    """Migrate a single file to the new pattern."""
    content = filepath.read_text()
    original_content = content
    
    # Skip if already fully migrated (has benchmark_main and no harness.benchmark in __main__)
    if 'benchmark_main(' in content:
        # Check if there's still old pattern in __main__ block
        main_match = re.search(r"if __name__ == ['\"]__main__['\"]:\s*\n(.*?)$", content, re.DOTALL)
        if main_match and 'harness.benchmark(' not in main_match.group(1):
            return False
    
    # Find the __main__ block
    main_match = re.search(r"if __name__ == ['\"]__main__['\"]:\s*\n(.*?)$", content, re.DOTALL)
    
    if not main_match:
        return False
    
    main_block = main_match.group(0)
    
    # Check if this needs migration
    if 'BenchmarkHarness' not in content and 'harness.benchmark(' not in content:
        return False
    
    # Determine factory function
    if 'get_benchmark()' in content or 'get_benchmark(' in content:
        factory = 'get_benchmark'
    else:
        # Look for Benchmark class
        match = re.search(r'class (\w+Benchmark)', content)
        if match:
            factory = match.group(1)
        else:
            factory = 'get_benchmark'
    
    # Remove old main() function if it exists and only does benchmark stuff
    main_func_match = re.search(r'\ndef main\(\)[^:]*:\s*"""[^"]*""".*?(?=\ndef |\nclass |\nif __name__|$)', content, re.DOTALL)
    if main_func_match:
        func_content = main_func_match.group(0)
        # Only remove if it's just doing benchmark stuff
        if 'BenchmarkHarness' in func_content and 'harness.benchmark(' in func_content:
            content = content[:main_func_match.start()] + content[main_func_match.end():]
    
    # Build new __main__ block
    new_main = f'''if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main({factory})
'''
    
    # Find and replace __main__ block
    main_match = re.search(r"if __name__ == ['\"]__main__['\"]:\s*\n(.*?)$", content, re.DOTALL)
    if main_match:
        content = content[:main_match.start()] + new_main
    
    # Clean up any trailing whitespace
    content = content.rstrip() + '\n'
    
    if content == original_content:
        return False
    
    if dry_run:
        print(f"Would update: {filepath}")
        return True
    
    filepath.write_text(content)
    print(f"Updated: {filepath}")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate benchmark files to safe pattern')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed')
    parser.add_argument('--file', type=str, help='Migrate a specific file')
    args = parser.parse_args()
    
    if args.file:
        files = [Path(args.file)]
    else:
        files = find_benchmark_files()
    
    print(f"Found {len(files)} files to migrate")
    
    migrated = 0
    for f in files:
        try:
            if migrate_file(f, args.dry_run):
                migrated += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    print(f"\n{'Would migrate' if args.dry_run else 'Migrated'} {migrated} files")


if __name__ == '__main__':
    main()
