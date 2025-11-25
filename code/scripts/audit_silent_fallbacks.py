#!/usr/bin/env python3
"""Audit script to find silent fallback patterns that should emit warnings.

This script identifies `except Exception: pass` patterns that silently hide
configuration failures. These should be replaced with explicit warnings.

Usage:
    python scripts/audit_silent_fallbacks.py [--fix]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple


# Categories of silent fallbacks
CATEGORIES = {
    "precision": [
        r"torch\.set_float32_matmul_precision",
        r"\.half\(\)",
        r"\.bfloat16\(\)",
        r"dtype.*=.*torch\.float16",
    ],
    "sdpa_backend": [
        r"enable_flash_sdp",
        r"enable_math_sdp", 
        r"enable_mem_efficient_sdp",
        r"enable_cudnn_sdp",
    ],
    "compile": [
        r"torch\.compile",
        r"compile_fn\(",
        r"compile_model\(",
    ],
    "tma_config": [
        r"enable_tma",
        r"tma_support",
    ],
}


def find_silent_fallbacks(filepath: Path) -> List[Tuple[int, str, str]]:
    """Find silent `except Exception: pass` patterns in a file.
    
    Returns list of (line_number, category, context) tuples.
    """
    try:
        content = filepath.read_text()
    except Exception:
        return []
    
    issues = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # Look for `except Exception:` followed by `pass`
        if 'except Exception:' in line or 'except Exception as' in line:
            # Check next non-empty line for `pass`
            for j in range(i + 1, min(i + 3, len(lines))):
                next_line = lines[j].strip()
                if next_line == 'pass':
                    # Look back for context
                    context_start = max(0, i - 5)
                    context = '\n'.join(lines[context_start:i + 1])
                    
                    # Categorize the fallback
                    category = "unknown"
                    for cat, patterns in CATEGORIES.items():
                        for pattern in patterns:
                            if re.search(pattern, context, re.IGNORECASE):
                                category = cat
                                break
                        if category != "unknown":
                            break
                    
                    issues.append((i + 1, category, context.strip()))
                    break
                elif next_line and not next_line.startswith('#'):
                    break
    
    return issues


def main() -> None:
    fix_mode = '--fix' in sys.argv
    
    repo_root = Path(__file__).parent.parent
    
    # Directories to search
    search_dirs = ['ch*/', 'labs/', 'common/', 'tools/']
    
    total_issues = 0
    files_with_issues = 0
    category_counts: dict[str, int] = {}
    
    for pattern in search_dirs:
        for filepath in repo_root.glob(f'{pattern}**/*.py'):
            if filepath.is_file():
                issues = find_silent_fallbacks(filepath)
                if issues:
                    files_with_issues += 1
                    print(f"\n{filepath.relative_to(repo_root)}:")
                    for line_no, category, context in issues:
                        total_issues += 1
                        category_counts[category] = category_counts.get(category, 0) + 1
                        print(f"  Line {line_no} [{category}]:")
                        # Show just first 2 lines of context
                        context_lines = context.split('\n')[-2:]
                        for ctx_line in context_lines:
                            print(f"    {ctx_line}")
    
    print(f"\n{'=' * 60}")
    print(f"Summary: {total_issues} silent fallbacks in {files_with_issues} files")
    print(f"\nBy category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    print(f"\nRecommendation: REMOVE try/except blocks - use fail-fast:")
    print("""
    # DON'T do this (AI slop):
        do_thing()
    
    # DO this instead:
    do_thing()  # Let it fail if something is wrong
    """)


if __name__ == '__main__':
    main()

