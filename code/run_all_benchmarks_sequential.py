#!/usr/bin/env python3
"""
Script to run full benchmarks with deep profiling on EVERY chapter and lab,
ONE at a time, sequentially.

Requirements:
- NO LLM analysis
- Deep profiling (deep_dive)
- Sequential execution (one chapter/lab at a time)
- NO parallel execution
"""

import subprocess
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime

# Get the list of all chapters and labs
def get_all_chapters_and_labs():
    """Get list of all chapters and labs from the system."""
    try:
        result = subprocess.run(
            ["python", "-m", "cli.aisp", "bench", "list-chapters"],
            capture_output=True,
            text=True,
            check=True
        )
        chapters = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return chapters
    except subprocess.CalledProcessError as e:
        print(f"Error getting chapters list: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)

def run_benchmark_for_chapter(chapter, log_file, update_expectations=False, accept_regressions=False, 
                              cold_start=False, reproducible=False, suite_timeout=None, timeout_multiplier=None):
    """
    Run benchmark with deep profiling for a single chapter/lab.
    
    Args:
        chapter: Chapter/lab name (e.g., 'ch01', 'labs/decode_optimization')
        log_file: File handle to write logs to
        update_expectations: If True, add --update-expectations flag
        accept_regressions: If True, add --accept-regressions flag
        cold_start: If True, add --cold-start flag (reset GPU state between benchmarks)
        reproducible: If True, add --reproducible flag (set seeds to 42)
        suite_timeout: Optional timeout in seconds for the suite
        timeout_multiplier: Optional multiplier for benchmark timeouts
    """
    print(f"\n{'='*80}")
    print(f"Starting benchmark for: {chapter}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Starting benchmark for: {chapter}\n")
    log_file.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'='*80}\n")
    log_file.flush()
    
    start_time = time.time()
    
    try:
        # Run benchmark with deep profiling, NO LLM analysis (default is False)
        cmd = [
            "python", "-m", "cli.aisp", "bench", "run",
            "--targets", chapter,
            "--profile", "deep_dive"
            # Note: llm_analysis defaults to False, so we don't need to disable it
        ]
        
        # Add expectation update flags if requested
        if update_expectations:
            cmd.append("--update-expectations")
        if accept_regressions:
            cmd.append("--accept-regressions")
        if cold_start:
            cmd.append("--cold-start")
        if reproducible:
            cmd.append("--reproducible")
        if suite_timeout is not None:
            cmd.extend(["--suite-timeout", str(suite_timeout)])
        if timeout_multiplier is not None:
            cmd.extend(["--timeout-multiplier", str(timeout_multiplier)])
        
        print(f"Command: {' '.join(cmd)}")
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.flush()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't fail on non-zero exit, we'll handle it
        )
        
        duration = time.time() - start_time
        
        print(f"\nReturn code: {result.returncode}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        log_file.write(f"\nReturn code: {result.returncode}\n")
        log_file.write(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)\n")
        log_file.write(f"\n--- STDOUT ---\n{result.stdout}\n")
        log_file.write(f"\n--- STDERR ---\n{result.stderr}\n")
        log_file.flush()
        
        if result.returncode != 0:
            print(f"WARNING: Benchmark for {chapter} returned non-zero exit code!")
            log_file.write(f"WARNING: Benchmark for {chapter} returned non-zero exit code!\n")
            log_file.flush()
            return False
        else:
            print(f"SUCCESS: Benchmark for {chapter} completed successfully!")
            log_file.write(f"SUCCESS: Benchmark for {chapter} completed successfully!\n")
            log_file.flush()
            return True
            
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"ERROR running benchmark for {chapter}: {str(e)}"
        print(error_msg)
        log_file.write(f"{error_msg}\n")
        log_file.write(f"Duration: {duration:.2f} seconds\n")
        log_file.flush()
        return False

def main():
    """Main function to run benchmarks sequentially."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run full benchmarks with deep profiling on every chapter and lab sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--update-expectations",
        action="store_true",
        help="Force-write observed metrics into expectation files (overrides regressions)"
    )
    parser.add_argument(
        "--accept-regressions",
        action="store_true",
        help="Update expectation files when improvements are detected instead of flagging regressions"
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Reset GPU state between benchmarks for cold start measurements"
    )
    parser.add_argument(
        "--reproducible",
        action="store_true",
        help="Enable reproducible mode: set all seeds to 42 and force deterministic algorithms"
    )
    parser.add_argument(
        "--suite-timeout",
        type=int,
        default=None,
        help="Suite timeout in seconds (default: 14400 = 4 hours, 0 = disabled)"
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=None,
        help="Multiply all benchmark timeouts by this factor (e.g., 2.0 = double all timeouts)"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("Sequential Benchmark Runner with Deep Profiling")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Profile: deep_dive (always enabled)")
    if args.update_expectations:
        print("Mode: --update-expectations enabled")
    if args.accept_regressions:
        print("Mode: --accept-regressions enabled")
    if args.cold_start:
        print("Mode: --cold-start enabled")
    if args.reproducible:
        print("Mode: --reproducible enabled")
    if args.suite_timeout is not None:
        print(f"Suite timeout: {args.suite_timeout} seconds")
    if args.timeout_multiplier is not None:
        print(f"Timeout multiplier: {args.timeout_multiplier}x")
    
    # Get all chapters and labs
    print("\nFetching list of chapters and labs...")
    chapters = get_all_chapters_and_labs()
    
    print(f"\nFound {len(chapters)} chapters/labs to process:")
    for i, ch in enumerate(chapters, 1):
        print(f"  {i:3d}. {ch}")
    
    # Create log file
    log_dir = Path("artifacts")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"sequential_benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    print(f"\nLog file: {log_file_path}")
    print(f"\n{'='*80}")
    print("TO MONITOR PROGRESS IN ANOTHER TERMINAL, RUN:")
    print(f"tail -f {log_file_path}")
    print(f"{'='*80}\n")
    
    # Track results
    results = {
        "start_time": datetime.now().isoformat(),
        "chapters": [],
        "successful": 0,
        "failed": 0,
        "skipped": 0
    }
    
    with open(log_file_path, 'w') as log_file:
        log_file.write("Sequential Benchmark Runner with Deep Profiling\n")
        log_file.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("Profile: deep_dive (always enabled)\n")
        if args.update_expectations:
            log_file.write("Mode: --update-expectations enabled\n")
        if args.accept_regressions:
            log_file.write("Mode: --accept-regressions enabled\n")
        if args.cold_start:
            log_file.write("Mode: --cold-start enabled\n")
        if args.reproducible:
            log_file.write("Mode: --reproducible enabled\n")
        if args.suite_timeout is not None:
            log_file.write(f"Suite timeout: {args.suite_timeout} seconds\n")
        if args.timeout_multiplier is not None:
            log_file.write(f"Timeout multiplier: {args.timeout_multiplier}x\n")
        log_file.write(f"Total chapters/labs: {len(chapters)}\n")
        log_file.write(f"Chapters/labs list:\n")
        for ch in chapters:
            log_file.write(f"  - {ch}\n")
        log_file.write("\n")
        log_file.flush()
        
        # Process each chapter/lab sequentially
        for idx, chapter in enumerate(chapters, 1):
            print(f"\n\nProcessing {idx}/{len(chapters)}: {chapter}")
            
            chapter_start_time = time.time()
            success = run_benchmark_for_chapter(
                chapter, 
                log_file,
                update_expectations=args.update_expectations,
                accept_regressions=args.accept_regressions,
                cold_start=args.cold_start,
                reproducible=args.reproducible,
                suite_timeout=args.suite_timeout,
                timeout_multiplier=args.timeout_multiplier
            )
            chapter_duration = time.time() - chapter_start_time
            
            result_entry = {
                "chapter": chapter,
                "index": idx,
                "success": success,
                "duration_seconds": chapter_duration,
                "timestamp": datetime.now().isoformat()
            }
            results["chapters"].append(result_entry)
            
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
            
            # Write intermediate results
            results_file = log_dir / f"sequential_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Small delay between runs to ensure clean state
            if idx < len(chapters):
                print(f"\nWaiting 5 seconds before next chapter/lab...")
                time.sleep(5)
    
    # Final summary
    results["end_time"] = datetime.now().isoformat()
    total_duration = sum(ch["duration_seconds"] for ch in results["chapters"])
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total chapters/labs processed: {len(chapters)}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes, {total_duration/3600:.2f} hours)")
    print(f"Start time: {results['start_time']}")
    print(f"End time: {results['end_time']}")
    print(f"\nLog file: {log_file_path}")
    print(f"Results JSON: {results_file}")
    print("="*80)
    
    # Write final results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Show how to tail the log file
    print(f"\n{'='*80}")
    print("TO MONITOR PROGRESS, RUN:")
    print(f"tail -f {log_file_path}")
    print(f"{'='*80}\n")
    
    # Exit with error code if any failed
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

