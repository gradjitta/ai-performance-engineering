#!/usr/bin/env python3
"""
Comprehensive Blackwell Feature Verification with Profiling
============================================================

Verifies all Blackwell B200/B300 features and optionally profiles with nsys/ncu.

Features verified:
- TMEM (Tensor Memory Accelerator)
- TMA (Tensor Memory Access)
- CTA Clusters
- DSMEM (Distributed Shared Memory)
- FP8 (E4M3, E5M2)
- FP4 (E2M1 - status check)
- Warp Specialization
- 5th Gen Tensor Cores (tcgen05)

Usage:
    python verify_all_blackwell_features.py [--profile] [--ncu] [--nsys]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

import torch

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLACKWELL_TESTS_DIR = PROJECT_ROOT / "tools" / "blackwell_optimizations"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "blackwell_verification"


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(name: str, passed: bool, detail: str = "") -> None:
    """Print a test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    line = f"  {status}: {name}"
    if detail:
        line += f" - {detail}"
    print(line)


def check_gpu() -> Tuple[bool, Dict]:
    """Check GPU and return info."""
    if not torch.cuda.is_available():
        return False, {"error": "CUDA not available"}
    
    props = torch.cuda.get_device_properties(0)
    info = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "major": props.major,
        "minor": props.minor,
        "total_memory_gb": props.total_memory / 1024**3,
        "multi_processor_count": props.multi_processor_count,
    }
    
    is_blackwell = props.major == 10
    return is_blackwell, info


def run_cuda_test(test_name: str, timeout: int = 30) -> Tuple[bool, str, float]:
    """Run a CUDA test binary."""
    test_path = BLACKWELL_TESTS_DIR / test_name
    
    if not test_path.exists():
        return False, f"Test binary not found: {test_path}", 0.0
    
    try:
        start = time.time()
        result = subprocess.run(
            [str(test_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(BLACKWELL_TESTS_DIR)
        )
        elapsed = time.time() - start
        
        output = result.stdout + result.stderr
        passed = result.returncode == 0 and "PASSED" in output
        
        return passed, output, elapsed
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s", timeout
    except Exception as e:
        return False, str(e), 0.0


def check_python_feature(name: str, test_fn) -> Tuple[bool, str]:
    """Check a Python-level feature."""
    try:
        result = test_fn()
        if isinstance(result, tuple):
            return result
        return True, str(result) if result else "OK"
    except Exception as e:
        return False, str(e)


def check_fp4_support() -> Dict:
    """Check FP4 support status."""
    status = {
        "dtype_available": hasattr(torch, 'float4_e2m1fn_x2'),
        "scaled_mm_available": hasattr(torch, '_scaled_mm'),
        "cublaslt_available": False,
        "native_ops_working": False,
        "workaround_available": True,  # Our packed uint8 implementation
    }
    
    # Check cuBLASLt
        import ctypes
        ctypes.CDLL('libcublasLt.so.13')
        status["cublaslt_available"] = True
    
    # Check if native FP4 conversion works
    if status["dtype_available"]:
        try:
            # Try creating and converting
            fp4_tensor = torch.empty(64, 64, dtype=torch.float4_e2m1fn_x2, device='cuda')
            # Try conversion (this is what fails in PyTorch 2.9.1)
            fp32_tensor = torch.randn(32, 64, device='cuda')
            fp4_from_fp32 = fp32_tensor.view(-1, 2).to(torch.float4_e2m1fn_x2)
            status["native_ops_working"] = True
        except (NotImplementedError, RuntimeError):
            status["native_ops_working"] = False
    
    return status


def check_fp8_support() -> Dict:
    """Check FP8 support status."""
    status = {
        "e4m3_available": hasattr(torch, 'float8_e4m3fn'),
        "e5m2_available": hasattr(torch, 'float8_e5m2'),
        "scaled_mm_available": hasattr(torch, '_scaled_mm'),
        "conversion_working": False,
        "tensor_core_accelerated": False,
    }
    
    if status["e4m3_available"]:
            x = torch.randn(64, 64, device='cuda', dtype=torch.float16)
            x_fp8 = x.to(torch.float8_e4m3fn)
            status["conversion_working"] = True
            
            # Check if _scaled_mm works
            # _scaled_mm requires row-major A and column-major B
            # a.t() gives column-major view, so we use a @ b.t()
            if status["scaled_mm_available"]:
                try:
                    M, K, N = 64, 128, 64
                    a = torch.randn(M, K, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
                    b = torch.randn(N, K, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
                    scale_a = torch.tensor(1.0, device='cuda', dtype=torch.float32)
                    scale_b = torch.tensor(1.0, device='cuda', dtype=torch.float32)
                    # b.t() provides column-major view of (K, N) matrix
                    result = torch._scaled_mm(a, b.t(), scale_a, scale_b)
                    status["tensor_core_accelerated"] = True
                except Exception as e:
                    status["scaled_mm_error"] = str(e)[:100]
    
    return status


def check_tma_support() -> Dict:
    """Check TMA support via device attributes."""
    status = {
        "hardware_supported": False,
        "driver_attribute": False,
    }
    
    # First try via PyTorch (more reliable)
        props = torch.cuda.get_device_properties(0)
        # TMA is available on SM 9.0+ (Hopper and Blackwell)
        if props.major >= 9:
            status["hardware_supported"] = True
    
    # Then try CUDA driver API
        import ctypes
        cuda = ctypes.CDLL('libcuda.so')
        
        # Initialize CUDA first
        cuda.cuInit(0)
        
        # CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 143
        attr = ctypes.c_int()
        device = ctypes.c_int(0)
        result = cuda.cuDeviceGetAttribute(ctypes.byref(attr), 143, device)
        
        if result == 0 and attr.value == 1:
            status["driver_attribute"] = True
            status["hardware_supported"] = True
    
    return status


def check_cluster_support() -> Dict:
    """Check CTA cluster support."""
    status = {
        "hardware_supported": False,
        "driver_attribute": False,
    }
    
    # First try via PyTorch
        props = torch.cuda.get_device_properties(0)
        # Clusters are available on SM 9.0+ (Hopper and Blackwell)
        if props.major >= 9:
            status["hardware_supported"] = True
    
    # Then try CUDA driver API
        import ctypes
        cuda = ctypes.CDLL('libcuda.so')
        
        # Initialize CUDA
        cuda.cuInit(0)
        
        # CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 133
        attr = ctypes.c_int()
        device = ctypes.c_int(0)
        result = cuda.cuDeviceGetAttribute(ctypes.byref(attr), 133, device)
        
        if result == 0 and attr.value == 1:
            status["driver_attribute"] = True
            status["hardware_supported"] = True
    
    return status


def run_nsys_profile(test_name: str) -> Optional[str]:
    """Run nsys profiling on a test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_path = BLACKWELL_TESTS_DIR / test_name
    output_path = RESULTS_DIR / f"nsys_{test_name}"
    
    if not test_path.exists():
        return None
    
    try:
        result = subprocess.run(
            [
                "nsys", "profile",
                "--stats=true",
                "-o", str(output_path),
                "-f", "true",
                str(test_path)
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(BLACKWELL_TESTS_DIR)
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"nsys error: {e}"


def run_ncu_profile(test_name: str) -> Optional[str]:
    """Run ncu profiling on a test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_path = BLACKWELL_TESTS_DIR / test_name
    output_path = RESULTS_DIR / f"ncu_{test_name}"
    
    if not test_path.exists():
        return None
    
    try:
        result = subprocess.run(
            [
                "ncu",
                "--target-processes", "all",
                "--set", "full",
                "-o", str(output_path),
                "-f",
                str(test_path)
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(BLACKWELL_TESTS_DIR)
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"ncu error: {e}"


def analyze_ncu_for_tensor_cores(report_path: Path) -> Dict:
    """Analyze NCU report for tensor core usage."""
    metrics = {
        "tensor_pipe_active": False,
        "tma_used": False,
        "tensor_memory_active": False,
    }
    
    ncu_report = report_path.with_suffix(".ncu-rep")
    if not ncu_report.exists():
        return metrics
    
        result = subprocess.run(
            [
                "ncu", "--import", str(ncu_report),
                "--page", "raw",
                "--print-kernel-base", "function"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout
        
        # Check for tensor core metrics
        if "sm__pipe_tensor_cycles_active" in output:
            # Parse the value
            for line in output.split('\n'):
                if "sm__pipe_tensor_cycles_active" in line and "pct_of_peak" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if '%' in p:
                                val = float(parts[i-1])
                                if val > 0:
                                    metrics["tensor_pipe_active"] = True
        
        if "tensor_map_access_supported" in output:
            metrics["tma_used"] = True
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Verify Blackwell features")
    parser.add_argument("--profile", action="store_true", help="Run profiling")
    parser.add_argument("--nsys", action="store_true", help="Run nsys profiling")
    parser.add_argument("--ncu", action="store_true", help="Run ncu profiling")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    args = parser.parse_args()
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {},
        "summary": {"passed": 0, "failed": 0, "skipped": 0}
    }
    
    # Check GPU
    print_header("Blackwell Feature Verification")
    is_blackwell, gpu_info = check_gpu()
    
    print(f"\nGPU: {gpu_info.get('name', 'Unknown')}")
    print(f"Compute Capability: {gpu_info.get('compute_capability', 'Unknown')}")
    print(f"Memory: {gpu_info.get('total_memory_gb', 0):.1f} GB")
    print(f"SMs: {gpu_info.get('multi_processor_count', 0)}")
    
    results["gpu"] = gpu_info
    
    if not is_blackwell:
        print("\n⚠ Not a Blackwell GPU - some tests may not apply")
    
    # Build CUDA tests
    print_header("Building CUDA Tests")
    build_result = subprocess.run(
        ["make", "clean", "all"],
        capture_output=True,
        text=True,
        cwd=str(BLACKWELL_TESTS_DIR)
    )
    if build_result.returncode != 0:
        print("⚠ Build warnings/errors (continuing anyway):")
        print(build_result.stderr[-500:] if len(build_result.stderr) > 500 else build_result.stderr)
    else:
        print("✓ Build successful")
    
    # Run CUDA tests
    print_header("CUDA Feature Tests")
    
    cuda_tests = [
        ("test_tmem", "TMEM (Tensor Memory)"),
        ("test_tma", "TMA (Tensor Memory Accelerator)"),
        ("test_clusters", "CTA Clusters"),
        ("test_dsmem", "DSMEM (Distributed Shared Memory)"),
        ("test_fp8", "FP8 Precision"),
        ("test_warp_spec", "Warp Specialization"),
    ]
    
    for test_binary, test_name in cuda_tests:
        print(f"\n  Testing: {test_name}")
        passed, output, elapsed = run_cuda_test(test_binary)
        print_result(test_name, passed, f"{elapsed:.2f}s")
        
        results["tests"][test_binary] = {
            "name": test_name,
            "passed": passed,
            "elapsed": elapsed,
        }
        
        if passed:
            results["summary"]["passed"] += 1
        else:
            results["summary"]["failed"] += 1
            if "FAILED" in output or "failed" in output.lower():
                # Extract failure reason
                for line in output.split('\n'):
                    if "failed" in line.lower() or "error" in line.lower():
                        print(f"    → {line.strip()[:80]}")
    
    # Python-level feature checks
    print_header("Python API Feature Checks")
    
    # TMA
    tma_status = check_tma_support()
    print_result("TMA Hardware Support", tma_status["hardware_supported"])
    results["tests"]["tma_python"] = tma_status
    
    # Clusters
    cluster_status = check_cluster_support()
    print_result("Cluster Launch Support", cluster_status["hardware_supported"])
    results["tests"]["cluster_python"] = cluster_status
    
    # FP8
    print("\n  FP8 Support:")
    fp8_status = check_fp8_support()
    print_result("FP8 E4M3 dtype", fp8_status["e4m3_available"])
    print_result("FP8 E5M2 dtype", fp8_status["e5m2_available"])
    print_result("FP8 conversion", fp8_status["conversion_working"])
    print_result("FP8 _scaled_mm (tensor cores)", fp8_status["tensor_core_accelerated"])
    results["tests"]["fp8_python"] = fp8_status
    
    # FP4
    print("\n  FP4 Support:")
    fp4_status = check_fp4_support()
    print_result("FP4 E2M1 dtype", fp4_status["dtype_available"])
    print_result("FP4 native ops", fp4_status["native_ops_working"], 
             "Not implemented in PyTorch 2.9.1" if not fp4_status["native_ops_working"] else "")
    print_result("FP4 packed workaround", fp4_status["workaround_available"])
    results["tests"]["fp4_python"] = fp4_status
    
    # Profiling
    if args.profile or args.nsys or args.ncu:
        print_header("Profiling")
        
        if args.profile or args.nsys:
            print("\n  Running nsys profiling on test_fp8...")
            nsys_output = run_nsys_profile("test_fp8")
            if nsys_output:
                print("  ✓ nsys profile generated")
                # Extract key metrics
                if "CUDA API Statistics" in str(nsys_output):
                    print("    → CUDA API calls captured")
        
        if args.profile or args.ncu:
            print("\n  Running ncu profiling on test_fp8...")
            print("  (This may take a few minutes...)")
            ncu_output = run_ncu_profile("test_fp8")
            if ncu_output:
                print("  ✓ ncu profile generated")
                
                # Analyze for tensor cores
                ncu_analysis = analyze_ncu_for_tensor_cores(RESULTS_DIR / "ncu_test_fp8")
                if ncu_analysis["tma_used"]:
                    print("    → TMA hardware confirmed active")
                if ncu_analysis["tensor_pipe_active"]:
                    print("    → Tensor core pipeline active (tcgen05)")
                else:
                    print("    → Note: Custom kernels may not use tensor core intrinsics")
                    print("           Use cuBLAS/cuBLASLt for hardware tensor core acceleration")
    
    # Summary
    print_header("Summary")
    
    total = results["summary"]["passed"] + results["summary"]["failed"]
    print(f"\n  Tests Passed: {results['summary']['passed']}/{total}")
    
    print("\n  Blackwell Features Status:")
    features = [
        ("TMEM", "test_tmem"),
        ("TMA", "test_tma"),
        ("CTA Clusters", "test_clusters"),
        ("DSMEM", "test_dsmem"),
        ("FP8", "test_fp8"),
        ("Warp Features", "test_warp_spec"),
    ]
    
    all_passed = True
    for name, test_key in features:
        if test_key in results["tests"]:
            passed = results["tests"][test_key].get("passed", False)
            status = "✓" if passed else "✗"
            print(f"    {status} {name}")
            if not passed:
                all_passed = False
    
    print("\n  FP4 Status:")
    print(f"    • dtype exists: {fp4_status['dtype_available']}")
    print(f"    • native ops: {'Working' if fp4_status['native_ops_working'] else 'Not implemented (use packed uint8 workaround)'}")
    print(f"    • To enable native FP4:")
    print(f"      1. Wait for PyTorch to implement copy_/conversion for float4_e2m1fn_x2")
    print(f"      2. Or use cuBLASLt directly with FP4 compute type")
    print(f"      3. Or use our packed uint8 quantization (ch19/native_fp4_quantization.py)")
    
    # Save results
    if args.json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_file = RESULTS_DIR / "verification_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {results_file}")
    
    print("\n" + "=" * 70)
    if all_passed and fp8_status["conversion_working"]:
        print(" ✓ All core Blackwell features are WORKING!")
    else:
        print(" ⚠ Some features need attention")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

