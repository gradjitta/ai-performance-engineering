"""
PerformanceCoreBase: shared, non-HTTP performance helpers.

This is the logic side of the old dashboard handler, split out so CLI/MCP
can reuse data loading, profiling artifacts, GPU/system inspection, and
benchmark discovery without depending on the HTTP server.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.analysis.performance_analyzer import (
    PerformanceAnalyzer,
    load_benchmark_data as load_benchmark_results,
)
from core import profile_artifacts
from core.compile_analysis import load_compile_analysis

CODE_ROOT = Path(__file__).resolve().parents[1]


class PerformanceCoreBase:
    """Shared performance logic without HTTP concerns."""

    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = data_file
        self._analyzer: Optional[PerformanceAnalyzer] = PerformanceAnalyzer(
            lambda: load_benchmark_results(self.data_file)
        )

    @property
    def analyzer(self) -> PerformanceAnalyzer:
        if not hasattr(self, "_analyzer") or self._analyzer is None:
            data_path = getattr(self, "data_file", None)
            self._analyzer = PerformanceAnalyzer(lambda: load_benchmark_results(data_path))
        return self._analyzer

    # ------------------------------------------------------------------
    # Benchmark data + exports
    # ------------------------------------------------------------------
    def load_benchmark_data(self) -> dict:
        return load_benchmark_results(self.data_file)

    def export_benchmarks_csv(self) -> str:
        data = self.load_benchmark_data()
        return profile_artifacts.export_benchmarks_csv(data)

    def export_detailed_csv(self) -> str:
        data = self.load_benchmark_data()
        return profile_artifacts.export_detailed_csv(data)

    # ------------------------------------------------------------------
    # Profiling artifact helpers
    # ------------------------------------------------------------------
    def get_flame_graph_data(self) -> dict:
        return profile_artifacts.load_flame_graph_data(CODE_ROOT)

    def get_memory_timeline(self) -> dict:
        return profile_artifacts.load_memory_timeline(CODE_ROOT)

    def get_cpu_gpu_timeline(self) -> dict:
        return profile_artifacts.load_cpu_gpu_timeline(CODE_ROOT)

    def get_kernel_breakdown(self) -> dict:
        return profile_artifacts.load_kernel_breakdown(self.get_flame_graph_data())

    def get_hta_analysis(self) -> dict:
        hta_data = profile_artifacts.load_hta_analysis(CODE_ROOT)
        if not hta_data.get("top_kernels"):
            kernel_data = self.get_kernel_breakdown()
            total_time = kernel_data.get("summary", {}).get("total_time_us", 0)
            if total_time > 0:
                for kernel in kernel_data.get("kernels", [])[:10]:
                    hta_data.setdefault("top_kernels", []).append({
                        "name": kernel.get("name"),
                        "time_us": kernel.get("time_us"),
                        "pct": kernel.get("time_us", 0) / total_time * 100 if total_time else 0,
                    })
            if kernel_data.get("by_type"):
                top_type = max(kernel_data["by_type"].items(), key=lambda x: x[1])
                hta_data.setdefault("recommendations", []).append(
                    f"Optimize {top_type[0]} operations ({top_type[1]/1000:.1f}ms total)"
                )
        return hta_data

    def get_compile_analysis(self) -> dict:
        benchmarks = self.load_benchmark_data().get("benchmarks", [])
        return load_compile_analysis(CODE_ROOT, benchmarks)

    def get_roofline_data(self) -> dict:
        roofline_data = {
            "has_real_data": False,
            "baseline_points": [],
            "optimized_points": [],
            "hardware_specs": {},
            "benchmark_details": [],
        }

        gpu_info = self.get_gpu_info()
        gpu_name = gpu_info.get("name", "Unknown GPU")
        if "B200" in gpu_name or "B300" in gpu_name:
            roofline_data["hardware_specs"] = {
                "name": gpu_name,
                "memory_bandwidth_gb_s": 8000,
                "peak_tflops": 2500,
            }
        elif "H100" in gpu_name:
            roofline_data["hardware_specs"] = {
                "name": gpu_name,
                "memory_bandwidth_gb_s": 3350,
                "peak_tflops": 120,
            }
        else:
            roofline_data["hardware_specs"] = {
                "name": gpu_name,
                "memory_bandwidth_gb_s": None,
                "peak_tflops": None,
            }

        try:
            data = self.load_benchmark_data().get("benchmarks", [])
            for bench in data:
                if "baseline_time_ms" in bench and "optimized_time_ms" in bench:
                    baseline_ms = bench["baseline_time_ms"]
                    optimized_ms = bench["optimized_time_ms"]
                    speedup = bench.get("speedup", baseline_ms / optimized_ms if optimized_ms else 0)
                    ai_estimate = bench.get("arithmetic_intensity", None)

                    roofline_data["baseline_points"].append({
                        "name": bench.get("name", ""),
                        "intensity": ai_estimate or 0.5,
                        "performance_tflops": bench.get("baseline_tflops", 0),
                    })
                    roofline_data["optimized_points"].append({
                        "name": bench.get("name", ""),
                        "intensity": ai_estimate or 0.5,
                        "performance_tflops": bench.get("optimized_tflops", 0),
                        "speedup": speedup,
                    })

                    roofline_data["benchmark_details"].append({
                        "name": bench.get("name", ""),
                        "chapter": bench.get("chapter", ""),
                        "arithmetic_intensity": ai_estimate,
                        "baseline_gflops": bench.get("baseline_tflops", 0) * 1000,
                        "optimized_gflops": bench.get("optimized_tflops", 0) * 1000,
                        "speedup": speedup,
                    })
            roofline_data["has_real_data"] = len(roofline_data["baseline_points"]) > 0
        except Exception:
            pass

        return roofline_data

    # ------------------------------------------------------------------
    # Benchmark discovery
    # ------------------------------------------------------------------
    def get_available_benchmarks(self) -> dict:
        available = {
            "chapters": [],
            "labs": [],
            "total_chapters": 0,
            "total_labs": 0,
            "total_benchmarks": 0,
        }

        for ch_dir in sorted(CODE_ROOT.glob("ch[0-9]*")):
            if ch_dir.is_dir():
                chapter_info = self._scan_directory(ch_dir, "chapter")
                if chapter_info["benchmarks"]:
                    available["chapters"].append(chapter_info)

        labs_dir = CODE_ROOT / "labs"
        if labs_dir.exists():
            for lab_dir in sorted(labs_dir.iterdir()):
                if lab_dir.is_dir() and not lab_dir.name.startswith("."):
                    lab_info = self._scan_directory(lab_dir, "lab")
                    if lab_info["benchmarks"]:
                        available["labs"].append(lab_info)

        available["total_chapters"] = len(available["chapters"])
        available["total_labs"] = len(available["labs"])
        available["total_benchmarks"] = sum(
            len(ch["benchmarks"]) for ch in available["chapters"]
        ) + sum(len(lab["benchmarks"]) for lab in available["labs"])

        return available

    def _scan_directory(self, directory: Path, dir_type: str) -> dict:
        info = {
            "name": directory.name,
            "path": str(directory.relative_to(CODE_ROOT)),
            "type": dir_type,
            "benchmarks": [],
            "has_expectations": False,
            "has_profiles": False,
        }

        baseline_files = list(directory.glob("baseline_*.py")) + list(directory.glob("baseline_*.cu"))
        for baseline in baseline_files:
            name = baseline.stem.replace("baseline_", "")
            file_type = "python" if baseline.suffix == ".py" else "cuda"
            optimized_files = list(directory.glob(f"optimized_{name}*.py")) + list(directory.glob(f"optimized_{name}*.cu"))
            benchmark_info = {
                "name": name,
                "type": file_type,
                "baseline_file": baseline.name,
                "optimized_files": [f.name for f in optimized_files],
                "optimization_count": len(optimized_files),
            }
            info["benchmarks"].append(benchmark_info)

        info["has_expectations"] = (
            (directory / "expectations_b200.json").exists()
            or (directory / "expectations_gb10.json").exists()
        )
        profile_dir = CODE_ROOT / "benchmark_profiles" / directory.name
        info["has_profiles"] = profile_dir.exists() and any(profile_dir.iterdir()) if profile_dir.exists() else False
        return info

    # ------------------------------------------------------------------
    # GPU + software info
    # ------------------------------------------------------------------
    def get_gpu_info(self) -> dict:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,temperature.memory,power.draw,power.limit,memory.used,memory.total,utilization.gpu,utilization.memory,clocks.current.graphics,clocks.current.memory,fan.speed,persistence_mode,pstate",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                hbm_temp = None
                try:
                    if parts[2] and parts[2] != "[N/A]":
                        hbm_temp = float(parts[2])
                except (ValueError, IndexError):
                    pass
                fan_speed = None
                try:
                    if len(parts) > 11 and parts[11] and parts[11] != "[N/A]":
                        fan_speed = int(float(parts[11]))
                except (ValueError, IndexError):
                    pass
                ecc_mode = None
                try:
                    if len(parts) > 14 and parts[14].strip() not in ["[N/A]", "N/A", ""]:
                        ecc_mode = parts[14].strip() == "Enabled"
                except (ValueError, IndexError):
                    pass

                return {
                    "name": parts[0],
                    "temperature": float(parts[1]),
                    "temperature_hbm": hbm_temp,
                    "power": float(parts[3]),
                    "power_limit": float(parts[4]) if parts[4] != "[N/A]" else None,
                    "memory_used": float(parts[5]),
                    "memory_total": float(parts[6]),
                    "utilization": float(parts[7]),
                    "utilization_memory": float(parts[8]) if parts[8] != "[N/A]" else None,
                    "clock_graphics": int(float(parts[9])) if len(parts) > 9 else None,
                    "clock_memory": int(float(parts[10])) if len(parts) > 10 else None,
                    "fan_speed": fan_speed,
                    "persistence_mode": parts[12].strip() == "Enabled" if len(parts) > 12 else None,
                    "pstate": parts[13].strip() if len(parts) > 13 else None,
                    "ecc_mode": ecc_mode,
                    "live": True,
                }
        except Exception:
            pass
        return {
            "name": "GPU Not Detected",
            "temperature": None,
            "temperature_hbm": None,
            "power": None,
            "power_limit": None,
            "memory_used": None,
            "memory_total": None,
            "utilization": None,
            "utilization_memory": None,
            "clock_graphics": None,
            "clock_memory": None,
            "fan_speed": None,
            "persistence_mode": None,
            "pstate": None,
            "ecc_mode": None,
            "live": False,
            "error": "nvidia-smi failed or no GPU available",
        }

    def get_gpu_topology(self) -> dict:
        topology = {
            "gpu_count": 0,
            "gpus": [],
            "topology_matrix": [],
            "nvlink_available": False,
            "p2p_matrix": [],
        }
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        topology["gpus"].append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "uuid": parts[2] if len(parts) > 2 else "",
                                "pci_bus": parts[3] if len(parts) > 3 else "",
                            }
                        )
                topology["gpu_count"] = len(topology["gpus"])

            result = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                topology["topology_raw"] = result.stdout
                for line in lines:
                    if "GPU" in line and ("NV" in line or "PIX" in line or "PHB" in line or "SYS" in line):
                        topology["nvlink_available"] = "NV" in line
                        parts = line.split()
                        row = []
                        for p in parts[1:]:
                            if p in [
                                "X",
                                "NV1",
                                "NV2",
                                "NV3",
                                "NV4",
                                "NV5",
                                "NV6",
                                "NV7",
                                "NV8",
                                "NV9",
                                "NV10",
                                "NV11",
                                "NV12",
                                "NV18",
                                "PIX",
                                "PHB",
                                "SYS",
                                "NODE",
                            ]:
                                row.append(p)
                        if row:
                            topology["topology_matrix"].append(row)

            try:
                import torch

                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    p2p_matrix: List[List[str]] = []
                    for i in range(min(gpu_count, 8)):
                        row = []
                        for j in range(min(gpu_count, 8)):
                            if i == j:
                                row.append("self")
                            else:
                                try:
                                    can_access = torch.cuda.can_device_access_peer(i, j)
                                    row.append("yes" if can_access else "no")
                                except Exception:
                                    row.append("?")
                        p2p_matrix.append(row)
                    topology["p2p_matrix"] = p2p_matrix
            except Exception:
                pass
        except Exception as e:
            topology["error"] = str(e)

        return topology

    def get_nvlink_status(self) -> dict:
        nvlink = {"available": False, "links_per_gpu": {}, "total_bandwidth_gbs": 0, "link_details": []}
        try:
            result = subprocess.run(["nvidia-smi", "nvlink", "--status"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                nvlink["available"] = True
                nvlink["raw_output"] = result.stdout
                current_gpu = None
                link_count = 0
                for line in result.stdout.split("\n"):
                    if "GPU" in line and ":" in line:
                        if current_gpu is not None:
                            nvlink["links_per_gpu"][current_gpu] = link_count
                        match = re.search(r"GPU (\\d+)", line)
                        if match:
                            current_gpu = int(match.group(1))
                            link_count = 0
                    elif "Link" in line and "GB/s" in line:
                        link_count += 1
                        bw_match = re.search(r"(\\d+)\\s*GB/s", line)
                        if bw_match:
                            nvlink["link_details"].append({"gpu": current_gpu, "bandwidth_gbs": int(bw_match.group(1))})
                if current_gpu is not None:
                    nvlink["links_per_gpu"][current_gpu] = link_count
                nvlink["total_bandwidth_gbs"] = sum(l.get("bandwidth_gbs", 0) for l in nvlink["link_details"])
        except Exception:
            pass
        return nvlink

    def get_software_info(self) -> dict:
        info: Dict[str, Any] = {
            "pytorch": None,
            "cuda_runtime": None,
            "cuda_driver": None,
            "triton": None,
            "cudnn": None,
            "cublas": None,
            "nccl": None,
            "flash_attn": None,
            "transformer_engine": None,
            "xformers": None,
            "python": None,
            "compute_capability": None,
            "architecture": None,
            "gpu_count": None,
            "torch_compile_backend": None,
        }

        try:
            import torch

            info["pytorch"] = torch.__version__
            info["cuda_runtime"] = torch.version.cuda
            if torch.cuda.is_available():
                device = torch.device("cuda")
                props = torch.cuda.get_device_properties(device)
                info["compute_capability"] = f"{props.major}.{props.minor}"
                info["architecture"] = props.name
                info["gpu_count"] = torch.cuda.device_count()
                info["torch_compile_backend"] = os.environ.get("TORCH_COMPILE_BACKEND")
        except Exception:
            pass

        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                lines = result.stdout.splitlines()
                if lines:
                    info["cuda_driver"] = lines[2].split()[2] if len(lines) > 2 else None
        except Exception:
            pass

        try:
            import importlib

            for pkg in ["triton", "transformer_engine", "flash_attn", "xformers"]:
                try:
                    module = importlib.import_module(pkg)
                    info[pkg] = getattr(module, "__version__", None)
                except Exception:
                    continue
        except Exception:
            pass

        try:
            import sys
            info["python"] = sys.version.split()[0]
        except Exception:
            pass

        return info

    def get_dependency_health(self) -> dict:
        project_root = CODE_ROOT
        third_party = project_root / "third_party"
        result = {
            "status": "ok",
            "issues": [],
            "warnings": [],
            "cutlass": {"version": None, "path": None, "sm100_headers": False},
            "transformer_engine": {"version": None, "cutlass_symlink": False, "cutlass_symlink_target": None, "cutlass_sm100_headers": False},
            "nvidia_cutlass_dsl": {"version": None, "path": None},
        }

        # CUTLASS version check
        cutlass_path = third_party / "cutlass"
        version_h = cutlass_path / "include" / "cutlass" / "version.h"
        if version_h.exists():
            result["cutlass"]["path"] = str(cutlass_path)
            try:
                content = version_h.read_text()
                major = minor = patch = 0
                for line in content.splitlines():
                    if "CUTLASS_VERSION_MAJOR" in line:
                        major = int(re.findall(r"\\d+", line)[0])
                    if "CUTLASS_VERSION_MINOR" in line:
                        minor = int(re.findall(r"\\d+", line)[0])
                    if "CUTLASS_VERSION_PATCH" in line:
                        patch = int(re.findall(r"\\d+", line)[0])
                result["cutlass"]["version"] = f"{major}.{minor}.{patch}"
            except Exception:
                pass

            # SM100 headers check
            sm100_header = cutlass_path / "include" / "cutlass" / "arch" / "sm100_smem_selector.h"
            result["cutlass"]["sm100_headers"] = sm100_header.exists()

        # Transformer Engine
        try:
            import transformer_engine

            te_path = Path(transformer_engine.__file__).resolve().parent
            result["transformer_engine"]["version"] = getattr(transformer_engine, "__version__", None)
            cutlass_link = te_path / "csrc" / "cutlass"
            if cutlass_link.exists():
                result["transformer_engine"]["cutlass_symlink"] = cutlass_link.is_symlink()
                try:
                    result["transformer_engine"]["cutlass_symlink_target"] = str(cutlass_link.resolve())
                except Exception:
                    pass
                sm100_header_te = cutlass_link / "include" / "cutlass" / "arch" / "sm100_smem_selector.h"
                result["transformer_engine"]["cutlass_sm100_headers"] = sm100_header_te.exists()
        except Exception:
            result["warnings"].append("transformer_engine not installed")

        # NVIDIA Cutlass DSL (optional)
        cutlass_dsl = third_party / "nvidia-cutlass-dsl"
        if cutlass_dsl.exists():
            result["nvidia_cutlass_dsl"]["path"] = str(cutlass_dsl)
            version_file = cutlass_dsl / "VERSION.txt"
            if version_file.exists():
                result["nvidia_cutlass_dsl"]["version"] = version_file.read_text().strip()

        return result

    def check_dependency_updates(self) -> dict:
        updates = {"outdated": [], "errors": []}
        try:
            result = subprocess.run(
                [os.environ.get("PYTHON_BIN", "python"), "-m", "pip", "list", "--outdated", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode == 0:
                try:
                    updates["outdated"] = json.loads(result.stdout)
                except Exception:
                    updates["errors"].append("Failed to parse pip output")
            else:
                updates["errors"].append(result.stderr.strip())
        except Exception as e:
            updates["errors"].append(str(e))
        return updates

    def get_full_system_context(self) -> dict:
        context = {
            "gpu_info": self.get_gpu_info(),
            "software_info": self.get_software_info(),
            "dependency_health": self.get_dependency_health(),
            "gpu_topology": self.get_gpu_topology(),
        }
        try:
            import torch

            context["cuda_available"] = torch.cuda.is_available()
        except Exception:
            context["cuda_available"] = False
        return context
