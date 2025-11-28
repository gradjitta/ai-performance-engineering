export interface Benchmark {
  name: string;
  chapter: string;
  type: string;
  status: 'succeeded' | 'failed' | 'skipped';
  baseline_time_ms: number;
  optimized_time_ms: number;
  speedup: number;
  raw_speedup?: number;
  speedup_capped?: boolean;
  optimization_goal?: 'speed' | 'memory';
  error?: string;
}

export interface BenchmarkData {
  benchmarks: Benchmark[];
  summary: {
    total: number;
    succeeded: number;
    failed: number;
    skipped: number;
    avg_speedup: number;
    max_speedup: number;
    min_speedup: number;
  };
  timestamp: string;
  speedup_cap?: number;
}

export interface GpuInfo {
  name: string;
  memory_total: number;      // MB
  memory_used: number;       // MB
  utilization: number;
  temperature: number;
  temperature_hbm?: number | null;
  power?: number;            // watts
  power_draw?: number;       // legacy key
  power_limit?: number;      // watts
  compute_capability?: string;
  driver_version?: string;
  cuda_version?: string;
  clock_graphics?: number;
  clock_memory?: number;
  fan_speed?: number | null;
  pstate?: string | null;
  live?: boolean;
}

export interface SoftwareInfo {
  python_version: string;
  pytorch_version: string;
  cuda_version: string;
  cudnn_version: string;
  triton_version?: string;
}

export interface LLMAnalysis {
  summary: string;
  key_findings: string[];
  recommendations: string[];
  bottlenecks: Array<{
    name: string;
    severity: 'high' | 'medium' | 'low';
    description: string;
    recommendation: string;
  }>;
}

export interface ProfilerData {
  kernels: Array<{
    name: string;
    duration_ms: number;
    memory_mb: number;
    occupancy: number;
  }>;
  memory_timeline: Array<{
    timestamp: number;
    allocated_mb: number;
    reserved_mb: number;
  }>;
  flame_data?: unknown;
}

export interface Tab {
  id: string;
  label: string;
  icon: string;
}
