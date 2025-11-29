'use client';

import { useEffect, useState } from 'react';
import {
  Activity,
  Battery,
  Calculator,
  Cpu,
  Gauge,
  Microscope,
  RefreshCw,
  Sparkles,
  Wand2,
  Workflow,
} from 'lucide-react';
import {
  getAnalysisAutoTune,
  getAnalysisBankConflicts,
  getAnalysisCompound,
  getAnalysisContainerLimits,
  getAnalysisCpuMemory,
  getAnalysisEnergy,
  getAnalysisFullSystem,
  getAnalysisMemoryAccess,
  getAnalysisMultiGpuScaling,
  getAnalysisOccupancy,
  getAnalysisOptimalStack,
  getAnalysisPredictScaling,
  getAnalysisRecommendations,
  getAnalysisSystemParams,
  getAnalysisTradeoffs,
  getAnalysisWhatIf,
  getAnalysisOptimizations,
  getAnalysisPlaybooks,
  getAnalysisWarpDivergence,
} from '@/lib/api';
import { getErrorMessage } from '@/lib/useApi';
import { formatNumber } from '@/lib/utils';
import { EmptyState, ErrorState, LoadingState } from './DataState';

type DeepData = {
  tradeoffs: any;
  recommendations: any;
  optimizations: any;
  playbooks: any;
  cpuMemory: any;
  systemParams: any;
  containerLimits: any;
  fullSystem: any;
  energy: any;
  scaling: any;
  multiGpu: any;
  compound: any;
  optimalStack: any;
};

export function AnalysisDeepDiveCard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<DeepData | null>(null);
  const [toolBusy, setToolBusy] = useState<string | null>(null);
  const [toolError, setToolError] = useState<string | null>(null);

  const [occupancyInputs, setOccupancyInputs] = useState({ threads: 256, shared: 0, registers: 32 });
  const [occupancyResult, setOccupancyResult] = useState<any>(null);

  const [warpCode, setWarpCode] = useState<string>('for (int i = threadIdx.x; i < N; i += blockDim.x) {...}');
  const [warpResult, setWarpResult] = useState<any>(null);

  const [memoryParams, setMemoryParams] = useState({ stride: 1, element_size: 4 });
  const [bankResult, setBankResult] = useState<any>(null);
  const [memoryAccessResult, setMemoryAccessResult] = useState<any>(null);

  const [autoTuneParams, setAutoTuneParams] = useState({ kernel: 'matmul', max_configs: 50 });
  const [autoTuneResult, setAutoTuneResult] = useState<any>(null);

  const [whatIfParams, setWhatIfParams] = useState({ vram: 24, latency: 50, throughput: 1000 });
  const [whatIfResult, setWhatIfResult] = useState<any>(null);

  async function loadData() {
    try {
      setLoading(true);
      setError(null);
      const [
        tradeoffs,
        recommendations,
        optimizations,
        playbooks,
        cpuMemory,
        systemParams,
        containerLimits,
        fullSystem,
        energy,
        scaling,
        multiGpu,
        compound,
        optimalStack,
      ] = await Promise.all([
        getAnalysisTradeoffs().catch(() => null),
        getAnalysisRecommendations().catch(() => null),
        getAnalysisOptimizations().catch(() => null),
        getAnalysisPlaybooks().catch(() => null),
        getAnalysisCpuMemory().catch(() => null),
        getAnalysisSystemParams().catch(() => null),
        getAnalysisContainerLimits().catch(() => null),
        getAnalysisFullSystem().catch(() => null),
        getAnalysisEnergy({ gpu: 'H100' }).catch(() => null),
        getAnalysisPredictScaling({ from_gpu: 'H100', to_gpu: 'B200', workload: 'inference' }).catch(() => null),
        getAnalysisMultiGpuScaling({ gpus: 8, nvlink: true, workload: 'training' }).catch(() => null),
        getAnalysisCompound(['flash', 'quant', 'fuse']).catch(() => null),
        getAnalysisOptimalStack({ target: 10, difficulty: 'medium' }).catch(() => null),
      ]);
      setData({
        tradeoffs,
        recommendations,
        optimizations,
        playbooks,
        cpuMemory,
        systemParams,
        containerLimits,
        fullSystem,
        energy,
        scaling,
        multiGpu,
        compound,
        optimalStack,
      });
    } catch (e) {
      setError(getErrorMessage(e, 'Failed to load deep analysis'));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadData();
  }, []);

  const runOccupancy = async () => {
    setToolError(null);
    setToolBusy('occupancy');
    try {
      const res = await getAnalysisOccupancy(occupancyInputs);
      setOccupancyResult(res);
    } catch (e) {
      setToolError(getErrorMessage(e, 'Occupancy calculator failed'));
    } finally {
      setToolBusy(null);
    }
  };

  const runWarpAnalysis = async () => {
    setToolError(null);
    setToolBusy('warp');
    try {
      const res = await getAnalysisWarpDivergence(warpCode);
      setWarpResult(res);
    } catch (e) {
      setToolError(getErrorMessage(e, 'Warp divergence analysis failed'));
    } finally {
      setToolBusy(null);
    }
  };

  const runMemoryTools = async () => {
    setToolError(null);
    setToolBusy('memory');
    try {
      const [bank, mem] = await Promise.all([
        getAnalysisBankConflicts(memoryParams),
        getAnalysisMemoryAccess(memoryParams),
      ]);
      setBankResult(bank);
      setMemoryAccessResult(mem);
    } catch (e) {
      setToolError(getErrorMessage(e, 'Memory access analysis failed'));
    } finally {
      setToolBusy(null);
    }
  };

  const runAutoTune = async () => {
    setToolError(null);
    setToolBusy('autotune');
    try {
      const res = await getAnalysisAutoTune(autoTuneParams);
      setAutoTuneResult(res);
    } catch (e) {
      setToolError(getErrorMessage(e, 'Auto-tune failed'));
    } finally {
      setToolBusy(null);
    }
  };

  const runWhatIf = async () => {
    setToolError(null);
    setToolBusy('whatif');
    try {
      const res = await getAnalysisWhatIf(whatIfParams);
      setWhatIfResult(res);
    } catch (e) {
      setToolError(getErrorMessage(e, 'What-if analysis failed'));
    } finally {
      setToolBusy(null);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading deep analysis..." />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body">
          <ErrorState message={error} onRetry={loadData} />
        </div>
      </div>
    );
  }

  const summaryTiles = [
    { label: 'Trade-offs', icon: Activity, value: data?.tradeoffs?.items?.length || data?.tradeoffs?.length || '—' },
    { label: 'Recommendations', icon: Sparkles, value: data?.recommendations?.items?.length || data?.recommendations?.length || '—' },
    { label: 'Optimizations', icon: Wand2, value: data?.optimizations?.length || data?.optimizations?.items?.length || '—' },
    { label: 'Playbooks', icon: Workflow, value: data?.playbooks?.length || data?.playbooks?.items?.length || '—' },
  ];

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Microscope className="w-5 h-5 text-accent-info" />
          <h3 className="font-medium text-white">Deep Analysis Coverage</h3>
        </div>
        <button onClick={loadData} className="p-2 rounded hover:bg-white/5" aria-label="Refresh deep analysis">
          <RefreshCw className="w-4 h-4 text-white/60" />
        </button>
      </div>
      <div className="card-body space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {summaryTiles.map((tile, idx) => (
            <div key={idx} className="p-3 rounded-lg bg-white/5 border border-white/10">
              <div className="flex items-center gap-2 text-xs text-white/50 uppercase">
                <tile.icon className="w-4 h-4 text-accent-primary" />
                {tile.label}
              </div>
              <div className="text-2xl font-bold text-accent-primary mt-1">{tile.value}</div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Panel title="System Profile" icon={Cpu}>
            {data?.cpuMemory ? (
              <div className="space-y-2 text-sm text-white/80">
                <div className="flex justify-between">
                  <span>CPU/Memory</span>
                  <span className="text-accent-primary font-mono">
                    {formatNumber(data.cpuMemory.memory_gb || data.cpuMemory.memory || 0, 1)} GB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Container limits</span>
                  <span className="text-accent-secondary font-mono">
                    {data.containerLimits?.memory_limit_gb
                      ? `${formatNumber(data.containerLimits.memory_limit_gb, 1)} GB`
                      : '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>System params</span>
                  <span className="text-white/60 text-xs">
                    {Object.keys(data.systemParams || {}).slice(0, 2).join(', ') || 'N/A'}
                  </span>
                </div>
              </div>
            ) : (
              <EmptyState title="No system data" description="Backend did not return system limits." />
            )}
          </Panel>

          <Panel title="Recommendations" icon={Sparkles}>
            {data?.recommendations?.items?.length ? (
              <ul className="space-y-2 text-sm text-white/80">
                {data.recommendations.items.slice(0, 3).map((item: any, idx: number) => (
                  <li key={idx} className="p-2 rounded bg-white/5 border border-white/10">
                    {item.title || item.name || item}
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState title="No recommendations" description="Run an analysis to populate guidance." />
            )}
          </Panel>

          <Panel title="Playbooks" icon={Workflow}>
            {data?.playbooks?.length ? (
              <ul className="space-y-1 text-sm text-white/80">
                {data.playbooks.slice(0, 4).map((p: any, idx: number) => (
                  <li key={idx} className="flex items-center justify-between">
                    <span className="truncate">{p.name || p.title || `Playbook ${idx + 1}`}</span>
                    <span className="text-white/50 text-xs">{p.category || p.type || ''}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState title="No playbooks" description="Backend did not return optimization playbooks." />
            )}
          </Panel>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Panel title="Predictive Models" icon={Gauge}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-white/80">
              <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                <div className="text-xs text-white/50 uppercase">Energy</div>
                <div className="text-xl font-bold text-accent-success">
                  {data?.energy?.efficiency || data?.energy?.score || '—'}
                </div>
                <div className="text-xs text-white/50">
                  GPU {data?.energy?.gpu || 'H100'} {data?.energy?.power_limit ? `· ${data.energy.power_limit}W` : ''}
                </div>
              </div>
              <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                <div className="text-xs text-white/50 uppercase">Scaling</div>
                <div className="text-xl font-bold text-accent-warning">
                  {data?.scaling?.target || data?.scaling?.predicted || '—'}
                </div>
                <div className="text-xs text-white/50">From H100 → B200</div>
              </div>
              <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                <div className="text-xs text-white/50 uppercase">Multi-GPU</div>
                <div className="text-xl font-bold text-accent-primary">
                  {data?.multiGpu?.efficiency || data?.multiGpu?.speedup || '—'}
                </div>
                <div className="text-xs text-white/50">NVLink {data?.multiGpu?.nvlink ? 'On' : 'Off'}</div>
              </div>
              <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                <div className="text-xs text-white/50 uppercase">Optimal Stack</div>
                <div className="text-xl font-bold text-accent-secondary">
                  {data?.optimalStack?.stack || data?.optimalStack?.summary || '—'}
                </div>
                <div className="text-xs text-white/50">
                  Target {data?.optimalStack?.target || ''} · {data?.optimalStack?.difficulty || 'medium'}
                </div>
              </div>
            </div>
          </Panel>

          <Panel title="Compound Strategies" icon={Wand2}>
            {data?.compound ? (
              <pre className="text-xs text-white/80 bg-white/5 border border-white/10 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap">
                {JSON.stringify(data.compound, null, 2)}
              </pre>
            ) : (
              <EmptyState title="No compound analysis" description="Backend did not return combined strategy impact." />
            )}
          </Panel>
        </div>

        <div className="card-subsection space-y-4">
          <div className="flex items-center gap-2">
            <Calculator className="w-5 h-5 text-accent-primary" />
            <h4 className="text-white font-semibold">Interactive Calculators</h4>
            {toolError && <span className="text-sm text-accent-warning">{toolError}</span>}
          </div>
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
            <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
              <div className="text-xs text-white/50 uppercase flex items-center gap-2">
                <Gauge className="w-4 h-4 text-accent-info" /> Occupancy
              </div>
              <div className="grid grid-cols-3 gap-2 text-sm">
                {(['threads', 'shared', 'registers'] as const).map((field) => (
                  <label key={field} className="text-white/60 text-xs flex flex-col gap-1">
                    {field}
                    <input
                      type="number"
                      value={occupancyInputs[field]}
                      onChange={(e) =>
                        setOccupancyInputs((prev) => ({ ...prev, [field]: Number(e.target.value) }))
                      }
                      className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-white text-xs"
                    />
                  </label>
                ))}
              </div>
              <button
                onClick={runOccupancy}
                className="px-3 py-2 rounded bg-accent-primary/20 text-accent-primary text-xs"
                disabled={toolBusy === 'occupancy'}
              >
                {toolBusy === 'occupancy' ? 'Calculating…' : 'Calculate occupancy'}
              </button>
              <ResultBlock result={occupancyResult} />
            </div>

            <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
              <div className="text-xs text-white/50 uppercase flex items-center gap-2">
                <Activity className="w-4 h-4 text-accent-warning" /> Memory Access
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {(['stride', 'element_size'] as const).map((field) => (
                  <label key={field} className="text-white/60 text-xs flex flex-col gap-1">
                    {field}
                    <input
                      type="number"
                      value={memoryParams[field]}
                      onChange={(e) =>
                        setMemoryParams((prev) => ({ ...prev, [field]: Number(e.target.value) }))
                      }
                      className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-white text-xs"
                    />
                  </label>
                ))}
              </div>
              <button
                onClick={runMemoryTools}
                className="px-3 py-2 rounded bg-accent-secondary/20 text-accent-secondary text-xs"
                disabled={toolBusy === 'memory'}
              >
                {toolBusy === 'memory' ? 'Analyzing…' : 'Analyze access'}
              </button>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <ResultBlock title="Bank conflicts" result={bankResult} />
                <ResultBlock title="Access pattern" result={memoryAccessResult} />
              </div>
            </div>

            <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
              <div className="text-xs text-white/50 uppercase flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-accent-success" /> Auto-tune & What-if
              </div>
              <label className="text-white/60 text-xs flex flex-col gap-1">
                Kernel
                <input
                  value={autoTuneParams.kernel}
                  onChange={(e) => setAutoTuneParams((prev) => ({ ...prev, kernel: e.target.value }))}
                  className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-white text-xs"
                  placeholder="matmul"
                />
              </label>
              <div className="grid grid-cols-2 gap-2 text-xs text-white/60">
                <label className="flex flex-col gap-1">
                  Max configs
                  <input
                    type="number"
                    value={autoTuneParams.max_configs}
                    onChange={(e) =>
                      setAutoTuneParams((prev) => ({ ...prev, max_configs: Number(e.target.value) }))
                    }
                    className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-white text-xs"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  VRAM
                  <input
                    type="number"
                    value={whatIfParams.vram}
                    onChange={(e) => setWhatIfParams((prev) => ({ ...prev, vram: Number(e.target.value) }))}
                    className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-white text-xs"
                  />
                </label>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs text-white/60">
                <label className="flex flex-col gap-1">
                  Latency
                  <input
                    type="number"
                    value={whatIfParams.latency}
                    onChange={(e) => setWhatIfParams((prev) => ({ ...prev, latency: Number(e.target.value) }))}
                    className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-white text-xs"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  Throughput
                  <input
                    type="number"
                    value={whatIfParams.throughput}
                    onChange={(e) => setWhatIfParams((prev) => ({ ...prev, throughput: Number(e.target.value) }))}
                    className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-white text-xs"
                  />
                </label>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={runAutoTune}
                  className="px-3 py-2 rounded bg-accent-primary/20 text-accent-primary text-xs"
                  disabled={toolBusy === 'autotune'}
                >
                  {toolBusy === 'autotune' ? 'Tuning…' : 'Run auto-tune'}
                </button>
                <button
                  onClick={runWhatIf}
                  className="px-3 py-2 rounded bg-accent-info/20 text-accent-info text-xs"
                  disabled={toolBusy === 'whatif'}
                >
                  {toolBusy === 'whatif' ? 'Simulating…' : 'Run what-if'}
                </button>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <ResultBlock title="Auto-tune" result={autoTuneResult} />
                <ResultBlock title="What-if" result={whatIfResult} />
              </div>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
            <div className="text-xs text-white/50 uppercase flex items-center gap-2">
              <Battery className="w-4 h-4 text-accent-warning" /> Warp divergence analyzer
            </div>
            <textarea
              value={warpCode}
              onChange={(e) => setWarpCode(e.target.value)}
              className="w-full h-24 rounded bg-white/5 border border-white/10 text-white text-sm p-2"
            />
            <button
              onClick={runWarpAnalysis}
              className="px-3 py-2 rounded bg-accent-warning/20 text-accent-warning text-xs"
              disabled={toolBusy === 'warp'}
            >
              {toolBusy === 'warp' ? 'Analyzing…' : 'Analyze divergence'}
            </button>
            <ResultBlock result={warpResult} />
          </div>
        </div>
      </div>
    </div>
  );
}

function Panel({ title, icon: Icon, children }: { title: string; icon: any; children: React.ReactNode }) {
  return (
    <div className="p-4 rounded-lg bg-white/5 border border-white/10 space-y-2">
      <div className="flex items-center gap-2 text-white font-semibold">
        <Icon className="w-4 h-4 text-accent-primary" />
        {title}
      </div>
      {children}
    </div>
  );
}

function ResultBlock({ result, title }: { result: any; title?: string }) {
  if (!result) {
    return <div className="text-xs text-white/50">{title ? `No ${title} yet.` : 'No output yet.'}</div>;
  }
  if (typeof result === 'string') {
    return <div className="text-xs text-white/80 whitespace-pre-wrap">{result}</div>;
  }
  return (
    <pre className="text-xs text-white/80 bg-white/5 border border-white/10 rounded-lg p-2 whitespace-pre-wrap overflow-x-auto">
      {JSON.stringify(result, null, 2)}
    </pre>
  );
}
