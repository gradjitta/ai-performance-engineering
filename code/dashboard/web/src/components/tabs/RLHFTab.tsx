'use client';

import { useState, useEffect, useCallback } from 'react';
import { Gamepad2, Loader2, AlertTriangle, RefreshCw, TrendingUp, Target, Zap, HardDrive, Calculator } from 'lucide-react';
import { getRLHFMethods, getRLHFConfig, getRLHFMemoryEstimate } from '@/lib/api';

interface RLHFMethod {
  name: string;
  full_name: string;
  description: string;
  complexity: string;
  memory_multiplier: number;
  training_speedup: string;
  quality: string;
  use_cases: string[];
  memory_estimate_gb: number;
  frameworks: string[];
  pros: string[];
  cons: string[];
}

export function RLHFTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [methods, setMethods] = useState<RLHFMethod[]>([]);
  const [recommendation, setRecommendation] = useState<string>('');
  const [recommendationReason, setRecommendationReason] = useState<string>('');
  const [gpuMemory, setGpuMemory] = useState<number>(80);
  
  // Config calculator state
  const [configParams, setConfigParams] = useState({
    method: 'ppo',
    model_size: 7,
    gpus: 8,
    memory_gb: 80,
  });
  const [configResult, setConfigResult] = useState<any>(null);
  const [configLoading, setConfigLoading] = useState(false);
  
  // Memory estimator state
  const [memoryParams, setMemoryParams] = useState({
    model_size: 7,
    method: 'ppo',
    precision: 'bf16',
    use_lora: false,
    batch_size: 4,
    seq_length: 512,
  });
  const [memoryResult, setMemoryResult] = useState<any>(null);
  const [memoryLoading, setMemoryLoading] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getRLHFMethods();
      const result = data as any;
      setMethods(result.methods || []);
      setRecommendation(result.recommended || '');
      setRecommendationReason(result.recommendation_reason || '');
      setGpuMemory(result.gpu_memory_gb || 80);
      setConfigParams(prev => ({ ...prev, memory_gb: result.gpu_memory_gb || 80 }));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load RLHF data');
    } finally {
      setLoading(false);
    }
  }, []);

  const calculateConfig = useCallback(async () => {
    try {
      setConfigLoading(true);
      const result = await getRLHFConfig(configParams);
      setConfigResult(result);
    } catch (e) {
      setConfigResult({ error: e instanceof Error ? e.message : 'Failed to calculate config' });
    } finally {
      setConfigLoading(false);
    }
  }, [configParams]);

  const estimateMemory = useCallback(async () => {
    try {
      setMemoryLoading(true);
      const result = await getRLHFMemoryEstimate(memoryParams);
      setMemoryResult(result);
    } catch (e) {
      setMemoryResult({ error: e instanceof Error ? e.message : 'Failed to estimate memory' });
    } finally {
      setMemoryLoading(false);
    }
  }, [memoryParams]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  if (loading) {
    return (
      <div className="card">
        <div className="card-body flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-accent-tertiary" />
          <span className="ml-3 text-white/50">Loading RLHF data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body text-center py-16">
          <AlertTriangle className="w-12 h-12 text-accent-danger mx-auto mb-4" />
          <p className="text-white/70 mb-4">{error}</p>
          <button
            onClick={loadData}
            className="flex items-center gap-2 px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg hover:bg-accent-primary/30 mx-auto"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Gamepad2 className="w-5 h-5 text-accent-tertiary" />
            <h2 className="text-lg font-semibold text-white">RL/RLHF Optimization</h2>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-white/50">GPU Memory: {gpuMemory}GB</span>
            <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
              <RefreshCw className="w-4 h-4 text-white/50" />
            </button>
          </div>
        </div>
        {recommendation && (
          <div className="px-5 py-3 bg-accent-success/10 border-t border-accent-success/20">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-accent-success" />
              <span className="text-accent-success font-medium">Recommended: {recommendation}</span>
              <span className="text-white/50 text-sm">— {recommendationReason}</span>
            </div>
          </div>
        )}
      </div>

      {/* RLHF Methods Grid */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Training Methods Comparison</h3>
          <span className="text-sm text-white/50">{methods.length} methods available</span>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {methods.map((method, i) => (
              <div
                key={i}
                className={`p-4 rounded-lg border transition-all ${
                  method.name === recommendation
                    ? 'bg-accent-success/10 border-accent-success/30'
                    : 'bg-white/5 border-white/10 hover:border-accent-primary/30'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-lg font-bold text-accent-tertiary">{method.name}</h4>
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    method.complexity === 'Low' ? 'bg-accent-success/20 text-accent-success' :
                    method.complexity === 'Medium' ? 'bg-accent-warning/20 text-accent-warning' :
                    'bg-accent-danger/20 text-accent-danger'
                  }`}>
                    {method.complexity}
                  </span>
                </div>
                <p className="text-sm text-white/70 mb-1">{method.full_name}</p>
                <p className="text-xs text-white/50 mb-3">{method.description}</p>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-white/40">Speedup</span>
                    <span className="text-accent-success font-bold">{method.training_speedup}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/40">Quality</span>
                    <span className="text-white">{method.quality}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/40">Memory ~</span>
                    <span className="text-accent-info">{method.memory_estimate_gb.toFixed(0)}GB</span>
                  </div>
                </div>
                
                <div className="mt-3 pt-3 border-t border-white/5">
                  <div className="text-xs text-white/40 mb-1">Frameworks</div>
                  <div className="flex flex-wrap gap-1">
                    {method.frameworks.slice(0, 3).map((fw, j) => (
                      <span key={j} className="px-1.5 py-0.5 bg-white/5 rounded text-xs text-white/60">
                        {fw}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div className="mt-2">
                  <div className="text-xs text-white/40 mb-1">Best for</div>
                  <div className="text-xs text-white/60">
                    {method.use_cases.slice(0, 2).join(', ')}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Calculators Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Config Calculator */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Calculator className="w-5 h-5 text-accent-primary" />
              <h3 className="font-medium text-white">Config Generator</h3>
            </div>
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-white/50 block mb-1">Method</label>
                <select
                  value={configParams.method}
                  onChange={(e) => setConfigParams({ ...configParams, method: e.target.value })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                >
                  {methods.map((m) => (
                    <option key={m.name} value={m.name.toLowerCase()} className="bg-brand-bg">
                      {m.name}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-white/50 block mb-1">Model Size (B)</label>
                <input
                  type="number"
                  value={configParams.model_size}
                  onChange={(e) => setConfigParams({ ...configParams, model_size: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
              <div>
                <label className="text-xs text-white/50 block mb-1">GPUs</label>
                <input
                  type="number"
                  value={configParams.gpus}
                  onChange={(e) => setConfigParams({ ...configParams, gpus: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
              <div>
                <label className="text-xs text-white/50 block mb-1">Memory/GPU (GB)</label>
                <input
                  type="number"
                  value={configParams.memory_gb}
                  onChange={(e) => setConfigParams({ ...configParams, memory_gb: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
            </div>
            <button
              onClick={calculateConfig}
              disabled={configLoading}
              className="w-full px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg text-sm font-medium disabled:opacity-50"
            >
              {configLoading ? 'Generating...' : 'Generate Config'}
            </button>
            
            {configResult && (
              <div className="p-3 rounded-lg bg-white/5 border border-white/10 text-xs">
                {configResult.error ? (
                  <span className="text-accent-danger">{configResult.error}</span>
                ) : (
                  <div className="space-y-2">
                    <div className={`flex items-center gap-2 ${configResult.fits_in_memory ? 'text-accent-success' : 'text-accent-warning'}`}>
                      {configResult.fits_in_memory ? '✓' : '⚠'} 
                      {configResult.fits_in_memory ? 'Fits in memory' : 'May need optimization'}
                    </div>
                    <div className="text-white/60">
                      Memory: {configResult.memory_required_gb?.toFixed(1)}GB required
                    </div>
                    <div className="text-white/60">
                      Batch size: {configResult.recommended_batch_size}
                    </div>
                    <div className="mt-2 p-2 bg-black/20 rounded font-mono text-accent-primary">
                      {configResult.launch_command}
                    </div>
                    {configResult.optimizations?.length > 0 && (
                      <div className="mt-2">
                        <div className="text-white/40 mb-1">Recommended optimizations:</div>
                        {configResult.optimizations.map((opt: string, i: number) => (
                          <div key={i} className="text-accent-warning">• {opt}</div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Memory Estimator */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <HardDrive className="w-5 h-5 text-accent-secondary" />
              <h3 className="font-medium text-white">Memory Estimator</h3>
            </div>
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-white/50 block mb-1">Model Size (B)</label>
                <input
                  type="number"
                  value={memoryParams.model_size}
                  onChange={(e) => setMemoryParams({ ...memoryParams, model_size: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
              <div>
                <label className="text-xs text-white/50 block mb-1">Method</label>
                <select
                  value={memoryParams.method}
                  onChange={(e) => setMemoryParams({ ...memoryParams, method: e.target.value })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                >
                  {methods.map((m) => (
                    <option key={m.name} value={m.name.toLowerCase()} className="bg-brand-bg">
                      {m.name}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-white/50 block mb-1">Precision</label>
                <select
                  value={memoryParams.precision}
                  onChange={(e) => setMemoryParams({ ...memoryParams, precision: e.target.value })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                >
                  {['fp32', 'bf16', 'fp16', 'fp8', 'int8'].map((p) => (
                    <option key={p} value={p} className="bg-brand-bg">{p.toUpperCase()}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-white/50 block mb-1">Batch Size</label>
                <input
                  type="number"
                  value={memoryParams.batch_size}
                  onChange={(e) => setMemoryParams({ ...memoryParams, batch_size: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
            </div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={memoryParams.use_lora}
                onChange={(e) => setMemoryParams({ ...memoryParams, use_lora: e.target.checked })}
                className="w-4 h-4 accent-accent-primary"
              />
              <span className="text-sm text-white/70">Use LoRA (reduces memory ~90%)</span>
            </label>
            <button
              onClick={estimateMemory}
              disabled={memoryLoading}
              className="w-full px-4 py-2 bg-accent-secondary/20 text-accent-secondary rounded-lg text-sm font-medium disabled:opacity-50"
            >
              {memoryLoading ? 'Calculating...' : 'Estimate Memory'}
            </button>
            
            {memoryResult && (
              <div className="p-3 rounded-lg bg-white/5 border border-white/10 text-xs">
                {memoryResult.error ? (
                  <span className="text-accent-danger">{memoryResult.error}</span>
                ) : (
                  <div className="space-y-2">
                    <div className="text-lg font-bold text-accent-info">
                      {memoryResult.total_memory_gb?.toFixed(1)} GB Total
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-white/60">
                      <div>Model: {memoryResult.breakdown?.model_weights}GB</div>
                      <div>Optimizer: {memoryResult.breakdown?.optimizer_states}GB</div>
                      <div>Gradients: {memoryResult.breakdown?.gradients}GB</div>
                      <div>Activations: {memoryResult.breakdown?.activations}GB</div>
                    </div>
                    <div className={`mt-2 ${memoryResult.fits_single_gpu_80gb ? 'text-accent-success' : 'text-accent-warning'}`}>
                      {memoryResult.fits_single_gpu_80gb 
                        ? '✓ Fits on single 80GB GPU' 
                        : `⚠ Needs ${memoryResult.recommended_gpus} GPUs minimum`}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-accent-warning" />
            <h3 className="font-medium text-white">Best Practices</h3>
          </div>
        </div>
        <div className="card-body space-y-3">
          {[
            { title: 'Start with DPO', desc: 'Use DPO instead of PPO when possible for 5-10x faster training with similar quality' },
            { title: 'Enable gradient checkpointing', desc: 'Reduces memory by 60-80% with ~20% training slowdown' },
            { title: 'Use LoRA/QLoRA', desc: 'For efficient fine-tuning with minimal memory overhead (10% of full fine-tune)' },
            { title: 'Consider RLAIF', desc: 'Use AI feedback for automated preference data generation at scale' },
            { title: 'Freeze reference model', desc: 'Always freeze the reference model and use FP16/BF16 inference' },
            { title: 'Use vLLM for generation', desc: 'Use vLLM for fast response generation during PPO rollouts' },
          ].map((rec, i) => (
            <div
              key={i}
              className="p-4 bg-gradient-to-r from-accent-warning/10 to-transparent rounded-lg border-l-2 border-accent-warning"
            >
              <div className="font-medium text-white mb-1">{rec.title}</div>
              <p className="text-sm text-white/60">{rec.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
