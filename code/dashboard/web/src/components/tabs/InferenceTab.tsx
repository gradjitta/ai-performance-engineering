'use client';

import { useState, useEffect, useCallback } from 'react';
import { Gauge, RefreshCw, Zap, Clock, Server, HardDrive, Target, Cpu, Search } from 'lucide-react';
import {
  getInferenceEngines,
  getInferenceOptimizationTechniques,
  getInferenceModelsFit,
  getHfTrending,
  searchHfModels,
  getHfModel,
} from '@/lib/api';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';

interface InferenceEngine {
  name: string;
  version: string;
  description: string;
  throughput_multiplier: number;
  latency_reduction: number;
  memory_efficiency: number;
  features: string[];
  best_for: string[];
  supported_models: string[];
  pros: string[];
  cons: string[];
  install: string;
  example: string;
  estimated_throughput_tps: number;
  estimated_ttft_ms: number;
  estimated_memory_gb: number;
}

interface OptimizationTechnique {
  name: string;
  category: string;
  impact: string;
  description: string;
  complexity: string;
  supported_by: string[];
  implementation: string;
}

interface ModelFit {
  name: string;
  params_b: number;
  fits_bf16: boolean;
  fits_int8: boolean;
  fits_int4: boolean;
  memory_bf16_gb: number;
  memory_int8_gb: number;
  memory_int4_gb: number;
  recommended_precision: string;
}

export function InferenceTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [engines, setEngines] = useState<InferenceEngine[]>([]);
  const [techniques, setTechniques] = useState<OptimizationTechnique[]>([]);
  const [modelsFit, setModelsFit] = useState<ModelFit[]>([]);
  const [trending, setTrending] = useState<any[]>([]);
  const [recommendation, setRecommendation] = useState<string>('');
  const [gpuInfo, setGpuInfo] = useState<any>(null);
  const [selectedEngine, setSelectedEngine] = useState<string | null>(null);
  const [hfQuery, setHfQuery] = useState('llama');
  const [hfResults, setHfResults] = useState<any[]>([]);
  const [hfModelId, setHfModelId] = useState<string | null>(null);
  const [hfModel, setHfModel] = useState<any>(null);
  const [hfLoading, setHfLoading] = useState(false);
  const [hfError, setHfError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [enginesData, techniquesData, modelsData, trendingData] = await Promise.allSettled([
        getInferenceEngines(),
        getInferenceOptimizationTechniques(),
        getInferenceModelsFit(),
        getHfTrending(),
      ]);
      
      if (enginesData.status === 'fulfilled') {
        const data = enginesData.value as any;
        setEngines(data.engines || []);
        setRecommendation(data.recommendation || '');
        setGpuInfo(data.gpu_info || null);
      }
      
      if (techniquesData.status === 'fulfilled') {
        const data = techniquesData.value as any;
        setTechniques(data.techniques || []);
      }
      
      if (modelsData.status === 'fulfilled') {
        const data = modelsData.value as any;
        setModelsFit(data.models || []);
      }
      
      if (trendingData.status === 'fulfilled') {
        const data = trendingData.value as any;
        setTrending(data.models || data || []);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load inference data');
    } finally {
      setLoading(false);
    }
  }, []);

  const runHfSearch = useCallback(async () => {
    try {
      setHfLoading(true);
      setHfError(null);
      const res = await searchHfModels(hfQuery);
      const models = (res as any)?.models || (res as any)?.results || res || [];
      setHfResults(Array.isArray(models) ? models : []);
      const first = (Array.isArray(models) && models[0]) || null;
      const id = first?.modelId || first?.id || first?.model || first?.name;
      if (id) {
        setHfModelId(id);
        const detail = await getHfModel(id);
        setHfModel(detail);
      }
    } catch (e) {
      setHfError(e instanceof Error ? e.message : 'HuggingFace search failed');
    } finally {
      setHfLoading(false);
    }
  }, [hfQuery]);

  const loadHfModel = useCallback(
    async (modelId: string) => {
      if (!modelId) return;
      try {
        setHfLoading(true);
        setHfError(null);
        setHfModelId(modelId);
        const detail = await getHfModel(modelId);
        setHfModel(detail);
      } catch (e) {
        setHfError(e instanceof Error ? e.message : 'Failed to load model details');
      } finally {
        setHfLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    void runHfSearch();
  }, [runHfSearch]);

  if (loading) {
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading inference optimization data..." />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body">
          <ErrorState
            message={error}
            onRetry={loadData}
          />
        </div>
      </div>
    );
  }

  const selectedEngineData = engines.find(e => e.name === selectedEngine);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Gauge className="w-5 h-5 text-accent-info" />
            <h2 className="text-lg font-semibold text-white">Inference Optimization</h2>
          </div>
          <div className="flex items-center gap-3">
            {gpuInfo && (
              <span className="text-sm text-white/50">
                {gpuInfo.name} • {gpuInfo.memory_gb}GB
              </span>
            )}
            <button
              onClick={loadData}
              className="p-2 hover:bg-white/5 rounded-lg text-white/70"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
        {recommendation && (
          <div className="px-5 py-3 bg-accent-success/10 border-t border-accent-success/20">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-accent-success" />
              <span className="text-accent-success font-medium">Recommended: {recommendation}</span>
            </div>
          </div>
        )}
      </div>

      {/* Inference Engines */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Server className="w-5 h-5 text-accent-primary" />
            <h3 className="font-medium text-white">Inference Engines</h3>
          </div>
          <span className="text-sm text-white/50">{engines.length} engines compared</span>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            {engines.map((engine, i) => (
              <div
                key={i}
                className={`p-4 rounded-lg border cursor-pointer transition-all ${
                  engine.name === recommendation
                    ? 'bg-accent-success/10 border-accent-success/30'
                    : selectedEngine === engine.name
                    ? 'bg-accent-primary/10 border-accent-primary/30'
                    : 'bg-white/5 border-white/10 hover:border-accent-primary/30'
                }`}
                onClick={() => setSelectedEngine(selectedEngine === engine.name ? null : engine.name)}
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-lg font-bold text-accent-info">{engine.name}</h4>
                  {engine.name === recommendation && (
                    <span className="px-1.5 py-0.5 bg-accent-success/20 text-accent-success text-xs rounded">
                      Best
                    </span>
                  )}
                </div>
                <p className="text-xs text-white/50 mb-3">{engine.description}</p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-white/40">Throughput</span>
                    <span className="text-accent-success font-bold">{engine.estimated_throughput_tps.toLocaleString()} tok/s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/40">TTFT</span>
                    <span className="text-accent-primary font-bold">{engine.estimated_ttft_ms}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/40">Memory</span>
                    <span className="text-accent-info">{engine.estimated_memory_gb}GB</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Selected Engine Details */}
      {selectedEngineData && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">{selectedEngineData.name} Details</h3>
            <span className="text-sm text-white/50">v{selectedEngineData.version}</span>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-sm font-medium text-white/70 mb-2">Features</h4>
                <div className="flex flex-wrap gap-1">
                  {selectedEngineData.features.map((f, i) => (
                    <span key={i} className="px-2 py-1 bg-accent-primary/10 text-accent-primary text-xs rounded">
                      {f}
                    </span>
                  ))}
                </div>
                
                <h4 className="text-sm font-medium text-white/70 mt-4 mb-2">Best For</h4>
                <ul className="text-sm text-white/60 space-y-1">
                  {selectedEngineData.best_for.map((use, i) => (
                    <li key={i}>• {use}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-medium text-accent-success mb-2">Pros</h4>
                <ul className="text-sm text-white/60 space-y-1 mb-4">
                  {selectedEngineData.pros.map((pro, i) => (
                    <li key={i} className="flex items-center gap-2">
                      <span className="text-accent-success">✓</span> {pro}
                    </li>
                  ))}
                </ul>
                
                <h4 className="text-sm font-medium text-accent-warning mb-2">Cons</h4>
                <ul className="text-sm text-white/60 space-y-1">
                  {selectedEngineData.cons.map((con, i) => (
                    <li key={i} className="flex items-center gap-2">
                      <span className="text-accent-warning">•</span> {con}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            
            <div className="mt-4 p-3 bg-black/20 rounded-lg">
              <div className="text-xs text-white/40 mb-1">Install</div>
              <code className="text-sm text-accent-primary font-mono">{selectedEngineData.install}</code>
            </div>
            <div className="mt-2 p-3 bg-black/20 rounded-lg">
              <div className="text-xs text-white/40 mb-1">Example</div>
              <code className="text-sm text-accent-secondary font-mono break-all">{selectedEngineData.example}</code>
            </div>
          </div>
        </div>
      )}

      {/* Optimization Techniques */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-accent-warning" />
            <h3 className="font-medium text-white">Optimization Techniques</h3>
          </div>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {techniques.map((tech, i) => (
              <div key={i} className="p-4 bg-white/5 rounded-lg flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-medium text-white">{tech.name}</h4>
                    <span className={`px-1.5 py-0.5 text-xs rounded ${
                      tech.category === 'Throughput' ? 'bg-accent-success/20 text-accent-success' :
                      tech.category === 'Memory' ? 'bg-accent-info/20 text-accent-info' :
                      tech.category === 'Latency' ? 'bg-accent-primary/20 text-accent-primary' :
                      'bg-white/10 text-white/60'
                    }`}>
                      {tech.category}
                    </span>
                    <span className={`px-1.5 py-0.5 text-xs rounded ${
                      tech.complexity === 'Low' ? 'bg-accent-success/10 text-accent-success' :
                      tech.complexity === 'Medium' ? 'bg-accent-warning/10 text-accent-warning' :
                      'bg-accent-danger/10 text-accent-danger'
                    }`}>
                      {tech.complexity}
                    </span>
                  </div>
                  <p className="text-sm text-white/50 mb-2">{tech.description}</p>
                  <div className="flex flex-wrap gap-1">
                    {tech.supported_by.slice(0, 4).map((fw, j) => (
                      <span key={j} className="px-1.5 py-0.5 bg-white/5 text-white/40 text-xs rounded">
                        {fw}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-accent-success font-bold text-sm">{tech.impact}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Models That Fit */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-accent-secondary" />
            <h3 className="font-medium text-white">Models That Fit Your GPU</h3>
          </div>
          <span className="text-sm text-white/50">{modelsFit.length} models available</span>
        </div>
        <div className="card-body">
          {modelsFit.length === 0 ? (
            <EmptyState
              title="No models data"
              description="Unable to determine which models fit. Check GPU detection."
              actionLabel="Refresh"
              onAction={loadData}
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-white/50 border-b border-white/10">
                    <th className="text-left py-2 px-3">Model</th>
                    <th className="text-center py-2 px-3">Params</th>
                    <th className="text-center py-2 px-3">BF16</th>
                    <th className="text-center py-2 px-3">INT8</th>
                    <th className="text-center py-2 px-3">INT4</th>
                    <th className="text-center py-2 px-3">Recommended</th>
                  </tr>
                </thead>
                <tbody>
                  {modelsFit.map((model, i) => (
                    <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-2 px-3 font-medium text-white">{model.name}</td>
                      <td className="py-2 px-3 text-center text-white/60">{model.params_b}B</td>
                      <td className="py-2 px-3 text-center">
                        {model.fits_bf16 ? (
                          <span className="text-accent-success">✓ {model.memory_bf16_gb}GB</span>
                        ) : (
                          <span className="text-white/30">—</span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-center">
                        {model.fits_int8 ? (
                          <span className="text-accent-success">✓ {model.memory_int8_gb}GB</span>
                        ) : (
                          <span className="text-white/30">—</span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-center">
                        {model.fits_int4 ? (
                          <span className="text-accent-success">✓ {model.memory_int4_gb}GB</span>
                        ) : (
                          <span className="text-white/30">—</span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          model.recommended_precision === 'bf16' ? 'bg-accent-success/20 text-accent-success' :
                          model.recommended_precision === 'int8' ? 'bg-accent-warning/20 text-accent-warning' :
                          model.recommended_precision === 'int4' ? 'bg-accent-info/20 text-accent-info' :
                          'bg-accent-danger/20 text-accent-danger'
                        }`}>
                          {model.recommended_precision.toUpperCase()}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Trending Models */}
      {trending.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-accent-tertiary" />
              <h3 className="font-medium text-white">Trending on HuggingFace</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {trending.slice(0, 6).map((model: any, i: number) => (
                <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10">
                  <div className="flex items-center justify-between">
                    <div className="min-w-0">
                      <div className="text-white font-semibold truncate">{model.id || model.name || model.modelId}</div>
                      <div className="text-xs text-white/50">{model.library_name || model.framework || 'transformers'}</div>
                    </div>
                    {model.downloads && (
                      <div className="text-xs text-accent-primary text-right ml-2">
                        {Number(model.downloads).toLocaleString()} downloads
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Search className="w-5 h-5 text-accent-primary" />
            <h3 className="font-medium text-white">HuggingFace Browser</h3>
          </div>
          <div className="flex items-center gap-2">
            <input
              value={hfQuery}
              onChange={(e) => setHfQuery(e.target.value)}
              className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
              placeholder="Search models (e.g., llama)"
            />
            <button
              onClick={runHfSearch}
              className="px-3 py-2 rounded-lg bg-accent-primary/20 text-accent-primary text-sm"
              disabled={hfLoading}
            >
              {hfLoading ? 'Searching…' : 'Search'}
            </button>
          </div>
        </div>
        <div className="card-body">
          {hfError && <div className="text-sm text-accent-warning mb-2">{hfError}</div>}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="md:col-span-1 space-y-2">
              {hfLoading && <div className="text-sm text-white/60">Loading models…</div>}
              {(hfResults.length ? hfResults : trending).slice(0, 10).map((m: any, idx: number) => {
                const id = m.modelId || m.id || m.model || m.name;
                return (
                  <button
                    key={idx}
                    onClick={() => loadHfModel(id)}
                    className={`w-full text-left p-3 rounded-lg border ${
                      hfModelId === id ? 'border-accent-primary/50 bg-accent-primary/10' : 'border-white/10 bg-white/5'
                    }`}
                  >
                    <div className="text-white font-semibold truncate">{id}</div>
                    <div className="text-xs text-white/50">
                      {(m.downloads && `${Number(m.downloads).toLocaleString()} downloads`) || m.library_name || ''}
                    </div>
                  </button>
                );
              })}
            </div>
            <div className="md:col-span-2">
              {hfModel ? (
                <pre className="text-xs text-white/80 bg-white/5 border border-white/10 rounded-lg p-3 whitespace-pre-wrap overflow-x-auto">
                  {JSON.stringify(hfModel, null, 2)}
                </pre>
              ) : (
                <div className="text-sm text-white/50">
                  Select a model to view metadata, configs, and card details.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
