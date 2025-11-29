'use client';

import { useMemo, useState } from 'react';
import { AlertTriangle, FileJson, Loader2, RefreshCw, Zap } from 'lucide-react';
import {
  getParallelismAutoTune,
  getParallelismBatchSize,
  getParallelismBottleneck,
  getParallelismCommOverlap,
  getParallelismCompare,
  getParallelismEstimate,
  getParallelismExport,
  getParallelismInferenceOpt,
  getParallelismLargeScale,
  getParallelismLongContext,
  getParallelismMoe,
  getParallelismNccl,
  getParallelismRLHF,
  getParallelismScaling,
  getParallelismVLLM,
  getParallelismWhatif,
} from '@/lib/api';
import { getErrorMessage } from '@/lib/useApi';

type StrategySelection = {
  model?: string;
  tp?: number;
  pp?: number;
  dp?: number;
  batchSize?: number;
  seqLength?: number;
};

type FieldDef = {
  key: string;
  label: string;
  type?: 'text' | 'number' | 'select' | 'checkbox';
  defaultValue?: any;
  min?: number;
  step?: number;
  options?: { value: string; label: string }[];
  placeholder?: string;
};

type CardDef = {
  key: string;
  title: string;
  description: string;
  fields: FieldDef[];
  cta?: string;
  onRun: (form: Record<string, any>) => Promise<any>;
  extractStrategy?: (result: any) => StrategySelection | null | undefined;
};

function JsonPreview({ data }: { data: any }) {
  if (data === null || data === undefined) {
    return <div className="text-xs text-white/50">No data yet</div>;
  }
  return (
    <pre className="text-[11px] leading-snug text-white/80 bg-black/50 rounded-lg p-2 border border-white/10 max-h-56 overflow-auto whitespace-pre-wrap break-all">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

function EndpointCard({
  title,
  description,
  fields,
  cta = 'Run',
  onRun,
  extractStrategy,
  onApplyStrategy,
}: CardDef & { onApplyStrategy?: (s: StrategySelection) => void }) {
  const initial = useMemo(
    () =>
      fields.reduce<Record<string, any>>((acc, field) => {
        acc[field.key] = field.defaultValue ?? (field.type === 'number' ? 0 : field.type === 'checkbox' ? false : '');
        return acc;
      }, {}),
    [fields]
  );
  const [form, setForm] = useState<Record<string, any>>(initial);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleChange = (key: string, value: any, type?: string) => {
    setForm((prev) => ({
      ...prev,
      [key]: type === 'number' ? Number(value) : value,
    }));
  };

  const run = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await onRun(form);
      setResult(res);
    } catch (err) {
      setError(getErrorMessage(err, 'Request failed'));
    } finally {
      setLoading(false);
    }
  };

  const strategy = extractStrategy ? extractStrategy(result) : null;

  return (
    <div className="p-4 rounded-lg border border-white/10 bg-white/5 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-white font-medium">{title}</div>
          <div className="text-xs text-white/50">{description}</div>
        </div>
        <button onClick={run} disabled={loading} className="p-2 rounded hover:bg-white/10 disabled:opacity-50">
          {loading ? <Loader2 className="w-4 h-4 animate-spin text-white/60" /> : <RefreshCw className="w-4 h-4 text-white/60" />}
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {fields.map((field) => (
          <div key={field.key} className="space-y-1">
            <label className="text-xs text-white/60">{field.label}</label>
            {field.type === 'select' ? (
              <select
                value={form[field.key]}
                onChange={(e) => handleChange(field.key, e.target.value, field.type)}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none"
              >
                {field.options?.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            ) : field.type === 'checkbox' ? (
              <label className="flex items-center gap-2 text-sm text-white/70">
                <input
                  type="checkbox"
                  checked={!!form[field.key]}
                  onChange={(e) => handleChange(field.key, e.target.checked, field.type)}
                  className="w-4 h-4 accent-accent-primary"
                />
                {field.label}
              </label>
            ) : (
              <input
                type={field.type || 'text'}
                value={form[field.key]}
                min={field.min}
                step={field.step}
                placeholder={field.placeholder}
                onChange={(e) => handleChange(field.key, e.target.value, field.type)}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none"
              />
            )}
          </div>
        ))}
      </div>

      <button
        onClick={run}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-white/10 text-white hover:bg-white/20 text-sm disabled:opacity-60"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
        {loading ? 'Running...' : cta}
      </button>

      {error && (
        <div className="flex items-center gap-2 text-xs text-accent-warning bg-accent-warning/10 border border-accent-warning/30 rounded-md px-3 py-2">
          <AlertTriangle className="w-4 h-4" />
          {error}
        </div>
      )}

      <JsonPreview data={result} />

      {strategy && onApplyStrategy && (
        <button
          onClick={() => onApplyStrategy(strategy)}
          className="w-full text-xs px-3 py-2 rounded-lg bg-accent-primary/15 text-accent-primary hover:bg-accent-primary/25"
        >
          Apply recommended strategy
        </button>
      )}
    </div>
  );
}

export function ParallelismAdvancedGrid({ onApplyStrategy }: { onApplyStrategy?: (s: StrategySelection) => void }) {
  const cards: CardDef[] = [
    {
      key: 'estimate',
      title: 'Training Estimate',
      description: '/api/parallelism/estimate',
      cta: 'Estimate',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        { key: 'tokens', label: 'Tokens', type: 'number', defaultValue: 1_000_000_000_000 },
        { key: 'throughput', label: 'Throughput (t/s)', type: 'number', defaultValue: 100000 },
        { key: 'gpus', label: 'GPUs', type: 'number', defaultValue: 8 },
        { key: 'gpu_cost', label: 'GPU $/hr', type: 'number', step: 0.1, defaultValue: 4 },
      ],
      onRun: (f) =>
        getParallelismEstimate({
          model: f.model,
          tokens: Number(f.tokens),
          throughput: Number(f.throughput),
          gpus: Number(f.gpus),
          gpu_cost: Number(f.gpu_cost),
        }),
    },
    {
      key: 'compare',
      title: 'Model Compare',
      description: '/api/parallelism/compare',
      cta: 'Compare',
      fields: [{ key: 'models', label: 'Models (comma separated)', defaultValue: 'llama-3.1-8b,llama-3.1-70b' }],
      onRun: (f) => getParallelismCompare(String(f.models || '').split(',').map((m: string) => m.trim()).filter(Boolean)),
    },
    {
      key: 'bottleneck',
      title: 'Bottleneck Analysis',
      description: '/api/parallelism/bottleneck',
      cta: 'Analyze',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        { key: 'batch_size', label: 'Batch', type: 'number', defaultValue: 8 },
        { key: 'seq_length', label: 'Seq', type: 'number', defaultValue: 4096 },
        { key: 'tp', label: 'TP', type: 'number', defaultValue: 1 },
        { key: 'pp', label: 'PP', type: 'number', defaultValue: 1 },
        { key: 'dp', label: 'DP', type: 'number', defaultValue: 8 },
      ],
      onRun: (f) =>
        getParallelismBottleneck({
          model: f.model,
          batch_size: Number(f.batch_size),
          seq_length: Number(f.seq_length),
          tp: Number(f.tp),
          pp: Number(f.pp),
          dp: Number(f.dp),
        }),
    },
    {
      key: 'scaling',
      title: 'Scaling Analysis',
      description: '/api/parallelism/scaling',
      cta: 'Evaluate',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        { key: 'throughput', label: 'Throughput', type: 'number', defaultValue: 100000 },
        { key: 'gpus', label: 'GPUs', type: 'number', defaultValue: 8 },
        { key: 'max_gpus', label: 'Max GPUs', type: 'number', defaultValue: 512 },
      ],
      onRun: (f) =>
        getParallelismScaling({
          model: f.model,
          throughput: Number(f.throughput),
          gpus: Number(f.gpus),
          max_gpus: Number(f.max_gpus),
        }),
    },
    {
      key: 'whatif',
      title: 'What-If Explorer',
      description: '/api/parallelism/whatif',
      cta: 'Simulate',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        { key: 'tp', label: 'TP', type: 'number', defaultValue: 1 },
        { key: 'pp', label: 'PP', type: 'number', defaultValue: 1 },
        { key: 'dp', label: 'DP', type: 'number', defaultValue: 8 },
        { key: 'batch_size', label: 'Batch', type: 'number', defaultValue: 8 },
      ],
      onRun: (f) =>
        getParallelismWhatif({
          model: f.model,
          tp: Number(f.tp),
          pp: Number(f.pp),
          dp: Number(f.dp),
          batch_size: Number(f.batch_size),
        }),
    },
    {
      key: 'batchsize',
      title: 'Batch Size Optimizer',
      description: '/api/parallelism/batch-size',
      cta: 'Optimize',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        { key: 'seq_length', label: 'Seq', type: 'number', defaultValue: 4096 },
        { key: 'tp', label: 'TP', type: 'number', defaultValue: 1 },
        { key: 'pp', label: 'PP', type: 'number', defaultValue: 1 },
        { key: 'dp', label: 'DP', type: 'number', defaultValue: 8 },
        { key: 'target_batch', label: 'Target batch', type: 'number', defaultValue: 1024 },
      ],
      onRun: (f) =>
        getParallelismBatchSize({
          model: f.model,
          seq_length: Number(f.seq_length),
          tp: Number(f.tp),
          pp: Number(f.pp),
          dp: Number(f.dp),
          target_batch: Number(f.target_batch),
        }),
    },
    {
      key: 'autotune',
      title: 'Auto-Tune',
      description: '/api/parallelism/auto-tune',
      cta: 'Auto-tune',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        {
          key: 'goal',
          label: 'Goal',
          type: 'select',
          options: [
            { value: 'throughput', label: 'Throughput' },
            { value: 'latency', label: 'Latency' },
            { value: 'balanced', label: 'Balanced' },
          ],
          defaultValue: 'throughput',
        },
        { key: 'target_batch', label: 'Target batch', type: 'number', defaultValue: 1024 },
      ],
      onRun: (f) =>
        getParallelismAutoTune({
          model: f.model,
          goal: f.goal,
          target_batch: Number(f.target_batch),
        }),
      extractStrategy: (r) => r?.strategy || r?.recommended_strategy,
    },
    {
      key: 'export',
      title: 'Export Config',
      description: '/api/parallelism/export',
      cta: 'Export',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        { key: 'nodes', label: 'Nodes', type: 'number', defaultValue: 1 },
        { key: 'gpus', label: 'GPUs', type: 'number', defaultValue: 8 },
        { key: 'tp', label: 'TP', type: 'number', defaultValue: 1 },
        { key: 'pp', label: 'PP', type: 'number', defaultValue: 1 },
        { key: 'dp', label: 'DP', type: 'number', defaultValue: 8 },
        { key: 'batch_size', label: 'Batch', type: 'number', defaultValue: 256 },
        { key: 'zero_stage', label: 'ZeRO stage', type: 'number', defaultValue: 2 },
      ],
      onRun: (f) =>
        getParallelismExport({
          model: f.model,
          nodes: Number(f.nodes),
          gpus: Number(f.gpus),
          tp: Number(f.tp),
          pp: Number(f.pp),
          dp: Number(f.dp),
          batch_size: Number(f.batch_size),
          zero_stage: Number(f.zero_stage),
        }),
    },
    {
      key: 'moe',
      title: 'MoE Optimization',
      description: '/api/parallelism/moe',
      cta: 'Get MoE plan',
      fields: [{ key: 'model', label: 'Model', defaultValue: 'mixtral-8x7b' }],
      onRun: (f) => getParallelismMoe({ model: f.model }),
      extractStrategy: (r) => r?.moe_config,
    },
    {
      key: 'longcontext',
      title: 'Long Context',
      description: '/api/parallelism/long-context',
      cta: 'Optimize',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        { key: 'seq_length', label: 'Seq length', type: 'number', defaultValue: 128000 },
      ],
      onRun: (f) =>
        getParallelismLongContext({
          model: f.model,
          seq_length: Number(f.seq_length),
        }),
    },
    {
      key: 'commoverlap',
      title: 'Comm Overlap',
      description: '/api/parallelism/comm-overlap',
      cta: 'Analyze',
      fields: [{ key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' }],
      onRun: (f) => getParallelismCommOverlap({ model: f.model }),
    },
    {
      key: 'vllm',
      title: 'vLLM Config',
      description: '/api/parallelism/vllm',
      cta: 'Generate',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        {
          key: 'goal',
          label: 'Goal',
          type: 'select',
          options: [
            { value: 'throughput', label: 'Throughput' },
            { value: 'latency', label: 'Latency' },
          ],
          defaultValue: 'throughput',
        },
        { key: 'gpus', label: 'GPUs', type: 'number', defaultValue: 1 },
        { key: 'seq', label: 'Max seq', type: 'number', defaultValue: 8192 },
        { key: 'compare', label: 'Compare engines', type: 'checkbox', defaultValue: false },
      ],
      onRun: (f) =>
        getParallelismVLLM({
          model: f.model,
          goal: f.goal,
          gpus: Number(f.gpus),
          max_seq_len: Number(f.seq),
          compare: !!f.compare,
        }),
    },
    {
      key: 'rlhf',
      title: 'RLHF Parallelism',
      description: '/api/parallelism/rlhf',
      cta: 'Plan RLHF',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        {
          key: 'algorithm',
          label: 'Algorithm',
          type: 'select',
          options: [
            { value: 'ppo', label: 'PPO' },
            { value: 'dpo', label: 'DPO' },
            { value: 'kto', label: 'KTO' },
          ],
          defaultValue: 'ppo',
        },
        { key: 'compare', label: 'Compare', type: 'checkbox', defaultValue: false },
      ],
      onRun: (f) => getParallelismRLHF({ model: f.model, algorithm: f.algorithm, compare: !!f.compare }),
    },
    {
      key: 'inferenceopt',
      title: 'Inference Optimization',
      description: '/api/parallelism/inference-opt',
      cta: 'Get tips',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        {
          key: 'goal',
          label: 'Goal',
          type: 'select',
          options: [
            { value: 'throughput', label: 'Throughput' },
            { value: 'latency', label: 'Latency' },
          ],
          defaultValue: 'throughput',
        },
      ],
      onRun: (f) => getParallelismInferenceOpt({ model: f.model, goal: f.goal }),
    },
    {
      key: 'nccl',
      title: 'NCCL Advisor',
      description: '/api/parallelism/nccl',
      cta: 'Diagnose',
      fields: [
        { key: 'nodes', label: 'Nodes', type: 'number', defaultValue: 1 },
        { key: 'gpus', label: 'GPUs per node', type: 'number', defaultValue: 8 },
        { key: 'diagnose', label: 'Include diagnosis', type: 'checkbox', defaultValue: false },
      ],
      onRun: (f) => getParallelismNccl({ nodes: Number(f.nodes), gpus: Number(f.gpus), diagnose: !!f.diagnose }),
    },
    {
      key: 'largescale',
      title: 'Large-Scale Plan',
      description: '/api/parallelism/large-scale',
      cta: 'Generate',
      fields: [
        { key: 'model', label: 'Model', defaultValue: 'llama-3.1-70b' },
        { key: 'nodes', label: 'Nodes', type: 'number', defaultValue: 8 },
        { key: 'gpus_per_node', label: 'GPUs/node', type: 'number', defaultValue: 8 },
        { key: 'network', label: 'Network', defaultValue: 'infiniband' },
        { key: 'batch_size', label: 'Batch', type: 'number', defaultValue: 1024 },
      ],
      onRun: (f) =>
        getParallelismLargeScale({
          model: f.model,
          nodes: Number(f.nodes),
          gpus_per_node: Number(f.gpus_per_node),
          network: f.network,
          batch_size: Number(f.batch_size),
        }),
      extractStrategy: (r) => r?.strategy,
    },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <FileJson className="w-5 h-5 text-accent-info" />
        <div>
          <div className="text-white font-semibold">Parallelism Lab</div>
          <div className="text-xs text-white/50">All remaining endpoints wired with quick-run cards</div>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        {cards.map((card) => (
          <EndpointCard key={card.key} {...card} onApplyStrategy={onApplyStrategy} />
        ))}
      </div>
    </div>
  );
}
