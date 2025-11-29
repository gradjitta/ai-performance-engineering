'use client';

import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Check, Clipboard, Layers, Loader2, RefreshCw, Route } from 'lucide-react';
import {
  getParallelismCalibration,
  getParallelismClusters,
  getParallelismPareto,
  getShardingPlan,
  recommendParallelism,
} from '@/lib/api';
import { formatNumber } from '@/lib/utils';

type StrategySelection = {
  model?: string;
  tp?: number;
  pp?: number;
  dp?: number;
  batchSize?: number;
  seqLength?: number;
};

export function ParallelismExtrasCard({ onApply }: { onApply?: (s: StrategySelection) => void }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);
  const [copied, setCopied] = useState(false);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const [cal, clusters, pareto, sharding, rec] = await Promise.all([
        getParallelismCalibration().catch(() => null),
        getParallelismClusters().catch(() => null),
        getParallelismPareto().catch(() => null),
        getShardingPlan({
          model: 'llama-3.1-70b',
          dp: 8,
          tp: 1,
          pp: 1,
          batch: 8,
          seq: 4096,
        }).catch(() => null),
        recommendParallelism({
          model: 'llama-3.1-70b',
          batch_size: 8,
          seq_length: 4096,
          goal: 'throughput',
          is_training: true,
        }).catch(() => null),
      ]);
      setData({ cal, clusters, pareto, sharding, rec });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load parallelism extras');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const strategy = useMemo(() => {
    const rec = data?.rec || {};
    const s = rec.strategy || rec.recommended_strategy || {};
    const memory = rec.memory_estimate_gb ?? rec.memory_gb;
    return {
      tp: s.tensor_parallel ?? s.tp ?? s.tp_size ?? 1,
      pp: s.pipeline_parallel ?? s.pp ?? s.pp_size ?? 1,
      dp: s.data_parallel ?? s.dp ?? s.dp_size ?? 1,
      total: s.total_gpus_needed || (s.tensor_parallel && s.pipeline_parallel && s.data_parallel
        ? s.tensor_parallel * s.pipeline_parallel * s.data_parallel
        : undefined),
      memory,
      fits: rec.fits_current_setup,
      notes: rec.recommendations || rec.notes || [],
    };
  }, [data]);

  const sharding = useMemo(() => data?.sharding || data?.sharding_plan || {}, [data]);
  const pareto = useMemo(() => {
    const entries = data?.pareto?.items || data?.pareto?.strategies || data?.pareto || [];
    return Array.isArray(entries) ? entries : [];
  }, [data]);
  const clusters = useMemo(() => {
    const raw = data?.clusters?.clusters ?? data?.clusters ?? [];
    if (Array.isArray(raw)) return raw;
    return raw ? [raw] : [];
  }, [data]);
  const calibration = useMemo(() => {
    const raw = data?.cal?.calibration ?? data?.cal ?? [];
    if (Array.isArray(raw)) return raw;
    return raw ? [raw] : [];
  }, [data]);

  const launchCommand = useMemo(() => {
    if (!strategy.tp || !strategy.pp || !strategy.dp) return '';
    return `torchrun --nproc_per_node=${strategy.tp * strategy.pp} train.py --tp ${strategy.tp} --pp ${strategy.pp} --dp ${strategy.dp}`;
  }, [strategy.dp, strategy.pp, strategy.tp]);

  const handleCopy = async () => {
    if (!launchCommand) return;
    try {
      await navigator.clipboard.writeText(launchCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (err) {
      console.error('Failed to copy launch command', err);
    }
  };

  const NoteList = ({ items }: { items: any[] }) => (
    <ul className="space-y-1 text-sm text-white/70 list-disc list-inside">
      {items.filter(Boolean).slice(0, 4).map((item, idx) => (
        <li key={idx}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>
      ))}
      {items.length === 0 && <li className="text-white/50">No additional notes from backend</li>}
    </ul>
  );

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-accent-secondary" />
          <h3 className="font-medium text-white">Parallelism Extras</h3>
        </div>
        <button onClick={load} className="p-2 rounded hover:bg-white/5" aria-label="Refresh parallelism extras">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-4">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" /> Loading parallelism helpers...
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-sm text-accent-warning">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid md:grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="text-xs text-white/50 uppercase">Recommended strategy</div>
                  {launchCommand && (
                    <button
                      onClick={handleCopy}
                      className="text-xs flex items-center gap-1 px-2 py-1 rounded bg-white/10 text-white hover:bg-white/20"
                    >
                      {copied ? <Check className="w-3 h-3 text-accent-success" /> : <Clipboard className="w-3 h-3" />}
                      {copied ? 'Copied' : 'Copy'}
                    </button>
                  )}
                </div>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 rounded bg-accent-primary/15 text-accent-primary text-xs">TP {strategy.tp}</span>
                  <span className="px-2 py-1 rounded bg-accent-secondary/15 text-accent-secondary text-xs">PP {strategy.pp}</span>
                  <span className="px-2 py-1 rounded bg-accent-tertiary/15 text-accent-tertiary text-xs">DP {strategy.dp}</span>
                  {strategy.total && (
                    <span className="px-2 py-1 rounded bg-white/10 text-white/70 text-xs">{strategy.total} GPUs total</span>
                  )}
                </div>
                <div className="text-xs text-white/60 flex items-center gap-2">
                  <Route className="w-3 h-3 text-accent-info" />
                  {strategy.fits ? 'Fits current cluster' : 'May need more GPUs'}
                  {strategy.memory && (
                    <>
                      · Est. memory{' '}
                      {typeof strategy.memory === 'number'
                        ? `${formatNumber(strategy.memory, 1)} GB`
                        : strategy.memory}
                    </>
                  )}
                </div>
                <NoteList items={Array.isArray(strategy.notes) ? strategy.notes : strategy.notes ? [strategy.notes] : []} />
                {onApply && (
                  <button
                    onClick={() =>
                      onApply({
                        model: data?.rec?.model,
                        tp: strategy.tp,
                        pp: strategy.pp,
                        dp: strategy.dp,
                        batchSize: data?.rec?.batch_size,
                        seqLength: data?.rec?.seq_length,
                      })
                    }
                    className="w-full text-xs px-3 py-2 rounded-lg bg-accent-primary/15 text-accent-primary hover:bg-accent-primary/25"
                  >
                    Apply to launch & forms
                  </button>
                )}
              </div>

              <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
                <div className="text-xs text-white/50 uppercase">Sharding plan</div>
                <div className="text-sm text-white/70">
                  {sharding.reason || sharding.summary || 'Backend-proposed sharding layout'}
                </div>
                <div className="flex flex-wrap gap-2 text-xs text-white/80">
                  {['dp', 'tp', 'pp', 'cp', 'ep'].map((k) => {
                    const val = sharding[k] || sharding[`${k}_size`];
                    return val ? (
                      <span key={k} className="px-2 py-1 rounded bg-white/10">
                        {k.toUpperCase()} {val}
                      </span>
                    ) : null;
                  })}
                </div>
                {sharding.memory_per_gpu_gb && (
                  <div className="text-xs text-white/60">
                    Memory/GPU: {formatNumber(sharding.memory_per_gpu_gb, 2)} GB · Target batch{' '}
                    {sharding.recommended_batch_size || sharding.batch_size}
                  </div>
                )}
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
                <div className="text-xs text-white/50 uppercase">Pareto frontier</div>
                {pareto.length > 0 ? (
                  <div className="space-y-1">
                    {pareto.slice(0, 4).map((item: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between text-sm text-white/80">
                        <span>{item.name || item.strategy || `Option ${idx + 1}`}</span>
                        <span className="text-white/60 text-xs">
                          {item.throughput || item.performance || item.tokens_per_sec || ''}{' '}
                          {item.cost && <span className="ml-1">· ${item.cost}</span>}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-white/60">No Pareto points returned.</div>
                )}
              </div>

              <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
                <div className="text-xs text-white/50 uppercase">Clusters</div>
                {clusters.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {clusters.slice(0, 4).map((c: any, idx: number) => (
                      <span key={idx} className="px-2 py-1 rounded bg-accent-primary/10 text-accent-primary text-xs">
                        {c.name || c.id || `Cluster ${idx + 1}`}
                      </span>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-white/60">No cluster presets available.</div>
                )}
              </div>

              <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
                <div className="text-xs text-white/50 uppercase">Calibration</div>
                {Array.isArray(calibration) && calibration.length > 0 ? (
                  <div className="space-y-1 text-sm text-white/70">
                    {calibration.slice(0, 4).map((c: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between">
                        <span>{c.name || c.metric || `Metric ${idx + 1}`}</span>
                        <span className="text-white/60">{c.value ?? c.score ?? ''}</span>
                      </div>
                    ))}
                  </div>
                ) : typeof calibration === 'object' && calibration ? (
                  <div className="space-y-1 text-sm text-white/70">
                    {Object.entries(calibration)
                      .slice(0, 4)
                      .map(([key, val]) => (
                        <div key={key} className="flex items-center justify-between">
                          <span className="text-white/60">{key}</span>
                          <span className="text-white/80">{String(val)}</span>
                        </div>
                      ))}
                  </div>
                ) : (
                  <div className="text-sm text-white/60">No calibration data yet.</div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
