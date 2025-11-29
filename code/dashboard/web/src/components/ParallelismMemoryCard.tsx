'use client';

import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, Brain, Loader2, RefreshCw } from 'lucide-react';
import { getParallelismMemory } from '@/lib/api';
import { formatNumber } from '@/lib/utils';
import { getErrorMessage } from '@/lib/useApi';

type Props = {
  defaultModel?: string;
  defaultTp?: number;
  defaultPp?: number;
  defaultDp?: number;
  strategy?: {
    model?: string;
    tp?: number;
    pp?: number;
    dp?: number;
    batchSize?: number;
    seqLength?: number;
  };
};

type MemoryForm = {
  model: string;
  batchSize: number;
  seqLength: number;
  tp: number;
  pp: number;
  dp: number;
};

export function ParallelismMemoryCard({
  defaultModel,
  defaultTp = 1,
  defaultPp = 1,
  defaultDp = 8,
  strategy,
}: Props) {
  const [form, setForm] = useState<MemoryForm>({
    model: defaultModel || 'llama-3.1-70b',
    batchSize: 8,
    seqLength: 4096,
    tp: defaultTp,
    pp: defaultPp,
    dp: defaultDp,
  });
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (defaultModel) {
      setForm((prev) => ({ ...prev, model: defaultModel }));
    }
  }, [defaultModel]);

  useEffect(() => {
    if (!strategy) return;
    setForm((prev) => ({
      ...prev,
      model: strategy.model || prev.model,
      tp: strategy.tp ?? prev.tp,
      pp: strategy.pp ?? prev.pp,
      dp: strategy.dp ?? prev.dp,
      batchSize: strategy.batchSize ?? prev.batchSize,
      seqLength: strategy.seqLength ?? prev.seqLength,
    }));
  }, [strategy]);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getParallelismMemory({
        model: form.model,
        batch_size: form.batchSize,
        seq_length: form.seqLength,
        tp: form.tp,
        pp: form.pp,
        dp: form.dp,
      });
      setData(res);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to load memory analysis'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const breakdownEntries = useMemo(() => {
    if (!data) return [];
    const base = data.memory || data.memory_analysis || data.breakdown || data.usage || data;
    if (!base || typeof base !== 'object') return [];
    const entries: [string, number][] = [];
    Object.entries(base).forEach(([key, value]) => {
      if (typeof value === 'number' && Number.isFinite(value)) {
        entries.push([key, value]);
      } else if (value && typeof value === 'object' && typeof (value as any).gb === 'number') {
        entries.push([key, (value as any).gb]);
      }
    });
    return entries;
  }, [data]);

  const totalMemory = useMemo(() => {
    if (!data) return 0;
    const base = data.memory || data.memory_analysis || data.breakdown || data.usage || data;
    const explicit =
      base?.total ??
      base?.total_gb ??
      base?.total_memory ??
      base?.total_memory_gb ??
      base?.total_estimated ??
      base?.memory_gb;
    if (typeof explicit === 'number') return explicit;
    const sum = breakdownEntries.reduce((acc, [, value]) => acc + value, 0);
    return sum;
  }, [breakdownEntries, data]);

  const callouts = useMemo(() => {
    if (!data) return [];
    const raw = data.recommendations || data.memory_optimizations || data.notes || [];
    if (Array.isArray(raw)) return raw.filter(Boolean);
    return raw ? [raw] : [];
  }, [data]);

  const fits =
    data?.fits ??
    data?.fits_current_setup ??
    (data?.memory_analysis
      ? data.memory_analysis.memory_per_gpu_gb <= data.memory_analysis.gpu_memory_gb
      : undefined);

  return (
    <div className="card h-full">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-accent-warning" />
          <div>
            <h3 className="font-semibold text-white">Memory Breakdown</h3>
            <p className="text-xs text-white/50">Visualizes /api/parallelism/memory</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={load}
            className="p-2 rounded-lg hover:bg-white/5"
            aria-label="Refresh memory breakdown"
          >
            <RefreshCw className="w-4 h-4 text-white/50" />
          </button>
        </div>
      </div>

      <div className="card-body space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <label className="text-sm text-white/60">Model</label>
            <input
              value={form.model}
              onChange={(e) => setForm({ ...form, model: e.target.value })}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm text-white/60">Batch size</label>
            <input
              type="number"
              min={1}
              value={form.batchSize}
              onChange={(e) => setForm({ ...form, batchSize: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm text-white/60">Sequence length</label>
            <input
              type="number"
              min={128}
              value={form.seqLength}
              onChange={(e) => setForm({ ...form, seqLength: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
            />
          </div>
          <div className="grid grid-cols-3 gap-2">
            {(['tp', 'pp', 'dp'] as const).map((key) => (
              <div key={key} className="space-y-1">
                <label className="text-xs uppercase text-white/50">{key}</label>
                <input
                  type="number"
                  min={1}
                  value={form[key]}
                  onChange={(e) =>
                    setForm((prev) => ({ ...prev, [key]: Number(e.target.value) } as MemoryForm))
                  }
                  className="w-full px-2 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
                />
              </div>
            ))}
          </div>
        </div>

        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/10 text-white hover:bg-white/20 w-full md:w-auto disabled:opacity-60"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
          {loading ? 'Analyzing...' : 'Analyze memory footprint'}
        </button>

        {error && (
          <div className="flex items-center gap-2 text-sm text-accent-warning bg-accent-warning/10 border border-accent-warning/30 rounded-md px-3 py-2">
            <AlertTriangle className="w-4 h-4" />
            {error}
          </div>
        )}

        {breakdownEntries.length > 0 ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm text-white/70">
              <div>
                Total footprint{' '}
                <span className="text-white font-semibold">{formatNumber(totalMemory || 0, 1)} GB</span>
              </div>
              <div className={`px-2 py-1 rounded-full text-xs ${fits ? 'bg-accent-success/20 text-accent-success' : 'bg-accent-danger/20 text-accent-danger'}`}>
                {fits ? 'Fits current memory' : 'May exceed memory'}
              </div>
            </div>
            <div className="space-y-2">
              {breakdownEntries.map(([key, value]) => {
                const pct = totalMemory > 0 ? Math.min(100, (value / totalMemory) * 100) : 0;
                const label = key.replace(/_/g, ' ');
                return (
                  <div key={key}>
                    <div className="flex items-center justify-between text-xs text-white/60 mb-1">
                      <span className="uppercase tracking-wide">{label}</span>
                      <span className="font-mono text-white/80">{formatNumber(value, 2)} GB</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-accent-primary to-accent-secondary"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="text-sm text-white/60 flex items-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin text-white/30" />
            Waiting for backend to return a memory breakdown.
          </div>
        )}

        {callouts.length > 0 && (
          <div className="rounded-lg border border-white/10 bg-black/40 p-3 space-y-1">
            <div className="text-xs text-white/50 uppercase">Optimizer suggestions</div>
            <ul className="text-sm text-white/70 list-disc list-inside space-y-1">
              {callouts.slice(0, 4).map((item: any, idx: number) => (
                <li key={idx}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
