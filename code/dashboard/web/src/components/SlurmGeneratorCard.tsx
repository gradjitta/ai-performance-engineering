'use client';

import { useEffect, useState } from 'react';
import { Check, Copy, FileCode, Loader2, RefreshCw, ServerCog } from 'lucide-react';
import { getParallelismSlurm } from '@/lib/api';
import { getErrorMessage } from '@/lib/useApi';

type StrategySelection = {
  model?: string;
  tp?: number;
  pp?: number;
  dp?: number;
  batchSize?: number;
  seqLength?: number;
};

type Props = {
  defaultModel?: string;
  defaultGpus?: number;
  gpusPerNode?: number;
  strategy?: StrategySelection;
};

export function SlurmGeneratorCard({ defaultModel, defaultGpus, gpusPerNode, strategy }: Props) {
  const [form, setForm] = useState({
    model: defaultModel || 'llama-3.1-70b',
    nodes: 1,
    gpus: defaultGpus || 8,
    framework: 'torchrun',
  });
  const [script, setScript] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (defaultGpus) {
      setForm((prev) => ({ ...prev, gpus: defaultGpus }));
    }
    if (defaultModel) {
      setForm((prev) => ({ ...prev, model: defaultModel }));
    }
  }, [defaultGpus, defaultModel]);

  useEffect(() => {
    if (!strategy) return;
    const totalGpus = (strategy.tp || 1) * (strategy.pp || 1) * (strategy.dp || form.gpus);
    const perNode = gpusPerNode || form.gpus || 8;
    const nodes = Math.max(1, Math.ceil(totalGpus / perNode));
    setForm((prev) => ({
      ...prev,
      model: strategy.model || prev.model,
      nodes,
      gpus: perNode,
    }));
  }, [strategy, gpusPerNode]);

  const generate = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getParallelismSlurm(form);
      const rawScript = (res as any)?.slurm_script || (res as any)?.script || res;
      setScript(typeof rawScript === 'string' ? rawScript : JSON.stringify(rawScript, null, 2));
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to generate SLURM script'));
    } finally {
      setLoading(false);
    }
  };

  const copyScript = async () => {
    if (!script) return;
    try {
      await navigator.clipboard.writeText(script);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (err) {
      console.error('Failed to copy SLURM script', err);
    }
  };

  return (
    <div className="card h-full">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <ServerCog className="w-5 h-5 text-accent-info" />
          <div>
            <h3 className="font-semibold text-white">SLURM Script Generator</h3>
            <p className="text-xs text-white/50">Wire-up for /api/parallelism/slurm</p>
          </div>
        </div>
        <button
          onClick={generate}
          className="p-2 rounded-lg hover:bg-white/5"
          aria-label="Generate SLURM script"
        >
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-3">
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
            <label className="text-sm text-white/60">Framework</label>
            <select
              value={form.framework}
              onChange={(e) => setForm({ ...form, framework: e.target.value })}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
            >
              <option value="torchrun">torchrun</option>
              <option value="deepspeed">DeepSpeed</option>
              <option value="megablocks">Megatron</option>
            </select>
          </div>
          <div className="space-y-2">
            <label className="text-sm text-white/60">Nodes</label>
            <input
              type="number"
              min={1}
              value={form.nodes}
              onChange={(e) => setForm({ ...form, nodes: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm text-white/60">GPUs per node</label>
            <input
              type="number"
              min={1}
              value={form.gpus}
              onChange={(e) => setForm({ ...form, gpus: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
            />
          </div>
        </div>

        <button
          onClick={generate}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-white/10 text-white hover:bg-white/20 disabled:opacity-60"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileCode className="w-4 h-4" />}
          {loading ? 'Generating...' : 'Generate script'}
        </button>

        {error && <div className="text-xs text-accent-warning bg-accent-warning/10 border border-accent-warning/30 rounded-md px-3 py-2">{error}</div>}

        <div className="rounded-lg bg-black/60 border border-white/10 p-3 font-mono text-xs text-white/80 max-h-64 overflow-auto">
          {script ? script : 'Click "Generate script" to pull a SLURM launch snippet from the backend.'}
        </div>

        <div className="flex items-center justify-between">
          <div className="text-xs text-white/50">Ready-to-copy job script</div>
          <button
            onClick={copyScript}
            disabled={!script}
            className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-white/10 text-white hover:bg-white/20 disabled:opacity-40"
          >
            {copied ? <Check className="w-3 h-3 text-accent-success" /> : <Copy className="w-3 h-3" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>
    </div>
  );
}
