'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  ArrowRight,
  Brain,
  Check,
  CheckCircle2,
  Copy,
  Loader2,
  Server,
  Sparkles,
  Target,
  Wand2,
} from 'lucide-react';
import { recommendParallelism } from '@/lib/api';
import { formatNumber } from '@/lib/utils';
import { getErrorMessage } from '@/lib/useApi';

type StrategySelection = {
  model?: string;
  tp?: number;
  pp?: number;
  dp?: number;
  batchSize?: number;
  seqLength?: number;
};

type WizardProps = {
  topology?: any;
  initialStrategy?: StrategySelection;
  onApply?: (strategy: StrategySelection) => void;
};

type WizardForm = {
  model: string;
  batchSize: number;
  seqLength: number;
  goal: string;
  isTraining: boolean;
  gpus?: number;
};

export function ParallelismWizard({ topology, initialStrategy, onApply }: WizardProps) {
  const [activeStep, setActiveStep] = useState(0);
  const [form, setForm] = useState<WizardForm>({
    model: 'llama-3.1-70b',
    batchSize: 8,
    seqLength: 4096,
    goal: 'throughput',
    isTraining: true,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendation, setRecommendation] = useState<any | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (topology?.total_gpus && !form.gpus) {
      setForm((prev) => ({ ...prev, gpus: topology.total_gpus }));
    }
  }, [topology, form.gpus]);

  useEffect(() => {
    if (!initialStrategy) return;
    setForm((prev) => ({
      ...prev,
      model: initialStrategy.model || prev.model,
      batchSize: initialStrategy.batchSize ?? prev.batchSize,
      seqLength: initialStrategy.seqLength ?? prev.seqLength,
      gpus: initialStrategy.dp ? initialStrategy.dp : prev.gpus,
    }));
  }, [initialStrategy]);

  const steps = [
    { title: 'Workload', description: 'Model + objective' },
    { title: 'Batch & layout', description: 'Sequence + parallelism knobs' },
    { title: 'Strategy', description: 'Recommended TP/PP/DP' },
  ];

  const handleRecommend = async () => {
    try {
      setLoading(true);
      setError(null);
      const payload = {
        model: form.model,
        batch_size: Number(form.batchSize),
        seq_length: Number(form.seqLength),
        goal: form.goal,
        is_training: form.isTraining,
      };
      const res = await recommendParallelism(payload);
      setRecommendation(res);
      setActiveStep(2);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to generate strategy'));
    } finally {
      setLoading(false);
    }
  };

  const goalLabel =
    form.goal === 'latency' ? 'Low latency' : form.goal === 'balanced' ? 'Balanced' : 'Throughput';
  const chips = [
    { label: 'Training', active: form.isTraining },
    { label: 'Inference', active: !form.isTraining },
    { label: goalLabel, active: true },
  ];

  const strategy = recommendation?.strategy || recommendation?.recommended_strategy || {};
  const tp = strategy.tensor_parallel ?? strategy.tp ?? 1;
  const pp = strategy.pipeline_parallel ?? strategy.pp ?? 1;
  const dp = strategy.data_parallel ?? strategy.dp ?? strategy.dp_size ?? 1;
  const totalGpus = strategy.total_gpus_needed || tp * pp * dp || 1;
  const availableGpus = form.gpus || topology?.total_gpus || topology?.nodes || topology?.gpus_per_node || 8;
  const memoryEstimate = recommendation?.memory_estimate_gb || recommendation?.memory_gb;
  const memoryText = useMemo(() => {
    if (typeof memoryEstimate === 'number') return `${formatNumber(memoryEstimate, 1)} GB`;
    if (typeof memoryEstimate === 'string') return memoryEstimate;
    return null;
  }, [memoryEstimate]);

  const launchCommand = useMemo(() => {
    if (!tp && !pp && !dp) return '';
    const nnodes = Math.max(1, Math.ceil(totalGpus / (topology?.gpus_per_node || availableGpus || 8)));
    const nprocPerNode = Math.max(1, Math.min(totalGpus, topology?.gpus_per_node || availableGpus || totalGpus));
    return [
      `torchrun --nnodes=${nnodes}`,
      `--nproc_per_node=${nprocPerNode}`,
      'train.py',
      `--tp ${tp || 1}`,
      `--pp ${pp || 1}`,
      `--dp ${dp || 1}`,
      `--batch-size ${form.batchSize}`,
      `--seq-length ${form.seqLength}`,
    ].join(' ');
  }, [availableGpus, dp, form.batchSize, form.seqLength, pp, topology?.gpus_per_node, tp, totalGpus]);

  const noteList = useMemo(() => {
    if (!recommendation) return [];
    const raw = recommendation.recommendations ?? recommendation.notes ?? [];
    if (Array.isArray(raw)) return raw.filter(Boolean);
    return raw ? [raw] : [];
  }, [recommendation]);

  const copyCommand = async () => {
    if (!launchCommand) return;
    try {
      await navigator.clipboard.writeText(launchCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (err) {
      console.error('Failed to copy launch command', err);
    }
  };

  const StepBadge = ({ index, title, active }: { index: number; title: string; active: boolean }) => (
    <div
      className={`flex items-center gap-2 px-3 py-1 rounded-full border ${
        active ? 'border-accent-primary/50 bg-accent-primary/10 text-accent-primary' : 'border-white/10 text-white/60'
      }`}
    >
      <span className={`w-6 h-6 rounded-full flex items-center justify-center text-sm ${active ? 'bg-accent-primary/20' : 'bg-white/5'}`}>
        {index + 1}
      </span>
      <span className="text-sm font-medium">{title}</span>
    </div>
  );

  return (
    <div className="card h-full">
      <div className="card-header">
        <div className="flex items-center gap-3">
          <Wand2 className="w-5 h-5 text-accent-primary" />
          <div>
            <h3 className="font-semibold text-white">Parallelism Strategy Wizard</h3>
            <p className="text-xs text-white/50">Guided recommender powered by /api/parallelism/recommend</p>
          </div>
        </div>
        <div className="hidden md:flex items-center gap-2">
          {steps.map((s, i) => (
            <StepBadge key={s.title} index={i} title={s.title} active={activeStep >= i} />
          ))}
        </div>
      </div>
      <div className="card-body space-y-4">
        <div className="flex flex-wrap gap-2">
          {chips.map((chip) => (
            <span
              key={chip.label}
              className={`px-3 py-1 rounded-full text-xs border ${
                chip.active ? 'border-accent-secondary/40 text-accent-secondary bg-accent-secondary/10' : 'border-white/10 text-white/60'
              }`}
            >
              {chip.label}
            </span>
          ))}
          <span className="px-3 py-1 rounded-full text-xs border border-white/10 text-white/60 flex items-center gap-1">
            <Server className="w-3 h-3" />
            {availableGpus} GPUs detected
          </span>
        </div>

        <div className="grid lg:grid-cols-3 gap-4">
          <div className={`lg:col-span-2 space-y-4 rounded-lg border ${activeStep === 0 ? 'border-accent-primary/50 bg-accent-primary/5' : 'border-white/10 bg-white/5'}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-accent-primary" />
                <div>
                  <div className="text-white font-medium">Step 1 · Workload</div>
                  <div className="text-xs text-white/50">Tell the wizard what you are trying to optimize</div>
                </div>
              </div>
              <button
                onClick={() => setActiveStep(1)}
                className="text-xs text-accent-primary hover:text-white flex items-center gap-1"
              >
                Next
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>

            <div className="grid md:grid-cols-2 gap-3">
              <div className="space-y-2">
                <label className="text-sm text-white/60">Model (name or size)</label>
                <input
                  value={form.model}
                  onChange={(e) => setForm({ ...form, model: e.target.value })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
                  placeholder="llama-3.1-70b"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm text-white/60">Goal</label>
                <select
                  value={form.goal}
                  onChange={(e) => setForm({ ...form, goal: e.target.value })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
                >
                  <option value="throughput">Max throughput</option>
                  <option value="latency">Low latency</option>
                  <option value="balanced">Balanced</option>
                </select>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-3">
              <div className="space-y-2">
                <label className="text-sm text-white/60">Batch size</label>
                <input
                  type="number"
                  value={form.batchSize}
                  onChange={(e) => setForm({ ...form, batchSize: Number(e.target.value) })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
                  min={1}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm text-white/60">Sequence length</label>
                <input
                  type="number"
                  value={form.seqLength}
                  onChange={(e) => setForm({ ...form, seqLength: Number(e.target.value) })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
                  min={128}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm text-white/60">GPUs available</label>
                <input
                  type="number"
                  value={form.gpus ?? availableGpus}
                  onChange={(e) => setForm({ ...form, gpus: Number(e.target.value) })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
                  min={1}
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-1">
              <label className="flex items-center gap-2 text-white/70">
                <input
                  type="checkbox"
                  checked={form.isTraining}
                  onChange={(e) => setForm({ ...form, isTraining: e.target.checked })}
                  className="w-4 h-4 accent-accent-primary"
                />
                Optimize for training (uncheck for inference)
              </label>
              <button
                onClick={() => setActiveStep(1)}
                className="text-xs text-white/70 hover:text-white flex items-center gap-1"
              >
                Continue to sizing
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          </div>

          <div className="space-y-3 rounded-lg border border-white/10 bg-white/5 p-4">
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-accent-secondary" />
              <div className="text-white font-medium">Strategy preview</div>
            </div>
            <div className="text-sm text-white/60">
              Generates a TP/PP/DP plan plus launch command using your inputs and the backend recommender.
            </div>
            <button
              onClick={handleRecommend}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-accent-primary to-accent-secondary text-black font-semibold disabled:opacity-60"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Target className="w-4 h-4" />}
              {loading ? 'Generating...' : 'Generate strategy'}
            </button>
            {error && <div className="text-xs text-accent-warning bg-accent-warning/10 border border-accent-warning/30 rounded-md px-3 py-2">{error}</div>}
            <div className="text-xs text-white/50 flex items-center gap-1">
              <Check className="w-3 h-3 text-accent-success" />
              Uses live cluster context when available.
            </div>
          </div>
        </div>

        <div className={`rounded-lg border ${activeStep >= 2 ? 'border-accent-secondary/50 bg-accent-secondary/5' : 'border-white/10 bg-white/5'} p-4 space-y-3`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-accent-secondary" />
              <div>
                <div className="text-white font-medium">Step 3 · Recommended plan</div>
                <div className="text-xs text-white/50">Torch distributed settings + rationale</div>
              </div>
            </div>
            <button
              onClick={() => setActiveStep(1)}
              className="text-xs text-accent-primary hover:text-white flex items-center gap-1"
            >
              Refine inputs
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>

          {recommendation ? (
            <div className="grid md:grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-black/40 border border-white/10 space-y-2">
                <div className="text-xs text-white/50 uppercase">Parallelism</div>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 rounded bg-accent-primary/15 text-accent-primary text-xs">TP {tp || 1}</span>
                  <span className="px-2 py-1 rounded bg-accent-secondary/15 text-accent-secondary text-xs">PP {pp || 1}</span>
                  <span className="px-2 py-1 rounded bg-accent-tertiary/15 text-accent-tertiary text-xs">DP {dp || 1}</span>
                  <span className="px-2 py-1 rounded bg-white/10 text-white/70 text-xs">{totalGpus} GPUs needed</span>
                </div>
                <div className="text-xs text-white/60">
                  {memoryText ? `Est. memory: ${memoryText}` : 'Memory estimate from backend'}
                </div>
              </div>

              <div className="p-3 rounded-lg bg-black/40 border border-white/10 space-y-2">
                <div className="text-xs text-white/50 uppercase">Notes</div>
                <ul className="space-y-1 text-sm text-white/70 list-disc list-inside">
                  {noteList.slice(0, 4).map((note: any, idx: number) => (
                    <li key={idx}>{typeof note === 'string' ? note : JSON.stringify(note)}</li>
                  ))}
                  {noteList.length === 0 && (
                    <li className="text-white/50">Backend did not return extra notes.</li>
                  )}
                </ul>
              </div>

              <div className="p-3 rounded-lg bg-black/50 border border-white/10 space-y-2">
                <div className="text-xs text-white/50 uppercase">Launch</div>
                <div className="font-mono text-xs text-white/80 bg-black/60 rounded-md p-2 border border-white/5 max-h-28 overflow-auto">
                  {launchCommand || 'Generate a plan to see the launch command.'}
                </div>
                <div className="flex items-center justify-between">
                  <div className="text-xs text-white/50">
                    {recommendation.fits_current_setup ? 'Fits current cluster' : 'May need more GPUs'}
                  </div>
                  <button
                    onClick={copyCommand}
                    disabled={!launchCommand}
                    className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-white/10 text-white hover:bg-white/20 disabled:opacity-40"
                  >
                    {copied ? <Check className="w-3 h-3 text-accent-success" /> : <Copy className="w-3 h-3" />}
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                </div>
                {onApply && (
                  <button
                    onClick={() =>
                      onApply({
                        model: form.model,
                        tp,
                        pp,
                        dp,
                        batchSize: form.batchSize,
                        seqLength: form.seqLength,
                      })
                    }
                    disabled={!tp || !dp}
                    className="w-full text-xs mt-2 px-3 py-2 rounded-lg bg-accent-primary/20 text-accent-primary hover:bg-accent-primary/30 disabled:opacity-50"
                  >
                    Apply to launch & visualizations
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="text-sm text-white/60 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-white/30" />
              Run the wizard to see a tailored parallelism plan.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
