'use client';

import { useState } from 'react';
import { Beaker, Cloud, Cpu, Layers, Loader2, Sliders, Workflow } from 'lucide-react';
import {
  getBatchCloudCost,
  getBatchCompound,
  getBatchDeployConfig,
  getBatchFinetuneEstimate,
  getBatchLLMAdvisor,
  getBatchMultiGpuScaling,
  getBatchThroughput,
} from '@/lib/api';
import { getErrorMessage } from '@/lib/useApi';

type BatchResults = {
  throughput?: any;
  cloud?: any;
  deploy?: any;
  finetune?: any;
  multiGpu?: any;
  llmAdvisor?: any;
  compound?: any;
};

export function BatchAdvancedCard() {
  const [model, setModel] = useState('llama-3.1-70b');
  const [precision, setPrecision] = useState('fp16');
  const [paramsB, setParamsB] = useState(70);
  const [gpus, setGpus] = useState(8);
  const [payload, setPayload] = useState({ batch: 32, seq: 4096 });
  const [results, setResults] = useState<BatchResults>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [compoundOpts, setCompoundOpts] = useState<Record<string, boolean>>({
    fuse: true,
    quant: true,
    flash: true,
  });

  const updateResult = (key: keyof BatchResults, value: any) => {
    setResults((prev) => ({ ...prev, [key]: value }));
  };

  const runThroughput = async () => {
    setBusy('throughput');
    setError(null);
    try {
      const res = await getBatchThroughput({ params: paramsB, precision });
      updateResult('throughput', res);
    } catch (e) {
      setError(getErrorMessage(e, 'Throughput estimate failed'));
    } finally {
      setBusy(null);
    }
  };

  const runMultiGpu = async () => {
    setBusy('multigpu');
    setError(null);
    try {
      const res = await getBatchMultiGpuScaling({ model, gpus, ...payload });
      updateResult('multiGpu', res);
    } catch (e) {
      setError(getErrorMessage(e, 'Multi-GPU scaling failed'));
    } finally {
      setBusy(null);
    }
  };

  const runCloud = async () => {
    setBusy('cloud');
    setError(null);
    try {
      const res = await getBatchCloudCost({ model, gpus, precision });
      updateResult('cloud', res);
    } catch (e) {
      setError(getErrorMessage(e, 'Cloud cost estimate failed'));
    } finally {
      setBusy(null);
    }
  };

  const runDeploy = async () => {
    setBusy('deploy');
    setError(null);
    try {
      const res = await getBatchDeployConfig({ model, gpus, precision, ...payload });
      updateResult('deploy', res);
    } catch (e) {
      setError(getErrorMessage(e, 'Deploy config failed'));
    } finally {
      setBusy(null);
    }
  };

  const runFinetune = async () => {
    setBusy('finetune');
    setError(null);
    try {
      const res = await getBatchFinetuneEstimate({ model, gpus, precision, ...payload });
      updateResult('finetune', res);
    } catch (e) {
      setError(getErrorMessage(e, 'Finetune estimate failed'));
    } finally {
      setBusy(null);
    }
  };

  const runAdvisor = async () => {
    setBusy('advisor');
    setError(null);
    try {
      const res = await getBatchLLMAdvisor({ model, gpus, goal: 'throughput', batch: payload.batch, seq: payload.seq });
      updateResult('llmAdvisor', res);
    } catch (e) {
      setError(getErrorMessage(e, 'LLM advisor failed'));
    } finally {
      setBusy(null);
    }
  };

  const runCompound = async () => {
    setBusy('compound');
    setError(null);
    try {
      const active = Object.keys(compoundOpts).filter((k) => compoundOpts[k]);
      const res = await getBatchCompound({ opts: active });
      updateResult('compound', res);
    } catch (e) {
      setError(getErrorMessage(e, 'Compound analysis failed'));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Workflow className="w-5 h-5 text-accent-info" />
          <h3 className="font-medium text-white">Advanced Batch Ops</h3>
        </div>
        {error && <div className="text-sm text-accent-warning">{error}</div>}
      </div>
      <div className="card-body space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-2 text-xs text-white/70">
          <label className="flex flex-col gap-1">
            Model
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
            />
          </label>
          <label className="flex flex-col gap-1">
            Params (B)
            <input
              type="number"
              value={paramsB}
              onChange={(e) => setParamsB(Number(e.target.value))}
              className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
            />
          </label>
          <label className="flex flex-col gap-1">
            GPUs
            <input
              type="number"
              value={gpus}
              onChange={(e) => setGpus(Number(e.target.value))}
              className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
            />
          </label>
          <label className="flex flex-col gap-1">
            Precision
            <select
              value={precision}
              onChange={(e) => setPrecision(e.target.value)}
              className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
            >
              <option value="fp16">fp16</option>
              <option value="bf16">bf16</option>
              <option value="int8">int8</option>
              <option value="int4">int4</option>
            </select>
          </label>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <ActionCard
            title="Throughput estimate"
            icon={Cpu}
            onRun={runThroughput}
            busy={busy === 'throughput'}
            result={results.throughput}
          >
            <div className="text-xs text-white/60">Params: {paramsB}B · Precision {precision}</div>
          </ActionCard>

          <ActionCard
            title="Multi-GPU scaling"
            icon={Layers}
            onRun={runMultiGpu}
            busy={busy === 'multigpu'}
            result={results.multiGpu}
          >
            <div className="grid grid-cols-2 gap-2 text-xs text-white/70">
              <label className="flex flex-col gap-1">
                Batch
                <input
                  type="number"
                  value={payload.batch}
                  onChange={(e) => setPayload((p) => ({ ...p, batch: Number(e.target.value) }))}
                  className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                />
              </label>
              <label className="flex flex-col gap-1">
                Seq
                <input
                  type="number"
                  value={payload.seq}
                  onChange={(e) => setPayload((p) => ({ ...p, seq: Number(e.target.value) }))}
                  className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                />
              </label>
            </div>
          </ActionCard>

          <ActionCard
            title="Cloud cost"
            icon={Cloud}
            onRun={runCloud}
            busy={busy === 'cloud'}
            result={results.cloud}
          >
            <div className="text-xs text-white/60">Quick TCO snapshot</div>
          </ActionCard>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <ActionCard
            title="Deploy config"
            icon={Sliders}
            onRun={runDeploy}
            busy={busy === 'deploy'}
            result={results.deploy}
          >
            <div className="text-xs text-white/60">Generate launch/deploy snippet</div>
          </ActionCard>

          <ActionCard
            title="Finetune sizing"
            icon={Beaker}
            onRun={runFinetune}
            busy={busy === 'finetune'}
            result={results.finetune}
          >
            <div className="text-xs text-white/60">Estimate VRAM & steps</div>
          </ActionCard>

          <ActionCard
            title="LLM advisor"
            icon={Zap}
            onRun={runAdvisor}
            busy={busy === 'advisor'}
            result={results.llmAdvisor}
          >
            <div className="text-xs text-white/60">AI-backed recommendations</div>
          </ActionCard>
        </div>

        <div className="p-3 rounded-lg bg-white/5 border border-white/10">
          <div className="flex items-center gap-2 text-white font-semibold mb-2">
            <Workflow className="w-4 h-4 text-accent-primary" />
            Compound strategy
          </div>
          <div className="flex flex-wrap gap-3 text-sm text-white/70">
            {Object.keys(compoundOpts).map((key) => (
              <label key={key} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={compoundOpts[key]}
                  onChange={(e) => setCompoundOpts((prev) => ({ ...prev, [key]: e.target.checked }))}
                />
                {key}
              </label>
            ))}
            <button
              onClick={runCompound}
              className="px-3 py-1 rounded bg-accent-primary/20 text-accent-primary text-xs"
              disabled={busy === 'compound'}
            >
              {busy === 'compound' ? 'Combining…' : 'Run compound'}
            </button>
          </div>
          <Result payload={results.compound} />
        </div>
      </div>
    </div>
  );
}

function ActionCard({
  title,
  icon: Icon,
  onRun,
  busy,
  result,
  children,
}: {
  title: string;
  icon: any;
  onRun: () => void;
  busy: boolean;
  result: any;
  children?: React.ReactNode;
}) {
  return (
    <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
      <div className="flex items-center gap-2 text-white font-semibold">
        <Icon className="w-4 h-4 text-accent-primary" />
        {title}
      </div>
      {children}
      <button
        onClick={onRun}
        className="px-3 py-2 rounded bg-white/5 border border-white/10 text-xs text-white/80"
        disabled={busy}
      >
        {busy ? <Loader2 className="w-3 h-3 animate-spin inline mr-2" /> : null}
        {busy ? 'Working…' : 'Run'}
      </button>
      <Result payload={result} />
    </div>
  );
}

function Result({ payload }: { payload: any }) {
  if (!payload) {
    return <div className="text-xs text-white/40">No output yet.</div>;
  }
  if (typeof payload === 'string') {
    return <div className="text-xs text-white/80 whitespace-pre-wrap">{payload}</div>;
  }
  return (
    <pre className="text-[11px] text-white/80 bg-white/5 border border-white/10 rounded p-2 whitespace-pre-wrap overflow-x-auto">
      {JSON.stringify(payload, null, 2)}
    </pre>
  );
}
