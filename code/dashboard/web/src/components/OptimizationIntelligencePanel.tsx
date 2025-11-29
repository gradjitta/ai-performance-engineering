'use client';

import { useEffect, useState } from 'react';
import {
  BrainCircuit,
  Lightbulb,
  Network,
  RefreshCw,
  Rocket,
  Send,
  Terminal,
} from 'lucide-react';
import {
  callUnifiedApi,
  getIntelligenceDistributed,
  getIntelligenceRecommendation,
  getIntelligenceRL,
  getIntelligenceTechniques,
  getIntelligenceVllm,
  runAIAnalysis,
  runAIQuery,
} from '@/lib/api';
import { getErrorMessage } from '@/lib/useApi';
import { EmptyState } from './DataState';

type IntelligenceState = {
  recommendation: any;
  distributed: any;
  vllm: any;
  rl: any;
};

export function OptimizationIntelligencePanel() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [techniques, setTechniques] = useState<any[]>([]);
  const [intel, setIntel] = useState<IntelligenceState | null>(null);

  const [intelParams, setIntelParams] = useState({ model: 'llama-3.1-70b', goal: 'throughput', gpus: 8 });
  const [aiType, setAiType] = useState('bottleneck');
  const [aiAnalysisResult, setAiAnalysisResult] = useState<any>(null);
  const [aiQuestion, setAiQuestion] = useState('How can I reduce kernel launch overhead?');
  const [aiContext, setAiContext] = useState('');
  const [aiAnswer, setAiAnswer] = useState<any>(null);
  const [unifiedPath, setUnifiedPath] = useState('optimize/search');
  const [unifiedPayload, setUnifiedPayload] = useState('{"model": "llama-3.1-70b"}');
  const [unifiedResult, setUnifiedResult] = useState<any>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function loadIntel() {
    try {
      setLoading(true);
      setError(null);
      const [techRes, recRes, distRes, vllmRes, rlRes] = await Promise.all([
        getIntelligenceTechniques().catch(() => ({ techniques: [] })),
        getIntelligenceRecommendation({ model: intelParams.model, goal: intelParams.goal, gpus: intelParams.gpus }).catch(() => null),
        getIntelligenceDistributed({ model: intelParams.model, gpus: intelParams.gpus }).catch(() => null),
        getIntelligenceVllm({ model: intelParams.model }).catch(() => null),
        getIntelligenceRL({ model: intelParams.model }).catch(() => null),
      ]);
      setTechniques((techRes as any)?.techniques || techRes || []);
      setIntel({
        recommendation: recRes,
        distributed: distRes,
        vllm: vllmRes,
        rl: rlRes,
      });
    } catch (e) {
      setError(getErrorMessage(e, 'Failed to load intelligence data'));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadIntel();
  }, []);

  const runAnalysis = async () => {
    setActionError(null);
    setBusy('analyze');
    try {
      const res = await runAIAnalysis(aiType);
      setAiAnalysisResult(res);
    } catch (e) {
      setActionError(getErrorMessage(e, 'AI analysis failed'));
    } finally {
      setBusy(null);
    }
  };

  const runQueryAction = async () => {
    setActionError(null);
    setBusy('query');
    try {
      const res = await runAIQuery(aiQuestion, aiContext || undefined);
      setAiAnswer(res);
    } catch (e) {
      setActionError(getErrorMessage(e, 'AI query failed'));
    } finally {
      setBusy(null);
    }
  };

  const runUnified = async () => {
    setActionError(null);
    setBusy('unified');
    try {
      const payload = unifiedPayload ? JSON.parse(unifiedPayload) : {};
      const res = await callUnifiedApi(unifiedPath, payload);
      setUnifiedResult(res);
    } catch (e) {
      setActionError(getErrorMessage(e, 'Unified API call failed (check JSON)'));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <BrainCircuit className="w-5 h-5 text-accent-secondary" />
          <h3 className="font-medium text-white">Optimization Intelligence + Unified API</h3>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={intelParams.model}
            onChange={(e) => setIntelParams((p) => ({ ...p, model: e.target.value }))}
            className="px-2 py-1 rounded bg-white/5 border border-white/10 text-xs text-white"
            placeholder="llama-3.1-70b"
          />
          <input
            type="number"
            value={intelParams.gpus}
            onChange={(e) => setIntelParams((p) => ({ ...p, gpus: Number(e.target.value) }))}
            className="w-16 px-2 py-1 rounded bg-white/5 border border-white/10 text-xs text-white"
            placeholder="8"
          />
          <select
            value={intelParams.goal}
            onChange={(e) => setIntelParams((p) => ({ ...p, goal: e.target.value }))}
            className="px-2 py-1 rounded bg-white/5 border border-white/10 text-xs text-white"
          >
            <option value="throughput">Throughput</option>
            <option value="latency">Latency</option>
            <option value="cost">Cost</option>
          </select>
          <button
            onClick={loadIntel}
            className="p-2 rounded hover:bg-white/5"
            aria-label="Refresh intelligence"
          >
            <RefreshCw className="w-4 h-4 text-white/60" />
          </button>
        </div>
      </div>
      <div className="card-body space-y-4">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <RefreshCw className="w-4 h-4 animate-spin" /> Loading intelligence…
          </div>
        ) : error ? (
          <div className="text-sm text-accent-warning">{error}</div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <InfoCard title="Recommendation" icon={Lightbulb} payload={intel?.recommendation} />
              <InfoCard title="Distributed plan" icon={Network} payload={intel?.distributed} />
              <InfoCard title="vLLM / RL hints" icon={Rocket} payload={{ vllm: intel?.vllm, rl: intel?.rl }} />
            </div>

            <div className="p-3 rounded-lg bg-white/5 border border-white/10">
              <div className="text-xs text-white/50 uppercase mb-2">Techniques</div>
              {techniques.length ? (
                <div className="flex flex-wrap gap-2">
                  {techniques.slice(0, 10).map((t, idx) => (
                    <span key={idx} className="px-2 py-1 rounded bg-accent-primary/10 text-accent-primary text-xs">
                      {t.name || t}
                    </span>
                  ))}
                </div>
              ) : (
                <EmptyState title="No techniques" description="Backend did not return optimization techniques." />
              )}
            </div>
          </>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
            <div className="flex items-center gap-2 text-white font-semibold">
              <Terminal className="w-4 h-4 text-accent-primary" />
              AI Analysis
            </div>
            {actionError && <div className="text-sm text-accent-warning">{actionError}</div>}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs text-white/70">
              <label className="flex flex-col gap-1">
                Type
                <select
                  value={aiType}
                  onChange={(e) => setAiType(e.target.value)}
                  className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                >
                  <option value="bottleneck">Bottleneck</option>
                  <option value="cost">Cost</option>
                  <option value="latency">Latency</option>
                </select>
              </label>
              <label className="flex flex-col gap-1">
                Question
                <input
                  value={aiQuestion}
                  onChange={(e) => setAiQuestion(e.target.value)}
                  className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                />
              </label>
              <label className="flex flex-col gap-1">
                Context (optional)
                <input
                  value={aiContext}
                  onChange={(e) => setAiContext(e.target.value)}
                  className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                />
              </label>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={runAnalysis}
                className="px-3 py-2 rounded bg-accent-primary/20 text-accent-primary text-xs"
                disabled={busy === 'analyze'}
              >
                {busy === 'analyze' ? 'Running…' : 'Run analysis'}
              </button>
              <button
                onClick={runQueryAction}
                className="px-3 py-2 rounded bg-accent-secondary/20 text-accent-secondary text-xs"
                disabled={busy === 'query'}
              >
                {busy === 'query' ? 'Asking…' : 'Ask AI'}
              </button>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <ResultPanel title="AI analysis" payload={aiAnalysisResult} />
              <ResultPanel title="AI answer" payload={aiAnswer} />
            </div>
          </div>

          <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
            <div className="flex items-center gap-2 text-white font-semibold">
              <Send className="w-4 h-4 text-accent-info" />
              Unified API Console
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs text-white/70">
              <label className="flex flex-col gap-1 sm:col-span-1">
                Path
                <input
                  value={unifiedPath}
                  onChange={(e) => setUnifiedPath(e.target.value)}
                  className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                  placeholder="optimize/search"
                />
              </label>
              <label className="flex flex-col gap-1 sm:col-span-2">
                Payload (JSON)
                <textarea
                  value={unifiedPayload}
                  onChange={(e) => setUnifiedPayload(e.target.value)}
                  className="w-full h-20 px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                />
              </label>
            </div>
            <button
              onClick={runUnified}
              className="px-3 py-2 rounded bg-accent-info/20 text-accent-info text-xs"
              disabled={busy === 'unified'}
            >
              {busy === 'unified' ? 'Calling…' : 'Send unified request'}
            </button>
            <ResultPanel title="Unified response" payload={unifiedResult} />
          </div>
        </div>
      </div>
    </div>
  );
}

function InfoCard({ title, icon: Icon, payload }: { title: string; icon: any; payload: any }) {
  return (
    <div className="p-3 rounded-lg bg-white/5 border border-white/10">
      <div className="flex items-center gap-2 text-white font-semibold mb-2">
        <Icon className="w-4 h-4 text-accent-primary" />
        {title}
      </div>
      {payload ? (
        <pre className="text-xs text-white/80 bg-white/5 border border-white/10 rounded p-2 whitespace-pre-wrap overflow-x-auto">
          {JSON.stringify(payload, null, 2)}
        </pre>
      ) : (
        <EmptyState title="No data" description="Backend did not return a payload yet." />
      )}
    </div>
  );
}

function ResultPanel({ title, payload }: { title: string; payload: any }) {
  return (
    <div className="p-2 rounded bg-white/5 border border-white/10">
      <div className="text-xs text-white/50 uppercase mb-1">{title}</div>
      {payload ? (
        typeof payload === 'string' ? (
          <div className="text-xs text-white/80 whitespace-pre-wrap">{payload}</div>
        ) : (
          <pre className="text-xs text-white/80 bg-white/5 border border-white/10 rounded p-2 whitespace-pre-wrap overflow-x-auto">
            {JSON.stringify(payload, null, 2)}
          </pre>
        )
      ) : (
        <div className="text-xs text-white/50">No output yet.</div>
      )}
    </div>
  );
}
