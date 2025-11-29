'use client';

import { useEffect, useState } from 'react';
import { AlertTriangle, Network, Radio, RefreshCw, Shield, Siren, Zap } from 'lucide-react';
import {
  diagnoseClusterError,
  getClusterElasticScaling,
  getClusterFaultTolerance,
  getClusterSpotConfig,
  getDistributedCommOverlap,
  getDistributedLongContext,
  getDistributedMoe,
  getDistributedNccl,
  getDistributedRLHF,
  getDistributedVllm,
} from '@/lib/api';
import { getErrorMessage } from '@/lib/useApi';

type DistributedState = {
  nccl: any;
  commOverlap: any;
  moe: any;
  longContext: any;
  rlhf: any;
  vllm: any;
};

type ClusterState = {
  fault?: any;
  elastic?: any;
  spot?: any;
  diagnose?: any;
};

export function DistributedCoverageCard() {
  const [distParams, setDistParams] = useState({
    model: 'llama-3.1-70b',
    gpus: 8,
    seq_length: 4096,
    tp: 1,
    pp: 1,
    dp: 8,
    experts: 8,
  });
  const [clusterParams, setClusterParams] = useState({
    params: 70,
    nodes: 1,
    gpus: 8,
    hours: 24,
    spot: true,
    cloud: 'aws',
  });
  const [clusterError, setClusterError] = useState('NCCL unresponsive on node 3');
  const [distData, setDistData] = useState<DistributedState | null>(null);
  const [clusterData, setClusterData] = useState<ClusterState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  async function loadDistributed() {
    try {
      setBusy('dist');
      setError(null);
      const [nccl, commOverlap, moe, longContext, rlhf, vllm] = await Promise.all([
        getDistributedNccl({ nodes: 1, gpus: distParams.gpus, model_size: 70, tp: distParams.tp, pp: distParams.pp }).catch(() => null),
        getDistributedCommOverlap({
          model: distParams.model,
          tp: distParams.tp,
          pp: distParams.pp,
          dp: distParams.dp,
          batch_size: 8,
          seq_length: distParams.seq_length,
        }).catch(() => null),
        getDistributedMoe({ model: distParams.model, num_experts: distParams.experts, gpus: distParams.gpus }).catch(() => null),
        getDistributedLongContext({ model: distParams.model, seq_length: distParams.seq_length, gpus: distParams.gpus }).catch(() => null),
        getDistributedRLHF({ model: distParams.model, batch_size: 4, seq_length: 2048, memory: 80, compare: true }).catch(() => null),
        getDistributedVllm({ model: distParams.model, gpus: distParams.gpus, target: 'throughput', max_seq_length: distParams.seq_length }).catch(() => null),
      ]);
      setDistData({ nccl, commOverlap, moe, longContext, rlhf, vllm });
    } catch (e) {
      setError(getErrorMessage(e, 'Failed to load distributed diagnostics'));
    } finally {
      setBusy(null);
      setLoading(false);
    }
  }

  async function loadCluster() {
    try {
      setBusy('cluster');
      setError(null);
      const [fault, elastic, spot, diagnose] = await Promise.all([
        getClusterFaultTolerance(clusterParams).catch(() => null),
        getClusterElasticScaling({ params: clusterParams.params, nodes: clusterParams.nodes, traffic: 'variable' }).catch(() => null),
        getClusterSpotConfig({ params: clusterParams.params, cloud: clusterParams.cloud, budget: true }).catch(() => null),
        clusterError ? diagnoseClusterError(clusterError).catch(() => null) : Promise.resolve(null),
      ]);
      setClusterData({ fault, elastic, spot, diagnose });
    } catch (e) {
      setError(getErrorMessage(e, 'Failed to load cluster diagnostics'));
    } finally {
      setBusy(null);
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadDistributed();
    void loadCluster();
  }, []);

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Network className="w-5 h-5 text-accent-info" />
          <h3 className="font-medium text-white">Distributed Coverage + Cluster Health</h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={loadDistributed}
            className="px-3 py-1 rounded bg-white/5 border border-white/10 text-xs text-white/70"
          >
            Refresh distributed
          </button>
          <button
            onClick={loadCluster}
            className="px-3 py-1 rounded bg-white/5 border border-white/10 text-xs text-white/70"
          >
            Refresh cluster
          </button>
        </div>
      </div>
      <div className="card-body space-y-4">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <RefreshCw className="w-4 h-4 animate-spin" /> Loading distributed and cluster details...
          </div>
        ) : error ? (
          <div className="text-sm text-accent-warning">{error}</div>
        ) : (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
                <div className="flex items-center gap-2 text-white font-semibold">
                  <Radio className="w-4 h-4 text-accent-secondary" />
                  Distributed Diagnostics
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs text-white/70">
                  <label className="flex flex-col gap-1">
                    Model
                    <input
                      value={distParams.model}
                      onChange={(e) => setDistParams((p) => ({ ...p, model: e.target.value }))}
                      className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    GPUs
                    <input
                      type="number"
                      value={distParams.gpus}
                      onChange={(e) => setDistParams((p) => ({ ...p, gpus: Number(e.target.value) }))}
                      className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    Seq
                    <input
                      type="number"
                      value={distParams.seq_length}
                      onChange={(e) => setDistParams((p) => ({ ...p, seq_length: Number(e.target.value) }))}
                      className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    Experts
                    <input
                      type="number"
                      value={distParams.experts}
                      onChange={(e) => setDistParams((p) => ({ ...p, experts: Number(e.target.value) }))}
                      className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                    />
                  </label>
                </div>
                {busy === 'dist' && <div className="text-xs text-white/60">Refreshing distributed endpoints…</div>}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  <Result title="NCCL" payload={distData?.nccl} />
                  <Result title="Comm overlap" payload={distData?.commOverlap} />
                  <Result title="Long context" payload={distData?.longContext} />
                  <Result title="MoE" payload={distData?.moe} />
                  <Result title="RLHF" payload={distData?.rlhf} />
                  <Result title="vLLM" payload={distData?.vllm} />
                </div>
              </div>

              <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
                <div className="flex items-center gap-2 text-white font-semibold">
                  <Shield className="w-4 h-4 text-accent-primary" />
                  Cluster Resilience
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs text-white/70">
                  <label className="flex flex-col gap-1">
                    Params (B)
                    <input
                      type="number"
                      value={clusterParams.params}
                      onChange={(e) => setClusterParams((p) => ({ ...p, params: Number(e.target.value) }))}
                      className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    Nodes
                    <input
                      type="number"
                      value={clusterParams.nodes}
                      onChange={(e) => setClusterParams((p) => ({ ...p, nodes: Number(e.target.value) }))}
                      className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    GPUs/node
                    <input
                      type="number"
                      value={clusterParams.gpus}
                      onChange={(e) => setClusterParams((p) => ({ ...p, gpus: Number(e.target.value) }))}
                      className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                    />
                  </label>
                  <label className="flex flex-col gap-1 col-span-2">
                    Cloud
                    <input
                      value={clusterParams.cloud}
                      onChange={(e) => setClusterParams((p) => ({ ...p, cloud: e.target.value }))}
                      className="px-2 py-1 rounded bg-white/5 border border-white/10 text-white"
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-white/70">
                    <input
                      type="checkbox"
                      checked={clusterParams.spot}
                      onChange={(e) => setClusterParams((p) => ({ ...p, spot: e.target.checked }))}
                    />
                    Use spot
                  </label>
                </div>
                <div className="text-xs text-white/60 flex items-center gap-2">
                  <Siren className="w-4 h-4 text-accent-danger" />
                  Diagnose
                </div>
                <input
                  value={clusterError}
                  onChange={(e) => setClusterError(e.target.value)}
                  className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-white text-xs"
                  placeholder="Paste cluster error message"
                />
                {busy === 'cluster' && <div className="text-xs text-white/60">Refreshing cluster guidance…</div>}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  <Result title="Fault tolerance" payload={clusterData?.fault} />
                  <Result title="Elastic scaling" payload={clusterData?.elastic} />
                  <Result title="Spot config" payload={clusterData?.spot} />
                  <Result title="Diagnosis" payload={clusterData?.diagnose} />
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Result({ title, payload }: { title: string; payload: any }) {
  return (
    <div className="p-2 rounded bg-white/5 border border-white/10 min-h-[100px]">
      <div className="text-xs text-white/50 uppercase mb-1">{title}</div>
      {payload ? (
        typeof payload === 'string' ? (
          <div className="text-xs text-white/80 whitespace-pre-wrap">{payload}</div>
        ) : (
          <pre className="text-[11px] text-white/80 bg-white/5 border border-white/10 rounded p-2 whitespace-pre-wrap overflow-x-auto">
            {JSON.stringify(payload, null, 2)}
          </pre>
        )
      ) : (
        <div className="text-xs text-white/40">No data yet.</div>
      )}
    </div>
  );
}
