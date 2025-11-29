'use client';

import { useState, useMemo, useCallback, useEffect } from 'react';
import { Benchmark } from '@/types';
import { formatMs, getSpeedupColor, cn } from '@/lib/utils';
import { GitCompare, Search, ArrowRight, X, TrendingUp, Radar as RadarIcon, BarChart3, FileCode2 } from 'lucide-react';
import { compareRuns as compareRunsApi, getCodeDiff } from '@/lib/api';
import { useToast } from '@/components/Toast';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LineChart,
  Line,
} from 'recharts';

interface CompareTabProps {
  benchmarks: Benchmark[];
  pinnedBenchmarks?: Set<string>;
}

export function CompareTab({ benchmarks, pinnedBenchmarks }: CompareTabProps) {
  const [leftBenchmark, setLeftBenchmark] = useState<Benchmark | null>(null);
  const [rightBenchmark, setRightBenchmark] = useState<Benchmark | null>(null);
  const [searchLeft, setSearchLeft] = useState('');
  const [searchRight, setSearchRight] = useState('');
  const [mode, setMode] = useState<'single' | 'pinned'>('single');
  const [compareInputs, setCompareInputs] = useState({ baseline: 'benchmark_test_results.json', candidate: 'benchmark_test_results.json', top: 5 });
  const [compareResult, setCompareResult] = useState<{ regressions: any[]; improvements: any[] } | null>(null);
  const [codeDiffs, setCodeDiffs] = useState<{ left: any | null; right: any | null }>({ left: null, right: null });
  const [codeDiffLoading, setCodeDiffLoading] = useState(false);
  const [codeDiffError, setCodeDiffError] = useState<string | null>(null);
  const { showToast } = useToast();
  const trendColors = ['#22d3ee', '#a855f7', '#f97316'];
  
  const formatCodeSnippet = (text?: string) => {
    if (!text) return 'Not available';
    const trimmed = text.trim();
    return trimmed.length > 1200 ? `${trimmed.slice(0, 1200)}\n…` : trimmed;
  };

  useEffect(() => {
    // Clear stale diff results when selection changes
    setCodeDiffs({ left: null, right: null });
    setCodeDiffError(null);
  }, [leftBenchmark, rightBenchmark]);

  const succeededBenchmarks = benchmarks.filter((b) => b.status === 'succeeded');

  const filteredLeft = useMemo(() => {
    return succeededBenchmarks.filter(
      (b) =>
        b.name.toLowerCase().includes(searchLeft.toLowerCase()) ||
        b.chapter.toLowerCase().includes(searchLeft.toLowerCase())
    );
  }, [succeededBenchmarks, searchLeft]);

  const filteredRight = useMemo(() => {
    return succeededBenchmarks.filter(
      (b) =>
        b.name.toLowerCase().includes(searchRight.toLowerCase()) ||
        b.chapter.toLowerCase().includes(searchRight.toLowerCase())
    );
  }, [succeededBenchmarks, searchRight]);

  const comparison = useMemo(() => {
    if (!leftBenchmark || !rightBenchmark) return null;
    const speedupDiff = rightBenchmark.speedup - leftBenchmark.speedup;
    const baselineDiff = rightBenchmark.baseline_time_ms - leftBenchmark.baseline_time_ms;
    const optimizedDiff = rightBenchmark.optimized_time_ms - leftBenchmark.optimized_time_ms;
    return { speedupDiff, baselineDiff, optimizedDiff };
  }, [leftBenchmark, rightBenchmark]);

  // Radar chart data for multi-dimensional comparison
  const radarData = useMemo(() => {
    if (!leftBenchmark || !rightBenchmark) return [];
    
    // Normalize metrics to 0-100 scale for radar chart
    const maxSpeedup = Math.max(leftBenchmark.speedup, rightBenchmark.speedup, 1);
    const maxBaseline = Math.max(leftBenchmark.baseline_time_ms, rightBenchmark.baseline_time_ms, 1);
    const maxOptimized = Math.max(leftBenchmark.optimized_time_ms, rightBenchmark.optimized_time_ms, 1);
    
    return [
      {
        metric: 'Speedup',
        A: (leftBenchmark.speedup / maxSpeedup) * 100,
        B: (rightBenchmark.speedup / maxSpeedup) * 100,
        fullMark: 100,
      },
      {
        metric: 'Efficiency',
        A: Math.min(100, (leftBenchmark.speedup / (leftBenchmark.baseline_time_ms / 100)) * 10),
        B: Math.min(100, (rightBenchmark.speedup / (rightBenchmark.baseline_time_ms / 100)) * 10),
        fullMark: 100,
      },
      {
        metric: 'Baseline Speed',
        A: (1 - leftBenchmark.baseline_time_ms / maxBaseline) * 100 + 10,
        B: (1 - rightBenchmark.baseline_time_ms / maxBaseline) * 100 + 10,
        fullMark: 100,
      },
      {
        metric: 'Opt. Speed',
        A: (1 - leftBenchmark.optimized_time_ms / maxOptimized) * 100 + 10,
        B: (1 - rightBenchmark.optimized_time_ms / maxOptimized) * 100 + 10,
        fullMark: 100,
      },
      {
        metric: 'Improvement %',
        A: Math.min(100, ((leftBenchmark.baseline_time_ms - leftBenchmark.optimized_time_ms) / leftBenchmark.baseline_time_ms) * 100),
        B: Math.min(100, ((rightBenchmark.baseline_time_ms - rightBenchmark.optimized_time_ms) / rightBenchmark.baseline_time_ms) * 100),
        fullMark: 100,
      },
    ];
  }, [leftBenchmark, rightBenchmark]);

  // Bar chart data for side-by-side comparison
  const barChartData = useMemo(() => {
    if (!leftBenchmark || !rightBenchmark) return [];
    return [
      {
        metric: 'Baseline (ms)',
        A: leftBenchmark.baseline_time_ms,
        B: rightBenchmark.baseline_time_ms,
      },
      {
        metric: 'Optimized (ms)',
        A: leftBenchmark.optimized_time_ms,
        B: rightBenchmark.optimized_time_ms,
      },
      {
        metric: 'Speedup (x)',
        A: leftBenchmark.speedup,
        B: rightBenchmark.speedup,
      },
    ];
  }, [leftBenchmark, rightBenchmark]);

  // Chapter trend analysis
  const chapterTrends = useMemo(() => {
    const chapters: Record<string, { benchmarks: Benchmark[]; avgSpeedup: number; maxSpeedup: number; count: number }> = {};
    
    for (const b of succeededBenchmarks) {
      if (!chapters[b.chapter]) {
        chapters[b.chapter] = { benchmarks: [], avgSpeedup: 0, maxSpeedup: 0, count: 0 };
      }
      chapters[b.chapter].benchmarks.push(b);
      chapters[b.chapter].count++;
    }
    
    for (const ch of Object.values(chapters)) {
      const speeds = ch.benchmarks.map(b => b.speedup);
      ch.avgSpeedup = speeds.reduce((a, b) => a + b, 0) / speeds.length;
      ch.maxSpeedup = Math.max(...speeds);
    }
    
    return Object.entries(chapters)
      .map(([name, data]) => ({ name, ...data }))
      .sort((a, b) => b.avgSpeedup - a.avgSpeedup);
  }, [succeededBenchmarks]);

  const trends = useMemo(() => {
    if (!succeededBenchmarks.length) {
      return { data: [] as Record<string, any>[], chapters: [] as string[] };
    }

    const sorted = [...succeededBenchmarks]
      .map((b, idx) => {
        const rawTs = (b as any).timestamp as string | undefined;
        const ts = rawTs ? Date.parse(rawTs) || idx : idx;
        return {
          benchmark: b,
          ts,
          label: rawTs ? new Date(rawTs).toLocaleString() : `#${idx + 1}`,
        };
      })
      .sort((a, b) => a.ts - b.ts);

    const chapterCounts: Record<string, number> = {};
    sorted.forEach(({ benchmark }) => {
      chapterCounts[benchmark.chapter] = (chapterCounts[benchmark.chapter] || 0) + 1;
    });

    const topChapters = Object.entries(chapterCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([ch]) => ch);

    const data = sorted.map(({ benchmark, label }, idx) => {
      const row: Record<string, any> = {
        idx,
        label,
        overall: benchmark.speedup,
      };
      topChapters.forEach((ch) => {
        row[ch] = benchmark.chapter === ch ? benchmark.speedup : null;
      });
      return row;
    });

    return { data, chapters: topChapters };
  }, [succeededBenchmarks]);

  const pinnedList = useMemo(() => {
    if (!pinnedBenchmarks) return [];
    const keys = Array.from(pinnedBenchmarks);
    return keys
      .map((key) => {
        const [chapter, name] = key.split(':');
        return benchmarks.find((b) => b.chapter === chapter && b.name === name);
      })
      .filter(Boolean) as Benchmark[];
  }, [pinnedBenchmarks, benchmarks]);

  const handleRunDiff = async () => {
    try {
      const res = await compareRunsApi(compareInputs);
      setCompareResult(res as any);
      showToast('Compared runs', 'success');
    } catch (e) {
      showToast('Compare failed', 'error');
    }
  };

  const loadCodeDiffs = useCallback(async () => {
    if (!leftBenchmark && !rightBenchmark) {
      setCodeDiffError('Select benchmarks to load code differences.');
      return;
    }

    try {
      setCodeDiffLoading(true);
      setCodeDiffError(null);

      const [left, right] = await Promise.all([
        leftBenchmark ? getCodeDiff(leftBenchmark.chapter, leftBenchmark.name) : Promise.resolve(null),
        rightBenchmark ? getCodeDiff(rightBenchmark.chapter, rightBenchmark.name) : Promise.resolve(null),
      ]);

      setCodeDiffs({ left, right });
      showToast('Loaded code diffs', 'success');
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load code diff';
      setCodeDiffError(msg);
      showToast('Failed to load code diff', 'error');
    } finally {
      setCodeDiffLoading(false);
    }
  }, [leftBenchmark, rightBenchmark, showToast]);

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-accent-primary" />
            <h2 className="text-lg font-semibold text-white">Benchmark Comparison</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setMode('single')}
              className={cn(
                'px-3 py-1.5 rounded-lg text-sm',
                mode === 'single' ? 'bg-accent-primary/20 text-accent-primary' : 'text-white/50 hover:text-white'
              )}
            >
              Single Compare
            </button>
            <button
              onClick={() => setMode('pinned')}
              className={cn(
                'px-3 py-1.5 rounded-lg text-sm',
                mode === 'pinned' ? 'bg-accent-secondary/20 text-accent-secondary' : 'text-white/50 hover:text-white'
              )}
            >
              Pinned Grid
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="font-medium text-white">Compare Runs (JSON)</h3>
            <p className="text-xs text-white/60">Diff two benchmark_test_results.json files</p>
          </div>
        </div>
        <div className="card-body space-y-2 text-sm">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <input
              className="rounded bg-white/10 px-3 py-2 text-white"
              value={compareInputs.baseline}
              onChange={(e) => setCompareInputs((p) => ({ ...p, baseline: e.target.value }))}
              placeholder="baseline benchmark_test_results.json"
            />
            <input
              className="rounded bg-white/10 px-3 py-2 text-white"
              value={compareInputs.candidate}
              onChange={(e) => setCompareInputs((p) => ({ ...p, candidate: e.target.value }))}
              placeholder="candidate benchmark_test_results.json"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-white/60">Top</label>
            <input
              type="number"
              className="w-16 rounded bg-white/10 px-2 py-1 text-white text-sm"
              value={compareInputs.top}
              onChange={(e) => setCompareInputs((p) => ({ ...p, top: Number(e.target.value) || 0 }))}
            />
            <button
              onClick={handleRunDiff}
              className="ml-auto px-3 py-1.5 bg-accent-primary/20 text-accent-primary rounded text-sm"
            >
              Diff
            </button>
          </div>
          {compareResult && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <ChartList title="Regressions" colorClass="bg-accent-warning" data={compareResult.regressions || []} />
              <ChartList title="Improvements" colorClass="bg-accent-success" data={compareResult.improvements || []} />
            </div>
          )}
        </div>
      </div>

      {mode === 'pinned' && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Pinned Benchmarks ({pinnedList.length})</h3>
          </div>
          <div className="card-body">
            {pinnedList.length === 0 ? (
              <div className="text-sm text-white/50">Pin benchmarks from Overview to see them here.</div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {pinnedList.map((b, i) => (
                  <div key={`${b.chapter}-${b.name}-${i}`} className="p-4 rounded-lg bg-white/5 border border-white/10 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="text-white font-semibold text-sm">{b.name}</div>
                      <span className="text-xs text-white/50">{b.chapter}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-center p-2 bg-white/5 rounded-lg">
                        <div className="text-white/50 text-xs">Baseline</div>
                        <div className="text-accent-tertiary font-mono">{formatMs(b.baseline_time_ms)}</div>
                      </div>
                      <div className="text-center p-2 bg-white/5 rounded-lg">
                        <div className="text-white/50 text-xs">Optimized</div>
                        <div className="text-accent-success font-mono">{formatMs(b.optimized_time_ms)}</div>
                      </div>
                    </div>
                    <div className="text-center">
                      <span className="text-lg font-bold" style={{ color: getSpeedupColor(b.speedup) }}>
                        {b.speedup.toFixed(2)}x
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Selection panels */}
      {mode === 'single' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left benchmark selector */}
          <div className="card">
            <div className="card-header">
              <h3 className="font-medium text-white">Benchmark A</h3>
              {leftBenchmark && (
                <button
                  onClick={() => setLeftBenchmark(null)}
                  className="text-white/40 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            <div className="card-body">
              {leftBenchmark ? (
                <div className="p-4 bg-accent-primary/10 border border-accent-primary/30 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-white">{leftBenchmark.name}</span>
                    <span
                      className="text-lg font-bold"
                      style={{ color: getSpeedupColor(leftBenchmark.speedup) }}
                    >
                      {leftBenchmark.speedup.toFixed(2)}x
                    </span>
                  </div>
                  <div className="text-sm text-white/60">{leftBenchmark.chapter}</div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-white/40">Baseline:</span>{' '}
                      <span className="text-accent-tertiary font-mono">
                        {formatMs(leftBenchmark.baseline_time_ms)}
                      </span>
                    </div>
                    <div>
                      <span className="text-white/40">Optimized:</span>{' '}
                      <span className="text-accent-success font-mono">
                        {formatMs(leftBenchmark.optimized_time_ms)}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="relative mb-3">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
                    <input
                      type="text"
                      placeholder="Search benchmarks..."
                      value={searchLeft}
                      onChange={(e) => setSearchLeft(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder:text-white/40 focus:outline-none focus:border-accent-primary/50"
                    />
                  </div>
                  <div className="max-h-[300px] overflow-y-auto space-y-1">
                    {filteredLeft.slice(0, 20).map((b, i) => (
                      <button
                        key={`${b.chapter}-${b.name}-${i}`}
                        onClick={() => setLeftBenchmark(b)}
                        className="w-full flex items-center justify-between p-3 hover:bg-white/5 rounded-lg transition-colors text-left"
                      >
                        <div>
                          <div className="font-medium text-white text-sm">{b.name}</div>
                          <div className="text-xs text-white/40">{b.chapter}</div>
                        </div>
                        <span
                          className="font-bold"
                          style={{ color: getSpeedupColor(b.speedup) }}
                        >
                          {b.speedup.toFixed(2)}x
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Right benchmark selector */}
          <div className="card">
            <div className="card-header">
              <h3 className="font-medium text-white">Benchmark B</h3>
              {rightBenchmark && (
                <button
                  onClick={() => setRightBenchmark(null)}
                  className="text-white/40 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            <div className="card-body">
              {rightBenchmark ? (
                <div className="p-4 bg-accent-secondary/10 border border-accent-secondary/30 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-white">{rightBenchmark.name}</span>
                    <span
                      className="text-lg font-bold"
                      style={{ color: getSpeedupColor(rightBenchmark.speedup) }}
                    >
                      {rightBenchmark.speedup.toFixed(2)}x
                    </span>
                  </div>
                  <div className="text-sm text-white/60">{rightBenchmark.chapter}</div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-white/40">Baseline:</span>{' '}
                      <span className="text-accent-tertiary font-mono">
                        {formatMs(rightBenchmark.baseline_time_ms)}
                      </span>
                    </div>
                    <div>
                      <span className="text-white/40">Optimized:</span>{' '}
                      <span className="text-accent-success font-mono">
                        {formatMs(rightBenchmark.optimized_time_ms)}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="relative mb-3">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
                    <input
                      type="text"
                      placeholder="Search benchmarks..."
                      value={searchRight}
                      onChange={(e) => setSearchRight(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder:text-white/40 focus:outline-none focus:border-accent-primary/50"
                    />
                  </div>
                  <div className="max-h-[300px] overflow-y-auto space-y-1">
                    {filteredRight.slice(0, 20).map((b, i) => (
                      <button
                        key={`${b.chapter}-${b.name}-${i}`}
                        onClick={() => setRightBenchmark(b)}
                        className="w-full flex items-center justify-between p-3 hover:bg-white/5 rounded-lg transition-colors text-left"
                      >
                        <div>
                          <div className="font-medium text-white text-sm">{b.name}</div>
                          <div className="text-xs text-white/40">{b.chapter}</div>
                        </div>
                        <span
                          className="font-bold"
                          style={{ color: getSpeedupColor(b.speedup) }}
                        >
                          {b.speedup.toFixed(2)}x
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Comparison results */}
      {mode === 'single' && leftBenchmark && rightBenchmark && comparison && (
        <>
          <div className="card">
            <div className="card-header">
              <h3 className="font-medium text-white">Comparison Results</h3>
            </div>
            <div className="card-body">
              <div className="flex items-center justify-center gap-8 py-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-accent-primary">
                    {leftBenchmark.speedup.toFixed(2)}x
                  </div>
                  <div className="text-sm text-white/50">{leftBenchmark.name}</div>
                </div>

                <div className="flex flex-col items-center gap-2">
                  <ArrowRight className="w-8 h-8 text-white/20" />
                  <div
                    className={cn(
                      'px-4 py-2 rounded-full font-bold',
                      comparison.speedupDiff > 0
                        ? 'bg-accent-success/20 text-accent-success'
                        : comparison.speedupDiff < 0
                        ? 'bg-accent-danger/20 text-accent-danger'
                        : 'bg-white/10 text-white/60'
                    )}
                  >
                    {comparison.speedupDiff > 0 ? '+' : ''}
                    {comparison.speedupDiff.toFixed(2)}x
                  </div>
                </div>

                <div className="text-center">
                  <div className="text-2xl font-bold text-accent-secondary">
                    {rightBenchmark.speedup.toFixed(2)}x
                  </div>
                  <div className="text-sm text-white/50">{rightBenchmark.name}</div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 pt-4 border-t border-white/5">
                <div className="text-center p-4 bg-white/5 rounded-lg">
                  <div className="text-sm text-white/50 mb-1">Speedup Difference</div>
                  <div
                    className={cn(
                      'text-xl font-bold',
                      comparison.speedupDiff >= 0 ? 'text-accent-success' : 'text-accent-danger'
                    )}
                  >
                    {comparison.speedupDiff >= 0 ? '+' : ''}
                    {comparison.speedupDiff.toFixed(2)}x
                  </div>
                </div>
                <div className="text-center p-4 bg-white/5 rounded-lg">
                  <div className="text-sm text-white/50 mb-1">Baseline Δ</div>
                  <div className="text-xl font-bold text-white font-mono">
                    {formatMs(Math.abs(comparison.baselineDiff))}
                  </div>
                </div>
                <div className="text-center p-4 bg-white/5 rounded-lg">
                  <div className="text-sm text-white/50 mb-1">Optimized Δ</div>
                  <div className="text-xl font-bold text-white font-mono">
                    {formatMs(Math.abs(comparison.optimizedDiff))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Radar Chart Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-2">
                  <RadarIcon className="w-5 h-5 text-accent-tertiary" />
                  <h3 className="font-medium text-white">Multi-Metric Radar</h3>
                </div>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                    <PolarAngleAxis 
                      dataKey="metric" 
                      tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }}
                    />
                    <PolarRadiusAxis 
                      angle={90} 
                      domain={[0, 100]}
                      tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
                    />
                    <Radar
                      name={leftBenchmark.name.slice(0, 15)}
                      dataKey="A"
                      stroke="#6366f1"
                      fill="#6366f1"
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                    <Radar
                      name={rightBenchmark.name.slice(0, 15)}
                      dataKey="B"
                      stroke="#06b6d4"
                      fill="#06b6d4"
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                    <Legend 
                      wrapperStyle={{ color: 'rgba(255,255,255,0.7)' }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Bar Chart Comparison */}
            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-accent-info" />
                  <h3 className="font-medium text-white">Side-by-Side Metrics</h3>
                </div>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={barChartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis type="number" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }} />
                    <YAxis 
                      dataKey="metric" 
                      type="category" 
                      width={100}
                      tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(17, 24, 39, 0.95)', 
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                      }}
                      labelStyle={{ color: 'white' }}
                    />
                    <Bar dataKey="A" name={leftBenchmark.name.slice(0, 15)} fill="#6366f1" radius={[0, 4, 4, 0]} />
                    <Bar dataKey="B" name={rightBenchmark.name.slice(0, 15)} fill="#06b6d4" radius={[0, 4, 4, 0]} />
                    <Legend />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Historical trends */}
          {trends.data.length > 0 && (
            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-accent-success" />
                  <h3 className="font-medium text-white">Historical Trend Lines</h3>
                </div>
                <span className="text-xs text-white/50">{trends.data.length} points</span>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={trends.data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis
                      dataKey="label"
                      tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                      stroke="rgba(255,255,255,0.3)"
                    />
                    <YAxis
                      tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                      stroke="rgba(255,255,255,0.3)"
                      domain={[0, 'auto']}
                      label={{ value: 'Speedup (x)', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(17, 24, 39, 0.95)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                        color: '#fff',
                      }}
                    />
                    <Line type="monotone" dataKey="overall" name="Overall" stroke="#22c55e" strokeWidth={2} dot={false} />
                    {trends.chapters.map((ch, idx) => (
                      <Line
                        key={ch}
                        type="monotone"
                        dataKey={ch}
                        name={ch}
                        stroke={trendColors[idx % trendColors.length]}
                        strokeWidth={2}
                        dot={false}
                        connectNulls
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Code diff viewer */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center gap-2">
                <FileCode2 className="w-5 h-5 text-accent-primary" />
                <h3 className="font-medium text-white">Code Diff Viewer</h3>
              </div>
              <div className="flex items-center gap-2">
                {codeDiffError && <span className="text-xs text-accent-danger">{codeDiffError}</span>}
                <button
                  onClick={loadCodeDiffs}
                  disabled={codeDiffLoading || (!leftBenchmark && !rightBenchmark)}
                  className={cn(
                    'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all',
                    codeDiffLoading || (!leftBenchmark && !rightBenchmark)
                      ? 'bg-white/10 text-white/40 cursor-not-allowed'
                      : 'bg-accent-primary/20 text-accent-primary hover:bg-accent-primary/30'
                  )}
                >
                  {codeDiffLoading ? 'Loading…' : 'Load Code Diff'}
                </button>
              </div>
            </div>
            <div className="card-body grid grid-cols-1 lg:grid-cols-2 gap-4">
              {(['left', 'right'] as const).map((side) => {
                const selection = side === 'left' ? leftBenchmark : rightBenchmark;
                const diff = side === 'left' ? codeDiffs.left : codeDiffs.right;
                return (
                  <div key={side} className="p-4 rounded-lg border border-white/10 bg-white/[0.03] space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-xs text-white/40 uppercase">Benchmark {side === 'left' ? 'A' : 'B'}</div>
                        <div className="text-white font-semibold">
                          {selection ? selection.name : 'Select a benchmark'}
                        </div>
                        <div className="text-xs text-white/50">{selection?.chapter || '—'}</div>
                      </div>
                      {!diff && <span className="text-xs text-white/40">No diff loaded</span>}
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="space-y-1">
                        <div className="text-xs text-white/50">Baseline</div>
                        <pre className="bg-black/30 border border-white/5 rounded-lg p-3 text-xs text-white/70 overflow-auto max-h-64">
                          {formatCodeSnippet((diff as any)?.baseline)}
                        </pre>
                      </div>
                      <div className="space-y-1">
                        <div className="text-xs text-white/50">Optimized</div>
                        <pre className="bg-black/30 border border-white/5 rounded-lg p-3 text-xs text-white/70 overflow-auto max-h-64">
                          {formatCodeSnippet((diff as any)?.optimized)}
                        </pre>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}

      {/* Chapter Trends */}
      {chapterTrends.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-accent-success" />
              <h3 className="font-medium text-white">Chapter Performance Trends</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              {chapterTrends.slice(0, 8).map((ch, i) => {
                const widthPct = (ch.avgSpeedup / Math.max(...chapterTrends.map(c => c.avgSpeedup))) * 100;
                return (
                  <div key={ch.name} className="flex items-center gap-4">
                    <div className="w-24 text-sm text-white/60 truncate" title={ch.name}>
                      {ch.name}
                    </div>
                    <div className="flex-1 h-8 bg-white/5 rounded-lg overflow-hidden relative">
                      <div
                        className={cn(
                          'h-full rounded-lg transition-all',
                          i === 0 ? 'bg-gradient-to-r from-accent-success to-accent-success/50' :
                          i === 1 ? 'bg-gradient-to-r from-accent-primary to-accent-primary/50' :
                          'bg-gradient-to-r from-accent-info/70 to-accent-info/30'
                        )}
                        style={{ width: `${widthPct}%` }}
                      />
                      <div className="absolute inset-0 flex items-center px-3">
                        <span className="text-sm font-medium text-white">
                          {ch.avgSpeedup.toFixed(2)}x avg
                        </span>
                        <span className="ml-auto text-xs text-white/50">
                          {ch.count} benchmarks • max {ch.maxSpeedup.toFixed(1)}x
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ChartList({
  title,
  colorClass,
  data,
}: {
  title: string;
  colorClass: string;
  data: { name: string; delta: number; baseline?: number; candidate?: number }[];
}) {
  const maxAbs = Math.max(...data.map((d) => Math.abs(d.delta || 0)), 1);
  return (
    <div className="text-xs text-white/80 space-y-1 max-h-60 overflow-y-auto">
      <div className="font-semibold">{title}</div>
      {data.length === 0 && <div className="text-white/50">None</div>}
      {data.map((r, idx) => {
        const width = Math.min(100, (Math.abs(r.delta || 0) / maxAbs) * 100);
        return (
          <div key={idx} className="flex items-center gap-2">
            <span className="flex-1 truncate" title={r.name}>
              {r.name}
            </span>
            <div className="w-32 h-2 bg-white/5 rounded">
              <div className={`h-2 ${colorClass} rounded`} style={{ width: `${width}%` }} />
            </div>
            <span className="w-28 text-right">
              {r.baseline?.toFixed?.(2)}x → {r.candidate?.toFixed?.(2)}x
            </span>
          </div>
        );
      })}
    </div>
  );
}
