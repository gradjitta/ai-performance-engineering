'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, Thermometer, Zap, HardDrive, Cpu, Play, Pause, RefreshCw } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { getGpuStreamUrl, getGpuHistory } from '@/lib/api';
import { cn } from '@/lib/utils';

interface GpuMetric {
  timestamp: number;
  iso_time: string;
  temperature: number;
  temperature_hbm?: number;
  power: number;
  power_limit: number;
  utilization: number;
  memory_used: number;
  memory_total: number;
  memory_percent: number;
  gpu_name: string;
}

interface LiveGpuMonitorProps {
  className?: string;
  compact?: boolean;
}

export function LiveGpuMonitor({ className, compact = false }: LiveGpuMonitorProps) {
  const [metrics, setMetrics] = useState<GpuMetric[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentMetric, setCurrentMetric] = useState<GpuMetric | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Load history on mount
  useEffect(() => {
    async function loadHistory() {
      try {
        const data = await getGpuHistory() as any;
        if (data.history && data.history.length > 0) {
          setMetrics(data.history);
          setCurrentMetric(data.history[data.history.length - 1]);
        }
      } catch (e) {
        // Ignore history load errors
      }
    }
    loadHistory();
  }, []);

  const startStreaming = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const url = getGpuStreamUrl();
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as GpuMetric;
        setCurrentMetric(data);
        setMetrics(prev => {
          const newMetrics = [...prev, data];
          // Keep last 300 samples (5 minutes at 1/sec)
          return newMetrics.slice(-300);
        });
        setError(null);
      } catch (e) {
        console.error('Failed to parse GPU metric:', e);
      }
    };

    eventSource.onerror = () => {
      setError('Connection lost. Click to reconnect.');
      setIsStreaming(false);
      eventSource.close();
    };

    setIsStreaming(true);
    setError(null);
  }, []);

  const stopStreaming = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Format time for chart
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-brand-card border border-white/10 rounded-lg p-3 shadow-xl">
          <p className="text-white/60 text-xs mb-2">{formatTime(label)}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value?.toFixed?.(1) || entry.value}{entry.unit || ''}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (compact) {
    return (
      <div className={cn('card', className)}>
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-accent-primary" />
            <h3 className="font-medium text-white">Live GPU Monitor</h3>
          </div>
          <button
            onClick={isStreaming ? stopStreaming : startStreaming}
            className={cn(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all',
              isStreaming
                ? 'bg-accent-success/20 text-accent-success'
                : 'bg-accent-primary/20 text-accent-primary hover:bg-accent-primary/30'
            )}
          >
            {isStreaming ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isStreaming ? 'Streaming' : 'Start'}
          </button>
        </div>
        <div className="card-body">
          {currentMetric ? (
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center">
                <Thermometer className="w-5 h-5 mx-auto mb-1 text-accent-danger" />
                <div className="text-2xl font-bold text-white">{currentMetric.temperature}°C</div>
                <div className="text-xs text-white/50">Temperature</div>
              </div>
              <div className="text-center">
                <Zap className="w-5 h-5 mx-auto mb-1 text-accent-warning" />
                <div className="text-2xl font-bold text-white">{currentMetric.power?.toFixed(0)}W</div>
                <div className="text-xs text-white/50">Power</div>
              </div>
              <div className="text-center">
                <Cpu className="w-5 h-5 mx-auto mb-1 text-accent-primary" />
                <div className="text-2xl font-bold text-white">{currentMetric.utilization}%</div>
                <div className="text-xs text-white/50">Utilization</div>
              </div>
              <div className="text-center">
                <HardDrive className="w-5 h-5 mx-auto mb-1 text-accent-secondary" />
                <div className="text-2xl font-bold text-white">{currentMetric.memory_percent}%</div>
                <div className="text-xs text-white/50">Memory</div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-white/50">
              Click &quot;Start&quot; to begin monitoring
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Header Card */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-accent-primary" />
            <h2 className="text-lg font-semibold text-white">Live GPU Monitor</h2>
            {currentMetric && (
              <span className="text-sm text-white/50">• {currentMetric.gpu_name}</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {error && (
              <span className="text-xs text-accent-danger">{error}</span>
            )}
            <button
              onClick={isStreaming ? stopStreaming : startStreaming}
              className={cn(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
                isStreaming
                  ? 'bg-accent-success/20 text-accent-success border border-accent-success/30'
                  : 'bg-accent-primary/20 text-accent-primary hover:bg-accent-primary/30'
              )}
            >
              {isStreaming ? (
                <>
                  <div className="w-2 h-2 rounded-full bg-accent-success animate-pulse" />
                  Live
                  <Pause className="w-4 h-4" />
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start Monitoring
                </>
              )}
            </button>
          </div>
        </div>

        {/* Current Stats */}
        {currentMetric && (
          <div className="px-5 py-4 border-t border-white/5 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 rounded-lg bg-gradient-to-br from-red-500/10 to-orange-500/10 border border-red-500/20">
              <div className="flex items-center gap-2 mb-1">
                <Thermometer className="w-4 h-4 text-red-400" />
                <span className="text-xs text-white/60">Temperature</span>
              </div>
              <div className="text-2xl font-bold text-white">
                {currentMetric.temperature}°C
                {currentMetric.temperature_hbm && (
                  <span className="text-sm text-white/50 ml-2">HBM: {currentMetric.temperature_hbm}°C</span>
                )}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-gradient-to-br from-yellow-500/10 to-amber-500/10 border border-yellow-500/20">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span className="text-xs text-white/60">Power</span>
              </div>
              <div className="text-2xl font-bold text-white">
                {currentMetric.power?.toFixed(0)}W
                <span className="text-sm text-white/50 ml-2">/ {currentMetric.power_limit}W</span>
              </div>
            </div>
            <div className="p-3 rounded-lg bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border border-cyan-500/20">
              <div className="flex items-center gap-2 mb-1">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <span className="text-xs text-white/60">GPU Utilization</span>
              </div>
              <div className="text-2xl font-bold text-white">{currentMetric.utilization}%</div>
            </div>
            <div className="p-3 rounded-lg bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20">
              <div className="flex items-center gap-2 mb-1">
                <HardDrive className="w-4 h-4 text-purple-400" />
                <span className="text-xs text-white/60">Memory</span>
              </div>
              <div className="text-2xl font-bold text-white">
                {(currentMetric.memory_used / 1024).toFixed(1)}GB
                <span className="text-sm text-white/50 ml-2">/ {(currentMetric.memory_total / 1024).toFixed(0)}GB</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Charts */}
      {metrics.length > 1 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Temperature & Power Chart */}
          <div className="card">
            <div className="card-header">
              <h3 className="font-medium text-white">Temperature & Power</h3>
              <span className="text-xs text-white/50">{metrics.length} samples</span>
            </div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatTime}
                    stroke="rgba(255,255,255,0.3)"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                  />
                  <YAxis 
                    yAxisId="temp"
                    domain={[0, 100]}
                    stroke="rgba(255,255,255,0.3)"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                    label={{ value: '°C', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                  />
                  <YAxis 
                    yAxisId="power"
                    orientation="right"
                    domain={[0, 'auto']}
                    stroke="rgba(255,255,255,0.3)"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                    label={{ value: 'W', angle: 90, position: 'insideRight', fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Line 
                    yAxisId="temp"
                    type="monotone" 
                    dataKey="temperature" 
                    stroke="#ef4444" 
                    strokeWidth={2}
                    dot={false}
                    name="Temp"
                    unit="°C"
                  />
                  <Line 
                    yAxisId="power"
                    type="monotone" 
                    dataKey="power" 
                    stroke="#f59e0b" 
                    strokeWidth={2}
                    dot={false}
                    name="Power"
                    unit="W"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Utilization & Memory Chart */}
          <div className="card">
            <div className="card-header">
              <h3 className="font-medium text-white">Utilization & Memory</h3>
            </div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatTime}
                    stroke="rgba(255,255,255,0.3)"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                  />
                  <YAxis 
                    domain={[0, 100]}
                    stroke="rgba(255,255,255,0.3)"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                    label={{ value: '%', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Area 
                    type="monotone" 
                    dataKey="utilization" 
                    stroke="#06b6d4" 
                    fill="#06b6d4"
                    fillOpacity={0.2}
                    strokeWidth={2}
                    name="GPU Util"
                    unit="%"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="memory_percent" 
                    stroke="#a855f7" 
                    fill="#a855f7"
                    fillOpacity={0.2}
                    strokeWidth={2}
                    name="Memory"
                    unit="%"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {metrics.length === 0 && !isStreaming && (
        <div className="card">
          <div className="card-body text-center py-12">
            <Activity className="w-12 h-12 mx-auto mb-4 text-white/20" />
            <h3 className="text-lg font-medium text-white mb-2">No Data Yet</h3>
            <p className="text-white/50 mb-4">Click &quot;Start Monitoring&quot; to begin collecting GPU metrics</p>
            <button
              onClick={startStreaming}
              className="px-6 py-2 bg-accent-primary/20 text-accent-primary rounded-lg hover:bg-accent-primary/30 transition-colors"
            >
              <Play className="w-4 h-4 inline mr-2" />
              Start Monitoring
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

