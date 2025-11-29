'use client';

import { useEffect, useState, useMemo } from 'react';
import { getCodeDiff } from '@/lib/api';
import { Loader2, FileCode, AlertTriangle, GitCompare, Columns2, Rows2 } from 'lucide-react';
import ReactDiffViewer, { DiffMethod } from 'react-diff-viewer-continued';
import { Highlight, themes } from 'prism-react-renderer';

interface CodeDiffModalProps {
  isOpen: boolean;
  onClose: () => void;
  chapter: string;
  name: string;
}

// Custom dark theme matching our dashboard aesthetic
const diffStyles = {
  variables: {
    dark: {
      diffViewerBackground: '#0d0d0d',
      diffViewerColor: '#e6e6e6',
      addedBackground: '#0a2e1a',
      addedColor: '#4ade80',
      removedBackground: '#2e0a0a',
      removedColor: '#f87171',
      wordAddedBackground: '#166534',
      wordRemovedBackground: '#991b1b',
      addedGutterBackground: '#0d3320',
      removedGutterBackground: '#3d0d0d',
      gutterBackground: '#111111',
      gutterBackgroundDark: '#0a0a0a',
      highlightBackground: '#1a1a2e',
      highlightGutterBackground: '#1a1a2e',
      codeFoldGutterBackground: '#161616',
      codeFoldBackground: '#161616',
      emptyLineBackground: '#0d0d0d',
      gutterColor: '#6b7280',
      addedGutterColor: '#4ade80',
      removedGutterColor: '#f87171',
      codeFoldContentColor: '#9ca3af',
      diffViewerTitleBackground: '#161616',
      diffViewerTitleColor: '#e6e6e6',
      diffViewerTitleBorderColor: '#2d2d2d',
    },
  },
  line: {
    padding: '4px 8px',
    fontSize: '13px',
    fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
  },
  gutter: {
    padding: '4px 12px',
    minWidth: '50px',
    fontSize: '12px',
  },
  contentText: {
    fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
  },
  titleBlock: {
    padding: '12px 16px',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
  },
  marker: {
    padding: '0 8px',
  },
  codeFold: {
    fontSize: '12px',
  },
};

// Syntax highlighter for Python code
const highlightSyntax = (str: string) => (
  <Highlight theme={themes.nightOwl} code={str} language="python">
    {({ tokens, getLineProps, getTokenProps }) => (
      <pre style={{ display: 'inline', background: 'transparent', padding: 0, margin: 0 }}>
        {tokens.map((line, i) => {
          const lineProps = getLineProps({ line, key: i });
          return (
            <span key={i} {...lineProps} style={{ ...lineProps.style, display: 'inline' }}>
              {line.map((token, key) => {
                const tokenProps = getTokenProps({ token, key });
                return <span key={key} {...tokenProps} />;
              })}
            </span>
          );
        })}
      </pre>
    )}
  </Highlight>
);

export function CodeDiffModal({ isOpen, onClose, chapter, name }: CodeDiffModalProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);
  const [splitView, setSplitView] = useState(true);

  useEffect(() => {
    if (!isOpen) return;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await getCodeDiff(chapter, name);
        setData(res);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load code diff');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [isOpen, chapter, name]);

  const baselineCode = useMemo(() => data?.baseline || '', [data]);
  const optimizedCode = useMemo(() => data?.optimized || '', [data]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[9998] bg-black/80 backdrop-blur-md flex items-start justify-center pt-[4vh]"
      onClick={onClose}
    >
      <div
        className="w-[1400px] max-w-[98vw] bg-[#0d0d0d] border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-[#111]">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-accent-info/20 to-accent-info/5 rounded-lg">
              <GitCompare className="w-5 h-5 text-accent-info" />
            </div>
            <div>
              <div className="text-xs uppercase tracking-wider text-white/40 font-medium">Code Comparison</div>
              <div className="text-lg font-semibold text-white">{chapter}: {name}</div>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {/* View Toggle */}
            <div className="flex items-center gap-1 p-1 bg-white/5 rounded-lg">
              <button
                onClick={() => setSplitView(true)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                  splitView 
                    ? 'bg-accent-info/20 text-accent-info' 
                    : 'text-white/50 hover:text-white/70'
                }`}
              >
                <Columns2 className="w-4 h-4" />
                Split
              </button>
              <button
                onClick={() => setSplitView(false)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                  !splitView 
                    ? 'bg-accent-info/20 text-accent-info' 
                    : 'text-white/50 hover:text-white/70'
                }`}
              >
                <Rows2 className="w-4 h-4" />
                Unified
              </button>
            </div>
            
            <button
              onClick={onClose}
              className="px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm font-medium text-white/70 hover:text-white transition-all"
            >
              Close
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto">
          {loading ? (
            <div className="flex items-center justify-center py-20 text-white/60">
              <Loader2 className="w-6 h-6 animate-spin mr-3" />
              <span className="text-lg">Loading baseline vs optimized code...</span>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center gap-3 py-16">
              <AlertTriangle className="w-6 h-6 text-accent-warning" />
              <span className="text-accent-warning text-lg">{error}</span>
            </div>
          ) : (
            <div className="diff-viewer-container">
              <ReactDiffViewer
                oldValue={baselineCode}
                newValue={optimizedCode}
                splitView={splitView}
                useDarkTheme={true}
                leftTitle="📄 Baseline (Unoptimized)"
                rightTitle="⚡ Optimized"
                styles={diffStyles}
                renderContent={highlightSyntax}
                compareMethod={DiffMethod.WORDS}
                showDiffOnly={false}
                extraLinesSurroundingDiff={3}
              />
            </div>
          )}
        </div>

        {/* Legend Footer */}
        {!loading && !error && (
          <div className="flex items-center justify-center gap-8 px-6 py-3 border-t border-white/10 bg-[#111]">
            <div className="flex items-center gap-2 text-xs text-white/50">
              <span className="w-3 h-3 rounded bg-[#2e0a0a] border border-[#f87171]/30"></span>
              <span>Removed from baseline</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-white/50">
              <span className="w-3 h-3 rounded bg-[#0a2e1a] border border-[#4ade80]/30"></span>
              <span>Added in optimized</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-white/50">
              <span className="px-1.5 py-0.5 rounded bg-[#991b1b] text-[10px] font-mono">word</span>
              <span>Changed words highlighted</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
