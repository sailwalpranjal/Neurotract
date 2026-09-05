'use client';

import React, { useState } from 'react';
import { apiClient } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { SensitivityResult } from '@/lib/types';

interface SensitivityLabProps {
  subjectId?: string;
}

export default function SensitivityLab({ subjectId = 'SUB1' }: SensitivityLabProps) {
  const { sensitivityResult, setSensitivityResult, addNotification } = useAppStore();
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedThresholds, setSelectedThresholds] = useState<number[]>([1, 2, 3, 5, 10]);

  const runAnalysis = async () => {
    setRunning(true);
    setError(null);
    const notifId = addNotification({
      type: 'loading',
      title: 'Evaluating Parameter Sensitivity',
      message: `Analyzing connectome topological divergence across thresholds: [${selectedThresholds.join(', ')}]...`,
      duration: 0,
    });

    try {
      const res = await apiClient.runSensitivityAnalysis({
        subject_id: subjectId,
        thresholds: selectedThresholds,
      });
      setSensitivityResult(res);
      useAppStore.getState().removeNotification(notifId);
      addNotification({
        type: 'success',
        title: 'Sensitivity Analysis Complete',
        message: `Stability ratio evaluated at ${(res.summary.stability_ratio * 100).toFixed(1)}% across thresholds.`,
        duration: 5000,
      });
    } catch (err: any) {
      useAppStore.getState().removeNotification(notifId);
      setError(err.message || 'Sensitivity analysis failed');
      addNotification({
        type: 'error',
        title: 'Analysis Error',
        message: err.message || 'Could not complete sensitivity analysis',
        duration: 8000,
      });
    } finally {
      setRunning(false);
    }
  };

  const sens = sensitivityResult;

  return (
    <div className="glass rounded-xl p-6 border border-neutral-800 space-y-6 text-gray-100">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-neutral-800">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="p-2 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
            </span>
            <div>
              <h2 className="text-xl font-bold text-white tracking-wide">
                Parameter Sensitivity Laboratory
              </h2>
              <p className="text-xs text-neutral-400">
                Evaluating structural connectome edge stability &amp; graph metric divergence across streamline filtering thresholds
              </p>
            </div>
          </div>
        </div>

        <button
          onClick={runAnalysis}
          disabled={running}
          className="inline-flex items-center justify-center px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:bg-neutral-800 disabled:text-neutral-500 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-xl shadow-lg shadow-indigo-950/50 transition-all gap-2"
        >
          {running ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Analyzing Sensitivity...</span>
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <span>{sens ? 'Re-run Sensitivity Sweep' : 'Run Sensitivity Sweep'}</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-800/60 text-red-300 text-sm">
          <span className="font-bold">Error:</span> {error}
        </div>
      )}

      {!sens && !running && (
        <div className="py-8 px-6 rounded-xl bg-neutral-950/40 border border-neutral-800/80 text-center space-y-3">
          <p className="text-neutral-300 text-sm max-w-xl mx-auto">
            Connectome topology is sensitive to algorithmic thresholding choices. Click &quot;Run Sensitivity Sweep&quot; to test edge conservation and metric divergence across varying streamline thresholds ([{selectedThresholds.join(', ')}]).
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4 text-xs text-neutral-400 font-mono pt-2">
            <span>• Jaccard Edge Agreement</span>
            <span>• Stable Core vs Spurious Edge Detection</span>
            <span>• Efficiency &amp; Modularity Divergence</span>
          </div>
        </div>
      )}

      {sens && (
        <div className="space-y-6 animate-in fade-in duration-300">
          {/* Summary KPIs */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <div className="p-4 bg-neutral-950/80 border border-neutral-800 rounded-xl space-y-1">
              <div className="text-[11px] text-neutral-400 uppercase tracking-wide">
                Stable Core Edges
              </div>
              <div className="text-2xl font-bold text-emerald-400 font-mono">
                {sens.summary.stable_edge_count}
              </div>
              <div className="text-[10px] text-neutral-500">
                Present in all thresholds
              </div>
            </div>

            <div className="p-4 bg-neutral-950/80 border border-neutral-800 rounded-xl space-y-1">
              <div className="text-[11px] text-neutral-400 uppercase tracking-wide">
                Variable Edges
              </div>
              <div className="text-2xl font-bold text-amber-400 font-mono">
                {sens.summary.variable_edge_count}
              </div>
              <div className="text-[10px] text-neutral-500">
                Threshold-dependent
              </div>
            </div>

            <div className="p-4 bg-neutral-950/80 border border-neutral-800 rounded-xl space-y-1">
              <div className="text-[11px] text-neutral-400 uppercase tracking-wide">
                Total Union Edges
              </div>
              <div className="text-2xl font-bold text-white font-mono">
                {sens.summary.union_edge_count}
              </div>
              <div className="text-[10px] text-neutral-500">
                Cumulative across sweep
              </div>
            </div>

            <div className="p-4 bg-neutral-950/80 border border-neutral-800 rounded-xl space-y-1">
              <div className="text-[11px] text-neutral-400 uppercase tracking-wide">
                Stability Ratio
              </div>
              <div className="text-2xl font-bold text-indigo-400 font-mono">
                {(sens.summary.stability_ratio * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-neutral-500">
                Core / Union fraction
              </div>
            </div>

            <div className="p-4 bg-neutral-950/80 border border-neutral-800 rounded-xl space-y-1 col-span-2 sm:col-span-1">
              <div className="text-[11px] text-neutral-400 uppercase tracking-wide">
                Mean Jaccard
              </div>
              <div className="text-2xl font-bold text-cyan-400 font-mono">
                {sens.summary.mean_pairwise_jaccard.toFixed(3)}
              </div>
              <div className="text-[10px] text-neutral-500">
                Pairwise edge overlap
              </div>
            </div>
          </div>

          {/* Metric Divergence Table */}
          <div className="space-y-2">
            <h3 className="text-sm font-bold uppercase tracking-wider text-neutral-300">
              Graph Metric Trajectory Across Streamline Thresholds
            </h3>
            <div className="overflow-x-auto rounded-xl border border-neutral-800 bg-neutral-950/60">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-neutral-900/90 text-neutral-400 uppercase tracking-wider text-[11px] border-b border-neutral-800">
                  <tr>
                    <th className="py-3 px-3">Threshold</th>
                    <th className="py-3 px-3">Edges</th>
                    <th className="py-3 px-3">Density</th>
                    <th className="py-3 px-3">Clustering</th>
                    <th className="py-3 px-3">Path Length</th>
                    <th className="py-3 px-3">Global Eff.</th>
                    <th className="py-3 px-3">Modularity</th>
                    <th className="py-3 px-3">Lost Edges</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800/60">
                  {sens.runs.map((r) => (
                    <tr key={r.run_index} className="hover:bg-white/[0.02] transition-colors">
                      <td className="py-3 px-3 font-bold text-indigo-400">
                        τ ≥ {r.parameter_value}
                      </td>
                      <td className="py-3 px-3 text-white font-semibold">
                        {r.edge_count}
                      </td>
                      <td className="py-3 px-3 text-neutral-300">
                        {r.density.toFixed(4)}
                      </td>
                      <td className="py-3 px-3 text-neutral-300">
                        {r.clustering_coefficient.toFixed(4)}
                      </td>
                      <td className="py-3 px-3 text-neutral-300">
                        {r.characteristic_path_length.toFixed(4)}
                      </td>
                      <td className="py-3 px-3 text-neutral-300">
                        {r.global_efficiency.toFixed(4)}
                      </td>
                      <td className="py-3 px-3 text-neutral-300">
                        {r.modularity.toFixed(4)}
                      </td>
                      <td className="py-3 px-3 text-amber-400">
                        -{r.lost_edges_vs_baseline}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Jaccard Matrix */}
          <div className="space-y-2">
            <h3 className="text-sm font-bold uppercase tracking-wider text-neutral-300">
              Pairwise Jaccard Similarity Matrix
            </h3>
            <div className="p-4 bg-neutral-950/70 border border-neutral-800 rounded-xl overflow-x-auto">
              <div className="inline-block min-w-full">
                <div className="grid grid-cols-6 gap-1 text-center font-mono text-xs">
                  <div className="text-neutral-500 font-bold p-1">τ</div>
                  {sens.parameter_values.map((v) => (
                    <div key={v} className="text-neutral-400 font-bold p-1">
                      τ={v}
                    </div>
                  ))}

                  {sens.pairwise_jaccard_matrix.map((row, i) => (
                    <React.Fragment key={i}>
                      <div className="text-neutral-400 font-bold p-1 text-left">
                        τ={sens.parameter_values[i]}
                      </div>
                      {row.map((val, j) => {
                        const intensity = Math.round(val * 100);
                        return (
                          <div
                            key={j}
                            className="p-1.5 rounded text-neutral-100 font-semibold"
                            style={{
                              backgroundColor: `rgba(99, 102, 241, ${Math.max(0.15, val * 0.85)})`,
                            }}
                            title={`Jaccard(τ=${sens.parameter_values[i]}, τ=${sens.parameter_values[j]}) = ${val.toFixed(4)}`}
                          >
                            {val.toFixed(2)}
                          </div>
                        );
                      })}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed pt-1">
              {sens.methodology}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
