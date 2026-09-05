'use client';

import React, { useState } from 'react';
import { apiClient } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { ValidationBenchmarkResult } from '@/lib/types';

interface ValidationCenterProps {
  subjectId?: string;
}

export default function ValidationCenter({ subjectId = 'SUB1' }: ValidationCenterProps) {
  const { validationBenchmark, setValidationBenchmark, addNotification } = useAppStore();
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runBenchmark = async () => {
    setRunning(true);
    setError(null);
    const notifId = addNotification({
      type: 'loading',
      title: 'Running Validation Benchmark',
      message: 'Comparing NeuroTract against DIPY TensorModel and NetworkX graph baselines...',
      duration: 0,
    });

    try {
      const res = await apiClient.runValidationBenchmark(subjectId);
      setValidationBenchmark(res);
      useAppStore.getState().removeNotification(notifId);
      addNotification({
        type: 'success',
        title: 'Validation Benchmark Completed',
        message: res.all_passed
          ? 'All scientific benchmarks passed within strict numerical tolerance!'
          : 'Benchmark completed with discrepancies detected.',
        duration: 5000,
      });
    } catch (err: any) {
      useAppStore.getState().removeNotification(notifId);
      setError(err.message || 'Validation benchmark failed');
      addNotification({
        type: 'error',
        title: 'Validation Benchmark Failed',
        message: err.message || 'Could not complete reference benchmark',
        duration: 8000,
      });
    } finally {
      setRunning(false);
    }
  };

  const bench = validationBenchmark;

  return (
    <div className="glass rounded-xl p-6 border border-neutral-800 space-y-6 text-gray-100">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-neutral-800">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="p-2 rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </span>
            <div>
              <h2 className="text-xl font-bold text-white tracking-wide">
                Reference Validation Laboratory
              </h2>
              <p className="text-xs text-neutral-400">
                Ground-truth numerical concordance verification against DIPY &amp; NetworkX baselines
              </p>
            </div>
          </div>
        </div>

        <button
          onClick={runBenchmark}
          disabled={running}
          className="inline-flex items-center justify-center px-4 py-2.5 bg-primary-600 hover:bg-primary-500 active:bg-primary-700 disabled:bg-neutral-800 disabled:text-neutral-500 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-xl shadow-lg shadow-primary-950/50 transition-all gap-2"
        >
          {running ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Verifying Computations...</span>
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{bench ? 'Re-run Reference Benchmark' : 'Run Reference Benchmark'}</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-800/60 text-red-300 text-sm">
          <span className="font-bold">Error:</span> {error}
        </div>
      )}

      {!bench && !running && (
        <div className="py-8 px-6 rounded-xl bg-neutral-950/40 border border-neutral-800/80 text-center space-y-3">
          <p className="text-neutral-300 text-sm max-w-xl mx-auto">
            NeuroTract enforces mathematical transparency. Click &quot;Run Reference Benchmark&quot; to execute real-time comparison tests of our DTI fitting equations and graph metrics directly against DIPY and NetworkX implementations on verified subject data.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4 text-xs text-neutral-400 font-mono pt-2">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              DIPY TensorModel (FA &amp; MD)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              NetworkX Graph Topologies
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              Zero Discrepancy Tolerance
            </span>
          </div>
        </div>
      )}

      {bench && (
        <div className="space-y-6 animate-in fade-in duration-300">
          {/* Status Badge & Summary */}
          <div className="p-4 rounded-xl bg-neutral-950/70 border border-neutral-800 flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <span
                className={`px-3 py-1 rounded-full text-xs font-bold tracking-wide uppercase ${
                  bench.all_passed
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                    : 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                }`}
              >
                {bench.all_passed ? 'ALL BENCHMARKS VERIFIED' : 'DISCREPANCY DETECTED'}
              </span>
              <span className="text-xs text-neutral-400">
                Verified against {bench.dti_benchmark.reference_toolkit} &amp; {bench.graph_benchmark.reference_toolkit}
              </span>
            </div>
            <div className="text-xs text-neutral-400 font-mono">
              Timestamp: {new Date(bench.dti_benchmark.timestamp).toLocaleTimeString()}
            </div>
          </div>

          {/* DTI Benchmark Grid */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold uppercase tracking-wider text-neutral-300 flex items-center gap-2">
              <span>1. DTI Microstructure vs DIPY TensorModel</span>
              <span className="text-xs text-neutral-400 font-normal">
                (Synthetic &amp; In-Vivo Diffusion Tensors)
              </span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* FA Comparison Card */}
              <div className="p-4 rounded-xl bg-neutral-900/80 border border-neutral-800 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="font-semibold text-white text-sm">
                    Fractional Anisotropy (FA)
                  </div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-md font-mono ${
                      bench.dti_benchmark.metrics.fractional_anisotropy.passed
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                    }`}
                  >
                    {bench.dti_benchmark.metrics.fractional_anisotropy.passed ? 'PASS' : 'FAIL'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <div className="p-2 bg-neutral-950 rounded-lg">
                    <div className="text-neutral-400 text-[10px] uppercase">Pearson r</div>
                    <div className="text-emerald-400 font-bold text-sm">
                      {bench.dti_benchmark.metrics.fractional_anisotropy.pearson_r.toFixed(6)}
                    </div>
                  </div>
                  <div className="p-2 bg-neutral-950 rounded-lg">
                    <div className="text-neutral-400 text-[10px] uppercase">Mean Abs Error</div>
                    <div className="text-white font-bold text-sm">
                      {bench.dti_benchmark.metrics.fractional_anisotropy.mean_absolute_error.toExponential(3)}
                    </div>
                  </div>
                  <div className="p-2 bg-neutral-950 rounded-lg">
                    <div className="text-neutral-400 text-[10px] uppercase">Max Abs Error</div>
                    <div className="text-white font-bold text-sm">
                      {bench.dti_benchmark.metrics.fractional_anisotropy.max_absolute_error.toExponential(3)}
                    </div>
                  </div>
                  <div className="p-2 bg-neutral-950 rounded-lg">
                    <div className="text-neutral-400 text-[10px] uppercase">RMSE</div>
                    <div className="text-white font-bold text-sm">
                      {bench.dti_benchmark.metrics.fractional_anisotropy.rmse.toExponential(3)}
                    </div>
                  </div>
                </div>
                <p className="text-[11px] text-neutral-400">
                  Tolerance: &lt; {bench.dti_benchmark.metrics.fractional_anisotropy.tolerance_threshold}. Confirms accurate Basser &amp; Pierpaoli (1996) scaling.
                </p>
              </div>

              {/* MD Comparison Card */}
              <div className="p-4 rounded-xl bg-neutral-900/80 border border-neutral-800 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="font-semibold text-white text-sm">
                    Mean Diffusivity (MD)
                  </div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-md font-mono ${
                      bench.dti_benchmark.metrics.mean_diffusivity.passed
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                    }`}
                  >
                    {bench.dti_benchmark.metrics.mean_diffusivity.passed ? 'PASS' : 'FAIL'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <div className="p-2 bg-neutral-950 rounded-lg">
                    <div className="text-neutral-400 text-[10px] uppercase">Pearson r</div>
                    <div className="text-emerald-400 font-bold text-sm">
                      {bench.dti_benchmark.metrics.mean_diffusivity.pearson_r.toFixed(6)}
                    </div>
                  </div>
                  <div className="p-2 bg-neutral-950 rounded-lg">
                    <div className="text-neutral-400 text-[10px] uppercase">Mean Abs Error</div>
                    <div className="text-white font-bold text-sm">
                      {bench.dti_benchmark.metrics.mean_diffusivity.mean_absolute_error_mm2_s.toExponential(3)}
                    </div>
                  </div>
                  <div className="p-2 bg-neutral-950 rounded-lg">
                    <div className="text-neutral-400 text-[10px] uppercase">Max Abs Error</div>
                    <div className="text-white font-bold text-sm">
                      {bench.dti_benchmark.metrics.mean_diffusivity.max_absolute_error_mm2_s.toExponential(3)}
                    </div>
                  </div>
                  <div className="p-2 bg-neutral-950 rounded-lg">
                    <div className="text-neutral-400 text-[10px] uppercase">RMSE</div>
                    <div className="text-white font-bold text-sm">
                      {bench.dti_benchmark.metrics.mean_diffusivity.rmse_mm2_s.toExponential(3)}
                    </div>
                  </div>
                </div>
                <p className="text-[11px] text-neutral-400">
                  Tolerance: &lt; {bench.dti_benchmark.metrics.mean_diffusivity.tolerance_threshold} mm²/s. Verifies arithmetic mean of trace / 3.
                </p>
              </div>
            </div>
          </div>

          {/* Graph Metrics Benchmark Table */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold uppercase tracking-wider text-neutral-300 flex items-center gap-2">
                <span>2. Structural Connectome Graph Metrics vs NetworkX</span>
                <span className="text-xs text-neutral-400 font-normal">
                  ({bench.graph_benchmark.matrix_nodes} nodes, {bench.graph_benchmark.matrix_edges} edges)
                </span>
              </h3>
            </div>

            <div className="overflow-x-auto rounded-xl border border-neutral-800 bg-neutral-950/60">
              <table className="w-full text-left text-xs">
                <thead className="bg-neutral-900/90 text-neutral-400 uppercase tracking-wider text-[11px] border-b border-neutral-800">
                  <tr>
                    <th className="py-3 px-4">Metric</th>
                    <th className="py-3 px-4 font-mono">NeuroTract</th>
                    <th className="py-3 px-4 font-mono">NetworkX Reference</th>
                    <th className="py-3 px-4 font-mono">Absolute Delta (|Δ|)</th>
                    <th className="py-3 px-4">Concordance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800/60 font-mono">
                  {bench.graph_benchmark.comparisons.map((c, i) => (
                    <tr key={i} className="hover:bg-white/[0.02] transition-colors">
                      <td className="py-3 px-4 font-sans font-medium text-white">
                        {c.metric}
                      </td>
                      <td className="py-3 px-4 text-neutral-200">
                        {c.neurotract_value.toFixed(6)}
                      </td>
                      <td className="py-3 px-4 text-neutral-300">
                        {c.reference_nx_value.toFixed(6)}
                      </td>
                      <td className="py-3 px-4 text-emerald-400 font-bold">
                        {c.absolute_diff.toExponential(2)}
                      </td>
                      <td className="py-3 px-4 font-sans">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold ${
                            c.passed
                              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                              : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                          }`}
                        >
                          {c.passed ? 'EXACT MATCH' : 'FAIL'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Summary: {bench.graph_benchmark.summary}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
