'use client';

import React, { useState } from 'react';
import { useAppStore } from '@/lib/store';

interface RunMetricItem {
  key: string;
  name: string;
  unit: string;
  valA: number;
  valB: number;
  description: string;
}

export default function RunComparisonView() {
  const { activeSubject } = useAppStore();

  const [compareMode, setCompareMode] = useState<'threshold' | 'subject'>('threshold');

  // Baseline Run A (SUB1 standard pipeline, threshold tau = 1)
  const runA = {
    id: 'SUB1_baseline_tau1',
    subject: 'SUB1',
    dataset: 'Stanford HARDI (b=1000 s/mm², 150 dirs)',
    threshold: 1,
    runtime_s: 34.2,
    seeds: 10000,
  };

  // Comparative Run B depending on mode
  const runB = compareMode === 'threshold'
    ? {
        id: 'SUB1_filtered_tau5',
        subject: 'SUB1',
        dataset: 'Stanford HARDI (b=1000 s/mm², 150 dirs)',
        threshold: 5,
        runtime_s: 34.2,
        seeds: 10000,
        label: 'SUB1 (Filtered τ ≥ 5)',
      }
    : {
        id: 'SUB2_baseline_tau1',
        subject: 'SUB2',
        dataset: 'Stanford HARDI SUB2 (b=1000 s/mm², 150 dirs)',
        threshold: 1,
        runtime_s: 36.8,
        seeds: 10000,
        label: 'SUB2 (Subject Comparison)',
      };

  const metricsComparison: RunMetricItem[] = compareMode === 'threshold'
    ? [
        {
          key: 'retained_edges',
          name: 'Retained Connectome Edges',
          unit: 'edges',
          valA: 394,
          valB: 274,
          description: 'Number of undirected pairwise white matter connections retained.',
        },
        {
          key: 'density',
          name: 'Network Connection Density',
          unit: 'ratio [0, 1]',
          valA: 0.1006,
          valB: 0.0700,
          description: 'Fraction of all possible 3,916 inter-parcel connections that exist.',
        },
        {
          key: 'clustering_coefficient',
          name: 'Clustering Coefficient (C)',
          unit: 'dimensionless',
          valA: 0.0149,
          valB: 0.0241,
          description: 'Prevalence of clustered local connectivity among neighboring nodes.',
        },
        {
          key: 'characteristic_path_length',
          name: 'Characteristic Path Length (L)',
          unit: 'hops / steps',
          valA: 12.4954,
          valB: 41.2496,
          description: 'Average harmonic shortest distance between all parcel pairs.',
        },
        {
          key: 'global_efficiency',
          name: 'Global Efficiency (E)',
          unit: 'dimensionless',
          valA: 0.4898,
          valB: 0.4191,
          description: 'Parallel information transfer efficiency via inverse path lengths.',
        },
        {
          key: 'louvain_modularity',
          name: 'Louvain Modularity (Q)',
          unit: 'dimensionless',
          valA: 0.2523,
          valB: 0.2549,
          description: 'Degree of connectome segregation into non-overlapping communities.',
        },
        {
          key: 'mean_fa',
          name: 'Mean White Matter FA',
          unit: 'fraction [0, 1]',
          valA: 0.428,
          valB: 0.428,
          description: 'DTI fractional anisotropy averaged over reconstructed streamline voxels.',
        },
      ]
    : [
        {
          key: 'streamlines_tracked',
          name: 'Total Streamlines Tracked',
          unit: 'trajectories',
          valA: 22392,
          valB: 18660,
          description: 'Algorithmic white matter streamlines retained after tracking.',
        },
        {
          key: 'retained_edges',
          name: 'Retained Connectome Edges',
          unit: 'edges',
          valA: 394,
          valB: 328,
          description: 'Number of pairwise inter-parcel connections.',
        },
        {
          key: 'density',
          name: 'Network Connection Density',
          unit: 'ratio [0, 1]',
          valA: 0.1006,
          valB: 0.0837,
          description: 'Ratio of existing connections to total possible combinations.',
        },
        {
          key: 'clustering_coefficient',
          name: 'Clustering Coefficient (C)',
          unit: 'dimensionless',
          valA: 0.0149,
          valB: 0.0162,
          description: 'Local neighborhood interconnectedness.',
        },
        {
          key: 'characteristic_path_length',
          name: 'Characteristic Path Length (L)',
          unit: 'hops / steps',
          valA: 12.4954,
          valB: 14.1205,
          description: 'Average shortest path between regional nodes.',
        },
        {
          key: 'global_efficiency',
          name: 'Global Efficiency (E)',
          unit: 'dimensionless',
          valA: 0.4898,
          valB: 0.4721,
          description: 'Global communication capacity across distributed systems.',
        },
        {
          key: 'louvain_modularity',
          name: 'Louvain Modularity (Q)',
          unit: 'dimensionless',
          valA: 0.2523,
          valB: 0.2618,
          description: 'Community partition modularity index.',
        },
      ];

  return (
    <div className="space-y-6">
      {/* Header & Mode Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-2xl bg-neutral-900/90 border border-neutral-800">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-bold text-white tracking-wide">
              Reproducible Pipeline Run Comparison
            </h3>
            <span className="text-[10px] px-2 py-0.5 rounded font-mono font-bold bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
              RUN A vs RUN B
            </span>
          </div>
          <p className="text-xs text-neutral-400 mt-1">
            Exact mathematical quantification of topological divergence, parameter sensitivity, and cross-subject variance
          </p>
        </div>

        {/* Mode Selector */}
        <div className="flex items-center gap-1.5 p-1 bg-neutral-950 rounded-xl border border-neutral-800 text-xs font-mono">
          <button
            onClick={() => setCompareMode('threshold')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              compareMode === 'threshold'
                ? 'bg-cyan-600 text-white shadow-md'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-900'
            }`}
          >
            Threshold Sweep (τ=1 vs τ=5)
          </button>
          <button
            onClick={() => setCompareMode('subject')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              compareMode === 'subject'
                ? 'bg-cyan-600 text-white shadow-md'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-900'
            }`}
          >
            Cross-Subject (SUB1 vs SUB2)
          </button>
        </div>
      </div>

      {/* Run Metadata Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Run A Card */}
        <div className="p-4 rounded-xl bg-neutral-950 border border-neutral-800/90 font-mono text-xs space-y-2">
          <div className="flex items-center justify-between pb-2 border-b border-neutral-800">
            <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 font-bold">
              RUN A (Baseline Reference)
            </span>
            <span className="text-neutral-500">{runA.id}</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-[11px] text-neutral-300">
            <div>Subject: <strong className="text-white">{runA.subject}</strong></div>
            <div>Threshold: <strong className="text-cyan-400">τ ≥ {runA.threshold}</strong></div>
            <div>Seeds: <strong className="text-white">{runA.seeds.toLocaleString()}</strong></div>
            <div>Runtime: <strong className="text-white">{runA.runtime_s}s</strong></div>
          </div>
          <div className="text-[10px] text-neutral-400 truncate">
            Dataset: {runA.dataset}
          </div>
        </div>

        {/* Run B Card */}
        <div className="p-4 rounded-xl bg-neutral-950 border border-neutral-800/90 font-mono text-xs space-y-2">
          <div className="flex items-center justify-between pb-2 border-b border-neutral-800">
            <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-bold">
              RUN B ({runB.label})
            </span>
            <span className="text-neutral-500">{runB.id}</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-[11px] text-neutral-300">
            <div>Subject: <strong className="text-white">{runB.subject}</strong></div>
            <div>Threshold: <strong className="text-amber-400">τ ≥ {runB.threshold}</strong></div>
            <div>Seeds: <strong className="text-white">{runB.seeds.toLocaleString()}</strong></div>
            <div>Runtime: <strong className="text-white">{runB.runtime_s}s</strong></div>
          </div>
          <div className="text-[10px] text-neutral-400 truncate">
            Dataset: {runB.dataset}
          </div>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="border border-neutral-800 rounded-xl overflow-hidden bg-neutral-950 font-mono text-xs">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="bg-neutral-900 border-b border-neutral-800 text-[11px] text-neutral-400 uppercase tracking-wider">
                <th className="p-3.5">Metric &amp; Description</th>
                <th className="p-3.5 text-right">Run A (Ref)</th>
                <th className="p-3.5 text-right">Run B</th>
                <th className="p-3.5 text-right">Absolute Δ</th>
                <th className="p-3.5 text-right">% Delta (%Δ)</th>
                <th className="p-3.5 text-center">Stability Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-900 text-[12px]">
              {metricsComparison.map((m) => {
                const delta = m.valB - m.valA;
                const pctDelta = m.valA !== 0 ? (delta / Math.abs(m.valA)) * 100 : 0;
                const absPct = Math.abs(pctDelta);

                const statusColor =
                  absPct < 5
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                    : absPct < 25
                    ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                    : 'bg-rose-500/10 text-rose-400 border-rose-500/30';

                const statusText =
                  absPct < 5
                    ? 'PRESERVED'
                    : absPct < 25
                    ? 'MODERATE SHIFT'
                    : 'HIGH SENSITIVITY';

                return (
                  <tr key={m.key} className="hover:bg-neutral-900/50 transition-colors">
                    <td className="p-3.5">
                      <div className="font-semibold text-white">{m.name}</div>
                      <div className="text-[10px] text-neutral-400 font-sans mt-0.5">{m.description}</div>
                    </td>
                    <td className="p-3.5 text-right text-neutral-300 font-bold">
                      {m.valA > 10 ? m.valA.toLocaleString() : m.valA.toFixed(4)}
                    </td>
                    <td className="p-3.5 text-right text-neutral-300 font-bold">
                      {m.valB > 10 ? m.valB.toLocaleString() : m.valB.toFixed(4)}
                    </td>
                    <td className={`p-3.5 text-right font-bold ${delta >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {delta >= 0 ? '+' : ''}
                      {Math.abs(delta) > 10 ? delta.toFixed(0) : delta.toFixed(4)}
                    </td>
                    <td className={`p-3.5 text-right font-bold ${pctDelta >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {pctDelta >= 0 ? '+' : ''}
                      {pctDelta.toFixed(2)}%
                    </td>
                    <td className="p-3.5 text-center">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${statusColor}`}>
                        {statusText}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
