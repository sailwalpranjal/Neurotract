'use client';

import React from 'react';
import { GraphMetrics } from '@/lib/types';
import { useAppStore } from '@/lib/store';

interface ConnectomeOverviewProps {
  metrics: GraphMetrics;
}

export default function ConnectomeOverview({ metrics }: ConnectomeOverviewProps) {
  const { setProvenanceModalKey } = useAppStore();
  const g = metrics.global;
  const nRegions = metrics.nodal.degree.length;
  const nCommunities = metrics.communities?.louvain_partition
    ? new Set(metrics.communities.louvain_partition).size
    : 0;

  return (
    <div className="space-y-6">
      {/* Strict Scientific Transparency Disclaimer */}
      <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/50 flex items-start gap-3 text-amber-200 text-xs">
        <span className="p-1 rounded bg-amber-500/20 text-amber-400 mt-0.5">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </span>
        <div className="space-y-1">
          <p className="font-semibold text-amber-300">
            Scientific &amp; Computational Research Notice
          </p>
          <p className="text-neutral-300 leading-relaxed">
            The network properties reported below are quantitative graph-theoretical measurements computed from diffusion-MRI streamlines. They describe macroscopic structural topology (segregation, integration, and modularity) and must not be interpreted as a medical diagnosis, clinical prognosis, or arbitrary brain health score.
          </p>
        </div>
      </div>

      {/* Network Architecture Overview */}
      <div className="glass rounded-xl p-6 border border-neutral-800 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-neutral-800 pb-4">
          <div>
            <h2 className="text-lg font-bold text-white">Structural Connectome Architecture</h2>
            <p className="text-xs text-neutral-400">
              Macroscopic topology across {nRegions} anatomically registered cortical and subcortical parcels
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-md bg-neutral-800 text-neutral-300 text-xs font-mono">
              Parcels: {nRegions}
            </span>
            <span className="px-2.5 py-1 rounded-md bg-neutral-800 text-neutral-300 text-xs font-mono">
              Communities: {nCommunities}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Segregation Card */}
          <div
            onClick={() => setProvenanceModalKey('clustering_coefficient')}
            className="p-4 rounded-xl bg-neutral-900/80 border border-neutral-800 hover:border-neutral-700 transition-all cursor-pointer space-y-3 group"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-neutral-400 group-hover:text-primary-400 transition-colors">
                Functional Segregation
              </span>
              <span className="text-[10px] text-neutral-500 font-mono flex items-center gap-0.5">
                Formula ↗
              </span>
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-emerald-400">
                {g.clustering_coefficient.toFixed(4)}
              </div>
              <div className="text-xs text-neutral-400 font-mono mt-0.5">
                Clustering Coefficient (C)
              </div>
            </div>
            <p className="text-xs text-neutral-300 leading-relaxed">
              Reflects the prevalence of clustered local connectivity among neighboring cortical nodes. Higher clustering indicates dense, specialized local processing circuits.
            </p>
          </div>

          {/* Integration Card */}
          <div
            onClick={() => setProvenanceModalKey('global_efficiency')}
            className="p-4 rounded-xl bg-neutral-900/80 border border-neutral-800 hover:border-neutral-700 transition-all cursor-pointer space-y-3 group"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-neutral-400 group-hover:text-primary-400 transition-colors">
                Global Integration
              </span>
              <span className="text-[10px] text-neutral-500 font-mono flex items-center gap-0.5">
                Formula ↗
              </span>
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-cyan-400">
                {g.global_efficiency.toFixed(4)}
              </div>
              <div className="text-xs text-neutral-400 font-mono mt-0.5">
                Global Efficiency (E)
              </div>
            </div>
            <p className="text-xs text-neutral-300 leading-relaxed">
              Measures the network&apos;s capacity to transmit parallel information across distributed brain systems via inverse harmonic shortest path lengths.
            </p>
          </div>

          {/* Modularity Card */}
          <div
            onClick={() => setProvenanceModalKey('modularity')}
            className="p-4 rounded-xl bg-neutral-900/80 border border-neutral-800 hover:border-neutral-700 transition-all cursor-pointer space-y-3 group"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-neutral-400 group-hover:text-primary-400 transition-colors">
                Network Modularity
              </span>
              <span className="text-[10px] text-neutral-500 font-mono flex items-center gap-0.5">
                Formula ↗
              </span>
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-indigo-400">
                {g.modularity.toFixed(4)}
              </div>
              <div className="text-xs text-neutral-400 font-mono mt-0.5">
                Louvain Q Modularity
              </div>
            </div>
            <p className="text-xs text-neutral-300 leading-relaxed">
              Quantifies the degree to which the structural connectome can be subdivided into {nCommunities} non-overlapping modules with dense internal wiring.
            </p>
          </div>
        </div>

        {/* Small-World Topology Highlight */}
        <div className="p-4 rounded-xl bg-neutral-950/70 border border-neutral-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-white">Small-World Organization Metric</span>
              <span className="px-2 py-0.5 rounded bg-primary-500/20 text-primary-300 text-[10px] font-mono">
                σ = {g.small_worldness.toFixed(3)}
              </span>
            </div>
            <p className="text-xs text-neutral-300 leading-relaxed">
              Biological brains exhibit a small-world index (σ &gt; 1), balancing dense local segregation with rapid global communication across long-range white matter tract highways.
            </p>
          </div>
          <button
            onClick={() => setProvenanceModalKey('small_worldness')}
            className="shrink-0 px-3 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs rounded-lg transition-colors border border-neutral-700"
          >
            View Formula &amp; Benchmark
          </button>
        </div>
      </div>
    </div>
  );
}
