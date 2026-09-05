'use client';

import React, { useEffect, useState } from 'react';
import { useAppStore } from '@/lib/store';
import { apiClient } from '@/lib/api';
import { MetricProvenance } from '@/lib/types';

interface ProvenanceInspectorProps {
  metricKey?: string | null;
  onClose?: () => void;
}

export default function ProvenanceInspector({
  metricKey: propKey,
  onClose,
}: ProvenanceInspectorProps) {
  const { provenanceModalKey, setProvenanceModalKey } = useAppStore();
  const activeKey = propKey !== undefined ? propKey : provenanceModalKey;

  const [provenance, setProvenance] = useState<MetricProvenance | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!activeKey) {
      setProvenance(null);
      return;
    }

    let isMounted = true;
    setLoading(true);
    setError(null);

    apiClient
      .getMetricProvenance(activeKey)
      .then((data) => {
        if (isMounted) {
          setProvenance(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Failed to load provenance record');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [activeKey]);

  if (!activeKey) return null;

  const handleClose = () => {
    if (onClose) {
      onClose();
    } else {
      setProvenanceModalKey(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl bg-neutral-900 border border-neutral-700/80 rounded-2xl shadow-2xl max-h-[90vh] flex flex-col overflow-hidden text-gray-100">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800 bg-neutral-950/60">
          <div className="flex items-center space-x-3">
            <span className="p-2 rounded-lg bg-primary-500/20 text-primary-400 border border-primary-500/30">
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </span>
            <div>
              <h3 className="text-lg font-bold text-white tracking-wide">
                Metric Provenance &amp; Mathematical Derivation
              </h3>
              <p className="text-xs text-neutral-400 font-mono">
                Key: {activeKey}
              </p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="p-1.5 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
            aria-label="Close"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-6">
          {loading && (
            <div className="py-12 text-center text-neutral-400">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-500 mb-3" />
              <p className="text-sm">Retrieving scientific provenance registry...</p>
            </div>
          )}

          {error && (
            <div className="p-4 rounded-xl bg-red-950/40 border border-red-800/60 text-red-300 text-sm">
              <p className="font-semibold mb-1">Provenance Lookup Failed</p>
              <p className="text-xs text-red-400">{error}</p>
            </div>
          )}

          {provenance && !loading && (
            <>
              {/* Metric Title & Unit */}
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 pb-4 border-b border-neutral-800">
                <div>
                  <h4 className="text-xl font-semibold text-white">
                    {provenance.name}
                  </h4>
                  <p className="text-sm text-neutral-300 mt-1 leading-relaxed">
                    {provenance.description}
                  </p>
                </div>
                <div className="self-start sm:self-auto px-3 py-1 bg-neutral-800 border border-neutral-700 rounded-lg text-xs font-mono text-primary-300 whitespace-nowrap">
                  Unit: {provenance.units}
                </div>
              </div>

              {/* Exact Formula Section */}
              <div className="space-y-2">
                <div className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                  Mathematical Formulation
                </div>
                <div className="p-4 bg-neutral-950 rounded-xl border border-neutral-800 font-mono text-sm text-emerald-400 overflow-x-auto select-all shadow-inner">
                  {provenance.formula}
                </div>
              </div>

              {/* Scientific Citation */}
              <div className="space-y-2">
                <div className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                  Authoritative Reference Citation
                </div>
                <div className="p-4 bg-neutral-800/40 rounded-xl border border-neutral-700/60 space-y-2">
                  <div className="text-sm font-medium text-white leading-relaxed">
                    {provenance.reference_citation}
                  </div>
                </div>
              </div>

              {/* Inputs & Parameters */}
              <div className="space-y-2">
                <div className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                  Input Parameters &amp; Upstream Variables
                </div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(provenance.input_properties || {}).map(([key, val], idx) => (
                    <span
                      key={idx}
                      className="px-2.5 py-1 bg-neutral-800 text-neutral-200 border border-neutral-700 rounded-md text-xs font-mono"
                    >
                      {key}: {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                    </span>
                  ))}
                </div>
              </div>

              {/* Software Environment */}
              <div className="space-y-2">
                <div className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                  Software Dependencies &amp; Versions
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                  {Object.entries(provenance.software_versions || {}).map(
                    ([pkg, version]) => (
                      <div
                        key={pkg}
                        className="p-2 bg-neutral-950/70 border border-neutral-800 rounded-lg flex flex-col"
                      >
                        <span className="text-neutral-500 uppercase tracking-wide text-[10px]">
                          {pkg}
                        </span>
                        <span className="text-neutral-200 font-semibold mt-0.5">
                          {version}
                        </span>
                      </div>
                    )
                  )}
                </div>
              </div>

              {/* Execution Trace */}
              <div className="pt-2 text-[11px] text-neutral-500 flex flex-col sm:flex-row sm:items-center sm:justify-between border-t border-neutral-800 gap-1">
                <div>
                  Execution ID:{' '}
                  <span className="font-mono text-neutral-400">
                    {provenance.execution_id}
                  </span>
                </div>
                <div>
                  Calculated at:{' '}
                  <span className="text-neutral-400">
                    {new Date(provenance.timestamp).toLocaleString()}
                  </span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-neutral-800 bg-neutral-950/80 flex items-center justify-between">
          <span className="text-xs text-neutral-400">
            NeuroTract 2.0 Scientific Transparency Engine
          </span>
          <button
            onClick={handleClose}
            className="px-4 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-white rounded-lg text-sm font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
