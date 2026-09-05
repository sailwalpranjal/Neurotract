'use client';

import { useState } from 'react';
import { useAppStore } from '@/lib/store';
import { interpretMetric, getMetricDescription } from '@/lib/interpretations';

interface MetricCardProps {
  label: string;
  metricKey: string;
  value?: number;
}

export default function MetricCard({ label, metricKey, value }: MetricCardProps) {
  const { userType, setProvenanceModalKey } = useAppStore();
  const [showTooltip, setShowTooltip] = useState(false);

  const interpretation = interpretMetric(metricKey, value ?? 0, userType);
  const description = getMetricDescription(metricKey, userType);

  const statusColors = {
    normal: 'border-green-500/30 bg-green-500/5 hover:border-green-500/50',
    elevated: 'border-yellow-500/30 bg-yellow-500/5 hover:border-yellow-500/50',
    reduced: 'border-red-500/30 bg-red-500/5 hover:border-red-500/50',
    abnormal: 'border-red-500/50 bg-red-500/10 hover:border-red-500/70',
  };

  const statusBadge = {
    normal: { text: 'Normal', color: 'text-green-400 bg-green-500/20' },
    elevated: { text: 'Elevated', color: 'text-yellow-400 bg-yellow-500/20' },
    reduced: { text: 'Reduced', color: 'text-red-400 bg-red-500/20' },
    abnormal: { text: 'Abnormal', color: 'text-red-400 bg-red-500/20' },
  };

  const status = interpretation.status || 'normal';

  return (
    <div
      className={`relative rounded-xl p-4 border transition-all cursor-pointer group ${statusColors[status]}`}
      onClick={() => setProvenanceModalKey(metricKey)}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
      title="Click to view mathematical derivation and scientific provenance"
    >
      <div className="flex items-start justify-between mb-1">
        <div className="text-sm font-medium text-gray-300 group-hover:text-white transition-colors">{label}</div>
        <div className="flex items-center gap-1.5">
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${statusBadge[status].color}`}>
            {statusBadge[status].text}
          </span>
          <span className="text-neutral-500 group-hover:text-primary-400 text-xs transition-colors" title="View derivation">
            ℹ️
          </span>
        </div>
      </div>
      <div className="text-2xl font-bold font-mono text-white">
        {typeof value === 'number' ? value.toFixed(4) : 'N/A'}
      </div>
      {description && (
        <div className="text-xs text-gray-400 mt-2 line-clamp-2">{description}</div>
      )}
      <div className="mt-3 pt-2 border-t border-white/5 flex items-center justify-between text-[11px] text-neutral-500 group-hover:text-primary-400 transition-colors">
        <span>Scientific Provenance</span>
        <span className="font-mono text-[10px]">Formula ↗</span>
      </div>

      {/* Tooltip */}
      {showTooltip && (
        <div className="absolute z-50 left-0 right-0 top-full mt-2 p-3 glass rounded-lg text-sm shadow-xl border border-white/10 pointer-events-none">
          <p className="text-gray-200">{interpretation[userType]}</p>
          {interpretation.normalRange && (
            <p className="text-xs text-gray-400 mt-2">
              Normative reference range: {interpretation.normalRange[0]} - {interpretation.normalRange[1]}
            </p>
          )}
          <p className="text-[10px] text-primary-400 mt-2 font-mono">
            Click card for exact scientific formula and citation
          </p>
        </div>
      )}
    </div>
  );
}
