'use client';

import { useState } from 'react';
import { useAppStore } from '@/lib/store';
import { interpretMetric, getMetricDescription } from '@/lib/interpretations';

interface MetricCardProps {
  label: string;
  metricKey: string;
  value: number;
}

export default function MetricCard({ label, metricKey, value }: MetricCardProps) {
  const { userType } = useAppStore();
  const [showTooltip, setShowTooltip] = useState(false);

  const interpretation = interpretMetric(metricKey, value, userType);
  const description = getMetricDescription(metricKey, userType);

  const statusColors = {
    normal: 'border-green-500/30 bg-green-500/5',
    elevated: 'border-yellow-500/30 bg-yellow-500/5',
    reduced: 'border-red-500/30 bg-red-500/5',
    abnormal: 'border-red-500/50 bg-red-500/10',
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
      className={`relative rounded-lg p-4 border transition-all cursor-pointer hover:bg-white/5 ${statusColors[status]}`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <div className="flex items-start justify-between mb-1">
        <div className="text-sm text-gray-400">{label}</div>
        <span className={`text-xs px-2 py-0.5 rounded-full ${statusBadge[status].color}`}>
          {statusBadge[status].text}
        </span>
      </div>
      <div className="text-2xl font-semibold">
        {typeof value === 'number' ? value.toFixed(4) : 'N/A'}
      </div>
      {description && (
        <div className="text-xs text-gray-500 mt-2 line-clamp-2">{description}</div>
      )}

      {/* Tooltip */}
      {showTooltip && (
        <div className="absolute z-50 left-0 right-0 top-full mt-2 p-3 glass rounded-lg text-sm shadow-xl border border-white/10">
          <p className="text-gray-200">{interpretation[userType]}</p>
          {interpretation.normalRange && (
            <p className="text-xs text-gray-500 mt-2">
              Normative range: {interpretation.normalRange[0]} - {interpretation.normalRange[1]}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
