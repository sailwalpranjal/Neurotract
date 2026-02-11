'use client';

import { GraphMetrics } from '@/lib/types';
import { getMetricStatus } from '@/lib/interpretations';

interface BrainHealthSummaryProps {
  metrics: GraphMetrics;
}

interface HealthIndicator {
  label: string;
  description: string;
  status: 'good' | 'attention' | 'concern';
  detail: string;
}

export default function BrainHealthSummary({ metrics }: BrainHealthSummaryProps) {
  const g = metrics.global;
  const nRegions = metrics.nodal.degree.length;
  const nCommunities = metrics.communities?.louvain_partition
    ? new Set(metrics.communities.louvain_partition).size
    : 0;

  // Build health indicators
  const indicators: HealthIndicator[] = [
    {
      label: 'Network Organization',
      description: 'How well-organized are your brain connections?',
      status: mapStatus(getMetricStatus('clustering_coefficient', g.clustering_coefficient)),
      detail: g.clustering_coefficient >= 0.2
        ? 'Your brain regions are well-organized into local groups that work together efficiently.'
        : 'The organization of local brain region groups may benefit from further review.',
    },
    {
      label: 'Communication Speed',
      description: 'How quickly can brain regions share information?',
      status: mapStatus(getMetricStatus('characteristic_path_length', g.characteristic_path_length)),
      detail: getMetricStatus('characteristic_path_length', g.characteristic_path_length) === 'normal'
        ? 'Information can travel between brain regions efficiently — this is a good sign.'
        : g.characteristic_path_length > 4
        ? 'Communication paths are longer than typical, which may affect processing speed.'
        : 'Communication paths are shorter than typical, indicating very direct connections.',
    },
    {
      label: 'Overall Efficiency',
      description: 'How effectively does your brain network transfer information?',
      status: mapStatus(getMetricStatus('global_efficiency', g.global_efficiency)),
      detail: g.global_efficiency >= 0.3
        ? 'Your brain network shows good overall information transfer efficiency.'
        : 'Overall information transfer efficiency is lower than typical ranges.',
    },
    {
      label: 'Brain Modules',
      description: 'Are brain regions organized into functional teams?',
      status: nCommunities >= 3 && nCommunities <= 10 ? 'good' : 'attention',
      detail: nCommunities >= 3
        ? `Your brain is organized into ${nCommunities} distinct modules, which is typical for healthy brains. These likely correspond to different functional systems (vision, movement, thinking, etc).`
        : 'The community structure shows fewer modules than typically expected.',
    },
    {
      label: 'Network Balance',
      description: 'Does your brain balance local and global processing?',
      status: g.small_worldness > 1.0 ? 'good' : 'attention',
      detail: g.small_worldness > 1.0
        ? 'Your brain has "small-world" organization — it efficiently balances local specialized processing with global communication. This is the hallmark of a well-organized brain.'
        : 'The balance between local and global processing could be improved.',
    },
  ];

  const goodCount = indicators.filter(i => i.status === 'good').length;
  const overallScore = Math.round((goodCount / indicators.length) * 100);
  const overallStatus = overallScore >= 80 ? 'good' : overallScore >= 50 ? 'attention' : 'concern';

  return (
    <div className="space-y-6">
      {/* Overall Score */}
      <div className="glass rounded-xl p-6 text-center border border-white/10">
        <h2 className="text-lg font-semibold mb-4">Brain Network Health Overview</h2>
        <div className="inline-flex items-center justify-center w-32 h-32 rounded-full border-4 mb-4"
          style={{
            borderColor: overallStatus === 'good' ? '#10b981' : overallStatus === 'attention' ? '#f59e0b' : '#ef4444',
          }}
        >
          <div className="text-center">
            <div className="text-3xl font-bold"
              style={{
                color: overallStatus === 'good' ? '#10b981' : overallStatus === 'attention' ? '#f59e0b' : '#ef4444',
              }}
            >
              {overallScore}%
            </div>
            <div className="text-xs text-gray-400">Score</div>
          </div>
        </div>
        <p className="text-gray-300 text-sm max-w-md mx-auto">
          {overallScore >= 80
            ? 'Your brain network shows healthy connectivity patterns across all major indicators.'
            : overallScore >= 50
            ? 'Most brain connectivity indicators are within normal ranges, with some areas to note.'
            : 'Several connectivity indicators may warrant further professional evaluation.'}
        </p>
        <div className="flex justify-center gap-6 mt-4 text-sm">
          <div className="text-gray-400">{nRegions} brain regions analyzed</div>
          <div className="text-gray-400">{nCommunities} functional modules detected</div>
        </div>
      </div>

      {/* Health Indicators */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {indicators.map((indicator) => (
          <div
            key={indicator.label}
            className={`rounded-xl p-5 border ${
              indicator.status === 'good'
                ? 'bg-green-500/5 border-green-500/20'
                : indicator.status === 'attention'
                ? 'bg-yellow-500/5 border-yellow-500/20'
                : 'bg-red-500/5 border-red-500/20'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <StatusDot status={indicator.status} />
              <h3 className="font-semibold text-sm">{indicator.label}</h3>
            </div>
            <p className="text-xs text-gray-400 mb-3">{indicator.description}</p>
            <p className="text-sm text-gray-300">{indicator.detail}</p>
          </div>
        ))}
      </div>

      {/* What This Means */}
      <div className="glass rounded-xl p-6 border border-white/10">
        <h3 className="font-semibold mb-3">What Do These Results Mean?</h3>
        <div className="space-y-3 text-sm text-gray-300">
          <p>
            This analysis looked at how {nRegions} different regions of your brain connect to each other
            through white matter pathways (the brain&apos;s internal wiring).
          </p>
          <p>
            A healthy brain has connections organized like a well-designed city: neighborhoods
            (clusters of closely connected regions) linked by highways (long-range connections)
            that allow fast communication across the entire network.
          </p>
          <p>
            The traffic light indicators above show whether each aspect of your brain&apos;s
            connectivity falls within ranges typically seen in healthy individuals. Green means
            normal, yellow suggests it&apos;s worth noting, and red means it should be discussed
            with a healthcare professional.
          </p>
        </div>
        <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg text-xs text-blue-300">
          Note: This is an automated analysis of brain connectivity structure. It is not a medical
          diagnosis. Always consult a qualified healthcare professional for interpretation of
          neuroimaging results.
        </div>
      </div>
    </div>
  );
}

function StatusDot({ status }: { status: 'good' | 'attention' | 'concern' }) {
  const colors = {
    good: 'bg-green-400',
    attention: 'bg-yellow-400',
    concern: 'bg-red-400',
  };
  return <div className={`w-3 h-3 rounded-full ${colors[status]}`} />;
}

function mapStatus(s: 'normal' | 'elevated' | 'reduced'): 'good' | 'attention' | 'concern' {
  if (s === 'normal') return 'good';
  return 'attention';
}
