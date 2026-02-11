'use client';

import { ParcellationLabel } from '@/lib/types';

// Community colors
const COMMUNITY_COLORS = [
  '#0ea5e9', '#e94560', '#10b981', '#f59e0b', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
  '#14b8a6', '#ef4444', '#a855f7', '#22c55e', '#eab308',
];

interface CommunityViewProps {
  partition: number[];
  labels: ParcellationLabel[];
  modularity?: number;
  nodeStrength?: number[];
}

export default function CommunityView({ partition, labels, modularity, nodeStrength }: CommunityViewProps) {
  // Group regions by community
  const communities: Record<number, { index: number; label: ParcellationLabel; strength: number }[]> = {};

  partition.forEach((communityId, idx) => {
    if (!communities[communityId]) communities[communityId] = [];
    communities[communityId].push({
      index: idx,
      label: labels[idx] || { index: idx, generic_name: `parcel_${idx}`, anatomical_name: `Region ${idx}`, abbreviation: `R${idx}`, hemisphere: 'unknown', lobe: 'unknown', description: '' },
      strength: nodeStrength?.[idx] || 0,
    });
  });

  const sortedCommunities = Object.entries(communities).sort(
    (a, b) => b[1].length - a[1].length
  );

  return (
    <div>
      {modularity !== undefined && (
        <div className="mb-4 flex items-center gap-4">
          <div className="text-sm text-gray-400">
            Modularity: <span className="text-white font-semibold">{modularity.toFixed(4)}</span>
          </div>
          <div className="text-sm text-gray-400">
            Communities: <span className="text-white font-semibold">{sortedCommunities.length}</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {sortedCommunities.map(([communityId, members], idx) => (
          <div
            key={communityId}
            className="bg-white/5 rounded-lg p-4 border border-white/10"
          >
            <div className="flex items-center gap-2 mb-3">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: COMMUNITY_COLORS[idx % COMMUNITY_COLORS.length] }}
              />
              <h4 className="font-semibold text-sm">
                Community {Number(communityId) + 1}
              </h4>
              <span className="text-xs text-gray-500 ml-auto">
                {members.length} regions
              </span>
            </div>

            <div className="space-y-1 max-h-40 overflow-y-auto">
              {members
                .sort((a, b) => b.strength - a.strength)
                .map((member) => (
                  <div
                    key={member.index}
                    className="flex items-center justify-between text-xs"
                  >
                    <span
                      className="text-gray-300 truncate"
                      title={member.label.anatomical_name}
                    >
                      {member.label.abbreviation}
                    </span>
                    <span className="text-gray-500 ml-2">
                      {member.label.hemisphere === 'left' ? 'L' : member.label.hemisphere === 'right' ? 'R' : ''}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
