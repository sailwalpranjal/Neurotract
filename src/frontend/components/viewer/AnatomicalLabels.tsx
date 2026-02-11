'use client';

import { Html } from '@react-three/drei';

// Major brain lobe label positions (approximate centroids in RAS space)
const LOBE_LABELS = [
  { name: 'Frontal Lobe', position: [0, 40, 40] as [number, number, number], color: '#60a5fa' },
  { name: 'Parietal Lobe', position: [0, -30, 55] as [number, number, number], color: '#34d399' },
  { name: 'Temporal Lobe (L)', position: [-50, -5, -10] as [number, number, number], color: '#fbbf24' },
  { name: 'Temporal Lobe (R)', position: [50, -5, -10] as [number, number, number], color: '#fbbf24' },
  { name: 'Occipital Lobe', position: [0, -75, 15] as [number, number, number], color: '#f472b6' },
  { name: 'Cerebellum', position: [0, -55, -30] as [number, number, number], color: '#a78bfa' },
  { name: 'Brain Stem', position: [0, -20, -35] as [number, number, number], color: '#fb923c' },
];

interface AnatomicalLabelsProps {
  lobeCentroids?: Record<string, number[]>;
}

export default function AnatomicalLabels({ lobeCentroids }: AnatomicalLabelsProps) {
  // Use server-provided centroids if available, otherwise defaults
  const labels = lobeCentroids
    ? Object.entries(lobeCentroids).map(([name, pos]) => ({
        name,
        position: pos as [number, number, number],
        color: '#94a3b8',
      }))
    : LOBE_LABELS;

  return (
    <group>
      {labels.map((label) => (
        <Html
          key={label.name}
          position={label.position}
          center
          distanceFactor={300}
          occlude={false}
          style={{ pointerEvents: 'none' }}
        >
          <div
            className="whitespace-nowrap select-none"
            style={{
              background: 'rgba(0,0,0,0.7)',
              color: label.color,
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '11px',
              fontWeight: 600,
              border: `1px solid ${label.color}40`,
              backdropFilter: 'blur(4px)',
            }}
          >
            {label.name}
          </div>
        </Html>
      ))}
    </group>
  );
}
