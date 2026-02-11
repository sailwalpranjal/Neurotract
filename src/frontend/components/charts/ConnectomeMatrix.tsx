'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { ParcellationLabel } from '@/lib/types';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ConnectomeMatrixProps {
  matrix: number[][];
  labels?: ParcellationLabel[];
}

export default function ConnectomeMatrix({ matrix, labels }: ConnectomeMatrixProps) {
  const [useLogScale, setUseLogScale] = useState(false);
  const [threshold, setThreshold] = useState(0);

  const numNodes = matrix.length;
  const nodeLabels = labels && labels.length === numNodes
    ? labels.map((l) => l.abbreviation)
    : Array.from({ length: numNodes }, (_, i) => `R${i + 1}`);

  const processedMatrix = useMemo(() => {
    return matrix.map((row) =>
      row.map((val) => {
        if (val < threshold) return 0;
        if (useLogScale && val > 0) return Math.log10(val + 1);
        return val;
      })
    );
  }, [matrix, useLogScale, threshold]);

  const maxVal = useMemo(() => {
    let max = 0;
    for (const row of matrix) {
      for (const val of row) {
        if (val > max) max = val;
      }
    }
    return max;
  }, [matrix]);

  return (
    <div>
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
          <input
            type="checkbox"
            checked={useLogScale}
            onChange={(e) => setUseLogScale(e.target.checked)}
            className="rounded"
          />
          Log Scale
        </label>

        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-400">Threshold:</label>
          <input
            type="range"
            min="0"
            max={maxVal}
            step={maxVal / 100}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-32"
          />
          <span className="text-xs text-gray-500 w-16">{threshold.toFixed(1)}</span>
        </div>
      </div>

      <div className="plotly-container">
        <Plot
          data={[
            {
              z: processedMatrix,
              x: nodeLabels,
              y: nodeLabels,
              type: 'heatmap',
              colorscale: 'Viridis',
              showscale: true,
              hoverongaps: false,
              colorbar: {
                title: useLogScale ? 'log10(Strength+1)' : 'Connection<br>Strength',
                titleside: 'right',
              },
            } as any,
          ]}
          layout={{
            title: `Structural Connectome (${numNodes}x${numNodes})`,
            xaxis: {
              title: 'Target Region',
              side: 'bottom',
              tickangle: -45,
              tickfont: { size: 7 },
            },
            yaxis: {
              title: 'Source Region',
              autorange: 'reversed',
              tickfont: { size: 7 },
            },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#fff', size: 10 },
            autosize: true,
            margin: { l: 80, r: 80, t: 60, b: 100 },
          }}
          useResizeHandler
          style={{ width: '100%', height: '600px' }}
          config={{ responsive: true }}
        />
      </div>
    </div>
  );
}
