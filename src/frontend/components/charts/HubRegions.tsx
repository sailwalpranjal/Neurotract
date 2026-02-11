'use client';

import dynamic from 'next/dynamic';
import { ParcellationLabel } from '@/lib/types';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface HubRegionsProps {
  degree: number[];
  betweenness: number[];
  labels: ParcellationLabel[];
  topN?: number;
}

export default function HubRegions({ degree, betweenness, labels, topN = 10 }: HubRegionsProps) {
  // Find top N hubs by degree
  const indexed = degree.map((d, i) => ({ index: i, degree: d, betweenness: betweenness[i] || 0 }));
  const topByDegree = [...indexed].sort((a, b) => b.degree - a.degree).slice(0, topN);

  const regionNames = topByDegree.map((h) =>
    labels[h.index]?.abbreviation || `R${h.index + 1}`
  );

  return (
    <div className="grid md:grid-cols-2 gap-6">
      <div className="plotly-container">
        <Plot
          data={[
            {
              y: regionNames.reverse(),
              x: topByDegree.map((h) => h.degree).reverse(),
              type: 'bar',
              orientation: 'h',
              marker: { color: '#0ea5e9' },
              name: 'Degree',
            },
          ]}
          layout={{
            title: `Top ${topN} Hub Regions (by Degree)`,
            xaxis: { title: 'Degree' },
            yaxis: { automargin: true },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#fff', size: 10 },
            autosize: true,
            margin: { l: 100, r: 20, t: 40, b: 40 },
          }}
          useResizeHandler
          style={{ width: '100%', height: '400px' }}
          config={{ responsive: true }}
        />
      </div>

      <div className="plotly-container">
        <Plot
          data={[
            {
              y: regionNames,
              x: topByDegree.map((h) => h.betweenness).reverse(),
              type: 'bar',
              orientation: 'h',
              marker: { color: '#e94560' },
              name: 'Betweenness',
            },
          ]}
          layout={{
            title: `Top ${topN} Hub Regions (by Betweenness)`,
            xaxis: { title: 'Betweenness Centrality' },
            yaxis: { automargin: true },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#fff', size: 10 },
            autosize: true,
            margin: { l: 100, r: 20, t: 40, b: 40 },
          }}
          useResizeHandler
          style={{ width: '100%', height: '400px' }}
          config={{ responsive: true }}
        />
      </div>
    </div>
  );
}
